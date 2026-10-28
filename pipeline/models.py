"""Lazy model registry. Import/inspection never constructs a model or imports Torch."""
from __future__ import annotations


import importlib
import json
from pathlib import Path

from pipeline.training import CONTRACT, consistent_scene_identity, identity, validate_scene_identity

ADAPTERS = {"diffusion": "pipeline.models:Adapter", "bc_cnn": "pipeline.models:Adapter"}


def _adapter_class(model_type):
    if model_type not in ADAPTERS:
        raise ValueError("Unknown model type: " + str(model_type))
    module, name = ADAPTERS[model_type].split(":", 1)
    return getattr(importlib.import_module(module), name)


def create_adapter(model_type, *, normalizer, config, device, updates, seed):
    return _adapter_class(model_type)(model_type, normalizer=normalizer, config=config,
                                     device=device, updates=updates, seed=seed)


def resume_adapter(model_type, checkpoint_path, *, expected_metadata, device):
    """Explicit training recovery; never substitute the EMA inference loader."""
    cls = _adapter_class(model_type)
    method = getattr(cls, "resume", None)
    if not callable(method):
        raise ValueError("Model adapter does not support exact training resume: " + model_type)
    return method(checkpoint_path, expected_metadata=expected_metadata, device=device)


def validate_artifact_scene_identity(value):
    """Require the saved scene identity to match every admitted source episode.

    This is metadata-only. Never read today's setup or supply a missing identity.
    """
    scene = validate_scene_identity(value.get("scene_setup_identity"))
    identities = value.get("identities")
    if not isinstance(identities, dict) or not isinstance(identities.get("episodes"), list):
        raise ValueError("Model artifact requires source episode scene identities")
    source_scene = consistent_scene_identity(identities["episodes"])
    if scene != source_scene:
        raise ValueError("Model artifact scene identity differs from admitted source episodes")
    return scene


def inspect_checkpoint(checkpoint_path):
    """Authenticate adjacent artifact.json and bytes; never deserialize checkpoint state."""
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    sidecar = checkpoint.parent / "artifact.json"
    value = json.loads(sidecar.read_text(encoding="utf-8"))
    if value.get("schema") != "research-model-artifact/v1" or value.get("model_type") not in ADAPTERS:
        raise ValueError("Unknown research model artifact")
    if value.get("policy_contract") != CONTRACT:
        raise ValueError("Policy contract mismatch")
    validate_artifact_scene_identity(value)
    config = value.get("model_config")
    start = value.get("training_window_start")
    if not isinstance(config, dict) or type(start) is not int or start < 0 or config.get("training_window_start") != start:
        raise ValueError("Artifact/config training window mismatch")
    if value.get("live_compatible") != (start == 50) or value.get("checkpoint_selection") != "FINAL_EMA":
        raise ValueError("Unknown research artifact selection or compatibility declaration")
    if type(value.get("completed_updates")) is not int or value["completed_updates"] < 1:
        raise ValueError("Artifact requires completed optimizer updates")
    if not isinstance(value.get("script_options"), dict):
        raise ValueError("Artifact must bind the selected script options")
    if value.get("checkpoint_path") != checkpoint.name or identity(checkpoint) != value.get("checkpoint"):
        raise ValueError("Checkpoint identity mismatch")
    relative = Path(value["normalizer_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Normalizer must be an artifact-relative file")
    normalizer = (checkpoint.parent / relative).resolve(strict=True)
    if not normalizer.is_relative_to(checkpoint.parent) or identity(normalizer) != value["normalizer"]:
        raise ValueError("Normalizer identity mismatch")
    return dict(value, resolved_checkpoint_path=str(checkpoint), resolved_normalizer_path=str(normalizer))


def load_predictor(checkpoint_path, *, device="cuda:0", evaluation_seed=15345):
    """Explicit runtime-only load; returns (two raw observations, chunk_index)->[16,5]."""
    metadata = inspect_checkpoint(checkpoint_path)
    if metadata["training_window_start"] != 50:
        raise ValueError("Native V3C integration currently requires training_window_start=50")
    if type(evaluation_seed) is not int or evaluation_seed < 0:
        raise ValueError("Nonnegative evaluation seed required")
    adapter = _adapter_class(metadata["model_type"]).load(checkpoint_path, metadata=metadata, device=device)

    def predict(history, chunk_index):
        if type(chunk_index) is not int or chunk_index < 0:
            raise ValueError("Nonnegative chunk index required")
        import numpy as np
        result = np.asarray(adapter.predict(history, seed=evaluation_seed + 12 * chunk_index))
        if result.shape != (16, 5) or result.dtype.kind != "f" or not np.isfinite(result).all():
            raise ValueError("Model adapter must return finite floating full16x5 actions")
        return result.astype(np.float32, copy=True)

    predict.metadata = metadata
    predict.reset = adapter.reset
    return predict


"""Model execution adapter; Torch imports occur only in explicit runtime calls."""


import copy
import json
import math
import os
from pathlib import Path
import random

import numpy as np

from pipeline.training import CONTRACT, frozen_module, identity, pack_history


def validate_checkpoint_scene_identity(saved, artifact):
    """Compare stored plan/source bindings before any model is constructed.

    A historical checkpoint without this research-scene identity is rejected;
    the current scene or a sidecar alone cannot retroactively establish it.
    """
    artifact_scene = validate_artifact_scene_identity(artifact)
    metadata = saved.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("plan"), dict):
        raise ValueError("Checkpoint requires a saved plan with scene identity")
    checkpoint_scene = validate_artifact_scene_identity({
        "scene_setup_identity": metadata["plan"].get("scene_setup_identity"),
        "identities": metadata.get("identities"),
    })
    if checkpoint_scene != artifact_scene:
        raise ValueError("Checkpoint saved-plan scene identity differs from artifact")
    return checkpoint_scene


def restore_normalizer(metadata):
    api = frozen_module("overfit_policy")
    api._enable_official_imports()
    from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
    normalizer = LinearNormalizer()
    for name, field in metadata["fields"].items():
        normalizer[name] = SingleFieldLinearNormalizer.create_manual(
            np.asarray(field["scale"], dtype=np.float32), np.asarray(field["offset"], dtype=np.float32),
            {key: np.asarray(value, dtype=np.float32) for key, value in field["input_stats"].items()})
    normalizer.requires_grad_(False)
    return normalizer


def _recovery_payload(saved, expected, torch, device):
    """Validate all bindings and bookkeeping before constructing a model."""
    required = {"schema", "model_type", "model_config", "policy_contract", "normalization",
                "completed_updates", "planned_updates", "seed", "ema_state", "metadata", "recovery",
                "model_state", "optimizer_state", "scheduler_state", "ema_control", "rng"}
    if not isinstance(saved, dict) or not required <= saved.keys():
        raise ValueError("Incomplete recovery checkpoint")
    if (saved["schema"] != "research-policy-checkpoint/v1" or saved["recovery"] is not True
            or saved["policy_contract"] != CONTRACT):
        raise ValueError("Resume requires a complete research training recovery checkpoint")
    plan = expected["plan"]
    metadata = saved["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("Missing recovery metadata")
    for key in ("plan", "identities", "software_versions"):
        if key not in metadata or metadata[key] != expected[key]:
            raise ValueError("Recovery binding differs: " + key)
    scene_view = {"scene_setup_identity": plan.get("scene_setup_identity"), "identities": expected["identities"]}
    validate_artifact_scene_identity(scene_view)
    if (saved["model_type"] != plan["model_type"] or saved["model_config"] != plan["config"]
            or saved["planned_updates"] != plan["updates"] or saved["seed"] != plan["seed"]
            or plan["device"] != device or saved["normalization"] != expected["normalization"]):
        raise ValueError("Recovery model, total, seed, device, recipe or normalization changed")
    step = saved["completed_updates"]
    if type(step) is not int or not 0 <= step <= plan["updates"]:
        raise ValueError("Invalid recovery completed-update count")

    def finite(value):
        if isinstance(value, torch.Tensor):
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Nonfinite recovery tensor")
        elif isinstance(value, dict):
            for item in value.values(): finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value: finite(item)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite recovery scalar")
    finite(saved)
    for key in ("model_state", "ema_state"):
        if not isinstance(saved[key], dict) or not saved[key] or not all(isinstance(v, torch.Tensor) for v in saved[key].values()):
            raise ValueError("Invalid recovery weights: " + key)
    optimizer, schedule, control = (saved[k] for k in ("optimizer_state", "scheduler_state", "ema_control"))
    if not all(isinstance(v, dict) for v in (optimizer, schedule, control)):
        raise ValueError("Invalid optimizer, scheduler or EMA bookkeeping")
    groups = optimizer.get("param_groups")
    states = optimizer.get("state")
    if not isinstance(groups, list) or not groups or not isinstance(states, dict) or (step and not states):
        raise ValueError("Incomplete optimizer state")
    if (control.get("optimization_step") != step or schedule.get("last_epoch") != step
            or schedule.get("_step_count") != step + 1
            or schedule.get("_last_lr") != [group.get("lr") for group in groups]):
        raise ValueError("Recovery EMA/scheduler/learning-rate counters disagree")
    for state in states.values():
        if not isinstance(state, dict) or float(state.get("step", -1)) != step or not {"exp_avg", "exp_avg_sq"} <= state.keys():
            raise ValueError("Recovery optimizer moments or step disagree")
    rng = saved["rng"]
    if not isinstance(rng, dict) or set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("Incomplete recovery RNG")
    try:
        random.Random(0).setstate(rng["python"])
        values = rng["numpy"]
        if not isinstance(values, (list, tuple)) or len(values) != 5:
            raise ValueError("Invalid NumPy global state")
        numpy_state = (values[0], np.asarray(values[1], dtype=np.uint32), *values[2:])
        np.random.RandomState(0).set_state(numpy_state)
        for state in [rng["torch_cpu"], *rng["torch_cuda"]]:
            if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1 or state.numel() < 1:
                raise ValueError("Invalid Torch RNG tensor")
        torch.Generator(device="cpu").set_state(rng["torch_cpu"])
        cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if not isinstance(rng["torch_cuda"], list) or len(rng["torch_cuda"]) != cuda_count:
            raise ValueError("Recovery CUDA generator count changed")
        sampler = copy.deepcopy(metadata["sampler_state"])
        trial = np.random.default_rng(0)
        trial.bit_generator.state = sampler
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("Invalid recovery random state: " + str(exc)) from exc
    return {"completed_updates": step, "sampler_state": sampler, "numpy_state": numpy_state}


def _bc_model(normalizer):
    import torch
    from torch import nn

    class BehaviorCloning(nn.Module):
        """Two-view, two-frame CNN baseline; fixed recipe, no pretrained weights."""
        def __init__(self):
            super().__init__()
            self.normalizer = normalizer
            def cnn():
                return nn.Sequential(nn.Conv2d(3, 16, 5, 4, 2), nn.ReLU(),
                                     nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                                     nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                                     nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
            self.left = cnn()
            self.right = cnn()
            self.head = nn.Sequential(nn.Linear(4 * 64 + 2 * 8, 256), nn.ReLU(), nn.Linear(256, 16 * 5))

        def forward(self, obs):
            normalized = self.normalizer.normalize(dict(obs))
            batch = obs["jaw_state"].shape[0]
            features = []
            for key, encoder in (("camera_left_rgb", self.left), ("camera_right_rgb", self.right)):
                rgb = normalized[key].reshape(batch * 2, 3, 240, 320)
                mean = rgb.new_tensor([.485, .456, .406])[None, :, None, None]
                std = rgb.new_tensor([.229, .224, .225])[None, :, None, None]
                features.append(encoder((rgb - mean) / std).reshape(batch, -1))
            features.extend(normalized[key].reshape(batch, -1) for key in ("arm_joint_pos", "jaw_state"))
            raw = self.head(torch.cat(features, dim=-1)).reshape(batch, 16, 5)
            # Positions are learned in affine-normalized units; jaws are independent logits.
            xyz = self.normalizer["action"].unnormalize(torch.cat([raw[..., :3], torch.zeros_like(raw[..., 3:])], -1))[..., :3]
            return torch.cat([xyz, torch.sigmoid(raw[..., 3:])], -1), raw[..., 3:]

    return BehaviorCloning()


class Adapter:
    def __init__(self, model_type, *, normalizer, config, device, updates, seed, training=True):
        if config.get("model_options"):
            raise ValueError("The two baseline adapters have fixed architecture; model_options must be empty")
        import torch
        self.torch = torch
        self.model_type = model_type
        self.normalizer_metadata = copy.deepcopy(normalizer)
        self.config = copy.deepcopy(config)
        self.device = torch.device(device)
        self.updates = updates
        self.seed = seed
        self.completed = 0
        self.api = frozen_module("overfit_policy")
        self.api.set_deterministic(seed)
        norm = restore_normalizer(normalizer)
        self.constructor_receipts = []
        if model_type == "diffusion":
            if self.device.type == "cuda":
                if self.device.index not in (None, 0):
                    raise ValueError("The preserved CUDA constructor supports cuda:0 only")
                with frozen_module("exposure_cuda").cuda_constructor(self.constructor_receipts):
                    self.model = self.api.build_policy(device="cuda:0")
            else:
                self.model = self.api.build_policy(device=device)
            self.model.set_normalizer(norm)
        elif model_type == "bc_cnn":
            self.model = _bc_model(norm)
        else:
            raise ValueError("Unknown model type")
        self.model.to(self.device)
        self.averaged = copy.deepcopy(self.model) if training else self.model
        self.averaged.eval()
        self.ema = self.optimizer = self.scheduler = None
        if training:
            from diffusion_policy.model.diffusion.ema_model import EMAModel
            from diffusion_policy.model.common.lr_scheduler import get_scheduler
            self.ema = EMAModel(self.averaged, inv_gamma=1., power=.75, min_value=0., max_value=.9999, update_after_step=0)
            self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                lr=config["learning_rate"], betas=(.95, .999), eps=1e-8, weight_decay=1e-6, foreach=False)
            self.scheduler = get_scheduler("cosine", self.optimizer,
                num_warmup_steps=config["warmup_steps"], num_training_steps=updates)

    def reset(self):
        """History is supplied by the scheduler; these adapters have no hidden recurrent state."""
        return None

    def train_step(self, batch):
        from contextlib import nullcontext
        torch = self.torch
        if self.optimizer is None:
            raise ValueError("Inference-only adapter cannot train")
        self.api.validate_model_obs({k: torch.from_numpy(v) for k, v in batch["obs"].items()})
        tensors = {"obs": {k: torch.as_tensor(v, device=self.device) for k, v in batch["obs"].items()},
                   **{k: torch.as_tensor(batch[k], device=self.device) for k in ("action", "valid_mask", "loss_mask")}}
        count = len(tensors["action"])
        if count < 1 or bool((tensors["loss_mask"] & ~tensors["valid_mask"]).any()):
            raise ValueError("Invalid training mask")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.
        lr_used = float(self.optimizer.param_groups[0]["lr"])
        for i in range(count):
            sample = {"obs": {k: v[i:i+1] for k, v in tensors["obs"].items()},
                      **{k: tensors[k][i:i+1] for k in ("action", "valid_mask", "loss_mask")}}
            mask = sample["loss_mask"]
            if not bool(mask.any()):
                raise ValueError("Only supervised windows may be sampled")
            precision = torch.autocast("cuda", dtype=torch.bfloat16) if self.config["precision"] == "bf16" else nullcontext()
            with precision:
                if self.model_type == "diffusion":
                    loss = self.api.masked_diffusion_loss(self.model, sample)
                else:
                    prediction, jaw_logits = self.model(sample["obs"])
                    normalized_prediction = self.model.normalizer["action"].normalize(prediction)
                    normalized_target = self.model.normalizer["action"].normalize(sample["action"])
                    xyz = (normalized_prediction[..., :3].float() - normalized_target[..., :3].float()).square()[mask].mean()
                    jaws = torch.nn.functional.binary_cross_entropy_with_logits(jaw_logits.float()[mask], sample["action"][..., 3:][mask])
                    loss = xyz + jaws
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite training loss")
            (loss / count).backward()
            total += float(loss.detach()) / count
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise ValueError("Absent or nonfinite gradients")
        self.optimizer.step()
        self.scheduler.step()
        self.ema.step(self.model)
        self.optimizer.zero_grad(set_to_none=True)
        self.completed += 1
        return {"loss": total, "lr_used": lr_used, "lr_next": float(self.optimizer.param_groups[0]["lr"]),
                "optimizer_updates": self.completed, "scheduler_updates": self.scheduler.last_epoch,
                "ema_updates": self.ema.optimization_step, "loss_kind": "epsilon_mse" if self.model_type == "diffusion" else "normalized_xyz_mse_plus_jaw_bce"}

    def predict(self, history, *, seed=12345):
        torch = self.torch
        obs = {k: torch.as_tensor(v, device=self.device) for k, v in pack_history(history).items()}
        self.averaged.eval()
        if self.model_type == "diffusion":
            value = self.api.predict_windows(self.averaged, obs, seed=seed, microbatch_size=1)[0]
        else:
            with torch.no_grad():
                value = self.averaged(obs)[0][0].detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (16, 5) or not np.isfinite(value).all():
            raise ValueError("Invalid full prediction")
        return value.copy()

    def save(self, path, metadata, *, recovery=False):
        torch = self.torch
        def cpu_tree(value):
            if isinstance(value, torch.Tensor):
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("Nonfinite checkpoint tensor")
                return value.detach().cpu().clone()
            if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
            if isinstance(value, list): return [cpu_tree(v) for v in value]
            if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
            return value
        saved = {"schema": "research-policy-checkpoint/v1", "model_type": self.model_type,
                 "model_config": self.config, "policy_contract": CONTRACT, "normalization": self.normalizer_metadata,
                 "completed_updates": self.completed, "planned_updates": self.updates, "seed": self.seed,
                 "ema_state": cpu_tree(self.averaged.state_dict()), "metadata": metadata, "recovery": recovery}
        if recovery:
            state = np.random.get_state()
            saved.update(model_state=cpu_tree(self.model.state_dict()), optimizer_state=cpu_tree(self.optimizer.state_dict()),
                         scheduler_state=self.scheduler.state_dict(),
                         ema_control={k: v for k, v in vars(self.ema).items() if k != "averaged_model"},
                         rng={"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
                              "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})
        path = Path(path)
        pending = path.with_name(path.name + ".pending")
        with pending.open("xb") as stream:
            torch.save(saved, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)

    @classmethod
    def resume(cls, path, *, expected_metadata, device="cuda:0"):
        """Restore a training continuation; preserve the separate EMA-only loader."""
        import torch
        if identity(path) != expected_metadata["checkpoint"]:
            raise ValueError("Recovery checkpoint bytes differ from the planned receipt")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if identity(path) != expected_metadata["checkpoint"]:
            raise ValueError("Recovery checkpoint changed during loading")
        info = _recovery_payload(saved, expected_metadata, torch, device)
        adapter = cls(saved["model_type"], normalizer=saved["normalization"], config=saved["model_config"],
                      device=device, updates=saved["planned_updates"], seed=saved["seed"], training=True)
        control = saved["ema_control"]
        current = {k: v for k, v in vars(adapter.ema).items() if k != "averaged_model"}
        if set(control) != set(current) or any(control[k] != current[k] for k in current if k not in ("optimization_step", "decay")):
            raise ValueError("EMA recovery recipe changed")
        if adapter.ema.averaged_model is not adapter.averaged:
            raise ValueError("EMA target changed")
        groups = saved["optimizer_state"]["param_groups"]
        if len(groups) != len(adapter.optimizer.param_groups):
            raise ValueError("Optimizer recovery groups changed")
        for stored, active in zip(groups, adapter.optimizer.param_groups):
            for key in ("betas", "eps", "weight_decay", "foreach", "initial_lr"):
                if stored.get(key) != active.get(key):
                    raise ValueError("Optimizer recovery recipe changed: " + key)
        normalizer = {k: v for k, v in adapter.model.state_dict().items() if k.startswith("normalizer.")}
        if not normalizer:
            raise ValueError("Recovery normalizer tensors missing")
        for key, value in normalizer.items():
            for weights in (saved["model_state"], saved["ema_state"]):
                if key not in weights or not torch.equal(value.detach().cpu(), weights[key]):
                    raise ValueError("Recovery normalizer tensors differ")
        adapter.model.load_state_dict(saved["model_state"], strict=True)
        adapter.averaged.load_state_dict(saved["ema_state"], strict=True)
        # Normalizer state loading can create CPU parameters; restore placement
        # before optimizer loading, without changing dtype or parameter order.
        for model in (adapter.model, adapter.averaged):
            model.to(device=device)
            if not all(value.device == adapter.device for value in (*model.parameters(), *model.buffers())):
                raise ValueError("Restored tensors have mixed devices")
        adapter.optimizer.load_state_dict(saved["optimizer_state"])
        adapter.scheduler.load_state_dict(saved["scheduler_state"])
        for key, value in control.items(): setattr(adapter.ema, key, value)
        adapter.model.train()
        adapter.averaged.eval()
        adapter.optimizer.zero_grad(set_to_none=True)
        adapter.completed = info["completed_updates"]
        # Constructor and state placement can consume random values. Restore
        # every live generator only after those operations have finished.
        random.setstate(saved["rng"]["python"])
        np.random.set_state(info.pop("numpy_state"))
        torch.set_rng_state(saved["rng"]["torch_cpu"])
        if saved["rng"]["torch_cuda"]:
            torch.cuda.set_rng_state_all(saved["rng"]["torch_cuda"])
        return adapter, info

    @classmethod
    def load(cls, path, *, metadata, device="cuda:0"):
        import torch
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("schema") != "research-policy-checkpoint/v1" or saved["policy_contract"] != CONTRACT:
            raise ValueError("Unknown checkpoint contract")
        validate_checkpoint_scene_identity(saved, metadata)
        for key in ("model_type", "model_config"):
            if saved[key] != metadata[key]:
                raise ValueError("Checkpoint/sidecar mismatch: " + key)
        if saved["completed_updates"] != metadata["completed_updates"] or saved["metadata"]["plan"]["config"] != saved["model_config"]:
            raise ValueError("Checkpoint/sidecar update or recipe mismatch")
        if saved["metadata"]["plan"]["script_options"] != metadata["script_options"]:
            raise ValueError("Checkpoint/sidecar script options mismatch")
        if not all(bool(torch.isfinite(value).all()) for value in saved["ema_state"].values()):
            raise ValueError("Nonfinite checkpoint weights")
        expected = json.loads(Path(metadata["resolved_normalizer_path"]).read_text(encoding="utf-8"))
        if saved["normalization"] != expected or saved["metadata"]["identities"] != metadata["identities"]:
            raise ValueError("Checkpoint normalization or dataset identity mismatch")
        adapter = cls(saved["model_type"], normalizer=expected, config=saved["model_config"], device=device,
                      updates=saved["planned_updates"], seed=saved["seed"], training=False)
        adapter.averaged.load_state_dict(saved["ema_state"], strict=True)
        adapter.averaged.to(device)
        adapter.averaged.eval()
        adapter.completed = saved["completed_updates"]
        return adapter
