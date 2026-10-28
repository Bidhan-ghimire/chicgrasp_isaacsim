"""Small offline adapter around the unmodified pinned CNN Diffusion Policy.

Importing this module does not import Torch, official training workspaces, or Isaac.
Only padding-aware loss and project-specific construction live here; the neural
network and diffusion sampler are the official implementations.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np

from .episode_io import validate_policy_arrays
from .interface import POLICY_OBSERVATION_KEYS


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = REPOSITORY_ROOT / "third_party" / "chicgrasp_official"
OFFICIAL_COMMIT = "321a566cfdb070926c4620e8dafca327600920d9"
OFFICIAL_POLICY_PATH = "diffusion_policy/policy/diffusion_unet_image_policy.py"
OFFICIAL_CONFIG_PATH = "diffusion_policy/config/train_diffusion_unet_real_image_workspace.yaml"
OFFICIAL_POLICY_SHA256 = "ff3bf7d91caa14ff7987df620b69a983249424184f92048115baafc2f5a45c66"
OFFICIAL_CONFIG_SHA256 = "3478bfd7401a6838fd9a83cb25cb90a684b1e53a7be4214e87d6f0cd3c56b609"

DEFAULT_POLICY_CONFIG = {
    "shape_meta": {
        "obs": {
            "camera_left_rgb": {"shape": [3, 240, 320], "type": "rgb"},
            "camera_right_rgb": {"shape": [3, 240, 320], "type": "rgb"},
            "arm_joint_pos": {"shape": [6], "type": "low_dim"},
            "jaw_state": {"shape": [2], "type": "low_dim"},
        },
        "action": {"shape": [5]},
    },
    "scheduler": {
        "num_train_timesteps": 100,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "beta_schedule": "squaredcos_cap_v2",
        "clip_sample": True,
        "set_alpha_to_one": True,
        "steps_offset": 0,
        "prediction_type": "epsilon",
    },
    "encoder": {
        "resnet": "resnet18",
        "weights": None,
        "resize_shape": None,
        "crop_shape": None,
        "random_crop": False,
        "use_group_norm": True,
        "share_rgb_model": False,
        "imagenet_norm": True,
    },
    "policy": {
        "horizon": 16,
        "n_obs_steps": 2,
        "n_action_steps": 12,
        "num_inference_steps": 100,
        "obs_as_global_cond": True,
        "diffusion_step_embed_dim": 128,
        "down_dims": [512, 1024, 2048],
        "kernel_size": 5,
        "n_groups": 8,
        "cond_predict_scale": True,
    },
}


def _enable_official_imports() -> None:
    # Bytecode cache creation would write into the immutable copied source.
    sys.dont_write_bytecode = True
    root = str(OFFICIAL_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def official_provenance() -> dict[str, Any]:
    result: dict[str, Any] = {"commit": OFFICIAL_COMMIT, "root": str(OFFICIAL_ROOT)}
    for key, relative, expected in (
        ("policy", OFFICIAL_POLICY_PATH, OFFICIAL_POLICY_SHA256),
        ("cnn_config", OFFICIAL_CONFIG_PATH, OFFICIAL_CONFIG_SHA256),
    ):
        path = OFFICIAL_ROOT / relative
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected:
            raise ValueError(f"Pinned official {key} hash changed: {path}")
        result[key] = {"path": str(path), "byte_count": len(payload), "sha256": digest}
    return result


def set_deterministic(seed: int = 42) -> None:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)


def build_policy(config: Mapping[str, Any] | None = None, device: str = "cpu") -> Any:
    """Construct the exact pinned family, with only the declared local interface."""
    selected = copy.deepcopy(DEFAULT_POLICY_CONFIG if config is None else dict(config))
    if selected != DEFAULT_POLICY_CONFIG:
        raise ValueError("Gate 3 model configuration must match the fixed pinned adapter")
    official_provenance()
    _enable_official_imports()
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusion_policy.model.vision.model_getter import get_resnet
    from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
    from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy

    encoder_config = dict(selected["encoder"])
    resnet_name = encoder_config.pop("resnet")
    weights = encoder_config.pop("weights")
    encoder = MultiImageObsEncoder(
        shape_meta=selected["shape_meta"],
        rgb_model=get_resnet(resnet_name, weights=weights),
        **encoder_config,
    )
    policy = DiffusionUnetImagePolicy(
        shape_meta=selected["shape_meta"],
        noise_scheduler=DDIMScheduler(**selected["scheduler"]),
        obs_encoder=encoder,
        **selected["policy"],
    )
    policy.gate3_config = selected
    return policy.to(device)




def _fit_normalizer_from_validated_policy_rows(checked, *, fit_source):
    """Shared fit for validated episodes and authenticated derived row views."""
    _enable_official_imports()
    import torch
    from diffusion_policy.model.common.normalizer import (
        LinearNormalizer,
        SingleFieldLinearNormalizer,
    )

    range_eps = 1e-4
    normalizer = LinearNormalizer()
    for key in ("action", "arm_joint_pos", "jaw_state"):
        normalizer[key] = SingleFieldLinearNormalizer.create_fit(
            np.array(checked[key], dtype=np.float32, copy=True),
            last_n_dims=1,
            mode="limits",
            range_eps=range_eps,
        )
        if len(checked[key]) == 1:
            # Stock torch.std is unbiased; a one-row descriptive std is zero.
            with torch.no_grad():
                normalizer[key].params_dict["input_stats"]["std"].zero_()
    action_params = normalizer["action"].params_dict
    xyz_min = action_params["input_stats"]["min"][:3]
    xyz_ranges = action_params["input_stats"]["max"][:3] - xyz_min
    # Same strict comparison and float32 ranges as pinned normalizer._fit.
    # Only an ordinary XYZ range may supply the shared scale: a tiny range
    # must never become its own unrestricted min/max amplification source.
    degenerate_xyz = xyz_ranges < range_eps
    ordinary_xyz_ranges = xyz_ranges[~degenerate_xyz]
    reference_range = ordinary_xyz_ranges.max() if ordinary_xyz_ranges.numel() else None
    if bool(degenerate_xyz.any()) and reference_range is not None:
        fallback_scale = 2.0 / reference_range
        if not bool(torch.isfinite(fallback_scale)) or not bool(fallback_scale > 0):
            raise ValueError("Constant XYZ normalization requires a finite positive scale")
        with torch.no_grad():
            action_params["scale"][:3][degenerate_xyz] = fallback_scale
            action_params["offset"][:3][degenerate_xyz] = -xyz_min[degenerate_xyz] * fallback_scale
    for key in ("camera_left_rgb", "camera_right_rgb"):
        image = checked[key]
        stats = {
            "min": np.asarray([float(image.min()) / 255.0], dtype=np.float32),
            "max": np.asarray([float(image.max()) / 255.0], dtype=np.float32),
            "mean": np.asarray([float(image.mean()) / 255.0], dtype=np.float32),
            "std": np.asarray([float(image.std()) / 255.0], dtype=np.float32),
        }
        normalizer[key] = SingleFieldLinearNormalizer.create_manual(
            scale=np.ones(1, dtype=np.float32),
            offset=np.zeros(1, dtype=np.float32),
            input_stats_dict=stats,
        )
    normalizer.requires_grad_(False)
    fields: dict[str, Any] = {}
    for key, params in normalizer.params_dict.items():
        for value in params.values():
            values = value.values() if isinstance(value, torch.nn.ParameterDict) else (value,)
            if not all(bool(torch.isfinite(item).all()) for item in values):
                raise ValueError(f"Nonfinite normalization statistics: {key}")
        fields[key] = {
            "scale": params["scale"].detach().cpu().tolist(),
            "offset": params["offset"].detach().cpu().tolist(),
            "input_stats": {
                name: value.detach().cpu().tolist()
                for name, value in params["input_stats"].items()
            },
        }
    return normalizer, {
        "fit_source": fit_source,
        "row_count": int(len(checked["action"])),
        "mode": "limits",
        "range_eps": range_eps,
        "constant_feature_behavior": (
            "XYZ action range < range_eps: scale=2/largest ordinary XYZ range when available; "
            "otherwise stock scale=1; offset=-minimum*scale; finite inverse"
        ),
        "xyz_branch": {
            "axes": ["x", "y", "z"],
            "input_ranges_m": xyz_ranges.detach().cpu().tolist(),
            "degenerate_mask": degenerate_xyz.detach().cpu().tolist(),
            "comparison": "float32(max-min) < range_eps; pinned stock ignore_dim predicate",
            "reference_range_m": float(reference_range) if reference_range is not None else None,
            "scale": action_params["scale"][:3].detach().cpu().tolist(),
            "offset": action_params["offset"][:3].detach().cpu().tolist(),
            "forward": "normalized = metres * scale + offset",
            "inverse": "metres = (normalized - offset) / scale",
        },
        "fields": fields,
        "rgb": {
            "layout": "NCHW",
            "byte_divisor": 255.0,
            "linear_transform": "identity",
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
            "random_augmentation": False,
            "crop": None,
        },
    }


def validate_model_obs(obs: Mapping[str, Any]) -> int:
    import torch

    if not isinstance(obs, Mapping) or tuple(obs) != POLICY_OBSERVATION_KEYS:
        raise ValueError("Model observations must contain exactly the four policy keys")
    expected = {
        "camera_left_rgb": (2, 3, 240, 320),
        "camera_right_rgb": (2, 3, 240, 320),
        "arm_joint_pos": (2, 6),
        "jaw_state": (2, 2),
    }
    batch_size = None
    for key, value in obs.items():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ValueError(f"{key} must be a float32 tensor")
        if tuple(value.shape[1:]) != expected[key] or value.shape[0] < 1:
            raise ValueError(f"Wrong model observation shape for {key}: {tuple(value.shape)}")
        if batch_size is None:
            batch_size = value.shape[0]
        if value.shape[0] != batch_size or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfinite or unaligned model observation: {key}")
        if key.startswith("camera_") and bool(((value < 0) | (value > 1)).any()):
            raise ValueError("Model RGB must be in [0,1] before ImageNet normalization")
        if key == "jaw_state" and not bool(((value == 0) | (value == 1)).all()):
            raise ValueError("Measured input jaw states must be binary")
    return int(batch_size)


def masked_diffusion_loss(
    policy: Any,
    batch: Mapping[str, Any],
    noise: Any | None = None,
    timesteps: Any | None = None,
) -> Any:
    """Official epsilon MSE; optional deployment mask affects only loss reduction."""
    import torch

    if set(batch) not in ({"obs", "action", "valid_mask"}, {"obs", "action", "valid_mask", "loss_mask"}):
        raise ValueError("Loss batch keys must be obs, action, valid_mask and optional loss_mask")
    batch_size = validate_model_obs(batch["obs"])
    action, valid = batch["action"], batch["valid_mask"]
    if action.shape != (batch_size, 16, 5) or action.dtype != torch.float32:
        raise ValueError("Action targets must have float32 shape [B,16,5]")
    if valid.shape != (batch_size, 16) or valid.dtype != torch.bool:
        raise ValueError("valid_mask must have bool shape [B,16]")
    if not bool(valid.any(dim=1).all()) or not bool(torch.isfinite(action[valid]).all()):
        raise ValueError("Every window needs finite valid action targets")
    reduction_valid = batch.get("loss_mask", valid)
    if reduction_valid.shape != valid.shape or reduction_valid.dtype != torch.bool:
        raise ValueError("loss_mask must have bool shape [B,16]")
    if bool((reduction_valid & ~valid).any()):
        raise ValueError("loss_mask cannot include padded slots")
    nobs = policy.normalizer.normalize(dict(batch["obs"]))
    clean = torch.where(valid.unsqueeze(-1), action, torch.zeros_like(action))
    normalized = policy.normalizer["action"].normalize(clean)
    trajectory = torch.where(valid.unsqueeze(-1), normalized, torch.zeros_like(normalized))
    features = policy.obs_encoder({
        key: value[:, :2].reshape(-1, *value.shape[2:]) for key, value in nobs.items()
    })
    global_cond = features.reshape(batch_size, -1)
    condition_mask = policy.mask_generator(trajectory.shape).to(trajectory.device)
    if condition_mask.shape != trajectory.shape or condition_mask.dtype != torch.bool:
        raise ValueError("Official condition mask shape/dtype changed")
    if noise is None:
        noise = torch.randn(trajectory.shape, device=trajectory.device, dtype=trajectory.dtype)
    if noise.shape != trajectory.shape or not bool(torch.isfinite(noise).all()):
        raise ValueError("Diffusion noise must be finite with shape [B,16,5]")
    if timesteps is None:
        timesteps = torch.randint(
            0, policy.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=trajectory.device, dtype=torch.long,
        )
    if timesteps.shape != (batch_size,) or timesteps.dtype != torch.long:
        raise ValueError("Diffusion timesteps must be int64 with shape [B]")
    if bool(((timesteps < 0) | (timesteps >= policy.noise_scheduler.config.num_train_timesteps)).any()):
        raise ValueError("Diffusion timestep outside scheduler range")
    noisy = policy.noise_scheduler.add_noise(trajectory, noise, timesteps)
    noisy = torch.where(condition_mask, trajectory, noisy)
    predicted = policy.model(noisy, timesteps, local_cond=None, global_cond=global_cond)
    if predicted.shape != trajectory.shape or not bool(torch.isfinite(predicted).all()):
        raise ValueError("The official UNet produced invalid/nonfinite values")
    if policy.noise_scheduler.config.prediction_type != "epsilon":
        raise ValueError("The pinned adapter requires epsilon prediction")
    loss_mask = reduction_valid.unsqueeze(-1).expand_as(predicted) & ~condition_mask
    if bool(reduction_valid.any()) and not bool(loss_mask.any()):
        raise ValueError("No valid unconditioned loss targets")
    squared_error = (predicted.float() - noise.float()).square()
    selected_error = squared_error[loss_mask]
    # Keep empty-loss microbatches in the unchanged forward/backward schedule.
    loss = selected_error.mean() if selected_error.numel() else selected_error.sum()
    if not bool(torch.isfinite(loss)):
        raise ValueError("Masked diffusion loss is nonfinite")
    return loss


def predict_windows(
    policy: Any,
    obs: Mapping[str, Any],
    seed: int = 12345,
    microbatch_size: int = 2,
) -> np.ndarray:
    """Fixed-seed official DDIM sampling of full, observation-aligned horizons."""
    import torch

    count = validate_model_obs(obs)
    if type(microbatch_size) is not int or microbatch_size < 1:
        raise ValueError("microbatch_size must be positive")
    device = next(iter(obs.values())).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    was_training = policy.training
    outputs = []
    try:
        policy.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            for start in range(0, count, microbatch_size):
                torch.manual_seed(seed + start)
                if device.type == "cuda":
                    torch.cuda.manual_seed(seed + start)
                subset = {key: value[start:start + microbatch_size] for key, value in obs.items()}
                prediction = policy.predict_action(subset)["action_pred"]
                if prediction.shape != (len(subset["jaw_state"]), 16, 5):
                    raise ValueError("Official full action prediction must have shape [B,16,5]")
                if not bool(torch.isfinite(prediction).all()):
                    raise ValueError("Nonfinite diffusion evaluation prediction")
                outputs.append(prediction.detach().cpu().numpy().copy())
    finally:
        policy.train(was_training)
    return np.concatenate(outputs, axis=0)


def save_checkpoint(path: str | Path, policy: Any, metadata: Mapping[str, Any]) -> None:
    """Publish once; save errors propagate rather than becoming a false pass."""
    import torch

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    clean_metadata = json.loads(json.dumps(dict(metadata), allow_nan=False))
    config = copy.deepcopy(policy.gate3_config)
    if config != DEFAULT_POLICY_CONFIG:
        raise ValueError("Checkpoint architecture configuration differs from fixed adapter")
    state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
    if not all(bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("Refusing to save nonfinite model weights/statistics")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        torch.save({
            "format_version": 1,
            "model_config": config,
            "model_state": state,
            "metadata": clean_metadata,
        }, stream)
        stream.flush()
        os.fsync(stream.fileno())


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    import torch

    with Path(path).open("rb") as stream:
        saved = torch.load(stream, map_location="cpu", weights_only=True)
    if set(saved) != {"format_version", "model_config", "model_state", "metadata"} or saved["format_version"] != 1:
        raise ValueError("Unknown checkpoint schema")
    metadata = json.loads(json.dumps(saved["metadata"], allow_nan=False))
    if not all(bool(torch.isfinite(value).all()) for value in saved["model_state"].values()):
        raise ValueError("Checkpoint contains nonfinite values")
    policy = build_policy(saved["model_config"], device="cpu")
    policy.load_state_dict(saved["model_state"], strict=True)
    policy.to(device)
    policy.eval()
    return policy, metadata
