"""Research episode loading and alignment; NumPy only, never model construction."""
from __future__ import annotations


import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np

OBS_KEYS = ("camera_left_rgb", "camera_right_rgb", "arm_joint_pos", "jaw_state")
CONTRACT = {"schema": "chicgrasp-policy-contract/v1", "observation_keys": list(OBS_KEYS),
            "image_shape": [240, 320, 3], "arm_joint_count": 6, "jaw_state": "commanded-jaw-state/v1",
            "action_names": ["x", "y", "z", "gL", "gR"], "position_frame": "robot_base",
            "position_units": "m", "open_bit": 1, "closed_bit": 0, "policy_hz": 10,
            "history": 2, "prediction_horizon": 16, "execution_slice": [1, 13]}


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def frozen_module(name):
    root = Path(__file__).resolve().parents[1] / "frozen_runtime" / "src"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sys.dont_write_bytecode = True
    return importlib.import_module("chicgrasp_policy." + name)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _file_record(value):
    if (not isinstance(value, dict) or not isinstance(value.get("path"), str) or not value["path"]
            or type(value.get("bytes")) is not int or value["bytes"] < 0 or not _sha(value.get("sha256"))):
        raise ValueError("File identity requires path, byte count and SHA256")
    return value


def validate_scene_identity(value):
    """Validate saved metadata only; never read or execute today's editable scene."""
    required = {"schema", "base_scene_sha256", "source", "options", "definition", "assets", "bridge_files"}
    if not isinstance(value, dict) or not required <= set(value) or value.get("schema") != "research-scene-setup/v1":
        raise ValueError("Missing or invalid research scene_setup_identity; legacy provenance is not upgraded")
    if not _sha(value["base_scene_sha256"]) or not isinstance(value["options"], dict):
        raise ValueError("Scene identity requires base hash and options object")
    definition = value["definition"]
    if (not isinstance(definition, dict) or not isinstance(definition.get("id"), str) or not definition["id"]
            or definition.get("kind") not in ("table", "custom")):
        raise ValueError("Invalid scene definition identity")
    files = [_file_record(value["source"])]
    for key in ("assets", "bridge_files"):
        if not isinstance(value[key], list):
            raise ValueError("Scene asset and bridge identities must be lists")
        files.extend(_file_record(record) for record in value[key])
    for record in files:
        relative = Path(record["path"])
        if relative.is_absolute() or relative.drive or ".." in relative.parts or "\\" in record["path"]:
            raise ValueError("Scene identity paths must be repository-relative POSIX paths")
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def consistent_scene_identity(records):
    """One exact scene across complete train/validation rows; tests are excluded."""
    values = set()
    for row in records:
        if row.get("complete", True) is not True or row.get("split") not in ("train", "validation"):
            continue
        value = validate_scene_identity(row.get("scene_setup_identity"))
        if row.get("scene_sha256") != value["base_scene_sha256"]:
            raise ValueError("Episode base scene hash disagrees with its scene_setup_identity")
        values.add(json.dumps(value, sort_keys=True, allow_nan=False))
    if len(values) != 1:
        raise ValueError("Missing or mixed scene_setup_identity requires a separate explicit dataset")
    return json.loads(next(iter(values)))


def verify_episode_binding(episode_dir, record, scene, *, base_dir):
    """Authenticate the sibling receipt and its exact scene/episode manifest."""
    record = _file_record(record)
    episode_dir, base_dir = Path(episode_dir).resolve(), Path(base_dir).resolve()
    candidate = Path(record["path"])
    path = (candidate if candidate.is_absolute() else base_dir / candidate).resolve()
    if (not path.is_relative_to(base_dir) or not episode_dir.is_relative_to(base_dir)
            or path != episode_dir.parent / "episode_binding.json"):
        raise ValueError("Episode binding must be the contained sibling episode_binding.json")
    actual = identity(path)
    if actual != {key:record[key] for key in ("bytes", "sha256")}:
        raise ValueError("Episode binding byte identity mismatch")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("schema") != "research-episode-binding/v1":
        raise ValueError("Unknown episode binding schema")
    if validate_scene_identity(saved.get("scene_setup_identity")) != validate_scene_identity(scene):
        raise ValueError("Episode binding scene mismatch")
    manifest = identity(episode_dir / "manifest.json")
    if saved.get("episode_manifest") != manifest:
        raise ValueError("Episode binding manifest mismatch")
    return dict(path=str(path), **actual, episode_manifest=manifest)


def read_manifest(path):
    """Read metadata only. Test episode files are never opened by this module."""
    path = Path(path).resolve(strict=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "research-dataset/v1" or not isinstance(data.get("episodes"), list):
        raise ValueError("Expected research-dataset/v1 episodes list")
    seen = set()
    records = []
    for row in data["episodes"]:
        required = {"id", "path", "split", "position_id", "script_sha256", "scene_sha256", "complete"}
        if not isinstance(row, dict) or not required <= set(row):
            raise ValueError("Incomplete episode manifest record")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in seen:
            raise ValueError("Episode IDs must be nonempty and unique")
        seen.add(row["id"])
        if row["split"] not in {"train", "validation", "test"} or type(row["complete"]) is not bool:
            raise ValueError("Invalid split or completion flag")
        if not _sha(row["script_sha256"]) or not _sha(row["scene_sha256"]):
            raise ValueError("Episode script/scene SHA256 required")
        options = row.get("script_options", {})
        if not isinstance(options, dict):
            raise ValueError("script_options must be an object")
        json.dumps(options, sort_keys=True, allow_nan=False)
        if "commands_sha256" in row and not _sha(row["commands_sha256"]):
            raise ValueError("Invalid recorded command SHA256")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or not row["path"]:
            raise ValueError("Episode path must be relative and contained")
        resolved = (path.parent / relative).resolve()
        if not resolved.is_relative_to(path.parent):
            raise ValueError("Episode path escapes dataset directory")
        if row["complete"] and row["split"] in ("train", "validation"):
            validate_scene_identity(row.get("scene_setup_identity"))
            binding = _file_record(row.get("episode_binding"))
            binding_path = Path(binding["path"])
            if binding_path.is_absolute() or binding_path.drive or ".." in binding_path.parts:
                raise ValueError("Episode binding path must be relative and contained")
            if (path.parent / binding_path).resolve() != resolved.parent / "episode_binding.json":
                raise ValueError("Episode binding must be adjacent to the episode directory")
        records.append(dict(row, script_options=options, resolved_path=str(resolved)))
    if not any(r["split"] == "train" and r["complete"] for r in records):
        raise ValueError("At least one complete training episode required")
    consistent_scene_identity(records)
    return data, records


def consistent_script_options(records):
    """One live setup configuration per research artifact; test splits are irrelevant."""
    options = {json.dumps(row.get("script_options", {}), sort_keys=True, allow_nan=False)
               for row in records if row["complete"] and row["split"] in ("train", "validation")}
    if len(options) != 1:
        raise ValueError("Mixed script_options require a separate explicit training contract")
    return json.loads(next(iter(options)))


def window_indices(length, target_start, training_start=50):
    """Slot1 is target_start; pad tails without changing history/action alignment."""
    if any(type(v) is not int for v in (length, target_start, training_start)):
        raise ValueError("Window indices must be integers")
    if not 0 <= training_start <= target_start < length:
        raise ValueError("Window start outside selected episode segment")
    raw = target_start - 1 + np.arange(16, dtype=np.int64)
    valid = (raw >= training_start) & (raw < length)
    loss = valid & (np.arange(16) >= 1) & (np.arange(16) < 13)
    return {"observation_indices": np.clip(raw[:2], 0, length - 1),
            "padded_indices": np.clip(raw, training_start, length - 1),
            "source_indices": np.where(valid, raw, -1), "valid_mask": valid, "loss_mask": loss}


def pack_history(history):
    if len(history) != 2:
        raise ValueError("Exactly two observation frames required")
    frames = [row.as_mapping() if hasattr(row, "as_mapping") else row for row in history]
    result = {}
    for row in frames:
        if tuple(row) != OBS_KEYS:
            raise ValueError("Only the four ordered policy observation keys are accepted")
    shapes = {"camera_left_rgb": (240, 320, 3), "camera_right_rgb": (240, 320, 3),
              "arm_joint_pos": (6,), "jaw_state": (2,)}
    for key in OBS_KEYS:
        values = [np.asarray(row[key]) for row in frames]
        if any(v.shape != shapes[key] for v in values):
            raise ValueError("Observation shape mismatch: " + key)
        if key.endswith("_rgb"):
            if any(v.dtype != np.uint8 for v in values):
                raise ValueError("Raw RGB must be uint8")
            packed = np.stack(values).transpose(0, 3, 1, 2).astype(np.float32) / np.float32(255)
        else:
            if any(not np.isfinite(v).all() for v in values):
                raise ValueError("Nonfinite low-dimensional observation")
            if key == "jaw_state" and any(not np.isin(v, [0, 1]).all() for v in values):
                raise ValueError("Jaw observations must be binary")
            packed = np.stack(values).astype(np.float32)
        result[key] = np.ascontiguousarray(packed[None])
    return result


def make_window(arrays, target_start, training_start=50):
    indices = window_indices(len(arrays["action"]), target_start, training_start)
    obs = pack_history([{k: arrays[k][i] for k in OBS_KEYS} for i in indices["observation_indices"]])
    return {"obs": obs, "action": np.asarray(arrays["action"][indices["padded_indices"]], dtype=np.float32)[None],
            "valid_mask": indices["valid_mask"][None], "loss_mask": indices["loss_mask"][None]}


def concatenate_windows(windows):
    if not windows:
        raise ValueError("Empty batch")
    return {"obs": {key: np.concatenate([w["obs"][key] for w in windows]) for key in OBS_KEYS},
            **{key: np.concatenate([w[key] for w in windows]) for key in ("action", "valid_mask", "loss_mask")}}


class Dataset:
    """Authenticate once, then use read-only mmap arrays rather than an RGB tensor cache."""
    def __init__(self, path, *, training_start=50, stride=12):
        if type(training_start) is not int or training_start < 0 or stride != 12:
            raise ValueError("Nonnegative training start and baseline stride12 required")
        self.path = Path(path).resolve(strict=True)
        self.manifest, records = read_manifest(self.path)
        self.script_options = consistent_script_options(records)
        self.scene_setup_identity = consistent_scene_identity(records)
        self.training_start = training_start
        self.rows = []
        self.file_stats = {}
        io = frozen_module("episode_io")
        for row in records:
            if not row["complete"] or row["split"] == "test":
                continue
            root = Path(row["resolved_path"])
            binding = verify_episode_binding(root, row["episode_binding"], self.scene_setup_identity,
                                             base_dir=self.path.parent)
            verified = io.load_policy_episode(root)
            end = json.loads((root / "episode_end.json").read_text(encoding="utf-8"))
            if end["schema_version"] != 2:
                raise ValueError("Only commanded-jaw schema2 episodes are admitted")
            length = len(verified["action"])
            if length <= training_start:
                raise ValueError("Episode has no selected training rows: " + row["id"])
            del verified
            arrays = {key: np.load(root / "policy" / (key + ".npy"), mmap_mode="r", allow_pickle=False)
                      for key in ("timestamp_ns", *OBS_KEYS, "action")}
            starts = list(range(training_start, length, stride))
            counts = [int(window_indices(length, t, training_start)["loss_mask"].sum()) for t in starts]
            for file in root.rglob("*"):
                if file.is_file():
                    self.file_stats[str(file)] = (file.stat().st_size, file.stat().st_mtime_ns)
            binding_path = Path(binding["path"])
            self.file_stats[str(binding_path)] = (binding_path.stat().st_size, binding_path.stat().st_mtime_ns)
            self.rows.append(dict(row, arrays=arrays, starts=starts, counts=counts,
                                  episode_manifest=identity(root / "manifest.json")))
        self.train = [row for row in self.rows if row["split"] == "train"]
        self.validation = [row for row in self.rows if row["split"] == "validation"]
        self.identities = {"dataset": identity(self.path), "episodes": [
            {**{key: row[key] for key in ("id", "split", "position_id", "script_sha256", "script_options", "scene_sha256", "episode_manifest", "scene_setup_identity", "episode_binding")},
             "complete": True,
             "commands_sha256": row.get("commands_sha256")}
            for row in self.rows], "excluded_test_and_incomplete_ids": [r["id"] for r in records
            if r["split"] == "test" or not r["complete"]]}

    def unchanged(self):
        if identity(self.path) != self.identities["dataset"]:
            raise ValueError("Dataset manifest changed during training")
        for name, expected in self.file_stats.items():
            file = Path(name)
            if (file.stat().st_size, file.stat().st_mtime_ns) != expected:
                raise ValueError("Source episode changed: " + name)

    def sample(self, rng, count):
        selected, windows = [], []
        for _ in range(count):
            row = self.train[int(rng.integers(len(self.train)))]
            weights = np.asarray(row["counts"], dtype=np.float64)
            index = int(rng.choice(len(weights), p=weights / weights.sum()))
            start = row["starts"][index]
            selected.append({"episode_id": row["id"], "target_start": start})
            windows.append(make_window(row["arrays"], start, self.training_start))
        return concatenate_windows(windows), selected

    def normalizer_rows(self):
        """Original fitting function receives actual train-only low rows and RGB statistics."""
        low = {key: np.concatenate([r["arrays"][key][self.training_start if key == "action"
                    else max(0, self.training_start - 1):] for r in self.train])
               for key in ("action", "arm_joint_pos", "jaw_state")}
        for key in OBS_KEYS[:2]:
            histogram = np.zeros(256, dtype=np.int64)
            for row in self.train:
                # One frame at a time bounds temporary memory for long new recordings.
                for frame in row["arrays"][key][max(0, self.training_start - 1):]:
                    histogram += np.bincount(frame.reshape(-1), minlength=256)
            low[key] = ImageStatistics(histogram)
        return low


class ImageStatistics:
    """Scalar byte statistics, avoiding concatenating the entire training image corpus."""
    def __init__(self, histogram):
        self.histogram = np.asarray(histogram, dtype=np.int64)
        self.levels = np.arange(256, dtype=np.float64)
        if self.histogram.shape != (256,) or self.histogram.sum() <= 0:
            raise ValueError("Nonempty byte histogram required")
    def min(self): return int(np.flatnonzero(self.histogram)[0])
    def max(self): return int(np.flatnonzero(self.histogram)[-1])
    def mean(self): return float(self.levels @ self.histogram / self.histogram.sum())
    def std(self): return float(np.sqrt(((self.levels - self.mean()) ** 2) @ self.histogram / self.histogram.sum()))


"""Explicit research training entry point. Import/plan calls never load a model."""


import importlib.metadata
import copy
import json
from pathlib import Path
import time

import numpy as np



def resolve_config(config, updates, device):
    supplied = dict(config or {})
    defaults = {"recipe": "research-aligned12/v1", "training_window_start": 50,
                "training_window_stride": 12, "effective_batch": 13, "learning_rate": 1e-4,
                "warmup_steps": min(max(0, updates - 1), int(updates * .05)),
                "precision": "bf16" if str(device).startswith("cuda") else "float32", "checkpoint_every": 1000}
    defaults["model_options"] = {}
    if set(supplied) - set(defaults):
        raise ValueError("Unknown training configuration fields: " + repr(sorted(set(supplied) - set(defaults))))
    defaults.update(supplied)
    if not isinstance(defaults["model_options"], dict):
        raise ValueError("model_options must be a JSON object")
    json.dumps(defaults["model_options"], allow_nan=False)
    if defaults["recipe"] != "research-aligned12/v1" or defaults["training_window_stride"] != 12:
        raise ValueError("This trainer implements the named research aligned12 recipe")
    for key in ("training_window_start", "warmup_steps"):
        if type(defaults[key]) is not int or defaults[key] < 0:
            raise ValueError("Nonnegative integer required: " + key)
    if defaults["warmup_steps"] >= updates:
        raise ValueError("Warmup must be shorter than the update budget")
    for key in ("effective_batch", "checkpoint_every"):
        if type(defaults[key]) is not int or defaults[key] <= 0:
            raise ValueError("Positive integer required: " + key)
    if not isinstance(defaults["learning_rate"], (int, float)) or not np.isfinite(defaults["learning_rate"]) or defaults["learning_rate"] <= 0:
        raise ValueError("Positive finite learning rate required")
    if defaults["precision"] not in ("float32", "bf16") or (defaults["precision"] == "bf16" and not str(device).startswith("cuda")):
        raise ValueError("BF16 recipe requires CUDA; float32 is separately labelled")
    return defaults


def training_plan(dataset_path, *, model_type="diffusion", updates=142858, seed=42, device="cuda:0", config=None,
                  expected_scene_setup_identity=None):
    """Metadata-only plan, with no episode arrays, Torch, model or checkpoint loading."""
    from pipeline.models import ADAPTERS
    if model_type not in ADAPTERS or type(updates) is not int or updates <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("Known model, positive update budget and nonnegative seed required")
    _, records = read_manifest(dataset_path)
    scene = consistent_scene_identity(records)
    if expected_scene_setup_identity is not None and scene != validate_scene_identity(expected_scene_setup_identity):
        raise ValueError("Dataset scene does not match the approved experiment scene")
    cfg = resolve_config(config, updates, device)
    return {"schema": "research-training-plan/v1", "label": "RESEARCH", "model_type": model_type,
            "updates": updates, "seed": seed, "device": device, "config": cfg, "policy_contract": CONTRACT,
            "dataset": identity(dataset_path), "train_ids": [r["id"] for r in records if r["complete"] and r["split"] == "train"],
            "script_options": consistent_script_options(records),
            "scene_setup_identity": scene,
            "validation_ids": [r["id"] for r in records if r["complete"] and r["split"] == "validation"],
            "test_data_loaded": False, "runtime_validated": False, "automatic_validation_predictions": False,
            "checkpoint_selection": "FINAL_EMA", "live_compatible": cfg["training_window_start"] == 50,
            "original_142858_recipe_equivalent": False}


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _pre_checkpoint_progress(checkpoint, plan):
    """Identify a fresh run interrupted before its first durable save."""
    if any((checkpoint.parent / name).exists() for name in ("resume_source.json", "resume.json")):
        raise ValueError("This was a resumed run; use its prior saved checkpoint, not a fresh restart")
    log = checkpoint.parent / "updates.jsonl"
    before = identity(log) if log.is_file() else None
    count, partial = 0, False
    if before is not None:
        with log.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    at_end = exc.pos >= len(line.rstrip()) or exc.msg.startswith("Unterminated string")
                    if not line.endswith("\n") and at_end:
                        partial = True
                        break
                    raise ValueError("Interrupted update log contains a malformed complete row") from exc
                if not isinstance(row, dict) or type(row.get("step")) is not int or row["step"] != count + 1:
                    raise ValueError("Interrupted update log must be a contiguous fresh-run prefix")
                count = row["step"]
        if identity(log) != before:
            raise ValueError("Interrupted update log changed while inspecting it")
    first_save = min(plan["config"]["checkpoint_every"], plan["updates"])
    if count >= first_save:
        raise ValueError("LAST.pt is missing after a checkpoint was due; locate the saved checkpoint or explicitly start a new run")
    if checkpoint.exists():
        raise ValueError("A checkpoint appeared during inspection; retry resume using that saved file")
    return count, before, partial, first_save


def inspect_resume(path):
    """Bind LAST and adjacent JSON without loading Torch or deserializing state.

    This receipt fixes the bytes that the explicit worker may later load. The
    checkpoint's completed step and full source bindings are checked at runtime.
    Unfinished .pending files are never recovery candidates.
    """
    from pipeline.models import ADAPTERS
    checkpoint = Path(path).resolve()
    exists = checkpoint.exists()
    if checkpoint.name != "LAST.pt" or (exists and not checkpoint.is_file()):
        raise ValueError("Resume requires the published LAST.pt, never a .pending file")
    paths = {"plan": checkpoint.parent / "plan.json",
             "normalizer": checkpoint.parent / "normalization.json"}
    if exists:
        paths["checkpoint"] = checkpoint
    before = {key: identity(value) for key, value in paths.items()}
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    normalization = json.loads(paths["normalizer"].read_text(encoding="utf-8"))
    json.dumps([plan, normalization], allow_nan=False)
    if (plan.get("schema") != "research-training-plan/v1" or plan.get("policy_contract") != CONTRACT
            or plan.get("model_type") not in ADAPTERS or type(plan.get("updates")) is not int
            or plan["updates"] <= 0 or type(plan.get("seed")) is not int or plan["seed"] < 0
            or not isinstance(plan.get("device"), str)):
        raise ValueError("Invalid saved training plan")
    if resolve_config(plan.get("config"), plan["updates"], plan["device"]) != plan.get("config"):
        raise ValueError("Saved training plan must contain the complete original recipe")
    scene = validate_scene_identity(plan.get("scene_setup_identity"))
    if (normalization.get("schema") != "research-normalization/v1"
            or normalization.get("dataset_identity") != plan.get("dataset")
            or normalization.get("scene_setup_identity") != scene
            or normalization.get("training_window_start") != plan["config"]["training_window_start"]
            or normalization.get("train_ids") != plan.get("train_ids")
            or normalization.get("validation_rows_used") is not False
            or normalization.get("test_rows_used") is not False):
        raise ValueError("Saved normalization differs from the original training plan")
    if before != {key: identity(value) for key, value in paths.items()}:
        raise ValueError("Resume source changed while inspecting it")
    receipt = {"schema": "research-resume-receipt/v1", "checkpoint_path": str(checkpoint),
               "checkpoint": before.get("checkpoint"), "plan_path": str(paths["plan"]),
               "plan_identity": before["plan"], "plan": plan,
               "normalizer_path": str(paths["normalizer"]), "normalizer_identity": before["normalizer"],
               "normalization": normalization}
    if not exists:
        count, log_identity, partial, first_save = _pre_checkpoint_progress(checkpoint, plan)
        notice = (f"No saved checkpoint exists: training was interrupted before the first save at update {first_save:,}. "
                  f"Training will restart from update 0; {count:,} logged updates were not saved.")
        receipt.update(schema="research-restart-receipt/v1", restart_from_zero=True,
                       logged_updates=count, checkpoint_every=plan["config"]["checkpoint_every"],
                       updates_log_identity=log_identity, trailing_partial_log=partial, notice=notice)
    return receipt


def _resume_receipt(value):
    if value is None:
        return None
    inspected = inspect_resume(value["checkpoint_path"] if isinstance(value, dict) else value)
    if isinstance(value, dict) and inspected != value:
        raise ValueError("Resume source differs from the planned receipt")
    return inspected


def run_training(dataset_path, output, *, model_type="diffusion", updates=142858, seed=42, device="cuda:0", config=None,
                 expected_scene_setup_identity=None, resume=None):
    """Run only after the CLI's explicit execute gate. Always creates a new output directory.

    Model operations exist here but were not executed during pipeline preparation.
    Validation data is authenticated but never used for fitting or gradients. No test
    data is loaded. The exported artifact is final EMA; no implicit BEST claim.
    """
    plan = training_plan(dataset_path, model_type=model_type, updates=updates, seed=seed, device=device, config=config,
                         expected_scene_setup_identity=expected_scene_setup_identity)
    receipt = _resume_receipt(resume)
    restart = bool(receipt and receipt.get("restart_from_zero"))
    if receipt is not None and receipt["plan"] != plan:
        raise ValueError("Resume must preserve the original dataset, scene, model, total updates, seed, device and recipe")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "plan.json", plan)
    if restart:
        _write(output / "restart_from_zero.json", receipt)
        print("Note: " + receipt["notice"], flush=True)
    elif receipt is not None:
        _write(output / "resume_source.json", receipt)
    began = time.time()
    report = {"schema": "research-training-result/v1", "label": "RESEARCH", "status": "FAILED",
              "completed_updates": 0, "requested_updates": updates, "kit_starts": 0,
              "resumed": receipt is not None and not restart, "restarted_from_zero": restart,
              "discarded_logged_updates": receipt["logged_updates"] if restart else 0,
              "starting_updates": 0, "new_updates": 0,
              "physical_outcome": "NOT_MEASURED", "original_18of30_reproduction": False}
    try:
        data = Dataset(dataset_path, training_start=plan["config"]["training_window_start"], stride=12)
        if data.scene_setup_identity != plan["scene_setup_identity"] or data.identities["dataset"] != plan["dataset"]:
            raise ValueError("Dataset provenance changed after training plan")
        import torch
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device unavailable; no CPU fallback")
        if plan["config"]["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Requested CUDA BF16 unsupported; no precision fallback")
        versions = {}
        for name in ("torch", "torchvision", "numpy", "diffusers", "einops"):
            try: versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: versions[name] = None
        if receipt is None:
            api = frozen_module("overfit_policy")
            fitted, normalization = api._fit_normalizer_from_validated_policy_rows(
                data.normalizer_rows(), fit_source="research_train_only_unpadded_selected_rows")
            del fitted
            normalization.update(schema="research-normalization/v1", dataset_identity=data.identities["dataset"],
                                 training_window_start=data.training_start, train_ids=plan["train_ids"],
                                 scene_setup_identity=plan["scene_setup_identity"],
                                 validation_rows_used=False, test_rows_used=False)
        else:
            normalization = copy.deepcopy(receipt["normalization"])
        _write(output / "normalization.json", normalization)
        from pipeline.models import create_adapter, resume_adapter
        rng = np.random.default_rng(seed)
        starting_step = 0
        if receipt is None or restart:
            adapter = create_adapter(model_type, normalizer=normalization, config=plan["config"],
                                     device=device, updates=updates, seed=seed)
        else:
            _resume_receipt(receipt)
            adapter, restored = resume_adapter(model_type, receipt["checkpoint_path"], device=device,
                expected_metadata={"plan": plan, "identities": data.identities, "normalization": normalization,
                                   "software_versions": versions, "checkpoint": receipt["checkpoint"]})
            starting_step = restored["completed_updates"]
            if type(starting_step) is not int or not 0 <= starting_step <= updates:
                raise ValueError("Invalid restored update counter")
            rng.bit_generator.state = copy.deepcopy(restored["sampler_state"])
            report.update(starting_updates=starting_step, completed_updates=starting_step,
                          resume_checkpoint=receipt["checkpoint"],
                          unsaved_prior_work="Not credited; continuation starts after the saved checkpoint only")
            _write(output / "resume.json", {"source": receipt, "completed_updates": starting_step,
                                           "next_update": starting_step + 1, "automatic_retry": False})
        constructor_receipts = getattr(adapter, "constructor_receipts", [])
        _write(output / "constructor.json", {"model_type": model_type, "device": device, "receipts": constructor_receipts})
        metadata = {"identities": data.identities, "plan": plan, "software_versions": versions,
                    "constructor_receipts": constructor_receipts,
                    "checkpoint_selection": "FINAL_EMA", "label": "RESEARCH"}
        if restart:
            metadata["restart_from_zero"] = receipt
        if receipt is not None and not restart:
            metadata["resume_source"] = {"checkpoint": receipt["checkpoint"], "completed_updates": starting_step}
            # Establish a recovery point in the new directory even if export or
            # the next update is interrupted. The source remains untouched.
            adapter.save(output / "LAST.pt", {**metadata, "sampler_state": rng.bit_generator.state}, recovery=True)
        with (output / "updates.jsonl").open("x", encoding="utf-8") as log:
            for step in range(starting_step + 1, updates + 1):
                data.unchanged()
                tick = time.monotonic()
                batch, selected = data.sample(rng, plan["config"]["effective_batch"])
                metrics = adapter.train_step(batch)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                report["completed_updates"] = step
                report["new_updates"] = step - starting_step
                record = {"step": step, **metrics, "seconds": time.monotonic() - tick, "sampled_windows": selected}
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                if step % plan["config"]["checkpoint_every"] == 0 or step == updates:
                    adapter.save(output / "LAST.pt", {**metadata, "sampler_state": rng.bit_generator.state}, recovery=True)
        data.unchanged()
        adapter.save(output / "model.pt", metadata)
        artifact = {"schema": "research-model-artifact/v1", "label": "RESEARCH", "model_type": model_type,
                    "model_config": plan["config"], "policy_contract": CONTRACT,
                    "training_window_start": data.training_start, "live_compatible": data.training_start == 50,
                    "checkpoint_path": "model.pt", "checkpoint": identity(output / "model.pt"),
                    "normalizer_path": "normalization.json", "normalizer": identity(output / "normalization.json"),
                    "identities": data.identities, "scene_sha256s": sorted({r["scene_sha256"] for r in data.train}),
                    "scene_setup_identity": plan["scene_setup_identity"],
                    "script_options": plan["script_options"], "constructor_receipts": constructor_receipts,
                    "script_sha256s": sorted({r["script_sha256"] for r in data.train}),
                    "completed_updates": updates, "checkpoint_selection": "FINAL_EMA", "software_versions": versions,
                    "validation_evaluation": "NOT_RUN", "test_data_loaded": False,
                    "original_142858_recipe_equivalent": False}
        _write(output / "artifact.json", artifact)
        report.update(status="TRAINING_FINISHED", artifact=str(output / "artifact.json"),
                      validation_evaluation="NOT_RUN", checkpoint_selection="FINAL_EMA")
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        report["elapsed_seconds"] = time.time() - began
        _write(output / "training_report.json", report)
    return report


"""Separate explicitly launched training worker. Never imported by planning."""
import sys
from pipeline.settings import read

def worker_main():
    request = read(sys.argv[1])
    cfg = dict(request['training'])
    run_training(request['dataset'], request['output'], model_type=cfg.pop('model_type'),
                 updates=cfg.pop('updates'), seed=cfg.pop('seed'), device=cfg.pop('device'), config=cfg,
                 expected_scene_setup_identity=request['scene_setup_identity'], resume=request.get('resume'))

if __name__ == "__main__":
    worker_main()
