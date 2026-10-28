"""Narrow saved-Stage1 admission and offline deployment scoring. No native imports."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from .episode_io import load_policy_episode, load_episode_end
from .overfit_data import _json_object, sha256_file, make_training_windows
from .deployment_adapter import DeploymentActionMapping, make_deployment_loss_mask
from .interface import POLICY_OBSERVATION_KEYS, threshold_jaw_commands, validate_policy_action
from .interface import TCP_X_RANGE_M, TCP_Y_RANGE_M, TCP_Z_RANGE_M, JAW_ACTION_THRESHOLD

ACTION_BOUNDS = (TCP_X_RANGE_M, TCP_Y_RANGE_M, TCP_Z_RANGE_M, (0., 1.), (0., 1.))

STAGE1_COMMIT = "b7253a18f843369d728ef6e2d607ddf21d02eeb9"
SOURCE_BINDING = {
    "attempt_result.json": "aca4ffc8d3850020de11f4c52efd368856a4c908f2abb9de53701d6b1f73c527",
    "invocation.json": "33d708e95485b91f4f36c0269ad74de4bb7c42a0ba4280586023e1aed1bb07c5",
    "capture/process_result.json": "7f7b6034c40691d5c0cade0139f5f9878f0b78c4d3131c725bc81e7452ad3ebd",
    "capture/experiment/artifact_manifest.json": "08ab959f62d9eed99769c67f331f6cfd395e2cc9486f2103e63b8be47d792105",
    "capture/experiment/episode/manifest.json": "06eaa8ed9c8595fbc307d9d8e86d8ed78e34a370176b25993697376045637a59",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def verify_bound_evidence(episode, binding):
    """Authenticate saved checker and all its payloads, without reinterpreting physics."""
    episode = Path(episode).resolve(strict=True)
    require(episode.name == "episode" and episode.parent.name == "experiment"
            and episode.parent.parent.name == "capture", "Stage1 episode layout required")
    attempt = episode.parents[2]
    for relative, digest in binding.items():
        require(sha256_file(attempt / relative) == digest, "Stage1 binding mismatch: " + relative)
    result = _json_object(attempt / "attempt_result.json")
    for field in ("status", "CAPTURE_INTEGRITY", "PROCESS_COMPLETION", "END_STATE_RESULT", "DEMO_INTEGRITY"):
        require(result.get(field) == "PASS", "Stage1 checker did not pass: " + field)
    require(result["run_scope"]["run_type"] == "EVIDENCE"
            and result["run_scope"]["acceptance_eligible"] is True, "Stage1 EVIDENCE required")
    manifest = _json_object(episode.parent / "artifact_manifest.json")
    require(manifest["schema"] == "capture-artifacts/v1", "Stage1 manifest schema")
    actual = {p.relative_to(episode.parent).as_posix() for p in episode.parent.rglob("*") if p.is_file()}
    require(actual == set(manifest["files"]) | {"artifact_manifest.json"}, "Stage1 payload membership")
    for relative, expected in manifest["files"].items():
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "Unsafe payload")
        path = episode.parent / relative
        require(path.stat().st_size == expected["bytes"] and sha256_file(path) == expected["sha256"],
                "Stage1 payload changed: " + relative)
    process = _json_object(attempt / "capture/process_result.json")
    for field, relative in (("stdout", "stdout.bin"), ("stderr", "stderr.bin"),
                            ("completed_journal", "structured/events.jsonl")):
        path = attempt / "capture" / relative
        require(path.stat().st_size == process[field]["bytes"]
                and sha256_file(path) == process[field]["sha256"], "Stage1 stream changed: " + field)
    return {"stage1_report_commit": STAGE1_COMMIT, "episode": str(episode),
            "binding": dict(binding), "checker": result, "payload_files": len(manifest["files"])}


def check_causal_labels(arrays):
    expected = np.concatenate((np.ones((1, 2), dtype=np.uint8), arrays["action"][:-1, 3:]))
    require(np.array_equal(arrays["jaw_state"], expected), "Jaw input must be already-active prior command")


def load_stage1_source(episode):
    source = verify_bound_evidence(episode, SOURCE_BINDING)
    arrays = load_policy_episode(episode)
    metadata = load_episode_end(episode)
    require(metadata["episode_id"] == source["checker"]["end_state"]["episode_id"], "Episode ID mismatch")
    require(len(arrays["action"]) == 200, "Exactly 200 Stage1 actions required")
    check_causal_labels(arrays)
    source["episode_metadata"] = metadata
    return arrays, source


def deployment_mapping(episode_length):
    require(type(episode_length) is int and episode_length > 0, "Positive episode length required")
    return tuple(DeploymentActionMapping(start - 1, slot + 1, slot, slot, start + slot)
                 for start in range(0, episode_length, 12)
                 for slot in range(min(12, episode_length - start)))


def deployment_data(arrays):
    windows = make_training_windows(arrays)  # Unchanged 197-window builder.
    mapping = deployment_mapping(len(arrays["action"]))
    mask = make_deployment_loss_mask(window_starts=windows["window_starts"],
        source_indices=windows["source_indices"], valid_mask=windows["valid_mask"],
        episode_length=len(arrays["action"]), verified_mapping=mapping)
    rows = np.flatnonzero(mask.any(axis=1))
    require(np.array_equal(windows["source_indices"][mask], np.arange(len(arrays["action"]))), "Coverage")
    return windows, mask, rows, mapping


def select_executed(predictions, windows, mask, rows, demonstrated):
    values = np.asarray(predictions)
    require(values.shape == (len(rows), 16, 5), "Deployment prediction shape")
    require(values.dtype.kind in "iuf", "Predictions must be real numeric values")
    require(np.isfinite(values).all(), "All predicted outputs must be finite")
    selected = values[mask[rows]].astype(np.float64)
    targets = windows["action"][rows][mask[rows]].astype(np.float64)
    require(np.array_equal(targets, demonstrated), "Selected targets differ from source")
    require(np.array_equal(windows["source_indices"][rows][mask[rows]], np.arange(len(demonstrated))),
            "Selected rows must cover source exactly once")
    return selected, targets


def jaw_transitions(bits):
    return [[{"frame": i, "value": int(bits[i, j])} for i in range(len(bits))
             if i == 0 or bits[i, j] != bits[i - 1, j]] for j in range(2)]


def range_violations(predictions, mapping=None):
    p = np.asarray(predictions)
    require(p.ndim == 2 and p.shape[1] == 5 and p.dtype.kind in "iuf" and np.isfinite(p).all(),
            "Range diagnostics require finite real [N,5] values")
    if mapping is None:
        mapping = deployment_mapping(len(p))
    require(len(mapping) == len(p), "Range mapping length")
    violations = []
    for axis, (lower, upper) in enumerate(ACTION_BOUNDS):
        for row in np.flatnonzero((p[:, axis] < lower) | (p[:, axis] > upper)):
            value = float(p[row, axis]); side = "lower" if value < lower else "upper"
            limit = lower if side == "lower" else upper
            item = mapping[int(row)]
            violations.append({**item.to_dict(), "chunk_index": item.absolute_frame // 12,
                "axis": ("x", "y", "z", "gL", "gR")[axis], "value": value,
                "lower_limit": lower, "upper_limit": upper, "violated_limit": limit,
                "side": side, "excess": abs(value-limit),
                "units": "m" if axis < 3 else "dimensionless", "frame": "robot_base" if axis < 3 else "command"})
    return violations


def score_executed(predictions, targets, mapping=None):
    require(np.asarray(predictions).dtype.kind in "iuf" and np.asarray(targets).dtype.kind in "iuf",
            "Scoring requires real numeric arrays")
    p, t = np.asarray(predictions, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    require(p.shape == t.shape and p.ndim == 2 and p.shape[1] == 5 and len(p) > 0, "Executed shape")
    require(np.isfinite(p).all() and np.isfinite(t).all(), "Nonfinite executed output")
    require(np.isin(t[:, 3:], (0, 1)).all(), "Nonbinary target")
    # Offline classification preserves values, threshold and bounds. Runtime
    # threshold_jaw_commands remains strict; finite range failures are diagnostics.
    violations = range_violations(p, mapping)
    bits = (p[:, 3:] >= JAW_ACTION_THRESHOLD).astype(np.uint8)
    reference = t[:, 3:].astype(np.uint8)
    errors = np.abs(p[:, :3] - t[:, :3])
    mae, maximum = float(errors.mean()), float(errors.max())
    require(math.isfinite(mae) and math.isfinite(maximum), "Nonfinite Cartesian metrics")
    wrong = [{"absolute_frame": int(i), "jaw": ("gL", "gR")[j],
              "predicted_continuous": float(p[i, 3+j]), "predicted_command": int(bits[i, j]),
              "target_command": int(reference[i, j])} for i, j in np.argwhere(bits != reference)]
    result = dict(finite=True, cartesian_mae_m=mae, cartesian_max_abs_error_m=maximum,
        cartesian_mae_passed=mae <= .002, cartesian_max_abs_error_passed=maximum <= .005,
        jaw_accuracy=float(np.mean(bits == reference)), jaw_accuracy_passed=not wrong,
        jaw_sequence_order_passed=jaw_transitions(bits) == jaw_transitions(reference),
        source_rows_consistent=True, valid_target_count=len(p), valid_jaw_bit_count=len(p)*2,
        correct_jaw_bits=int((bits == reference).sum()), wrong_executed_jaw_bits=wrong,
        predicted_transitions=jaw_transitions(bits), target_transitions=jaw_transitions(reference),
        dispatch_valid=not violations, range_violations=violations)
    result["passed"] = all(result[k] for k in ("cartesian_mae_passed", "cartesian_max_abs_error_passed",
                                             "jaw_accuracy_passed", "jaw_sequence_order_passed", "dispatch_valid"))
    return result


def inspect_conditioning(windows, rows, mask):
    require(tuple(windows["obs"]) == POLICY_OBSERVATION_KEYS, "Exact conditioning fields required")
    fingerprints, conflicts = [], []
    for row in rows:
        digest = hashlib.sha256()
        for key, value in windows["obs"].items():
            digest.update(key.encode()); digest.update(np.ascontiguousarray(value[row]).tobytes())
        fingerprints.append(digest.hexdigest())
    for i, left in enumerate(rows):
        for j in range(i):
            right = rows[j]
            if fingerprints[i] != fingerprints[j]:
                continue
            if not all(np.array_equal(v[left], v[right]) for v in windows["obs"].values()):
                continue
            common = mask[left] & mask[right]
            if not np.array_equal(windows["action"][left, common], windows["action"][right, common]):
                conflicts.append({"window_starts": [int(windows["window_starts"][r]) for r in (right, left)],
                                  "conflicting_slots": np.flatnonzero(common & np.any(
                                      windows["action"][left] != windows["action"][right], axis=1)).tolist()})
    return {"scope": "Exact complete two-view/two-time RGB plus six joints and two commanded jaw bits; all deployment pairs",
            "deployment_chunks": len(rows), "pair_count": len(rows)*(len(rows)-1)//2,
            "complete_conditioning_sha256": fingerprints, "demonstrated_conflicts": conflicts,
            "limitation": "No claim that near-identical inputs are distinguishable by the learned encoder."}


def freeze_budget(update100_seconds, evaluation_seconds, checkpoint_seconds, reload_seconds, elapsed_seconds):
    values = (update100_seconds, evaluation_seconds, checkpoint_seconds, reload_seconds, elapsed_seconds)
    require(all(math.isfinite(x) and x > 0 for x in values), "Budget measurements must be finite positive")
    # Twenty remaining scheduled checkpoints/evaluations; final save/reload/eval.
    # Full recovery writes conservatively upper-bound the final smaller EMA save.
    total = elapsed_seconds + update100_seconds / 100 * 9900 + evaluation_seconds * 21 + checkpoint_seconds * 21 + reload_seconds
    return {"calibration_updates": 100, "measured_100_updates_seconds": update100_seconds,
            "measured_evaluation_seconds": evaluation_seconds, "measured_checkpoint_seconds": checkpoint_seconds,
            "measured_reload_seconds": reload_seconds, "elapsed_at_freeze_seconds": elapsed_seconds,
            "planned_total_seconds": total, "margin_factor": 1.5,
            "absolute_training_cap_seconds": math.ceil(total * 1.5), "max_optimizer_updates": 10000,
            "evaluation_steps": [100] + list(range(500, 10001, 500))}


def demo_dispatch_preflight(arrays, root):
    """Execute exact source methods with CPU tensors; no native module import."""
    import ast
    from types import SimpleNamespace
    import torch
    from . import interface
    paths = [Path(root) / "src/chicgrasp_policy/attachment_capture_runtime.py",
             Path(root) / "source/chicgrasp_isaac/chicgrasp_isaac/tasks/manager_based/chicgrasp_isaac/mdp/policy_interface.py"]
    namespace = {"torch": torch, **vars(interface)}
    records = []
    for path, method in zip(paths, ("action_tensor", "_validate_actions")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        matches = [f for cls in tree.body if isinstance(cls, ast.ClassDef)
                   for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == method]
        require(len(matches) == 1, "Exact dispatch method not found")
        function = matches[0]
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        records.append(dict(path=str(path), sha256=sha256_file(path), method=method,
                            first_line=function.lineno, last_line=function.end_lineno))
    actions = arrays["action"]
    require(actions.shape == (200, 5) and actions.dtype == np.float32, "Actual demo float32[200,5] required")
    dispatched, rejected = [], []
    for index, action in enumerate(actions):
        tensor = namespace["action_tensor"](SimpleNamespace(torch=torch), action, "cpu")
        value = tensor.to(torch.float64).numpy()[0]
        dispatched.append(value)
        try:
            namespace["_validate_actions"](SimpleNamespace(num_envs=1, device="cpu"), tensor)
            validate_policy_action(value)
        except ValueError as exc:
            rejected.append(dict(index=index, vector=value.tolist(), units=["m","m","m","binary","binary"],
                frame="robot_base", exception=str(exc), violations=range_violations(value[None],
                    (deployment_mapping(200)[index],))))
    values = np.stack(dispatched)
    margins = {axis: {"minimum_lower_margin": float((values[:, i]-lo).min()),
                      "minimum_upper_margin": float((hi-values[:, i]).min()),
                      "minimum_margin": float(np.minimum(values[:, i]-lo,hi-values[:, i]).min()),
                      "units": "m" if i < 3 else "dimensionless", "bounds": [lo, hi]}
               for i,(axis,(lo,hi)) in enumerate(zip(("x","y","z","gL","gR"),ACTION_BOUNDS))}
    return dict(status="FAIL" if rejected else "PASS", checked=200, accepted=200-len(rejected),
                rejected=rejected, margins=margins, source_methods=records,
                conversion="Stored np.float32 action -> unchanged action_tensor reshape[1,5]/torch.float32 -> unchanged _validate_actions float64 comparison; robot-base metres; CPU only",
                native_imports=False)
