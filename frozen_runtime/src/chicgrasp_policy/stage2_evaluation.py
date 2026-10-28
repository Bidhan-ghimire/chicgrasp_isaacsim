"""Durable offline evaluation: recovery first, raw output next, classification last."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np

from . import stage2_overfit as data
from .overfit_data import _json_object, sha256_file


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def identity(path):
    return dict(bytes=Path(path).stat().st_size, sha256=sha256_file(path))


def prediction_metadata(step, windows, mask, rows, demonstrated, normalization):
    result = dict(optimizer_step=step, denormalized=True, normalization=normalization,
        window_starts=windows["window_starts"][rows].tolist(), loss_mask=mask[rows].tolist(),
        observation_indices=windows["observation_indices"][rows].tolist(),
        source_indices=windows["source_indices"][rows].tolist(), action_units=["m","m","m","command","command"],
        coordinate_frame="robot_base", mapping=[m.to_dict() for m in data.deployment_mapping(len(demonstrated))])
    if 'new_to_original' in windows:
        result.update(new_to_original=windows['new_to_original'].tolist(),
            original_source_indices=windows['original_source_indices'][rows].tolist(),
            original_observation_indices=windows['original_observation_indices'][rows].tolist())
    return result


def save_recovery(path, *, policy, ema_policy, optimizer, scheduler, ema, step, metadata):
    """Save all restart state, without authorizing or performing a restart."""
    import torch
    from .overfit_policy import DEFAULT_POLICY_CONFIG
    data.require(policy.gate3_config == DEFAULT_POLICY_CONFIG, "Recovery architecture mismatch")
    np_state = np.random.get_state()
    rng = dict(python=random.getstate(), numpy=dict(name=np_state[0], keys=np_state[1].tolist(),
        position=np_state[2], has_gauss=np_state[3], cached_gaussian=np_state[4]),
        torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    def cpu_tree(value):
        if isinstance(value, torch.Tensor):
            data.require(bool(torch.isfinite(value).all()), "Nonfinite checkpoint tensor")
            return value.detach().cpu().clone()
        if isinstance(value, dict): return {k:cpu_tree(v) for k,v in value.items()}
        if isinstance(value, list): return [cpu_tree(v) for v in value]
        if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
        return value
    clean_metadata = json.loads(json.dumps(metadata, allow_nan=False))
    saved = dict(schema="stage2-recovery/v1", optimizer_step=step, model_config=policy.gate3_config,
        model_state=cpu_tree(policy.state_dict()), ema_state=cpu_tree(ema_policy.state_dict()),
        optimizer_state=cpu_tree(optimizer.state_dict()), scheduler_state=scheduler.state_dict(),
        ema_control={k:v for k,v in vars(ema).items() if k != "averaged_model"},
        rng=cpu_tree(rng), metadata=clean_metadata)
    with Path(path).open("xb") as stream:
        torch.save(saved, stream); stream.flush(); os.fsync(stream.fileno())


def load_recovery(path):
    import torch
    from .overfit_policy import DEFAULT_POLICY_CONFIG
    saved = torch.load(path, map_location="cpu", weights_only=True)
    required = {"schema","optimizer_step","model_config","model_state","ema_state","optimizer_state",
                "scheduler_state","ema_control","rng","metadata"}
    data.require(set(saved) == required and saved["schema"] == "stage2-recovery/v1", "Incomplete recovery checkpoint")
    data.require(saved["model_config"] == DEFAULT_POLICY_CONFIG, "Recovery model configuration changed")
    data.require(type(saved["optimizer_step"]) is int and saved["optimizer_step"] >= 0, "Invalid recovery step")
    for key in ("model_state","ema_state"):
        data.require(bool(saved[key]) and all(bool(torch.isfinite(t).all()) for t in saved[key].values()), "Invalid recovery model")
    data.require(set(saved["optimizer_state"]) == {"state","param_groups"}
                 and bool(saved["optimizer_state"]["param_groups"]), "Incomplete optimizer state")
    data.require(set(saved["rng"]) == {"python","numpy","torch_cpu","torch_cuda"}, "Incomplete RNG state")
    data.require(saved["ema_control"]["optimization_step"] == saved["optimizer_step"], "EMA step mismatch")
    return saved


def verify_evaluation(directory, windows, mask, rows, demonstrated, normalization):
    """Recompute acceptance; a cached PASS flag cannot establish it."""
    directory = Path(directory)
    result = _json_object(directory / "result.json")
    receipt = _json_object(directory / "checkpoint_receipt.json")
    data.require(receipt["identity"] == identity(directory / "recovery.pt"), "Recovery hash mismatch")
    recovery = load_recovery(directory / "recovery.pt")
    data.require(recovery["optimizer_step"] == receipt["optimizer_step"]
                 and recovery["metadata"]["normalization"] == normalization, "Recovery metadata mismatch")
    del recovery
    metadata = _json_object(directory / "prediction_metadata.json")
    data.require(metadata == prediction_metadata(receipt["optimizer_step"], windows, mask, rows,
                                                  demonstrated, normalization), "Prediction metadata mismatch")
    data.require(metadata["optimizer_step"] == receipt["optimizer_step"] == result["optimizer_step"], "Evaluation step mismatch")
    raw_path = directory / "raw_predictions.npz"
    data.require(identity(raw_path) == result["raw_predictions"], "Raw prediction hash mismatch")
    with np.load(raw_path, allow_pickle=False) as archive:
        data.require(set(archive.files) == {"predictions"}, "Raw prediction archive schema")
        selected, targets = data.select_executed(archive["predictions"], windows, mask, rows, demonstrated)
    metrics = data.score_executed(selected, targets, data.deployment_mapping(len(demonstrated)))
    with np.load(directory / "executed_predictions.npz", allow_pickle=False) as archive:
        data.require(set(archive.files) == {"predictions","targets","xyz_errors"}
                     and np.array_equal(archive["predictions"], selected)
                     and np.array_equal(archive["targets"], targets)
                     and np.array_equal(archive["xyz_errors"], selected[:,:3]-targets[:,:3]), "Executed archive mismatch")
    data.require(result["metrics"] == metrics and result["status"] == ("PASS" if metrics["passed"] else "FAIL"),
                 "Forged or inconsistent evaluation result")
    data.require(result["metadata_identity"] == identity(directory / "prediction_metadata.json"), "Metadata identity")
    data.require(np.isfinite(result["fixed_loss"]) and result["evaluation_seconds"] > 0, "Incomplete evaluation timing/loss")
    return metrics


def evaluate_and_publish(directory, *, step, checkpoint, predict, fixed_loss,
                         windows, mask, rows, demonstrated, normalization):
    """Shared production/test path. Only range failures return FAIL; errors propagate."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    checkpoint_seconds = None
    try:
        checkpoint_path = directory / "recovery.pt"
        checkpoint(checkpoint_path)
        write_json(directory / "checkpoint_receipt.json", dict(optimizer_step=step, identity=identity(checkpoint_path)))
        checkpoint_seconds = time.perf_counter()-start
        evaluation_start = time.perf_counter()
        loss = float(fixed_loss())
        data.require(np.isfinite(loss), "Nonfinite fixed evaluation loss")
        values = np.asarray(predict())
        # Persist every returned array, even if subsequent structural validation blocks.
        with (directory / "raw_predictions.npz").open("xb") as stream:
            np.savez_compressed(stream, predictions=values)
            stream.flush(); os.fsync(stream.fileno())
        write_json(directory / "prediction_metadata.json", prediction_metadata(step, windows, mask, rows,
                                                                                 demonstrated, normalization))
        selected, targets = data.select_executed(values, windows, mask, rows, demonstrated)
        metrics = data.score_executed(selected, targets)
        with (directory / "executed_predictions.npz").open("xb") as stream:
            np.savez_compressed(stream, predictions=selected, targets=targets, xyz_errors=selected[:,:3]-targets[:,:3])
            stream.flush(); os.fsync(stream.fileno())
        measured = dict(status="PASS" if metrics["passed"] else "FAIL", optimizer_step=step,
            checkpoint_seconds=checkpoint_seconds, evaluation_seconds=time.perf_counter()-evaluation_start,
            fixed_loss=loss, metrics=metrics, raw_predictions=identity(directory/"raw_predictions.npz"),
            metadata_identity=identity(directory/"prediction_metadata.json"))
        write_json(directory / "result.json", measured)
        verify_evaluation(directory, windows, mask, rows, demonstrated, normalization)
        measured["complete_evaluation_seconds"] = time.perf_counter()-evaluation_start
        write_json(directory / "verification.json", dict(**measured, verification_status="VERIFIED"))
        return loss, values, selected, metrics, measured["complete_evaluation_seconds"], checkpoint_seconds
    except Exception as exc:
        # Failure to publish this exception also propagates. Never turn I/O errors into FAIL/PASS.
        write_json(directory / "exception.json", dict(status="BLOCKED", optimizer_step=step,
            elapsed_seconds=time.perf_counter()-start, checkpoint_seconds=checkpoint_seconds,
            error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()))
        raise
