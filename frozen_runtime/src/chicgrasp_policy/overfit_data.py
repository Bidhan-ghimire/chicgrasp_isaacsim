"""Isaac- and Torch-independent data checks for the one-trajectory overfit.

Provenance and diagnostic validation are kept separate from the four policy
observations.  No simulator, official-policy import, or write occurs on import.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import episode_io
from .interface import (
    EXECUTION_HORIZON,
    OBSERVATION_HORIZON,
    POLICY_ACTION_DIM,
    POLICY_OBSERVATION_KEYS,
    PREDICTION_HORIZON,
    threshold_jaw_commands,
)


CARTESIAN_MAE_LIMIT_M = 0.002
CARTESIAN_MAX_ABS_LIMIT_M = 0.005
ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"


class OverfitDataError(ValueError):
    """An invalid source, window, prediction, or evidence package."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OverfitDataError(message)


def _json_object(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            _require(key not in result, f"Duplicate JSON key {key!r} in {path}.")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise OverfitDataError(f"Nonfinite JSON value {value!r} in {path}.")

    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=nonfinite)
    _require(isinstance(result, dict), f"{path} must contain one JSON object.")
    return result


def _reject_link(path: Path) -> None:
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    _require(not path.is_symlink() and not attributes & 0x400, f"Links/reparse points are forbidden: {path}")


def sha256_file(path: str | Path) -> str:
    """Hash one ordinary file without mutating it."""
    source = Path(path)
    _reject_link(source)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()














def make_training_windows(arrays: Mapping[str, Any]) -> dict[str, Any]:
    """Use the pinned sampler's indices, keeping metadata outside ``obs``."""
    source = episode_io.validate_policy_arrays(arrays)
    count = len(source["action"])
    pad_before, pad_after = OBSERVATION_HORIZON - 1, EXECUTION_HORIZON - 1
    starts = np.arange(-pad_before, count - PREDICTION_HORIZON + pad_after + 1, dtype=np.int64)
    _require(len(starts) > 0, "The episode is too short for the pinned training-window convention.")
    indices = starts[:, None] + np.arange(PREDICTION_HORIZON, dtype=np.int64)
    valid = (indices >= 0) & (indices < count)
    padded = indices.clip(0, count - 1)
    obs_indices = padded[:, :OBSERVATION_HORIZON]
    obs = {}
    for key in POLICY_OBSERVATION_KEYS:
        values = source[key][obs_indices]
        if key.endswith("_rgb"):
            values = values.transpose(0, 1, 4, 2, 3).astype(np.float32) / np.float32(255)
        obs[key] = np.ascontiguousarray(values, dtype=np.float32)
    return {
        "obs": obs, "action": np.ascontiguousarray(source["action"][padded]), "valid_mask": valid,
        "source_indices": np.where(valid, indices, -1), "observation_indices": obs_indices,
        "window_starts": starts,
    }


def evaluate_predictions(predictions: Any, targets: Any, valid_mask: Any, source_indices: Any) -> dict[str, Any]:
    """Evaluate every real target; arbitrary padded values have no influence."""
    predicted, target = np.asarray(predictions), np.asarray(targets)
    valid, indices = np.asarray(valid_mask), np.asarray(source_indices)
    _require(predicted.ndim == 3 and predicted.shape[1:] == (PREDICTION_HORIZON, POLICY_ACTION_DIM) and predicted.shape[0] > 0, "Predictions must have shape [B,16,5].")
    _require(target.shape == predicted.shape and predicted.dtype.kind in "iuf" and target.dtype.kind in "iuf", "Targets and predictions must be matching real arrays.")
    _require(valid.shape == predicted.shape[:2] and valid.dtype == np.bool_ and bool(valid.any()), "valid_mask must be a nonempty boolean [B,16] mask.")
    _require(indices.shape == valid.shape and indices.dtype.kind in "iu" and bool((indices[~valid] == -1).all()) and bool((indices[valid] >= 0).all()), "Source indices must be real row indices, with -1 only at padding.")
    predicted = predicted[valid].astype(np.float64)
    target = target[valid].astype(np.float64)
    rows = indices[valid]
    _require(bool(np.isfinite(predicted).all()) and bool(np.isfinite(target).all()), "Valid predictions and targets must remain finite.")
    _require(bool(np.isin(target[:, 3:], (0, 1)).all()), "Valid jaw targets must be binary commands.")
    bits = threshold_jaw_commands(predicted[:, 3:])
    reference_bits = target[:, 3:].astype(np.uint8)
    try:
        with np.errstate(over="raise", invalid="raise"):
            errors = np.abs(predicted[:, :3] - target[:, :3])
            mae, maximum = float(errors.mean()), float(errors.max())
    except FloatingPointError as exc:
        raise OverfitDataError("Cartesian evaluation must remain finite.") from exc
    _require(bool(np.isfinite((mae, maximum)).all()), "Cartesian evaluation must remain finite.")
    accuracy = float(np.mean(bits == reference_bits))
    unique_rows = np.unique(rows)
    _require(np.array_equal(unique_rows, np.arange(len(unique_rows))), "Evaluation source rows must cover the episode contiguously from row zero.")
    predicted_sequence, target_sequence = [], []
    consistent = True
    for row in unique_rows:
        copies = rows == row
        row_targets, row_bits = target[copies], bits[copies]
        _require(np.array_equal(row_targets, np.broadcast_to(row_targets[0], row_targets.shape)), "Targets for the same source row disagree.")
        consistent = consistent and bool((row_bits == row_bits[0]).all())
        predicted_sequence.append(row_bits[0].tolist())
        target_sequence.append(row_targets[0, 3:].astype(np.uint8).tolist())

    def transitions(sequence: list[list[int]], jaw: int) -> list[int]:
        values = [pair[jaw] for pair in sequence]
        return [value for index, value in enumerate(values) if index == 0 or value != values[index - 1]]

    order_passed = consistent and all(transitions(predicted_sequence, jaw) == transitions(target_sequence, jaw) == [1, 0, 1] for jaw in range(2))
    result = {
        "finite": True, "cartesian_mae_m": mae, "cartesian_max_abs_error_m": maximum,
        "cartesian_mae_limit_m": CARTESIAN_MAE_LIMIT_M, "cartesian_max_abs_limit_m": CARTESIAN_MAX_ABS_LIMIT_M,
        "jaw_accuracy": accuracy, "jaw_accuracy_percent": accuracy * 100, "jaw_sequence_order_passed": bool(order_passed),
        "source_rows_consistent": consistent, "valid_target_count": int(valid.sum()), "valid_jaw_bit_count": int(valid.sum() * 2),
        "jaw_decision_rule": "finite [0,1]; >=0.5 means 1=open, <0.5 means 0=closed; no clipping",
        "predicted_source_jaw_states": predicted_sequence, "target_source_jaw_states": target_sequence,
        "cartesian_mae_passed": mae <= CARTESIAN_MAE_LIMIT_M, "cartesian_max_abs_error_passed": maximum <= CARTESIAN_MAX_ABS_LIMIT_M,
        "jaw_accuracy_passed": accuracy == 1.0,
    }
    result["passed"] = bool(result["cartesian_mae_passed"] and result["cartesian_max_abs_error_passed"] and result["jaw_accuracy_passed"] and order_passed)
    return result






