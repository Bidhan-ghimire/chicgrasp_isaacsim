"""Isaac-independent Gate 3 deployment mapping and executed-slice scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from .interface import (
    EXECUTION_HORIZON,
    OBSERVATION_HORIZON,
    POLICY_ACTION_DIM,
    PREDICTION_HORIZON,
    validate_policy_action,
)
from .overfit_data import (
    CARTESIAN_MAE_LIMIT_M,
    CARTESIAN_MAX_ABS_LIMIT_M,
    evaluate_predictions,
)


OFFICIAL_EXECUTION_START: Final[int] = OBSERVATION_HORIZON - 1
OFFICIAL_EXECUTION_STOP: Final[int] = OFFICIAL_EXECUTION_START + EXECUTION_HORIZON
DEPLOYMENT_WINDOW_STARTS: Final[tuple[int, int]] = (-1, 11)
DEPLOYMENT_EPISODE_FRAME_COUNT: Final[int] = 21


class DeploymentMappingError(ValueError):
    """A prediction archive or mapping does not satisfy the fixed Gate 3 contract."""


@dataclass(frozen=True)
class AdaptedSchedulerPrediction:
    scheduler_input: NDArray[np.float64]
    executable_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class DeploymentActionMapping:
    window_start: int
    local_full_prediction_slot: int
    returned_action_slot: int
    scheduler_slot: int
    absolute_frame: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class DeploymentSelection:
    predictions: NDArray[np.float64]
    targets: NDArray[np.float64]
    mappings: tuple[DeploymentActionMapping, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DeploymentMappingError(message)


def adapt_returned_actions_for_scheduler(returned_actions: Any) -> AdaptedSchedulerPrediction:
    """Put the official 12 returned actions into the scheduler's first 12 rows."""

    candidate = np.asarray(returned_actions)
    _require(
        candidate.shape == (EXECUTION_HORIZON, POLICY_ACTION_DIM),
        f"Official returned actions must have shape ({EXECUTION_HORIZON},{POLICY_ACTION_DIM}).",
    )
    validated = validate_policy_action(candidate)
    scheduler_input = np.empty((PREDICTION_HORIZON, POLICY_ACTION_DIM), dtype=np.float64)
    scheduler_input[:EXECUTION_HORIZON] = validated
    scheduler_input[EXECUTION_HORIZON:] = validated[-1]
    scheduler_input.setflags(write=False)
    executable_mask = np.zeros(PREDICTION_HORIZON, dtype=np.bool_)
    executable_mask[:EXECUTION_HORIZON] = True
    executable_mask.setflags(write=False)
    return AdaptedSchedulerPrediction(
        scheduler_input=scheduler_input,
        executable_mask=executable_mask,
    )




def make_deployment_loss_mask(
    *,
    window_starts: Any,
    source_indices: Any,
    valid_mask: Any,
    episode_length: int,
    verified_mapping: tuple[DeploymentActionMapping, ...],
    observation_horizon: int = OBSERVATION_HORIZON,
    prediction_horizon: int = PREDICTION_HORIZON,
    execution_horizon: int = EXECUTION_HORIZON,
    training_window_stride: int | None = None,
    training_window_offset: int = 0,
) -> NDArray[np.bool_]:
    """Default deployment loss, or explicit training-only candidate eligibility."""
    _require(type(training_window_offset) is int and training_window_offset >= 0,
             "training_window_offset must be a nonnegative integer.")
    if training_window_stride is None:
        _require(training_window_offset == 0, "Training offset requires an explicit stride.")
    else:
        _require(type(training_window_stride) is int and training_window_stride > 0,
                 "training_window_stride must be a positive integer.")
        _require(training_window_offset < training_window_stride,
                 "training_window_offset must be smaller than training_window_stride.")

    for name, value in (
        ("episode_length", episode_length), ("observation_horizon", observation_horizon),
        ("prediction_horizon", prediction_horizon), ("execution_horizon", execution_horizon),
    ):
        _require(type(value) is int and value > 0, f"{name} must be a positive integer.")
    offset = observation_horizon - 1
    _require(offset + execution_horizon <= prediction_horizon, "Execution slice exceeds the prediction horizon.")
    starts, indices, valid = np.asarray(window_starts), np.asarray(source_indices), np.asarray(valid_mask)
    _require(starts.ndim == 1 and starts.dtype.kind in "iu", "window_starts must be an integer vector.")
    _require(len(np.unique(starts)) == len(starts), "window_starts must be unique.")
    _require(valid.shape == (len(starts), prediction_horizon) and valid.dtype == np.bool_, "valid_mask must be boolean [B,P].")
    _require(indices.shape == valid.shape and indices.dtype.kind in "iu", "source_indices must be integer [B,P].")
    raw_indices = starts[:, None] + np.arange(prediction_horizon, dtype=np.int64)
    expected_valid = (raw_indices >= 0) & (raw_indices < episode_length)
    _require(np.array_equal(valid, expected_valid), "Dataset padding does not match the episode bounds.")
    _require(np.array_equal(indices, np.where(expected_valid, raw_indices, -1)), "Dataset source indices do not match window slots or padding.")
    expected_mapping = tuple(
        DeploymentActionMapping(
            window_start=chunk_start - offset,
            local_full_prediction_slot=offset + returned_slot,
            returned_action_slot=returned_slot,
            scheduler_slot=returned_slot,
            absolute_frame=chunk_start + returned_slot,
        )
        for chunk_start in range(0, episode_length, execution_horizon)
        for returned_slot in range(min(execution_horizon, episode_length - chunk_start))
    )
    _require(
        isinstance(verified_mapping, tuple)
        and all(isinstance(item, DeploymentActionMapping) for item in verified_mapping)
        and verified_mapping == expected_mapping,
        "Verified deployment mapping differs from the horizon-derived schedule.",
    )
    rows = {int(start): row for row, start in enumerate(starts)}
    mask = np.zeros(valid.shape, dtype=np.bool_)
    for item in expected_mapping:
        _require(item.window_start in rows, f"Dataset is missing deployment window {item.window_start}.")
        row, slot = rows[item.window_start], item.local_full_prediction_slot
        _require(bool(valid[row, slot]) and int(indices[row, slot]) == item.absolute_frame, "Deployment loss row is padded or misaligned.")
        mask[row, slot] = True
    _require(
        np.array_equal(np.sort(indices[mask]), np.arange(episode_length, dtype=np.int64)),
        "Deployment loss must include every episode frame exactly once.",
    )
    if training_window_stride is not None:
        # Candidate/padding validation and the native deployment mapping stay
        # unchanged. Only training-row eligibility is independent of that grid.
        first_executable = starts + offset
        eligible = ((first_executable >= training_window_offset)
                    & (first_executable < episode_length)
                    & ((first_executable - training_window_offset) % training_window_stride == 0))
        selected_slots = ((np.arange(prediction_horizon) >= offset)
                          & (np.arange(prediction_horizon) < offset + execution_horizon))
        mask = np.ascontiguousarray(valid & eligible[:, None] & selected_slots[None, :])
        _require(bool(mask.any()), "Training selection contains no valid targets.")
    mask.setflags(write=False)
    return mask




