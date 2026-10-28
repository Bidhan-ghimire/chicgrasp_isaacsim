"""Pure Gate 1 policy-interface contract.

This module deliberately has no Isaac, dataset, model, contact, attachment, or
success-classification dependency.  It validates the only values that may cross
the learned-policy boundary and owns the integer 10 Hz to 60 Hz scheduler.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray


POLICY_OBSERVATION_KEYS: Final[tuple[str, ...]] = (
    "camera_left_rgb",
    "camera_right_rgb",
    "arm_joint_pos",
    "jaw_state",
)

ARM_JOINT_NAMES: Final[tuple[str, ...]] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
JAW_STATE_NAMES: Final[tuple[str, str]] = ("gL", "gR")
POLICY_ACTION_NAMES: Final[tuple[str, ...]] = ("x", "y", "z", "gL", "gR")
POLICY_ACTION_DIM: Final[int] = 5

CAMERA_HEIGHT: Final[int] = 240
CAMERA_WIDTH: Final[int] = 320
CAMERA_CHANNELS: Final[int] = 3

FIXED_TCP_QUATERNION_BASE_WXYZ: Final[tuple[float, float, float, float]] = (
    0.7575648913025227,
    -0.6527596722606701,
    0.0003756575301500135,
    -0.0003234460691291637,
)
POLICY_TCP_PARENT_BODY: Final[str] = "gripper_base_link"
POLICY_TCP_LOCAL_POSITION_M: Final[tuple[float, float, float]] = (0.0, 0.12, 0.0)
POLICY_TCP_LOCAL_QUATERNION_WXYZ: Final[tuple[float, float, float, float]] = (1.0, 0.0, 0.0, 0.0)
ACTION_POSITION_FRAME: Final[str] = "robot_base"
ACTION_POSITION_UNITS: Final[str] = "m"
TCP_POSITION_FRAME: Final[str] = ACTION_POSITION_FRAME
TCP_POSITION_UNITS: Final[str] = ACTION_POSITION_UNITS

TCP_X_RANGE_M: Final[tuple[float, float]] = (0.64, 0.70)
TCP_Y_RANGE_M: Final[tuple[float, float]] = (-0.17, 0.19)
TCP_Z_RANGE_M: Final[tuple[float, float]] = (0.10, 0.57)

JAW_OPEN_BIT: Final[int] = 1
JAW_CLOSED_BIT: Final[int] = 0
# Gate 6 Stage 0: controller intent from the accepted, already-issued command.
# Historical measured-label recordings keep their original schema/definition.
JAW_STATE_SEMANTICS: Final[str] = "commanded-jaw-state/v1"
JAW_OBSERVATION_THRESHOLD_M: Final[float] = 0.0
JAW_ACTION_THRESHOLD: Final[float] = 0.5

POLICY_HZ: Final[int] = 10
CONTROL_HZ: Final[int] = 60
CONTROL_TICKS_PER_POLICY_ACTION: Final[int] = 6
PREDICTION_HORIZON: Final[int] = 16
EXECUTION_HORIZON: Final[int] = 12
OBSERVATION_HORIZON: Final[int] = 2

if CONTROL_HZ != POLICY_HZ * CONTROL_TICKS_PER_POLICY_ACTION:
    raise RuntimeError("The Gate 1 policy/control clock ratio is inconsistent.")


class PolicyInterfaceError(ValueError):
    """Base error for a rejected Gate 1 interface value."""


class InvalidPolicyObservation(PolicyInterfaceError):
    """Raised when a policy observation violates the exact four-field contract."""


class InvalidPolicyAction(PolicyInterfaceError):
    """Raised when a public policy action violates the exact five-value contract."""


class InvalidPolicyFrame(PolicyInterfaceError):
    """Raised when raw state cannot form a valid policy frame."""


class TerminalRowError(PolicyInterfaceError):
    """Raised when a lifecycle-terminal row is offered to a policy/data path."""


class SchedulerStateError(PolicyInterfaceError):
    """Raised when scheduler operations are attempted out of contract order."""


@dataclass(frozen=True, slots=True)
class PolicyObservation:
    """The complete learned observation at one 10 Hz policy boundary."""

    camera_left_rgb: NDArray[np.uint8]
    camera_right_rgb: NDArray[np.uint8]
    arm_joint_pos: NDArray[np.float32]
    jaw_state: NDArray[np.uint8]

    def as_mapping(self) -> dict[str, NDArray[Any]]:
        """Return the fields in the sole permitted policy-key order."""

        return {
            "camera_left_rgb": self.camera_left_rgb,
            "camera_right_rgb": self.camera_right_rgb,
            "arm_joint_pos": self.arm_joint_pos,
            "jaw_state": self.jaw_state,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticFrame:
    """Privileged sidecar values that are never valid policy observations.

    Requested TCP movement, actual TCP movement, and chicken movement have
    intentionally distinct fields.  No one is a proxy for either of the others.
    """

    requested_tcp_pose_base_wxyz: tuple[float, ...] | None = None
    actual_tcp_pose_base_wxyz: tuple[float, ...] | None = None
    chicken_pose_world_wxyz: tuple[float, ...] | None = None
    chicken_velocity_world: tuple[float, ...] | None = None
    requested_tcp_movement_m: tuple[float, float, float] | None = None
    actual_tcp_movement_m: tuple[float, float, float] | None = None
    chicken_movement_m: tuple[float, float, float] | None = None
    table_support: bool | None = None
    contact_valid: bool | None = None
    contact_identity: tuple[str, ...] = ()
    reset: bool = False
    terminated: bool = False
    truncated: bool = False
    automatic_reset: bool = False
    invalid_policy_frame: bool = False
    jaw_pair_disagreement: bool = False
    invalid_reason: str | None = None
    attachment_state: bool | None = None
    outcome_label: str | None = None


def _real_array(value: Any, *, name: str, error_type: type[PolicyInterfaceError]) -> NDArray[Any]:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise error_type(f"{name} must contain finite real values.")
    if not np.all(np.isfinite(array)):
        raise error_type(f"{name} must contain finite real values.")
    return array


def _frozen_copy(array: NDArray[Any]) -> NDArray[Any]:
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


def validate_policy_observation(
    observation: PolicyObservation | Mapping[str, Any],
) -> PolicyObservation:
    """Validate and freeze one exact four-field policy observation.

    Mappings must contain the four keys in the declared order.  Unknown or
    privileged extras are rejected rather than silently discarded.
    """

    if isinstance(observation, DiagnosticFrame):
        raise TypeError("DiagnosticFrame is sidecar-only and cannot be a policy observation.")

    if isinstance(observation, PolicyObservation):
        values = observation.as_mapping()
    elif isinstance(observation, Mapping):
        keys = tuple(observation.keys())
        if keys != POLICY_OBSERVATION_KEYS:
            raise InvalidPolicyObservation(
                f"Policy observation keys must be exactly {POLICY_OBSERVATION_KEYS}; received {keys}."
            )
        values = observation
    else:
        raise TypeError("A policy observation must be a PolicyObservation or exact ordered mapping.")

    left = np.asarray(values["camera_left_rgb"])
    right = np.asarray(values["camera_right_rgb"])
    arm = np.asarray(values["arm_joint_pos"])
    jaw = np.asarray(values["jaw_state"])

    image_shape = (CAMERA_HEIGHT, CAMERA_WIDTH, CAMERA_CHANNELS)
    for name, image in (("camera_left_rgb", left), ("camera_right_rgb", right)):
        if image.shape != image_shape:
            raise InvalidPolicyObservation(f"{name} must have shape {image_shape}; received {image.shape}.")
        if image.dtype != np.dtype(np.uint8):
            raise InvalidPolicyObservation(f"{name} must have dtype uint8; received {image.dtype}.")

    if arm.shape != (len(ARM_JOINT_NAMES),):
        raise InvalidPolicyObservation(
            f"arm_joint_pos must have shape ({len(ARM_JOINT_NAMES)},); received {arm.shape}."
        )
    if arm.dtype != np.dtype(np.float32):
        raise InvalidPolicyObservation(f"arm_joint_pos must have dtype float32; received {arm.dtype}.")
    if not np.all(np.isfinite(arm)):
        raise InvalidPolicyObservation("arm_joint_pos must contain finite values.")

    if jaw.shape != (len(JAW_STATE_NAMES),):
        raise InvalidPolicyObservation(f"jaw_state must have shape ({len(JAW_STATE_NAMES)},); received {jaw.shape}.")
    if jaw.dtype != np.dtype(np.uint8):
        raise InvalidPolicyObservation(f"jaw_state must have dtype uint8; received {jaw.dtype}.")
    if not np.all((jaw == JAW_CLOSED_BIT) | (jaw == JAW_OPEN_BIT)):
        raise InvalidPolicyObservation("jaw_state values must be exactly 0 or 1.")

    return PolicyObservation(
        camera_left_rgb=_frozen_copy(left),
        camera_right_rgb=_frozen_copy(right),
        arm_joint_pos=_frozen_copy(arm),
        jaw_state=_frozen_copy(jaw),
    )


def policy_normalizer_schema(observation: PolicyObservation | None = None) -> dict[str, dict[str, Any]]:
    """Return the fixed model-side schema without accepting diagnostic data."""

    if observation is not None:
        if not isinstance(observation, PolicyObservation):
            raise TypeError("Only PolicyObservation may be checked against the policy schema.")
        validate_policy_observation(observation)
    return {
        "camera_left_rgb": {"shape": (3, CAMERA_HEIGHT, CAMERA_WIDTH), "type": "rgb"},
        "camera_right_rgb": {"shape": (3, CAMERA_HEIGHT, CAMERA_WIDTH), "type": "rgb"},
        "arm_joint_pos": {"shape": (len(ARM_JOINT_NAMES),), "type": "low_dim"},
        "jaw_state": {"shape": (len(JAW_STATE_NAMES),), "type": "low_dim"},
    }


def _observation_windows(
    observations: Sequence[PolicyObservation] | Sequence[Sequence[PolicyObservation]],
) -> list[list[PolicyObservation]]:
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise TypeError("Model packing requires PolicyObservation sequences.")
    if len(observations) == 0:
        raise InvalidPolicyObservation("At least one two-observation window is required.")

    if all(isinstance(item, PolicyObservation) for item in observations):
        if len(observations) != OBSERVATION_HORIZON:
            raise InvalidPolicyObservation(
                f"A model window must contain exactly {OBSERVATION_HORIZON} observations."
            )
        return [list(observations)]  # type: ignore[list-item]

    windows: list[list[PolicyObservation]] = []
    for window in observations:
        if not isinstance(window, Sequence) or isinstance(window, (str, bytes)):
            raise TypeError("Each model batch item must be a PolicyObservation sequence.")
        if len(window) != OBSERVATION_HORIZON:
            raise InvalidPolicyObservation(
                f"Each model window must contain exactly {OBSERVATION_HORIZON} observations."
            )
        if not all(isinstance(item, PolicyObservation) for item in window):
            raise TypeError("Diagnostic or mapping records cannot enter model packing.")
        windows.append(list(window))
    return windows


def pack_model_observation(
    observations: Sequence[PolicyObservation] | Sequence[Sequence[PolicyObservation]],
) -> dict[str, NDArray[np.float32]]:
    """Pack one or more validated two-step windows into model tensors.

    RGB is converted once from byte NHWC storage to contiguous float32 NCHW
    in ``[0, 1]``.  Low-dimensional fields remain separate.
    """

    windows = _observation_windows(observations)
    validated = [[validate_policy_observation(frame) for frame in window] for window in windows]

    left = np.stack([[frame.camera_left_rgb for frame in window] for window in validated])
    right = np.stack([[frame.camera_right_rgb for frame in window] for window in validated])
    arm = np.stack([[frame.arm_joint_pos for frame in window] for window in validated])
    jaw = np.stack([[frame.jaw_state for frame in window] for window in validated])

    left_model = np.ascontiguousarray(left.transpose(0, 1, 4, 2, 3), dtype=np.float32)
    right_model = np.ascontiguousarray(right.transpose(0, 1, 4, 2, 3), dtype=np.float32)
    left_model /= np.float32(255.0)
    right_model /= np.float32(255.0)

    return {
        "camera_left_rgb": left_model,
        "camera_right_rgb": right_model,
        "arm_joint_pos": np.ascontiguousarray(arm, dtype=np.float32),
        "jaw_state": np.ascontiguousarray(jaw, dtype=np.float32),
    }


def jaw_positions_to_state(jaw_joint_positions_m: Any) -> NDArray[np.uint8]:
    """Historical measured-label decoder, not the current live jaw_state.

    Each joint is thresholded independently at exactly 0.0 m.  Disagreement
    within either physical pair invalidates the complete frame.
    """

    positions = _real_array(
        jaw_joint_positions_m,
        name="jaw_joint_positions_m",
        error_type=InvalidPolicyFrame,
    )
    if positions.ndim == 0 or positions.shape[-1] != 4:
        raise InvalidPolicyFrame(
            f"jaw_joint_positions_m must have final dimension 4; received {positions.shape}."
        )

    joint_bits = (positions >= JAW_OBSERVATION_THRESHOLD_M).astype(np.uint8)
    left_agrees = joint_bits[..., 0] == joint_bits[..., 1]
    right_agrees = joint_bits[..., 2] == joint_bits[..., 3]
    if not np.all(left_agrees & right_agrees):
        raise InvalidPolicyFrame("Jaw-pair disagreement invalidates the policy frame.")

    return np.stack((joint_bits[..., 0], joint_bits[..., 2]), axis=-1).astype(np.uint8, copy=False)


def threshold_jaw_commands(raw_jaw_commands: Any) -> NDArray[np.uint8]:
    """Validate raw ``[gL,gR]`` in ``[0,1]`` and threshold at 0.5."""

    commands = _real_array(raw_jaw_commands, name="raw_jaw_commands", error_type=InvalidPolicyAction)
    if commands.ndim == 0 or commands.shape[-1] != 2:
        raise InvalidPolicyAction(f"raw_jaw_commands must have final dimension 2; received {commands.shape}.")
    if np.any(commands < 0.0) or np.any(commands > 1.0):
        raise InvalidPolicyAction("Raw jaw commands must be within the inclusive interval [0,1].")
    return (commands >= JAW_ACTION_THRESHOLD).astype(np.uint8)


def expand_jaw_targets(
    jaw_bits: Any,
    *,
    open_position_m: float,
    closed_position_m: float,
) -> NDArray[np.float64]:
    """Fan ``[gL,gR]`` bits to four existing continuous actuator endpoints."""

    bits = _real_array(jaw_bits, name="jaw_bits", error_type=InvalidPolicyAction)
    if bits.ndim == 0 or bits.shape[-1] != 2:
        raise InvalidPolicyAction(f"jaw_bits must have final dimension 2; received {bits.shape}.")
    if not np.all((bits == JAW_CLOSED_BIT) | (bits == JAW_OPEN_BIT)):
        raise InvalidPolicyAction("jaw_bits must contain exactly 0 or 1.")

    endpoint_values: list[float] = []
    for name, endpoint in (
        ("open_position_m", open_position_m),
        ("closed_position_m", closed_position_m),
    ):
        value = _real_array(endpoint, name=name, error_type=InvalidPolicyAction)
        if value.shape != ():
            raise InvalidPolicyAction(f"{name} must be one finite real scalar.")
        endpoint_values.append(float(value))
    endpoints = np.asarray(endpoint_values, dtype=np.float64)
    left = np.where(bits[..., 0] == JAW_OPEN_BIT, endpoints[0], endpoints[1])
    right = np.where(bits[..., 1] == JAW_OPEN_BIT, endpoints[0], endpoints[1])
    return np.stack((left, left, right, right), axis=-1).astype(np.float64, copy=False)


def validate_policy_action(action: Any) -> NDArray[np.float64]:
    """Validate exact public actions ``[...,x,y,z,gL,gR]`` by rejection.

    Input values are converted to float64 before comparison with the exact
    approved bounds.  A lower-precision value whose represented value lies
    outside an interval is rejected even if its source literal named an
    endpoint; no dtype-dependent tolerance or rounded bound can broaden the
    approved workspace.
    """

    values = _real_array(action, name="policy action", error_type=InvalidPolicyAction)
    if values.ndim == 0 or values.shape[-1] != POLICY_ACTION_DIM:
        received = values.shape[-1] if values.ndim else 0
        raise InvalidPolicyAction(
            f"Policy action final dimension must be exactly {POLICY_ACTION_DIM}; received {received}."
        )

    validated = np.ascontiguousarray(values, dtype=np.float64).copy()
    for coordinate_index, (coordinate_name, bounds) in enumerate(
        (("x", TCP_X_RANGE_M), ("y", TCP_Y_RANGE_M), ("z", TCP_Z_RANGE_M))
    ):
        lower, upper = bounds
        coordinate = validated[..., coordinate_index]
        if np.any(coordinate < lower) or np.any(coordinate > upper):
            raise InvalidPolicyAction(
                f"{coordinate_name} must be within the inclusive interval [{bounds[0]},{bounds[1]}] m."
            )

    raw_jaws = validated[..., 3:5]
    if np.any(raw_jaws < 0.0) or np.any(raw_jaws > 1.0):
        raise InvalidPolicyAction("Raw gL and gR must each be within the inclusive interval [0,1].")

    validated.setflags(write=False)
    return validated


def append_fixed_tcp_quaternion(action: Any) -> NDArray[np.float64]:
    """Create internal absolute DLS pose commands from accepted 5-D actions."""

    validated = validate_policy_action(action)
    quaternion = np.asarray(FIXED_TCP_QUATERNION_BASE_WXYZ, dtype=np.float64)
    broadcast_shape = validated.shape[:-1] + (len(FIXED_TCP_QUATERNION_BASE_WXYZ),)
    fixed = np.broadcast_to(quaternion, broadcast_shape)
    return np.concatenate((validated[..., :3], fixed), axis=-1)


def row_is_policy_eligible(diagnostic: DiagnosticFrame) -> bool:
    """Return whether a row may enter policy history or dataset storage."""

    if not isinstance(diagnostic, DiagnosticFrame):
        raise TypeError("Lifecycle eligibility requires a DiagnosticFrame sidecar.")
    return not (
        diagnostic.reset
        or diagnostic.terminated
        or diagnostic.truncated
        or diagnostic.automatic_reset
        or diagnostic.invalid_policy_frame
        or diagnostic.jaw_pair_disagreement
    )


def row_is_success_eligible(diagnostic: DiagnosticFrame) -> bool:
    """Reject the same terminal/invalid rows before any later success predicate."""

    return row_is_policy_eligible(diagnostic)


def require_policy_row_eligible(diagnostic: DiagnosticFrame) -> None:
    """Raise instead of allowing a terminal/automatic-reset row downstream."""

    if row_is_policy_eligible(diagnostic):
        return
    reasons = [
        name
        for name in (
            "reset",
            "terminated",
            "truncated",
            "automatic_reset",
            "invalid_policy_frame",
            "jaw_pair_disagreement",
        )
        if getattr(diagnostic, name)
    ]
    raise TerminalRowError(f"Row is sidecar-only and ineligible: {', '.join(reasons)}.")


class PolicyChunkScheduler:
    """Integer scheduler for 16-action predictions and 12-action execution."""

    def __init__(self) -> None:
        self._observation_history: deque[PolicyObservation] = deque(maxlen=OBSERVATION_HORIZON)
        self._prediction: NDArray[np.float64] | None = None
        self._chunk_control_tick = 0
        self._policy_step = 0
        self._boundary_observation_due = False
        self._awaiting_post_chunk_observation = False

    @property
    def observation_history(self) -> tuple[PolicyObservation, ...]:
        return tuple(self._observation_history)

    @property
    def ready_for_inference(self) -> bool:
        return (
            len(self._observation_history) == OBSERVATION_HORIZON
            and self._prediction is None
            and not self._boundary_observation_due
            and not self._awaiting_post_chunk_observation
        )

    @property
    def has_active_chunk(self) -> bool:
        return self._prediction is not None

    @property
    def boundary_observation_due(self) -> bool:
        return self._boundary_observation_due

    @property
    def chunk_control_tick(self) -> int:
        return self._chunk_control_tick

    @property
    def policy_step(self) -> int:
        return self._policy_step

    @property
    def current_action_index(self) -> int | None:
        if self._prediction is None:
            return None
        return self._chunk_control_tick // CONTROL_TICKS_PER_POLICY_ACTION

    @property
    def current_repeat_index(self) -> int | None:
        if self._prediction is None:
            return None
        return self._chunk_control_tick % CONTROL_TICKS_PER_POLICY_ACTION

    @property
    def remaining_control_ticks(self) -> int:
        if self._prediction is None:
            return 0
        return EXECUTION_HORIZON * CONTROL_TICKS_PER_POLICY_ACTION - self._chunk_control_tick

    def _discard_all(self) -> None:
        self._prediction = None
        self._chunk_control_tick = 0
        self._policy_step = 0
        self._boundary_observation_due = False
        self._awaiting_post_chunk_observation = False
        self._observation_history.clear()

    def reset(self) -> None:
        """Explicitly discard the chunk and all observation history."""

        self._discard_all()

    def interrupt(self) -> None:
        """Discard all state after any lifecycle or invalid-frame interruption."""

        self._discard_all()

    def record_policy_observation(
        self,
        observation: PolicyObservation,
        *,
        reset: bool = False,
        terminated: bool = False,
        truncated: bool = False,
        automatic_reset: bool = False,
        invalid_policy_frame: bool = False,
    ) -> bool:
        """Append one valid boundary observation or reject a terminal row."""

        if reset or terminated or truncated or automatic_reset or invalid_policy_frame:
            self._discard_all()
            return False

        if self._prediction is not None and not self._boundary_observation_due:
            raise SchedulerStateError("A policy observation is accepted only after each sixth control tick.")
        if self._awaiting_post_chunk_observation and not self._boundary_observation_due:
            raise SchedulerStateError("The completed chunk's boundary observation is not due.")

        if not isinstance(observation, PolicyObservation):
            self._discard_all()
            raise TypeError("The scheduler accepts only a validated PolicyObservation frame.")
        try:
            frozen = validate_policy_observation(observation)
        except (InvalidPolicyObservation, TypeError):
            self._discard_all()
            raise

        self._observation_history.append(frozen)
        self._policy_step += 1
        if self._boundary_observation_due:
            self._boundary_observation_due = False
            self._awaiting_post_chunk_observation = False
        return True

    def load_prediction(self, prediction: Any) -> None:
        """Freeze one exact ``[16,5]`` prediction after a valid two-frame history."""

        if not self.ready_for_inference:
            raise SchedulerStateError("A fresh valid two-observation history is required before inference.")
        candidate = np.asarray(prediction)
        if candidate.shape != (PREDICTION_HORIZON, POLICY_ACTION_DIM):
            raise InvalidPolicyAction(
                "Prediction must have shape "
                f"({PREDICTION_HORIZON},{POLICY_ACTION_DIM}); received {candidate.shape}."
            )
        validated = validate_policy_action(candidate)
        self._prediction = np.ascontiguousarray(validated).copy()
        self._prediction.setflags(write=False)
        self._chunk_control_tick = 0

    def next_control_action(
        self,
        *,
        reset: bool = False,
        terminated: bool = False,
        truncated: bool = False,
        automatic_reset: bool = False,
        invalid_policy_frame: bool = False,
    ) -> NDArray[np.float64] | None:
        """Issue one of exactly six copies of the current executed action."""

        if reset or terminated or truncated or automatic_reset or invalid_policy_frame:
            self._discard_all()
            return None
        if self._boundary_observation_due:
            raise SchedulerStateError("Record the atomic policy observation before the next control tick.")
        if self._prediction is None:
            return None

        action_index = self._chunk_control_tick // CONTROL_TICKS_PER_POLICY_ACTION
        if action_index >= EXECUTION_HORIZON:
            raise SchedulerStateError("The scheduler attempted to execute outside indices 0 through 11.")
        action = self._prediction[action_index].copy()
        self._chunk_control_tick += 1

        if self._chunk_control_tick % CONTROL_TICKS_PER_POLICY_ACTION == 0:
            self._boundary_observation_due = True
        if self._chunk_control_tick == EXECUTION_HORIZON * CONTROL_TICKS_PER_POLICY_ACTION:
            self._prediction = None
            self._awaiting_post_chunk_observation = True
        return action
