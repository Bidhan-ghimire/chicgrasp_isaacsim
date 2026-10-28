"""Strict, Isaac-independent Gate 2 episode storage and replay checks.

The policy loader returns only the six declared policy arrays.  Privileged
``DiagnosticFrame`` records live in a separately loaded, manifest-covered JSON
sidecar.  Every payload is byte-counted and SHA-256 verified before parsing.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import tempfile
from .episode_publication import Publication, destination_absent
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from .interface import (
    CAMERA_CHANNELS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CONTROL_HZ,
    CONTROL_TICKS_PER_POLICY_ACTION,
    JAW_CLOSED_BIT,
    JAW_OPEN_BIT,
    JAW_STATE_SEMANTICS,
    POLICY_ACTION_DIM,
    POLICY_HZ,
    DiagnosticFrame,
    TerminalRowError,
    require_policy_row_eligible,
    validate_policy_action,
)


POLICY_ARRAY_KEYS: Final[tuple[str, ...]] = (
    "timestamp_ns",
    "camera_left_rgb",
    "camera_right_rgb",
    "arm_joint_pos",
    "jaw_state",
    "action",
)
POLICY_DIRECTORY: Final[str] = "policy"
EPISODE_END_FILENAME: Final[str] = "episode_end.json"
DIAGNOSTIC_SIDECAR_FILENAME: Final[str] = "diagnostic_sidecar.json"
MANIFEST_FILENAME: Final[str] = "manifest.json"

EPISODE_SCHEMA_VERSION: Final[int] = 1
# Version 1 remains the exact historical schema; writers must opt in explicitly.
COMMANDED_EPISODE_SCHEMA_VERSION: Final[int] = 2
DIAGNOSTIC_SCHEMA_VERSION: Final[int] = 1
MANIFEST_SCHEMA_VERSION: Final[int] = 1
TIMESTAMP_PERIOD_NS: Final[int] = 100_000_000
TIMESTAMP_TOLERANCE_NS: Final[int] = 16_666_667
DEFAULT_REPLAY_TOLERANCE_M: Final[float] = 0.002
TIMESTAMP_ORIGIN: Final[str] = "episode_relative_ns"
OBSERVATION_ACTION_ALIGNMENT: Final[str] = (
    "capture_observation_then_record_associated_action_then_execute_six_control_ticks"
)

EPISODE_END_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "episode_id",
    "task_id",
    "seed",
    "num_policy_steps",
    "policy_hz",
    "control_hz",
    "control_ticks_per_policy_action",
    "timestamp_origin",
    "observation_action_alignment",
)
COMMANDED_EPISODE_END_KEYS: Final[tuple[str, ...]] = (
    *EPISODE_END_KEYS, "jaw_state_semantics",
)

_DIAGNOSTIC_FIELD_NAMES: Final[tuple[str, ...]] = tuple(
    field.name for field in fields(DiagnosticFrame)
)
_MANIFEST_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "policy_fields",
    "files",
)
_MANIFEST_FILE_KEYS: Final[tuple[str, ...]] = ("path", "byte_count", "sha256")
_DIAGNOSTIC_ROOT_KEYS: Final[tuple[str, ...]] = ("schema_version", "frames")
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")

_POLICY_PAYLOAD_PATHS: Final[tuple[str, ...]] = tuple(
    f"{POLICY_DIRECTORY}/{key}.npy" for key in POLICY_ARRAY_KEYS
)
_PAYLOAD_PATHS: Final[tuple[str, ...]] = (
    *_POLICY_PAYLOAD_PATHS,
    EPISODE_END_FILENAME,
    DIAGNOSTIC_SIDECAR_FILENAME,
)
_ALL_FILE_PATHS: Final[frozenset[str]] = frozenset((*_PAYLOAD_PATHS, MANIFEST_FILENAME))


class EpisodeIOError(ValueError):
    """Raised when an episode package violates the Gate 2 contract."""


def _ordered_keys(mapping: Mapping[str, Any], *, name: str, expected: tuple[str, ...]) -> None:
    keys = tuple(mapping.keys())
    if keys != expected:
        raise EpisodeIOError(f"{name} keys must be exactly {expected}; received {keys}.")


def _frozen_c_copy(array: NDArray[Any]) -> NDArray[Any]:
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


def _require_finite(array: NDArray[Any], *, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise EpisodeIOError(f"{name} must contain only finite values.")


def validate_policy_arrays(arrays: Mapping[str, Any]) -> dict[str, NDArray[Any]]:
    """Validate and freeze the sole permitted policy-episode array mapping."""

    if not isinstance(arrays, Mapping):
        raise TypeError("Policy episode arrays must be an ordered mapping.")
    _ordered_keys(arrays, name="policy episode", expected=POLICY_ARRAY_KEYS)

    values = {key: np.asarray(arrays[key]) for key in POLICY_ARRAY_KEYS}
    timestamps = values["timestamp_ns"]
    if timestamps.ndim != 1:
        raise EpisodeIOError(
            f"timestamp_ns must have shape [N]; received {timestamps.shape}."
        )
    if timestamps.dtype != np.dtype(np.int64):
        raise EpisodeIOError(
            f"timestamp_ns must have dtype int64; received {timestamps.dtype}."
        )
    sample_count = int(timestamps.shape[0])
    if sample_count < 1:
        raise EpisodeIOError("A policy episode must contain at least one sample.")
    if np.any(timestamps[1:] <= timestamps[:-1]):
        raise EpisodeIOError("timestamp_ns must be strictly increasing.")
    if np.any(timestamps < 0):
        raise EpisodeIOError("episode-relative timestamp_ns values must be non-negative.")

    for index, timestamp in enumerate(timestamps):
        expected = index * TIMESTAMP_PERIOD_NS
        error = abs(int(timestamp) - expected)
        if error > TIMESTAMP_TOLERANCE_NS:
            raise EpisodeIOError(
                "timestamp_ns must stay within one 60 Hz tick of the 10 Hz "
                f"episode clock; sample {index} has error {error} ns."
            )

    expected_shapes = {
        "camera_left_rgb": (
            sample_count,
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
            CAMERA_CHANNELS,
        ),
        "camera_right_rgb": (
            sample_count,
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
            CAMERA_CHANNELS,
        ),
        "arm_joint_pos": (sample_count, 6),
        "jaw_state": (sample_count, 2),
        "action": (sample_count, POLICY_ACTION_DIM),
    }
    expected_dtypes = {
        "camera_left_rgb": np.dtype(np.uint8),
        "camera_right_rgb": np.dtype(np.uint8),
        "arm_joint_pos": np.dtype(np.float32),
        "jaw_state": np.dtype(np.uint8),
        "action": np.dtype(np.float32),
    }
    for key in POLICY_ARRAY_KEYS[1:]:
        value = values[key]
        if value.shape != expected_shapes[key]:
            raise EpisodeIOError(
                f"{key} must have shape {expected_shapes[key]}; received {value.shape}."
            )
        if value.dtype != expected_dtypes[key]:
            raise EpisodeIOError(
                f"{key} must have dtype {expected_dtypes[key]}; received {value.dtype}."
            )

    _require_finite(values["arm_joint_pos"], name="arm_joint_pos")
    _require_finite(values["action"], name="action")
    if not np.all(
        (values["jaw_state"] == JAW_CLOSED_BIT)
        | (values["jaw_state"] == JAW_OPEN_BIT)
    ):
        raise EpisodeIOError("jaw_state values must be exactly 0 or 1.")

    action_jaws = values["action"][:, 3:5]
    if not np.all(
        (action_jaws == np.float32(JAW_CLOSED_BIT))
        | (action_jaws == np.float32(JAW_OPEN_BIT))
    ):
        raise EpisodeIOError("Stored action gL and gR values must be exactly 0 or 1.")
    try:
        validate_policy_action(values["action"])
    except (TypeError, ValueError) as exc:
        raise EpisodeIOError(f"Stored action violates the frozen 5-D contract: {exc}") from exc

    return {key: _frozen_c_copy(values[key]) for key in POLICY_ARRAY_KEYS}


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _validate_episode_end(
    episode_end: Mapping[str, Any], *, sample_count: int
) -> dict[str, Any]:
    if not isinstance(episode_end, Mapping):
        raise TypeError("episode_end must be an ordered scalar mapping.")
    version = episode_end.get("schema_version")
    if not _is_int(version) or version not in (EPISODE_SCHEMA_VERSION, COMMANDED_EPISODE_SCHEMA_VERSION):
        raise EpisodeIOError("episode_end schema_version must be 1 (historical) or 2 (commanded).")
    expected_keys = COMMANDED_EPISODE_END_KEYS if version == COMMANDED_EPISODE_SCHEMA_VERSION else EPISODE_END_KEYS
    _ordered_keys(episode_end, name="episode_end", expected=expected_keys)
    if version == COMMANDED_EPISODE_SCHEMA_VERSION and episode_end["jaw_state_semantics"] != JAW_STATE_SEMANTICS:
        raise EpisodeIOError("episode_end jaw_state_semantics must identify commanded-jaw-state/v1.")
    for key in ("episode_id", "task_id"):
        value = episode_end[key]
        if not isinstance(value, str) or not value.strip():
            raise EpisodeIOError(f"episode_end {key} must be a non-empty string.")
    if not _is_int(episode_end["seed"]) or int(episode_end["seed"]) < 0:
        raise EpisodeIOError("episode_end seed must be a non-negative integer.")
    if not _is_int(episode_end["num_policy_steps"]):
        raise EpisodeIOError("episode_end num_policy_steps must be an integer.")
    if int(episode_end["num_policy_steps"]) != sample_count:
        raise EpisodeIOError(
            "episode_end num_policy_steps must equal the policy-array length."
        )

    required_constants = {
        "policy_hz": POLICY_HZ,
        "control_hz": CONTROL_HZ,
        "control_ticks_per_policy_action": CONTROL_TICKS_PER_POLICY_ACTION,
    }
    for key, expected in required_constants.items():
        if not _is_int(episode_end[key]) or int(episode_end[key]) != expected:
            raise EpisodeIOError(
                f"episode_end {key} must be {expected!r}; received {episode_end[key]!r}."
            )
    required_strings = {
        "timestamp_origin": TIMESTAMP_ORIGIN,
        "observation_action_alignment": OBSERVATION_ACTION_ALIGNMENT,
    }
    for key, expected in required_strings.items():
        if type(episode_end[key]) is not str or episode_end[key] != expected:
            raise EpisodeIOError(
                f"episode_end {key} must be {expected!r}; received {episode_end[key]!r}."
            )

    result: dict[str, Any] = {}
    for key in expected_keys:
        value = episode_end[key]
        if isinstance(value, (Mapping, list, tuple, set)) or value is None:
            raise EpisodeIOError(f"episode_end {key} must be one JSON scalar.")
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            raise EpisodeIOError(f"episode_end {key} must be finite.")
        if isinstance(value, np.generic):
            value = value.item()
        result[key] = value
    return result


def _validate_commanded_alignment(arrays: Mapping[str, Any], episode_end: Mapping[str, Any]) -> None:
    """V2 labels copy the accepted prior action; V1 labels are never redefined."""
    if episode_end["schema_version"] != COMMANDED_EPISODE_SCHEMA_VERSION:
        return
    commands = arrays["action"][:, 3:]
    if not np.all((commands == JAW_CLOSED_BIT) | (commands == JAW_OPEN_BIT)):
        raise EpisodeIOError("commanded episode requires accepted binary jaw actions.")
    states = arrays["jaw_state"]
    if not np.all(states[0] == JAW_OPEN_BIT) or not np.array_equal(states[1:], commands[:-1]):
        raise EpisodeIOError("commanded jaw_state must be reset-open then the preceding accepted action.")


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise EpisodeIOError(f"{name} must contain finite real values.")
    result = float(value)
    if not math.isfinite(result):
        raise EpisodeIOError(f"{name} must contain finite real values.")
    return result


def _numeric_tuple(
    value: Any,
    *,
    name: str,
    length: int,
    optional: bool = True,
) -> tuple[float, ...] | None:
    if value is None and optional:
        return None
    if not isinstance(value, tuple) or len(value) != length:
        raise EpisodeIOError(f"{name} must be a tuple of length {length} or None.")
    return tuple(_finite_real(item, name=name) for item in value)


def _numeric_tuple_any_length(
    value: Any, *, name: str
) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) == 0:
        raise EpisodeIOError(f"{name} must be a non-empty numeric tuple or None.")
    return tuple(_finite_real(item, name=name) for item in value)


def _normalize_diagnostic_frame(frame: DiagnosticFrame, *, index: int) -> DiagnosticFrame:
    if not isinstance(frame, DiagnosticFrame):
        raise TypeError(f"diagnostic_frames[{index}] must be a DiagnosticFrame.")
    try:
        require_policy_row_eligible(frame)
    except TerminalRowError as exc:
        raise EpisodeIOError(
            f"diagnostic_frames[{index}] is lifecycle-ineligible: {exc}"
        ) from exc

    optional_bool_names = ("table_support", "contact_valid", "attachment_state")
    for name in optional_bool_names:
        value = getattr(frame, name)
        if value is not None and type(value) is not bool:
            raise EpisodeIOError(f"diagnostic_frames[{index}].{name} must be bool or None.")
    for name in (
        "reset",
        "terminated",
        "truncated",
        "automatic_reset",
        "invalid_policy_frame",
        "jaw_pair_disagreement",
    ):
        if type(getattr(frame, name)) is not bool:
            raise EpisodeIOError(f"diagnostic_frames[{index}].{name} must be bool.")
    if not isinstance(frame.contact_identity, tuple) or not all(
        isinstance(item, str) for item in frame.contact_identity
    ):
        raise EpisodeIOError(
            f"diagnostic_frames[{index}].contact_identity must be a tuple of strings."
        )
    for name in ("invalid_reason", "outcome_label"):
        value = getattr(frame, name)
        if value is not None and not isinstance(value, str):
            raise EpisodeIOError(
                f"diagnostic_frames[{index}].{name} must be a string or None."
            )

    return DiagnosticFrame(
        requested_tcp_pose_base_wxyz=_numeric_tuple(
            frame.requested_tcp_pose_base_wxyz,
            name=f"diagnostic_frames[{index}].requested_tcp_pose_base_wxyz",
            length=7,
        ),
        actual_tcp_pose_base_wxyz=_numeric_tuple(
            frame.actual_tcp_pose_base_wxyz,
            name=f"diagnostic_frames[{index}].actual_tcp_pose_base_wxyz",
            length=7,
        ),
        chicken_pose_world_wxyz=_numeric_tuple(
            frame.chicken_pose_world_wxyz,
            name=f"diagnostic_frames[{index}].chicken_pose_world_wxyz",
            length=7,
        ),
        chicken_velocity_world=_numeric_tuple_any_length(
            frame.chicken_velocity_world,
            name=f"diagnostic_frames[{index}].chicken_velocity_world",
        ),
        requested_tcp_movement_m=_numeric_tuple(
            frame.requested_tcp_movement_m,
            name=f"diagnostic_frames[{index}].requested_tcp_movement_m",
            length=3,
        ),
        actual_tcp_movement_m=_numeric_tuple(
            frame.actual_tcp_movement_m,
            name=f"diagnostic_frames[{index}].actual_tcp_movement_m",
            length=3,
        ),
        chicken_movement_m=_numeric_tuple(
            frame.chicken_movement_m,
            name=f"diagnostic_frames[{index}].chicken_movement_m",
            length=3,
        ),
        table_support=frame.table_support,
        contact_valid=frame.contact_valid,
        contact_identity=frame.contact_identity,
        reset=frame.reset,
        terminated=frame.terminated,
        truncated=frame.truncated,
        automatic_reset=frame.automatic_reset,
        invalid_policy_frame=frame.invalid_policy_frame,
        jaw_pair_disagreement=frame.jaw_pair_disagreement,
        invalid_reason=frame.invalid_reason,
        attachment_state=frame.attachment_state,
        outcome_label=frame.outcome_label,
    )


def _diagnostic_to_mapping(frame: DiagnosticFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in _DIAGNOSTIC_FIELD_NAMES:
        value = getattr(frame, name)
        if isinstance(value, tuple):
            result[name] = list(value)
        else:
            result[name] = value
    return result


def _validate_diagnostic_frames(
    diagnostic_frames: Sequence[DiagnosticFrame], *, sample_count: int
) -> tuple[DiagnosticFrame, ...]:
    if not isinstance(diagnostic_frames, Sequence) or isinstance(
        diagnostic_frames, (str, bytes)
    ):
        raise TypeError("diagnostic_frames must be a sequence of DiagnosticFrame values.")
    if len(diagnostic_frames) != sample_count:
        raise EpisodeIOError(
            "diagnostic frame count must equal the policy-array length."
        )
    validated = tuple(
        _normalize_diagnostic_frame(frame, index=index)
        for index, frame in enumerate(diagnostic_frames)
    )
    if not isinstance(validated[-1].outcome_label, str) or not validated[
        -1
    ].outcome_label.strip():
        raise EpisodeIOError(
            "The final diagnostic frame must contain a non-empty terminal classification."
        )
    return validated


def _json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EpisodeIOError(f"Value is not strict JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _npy_bytes(array: NDArray[Any]) -> bytes:
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def _payload_manifest(payloads: Mapping[str, bytes]) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "policy_fields": list(POLICY_ARRAY_KEYS),
        "files": [
            {
                "path": path,
                "byte_count": len(payloads[path]),
                "sha256": hashlib.sha256(payloads[path]).hexdigest(),
            }
            for path in _PAYLOAD_PATHS
        ],
    }


def write_episode(
    episode_dir: str | Path,
    arrays: Mapping[str, Any],
    *,
    episode_end: Mapping[str, Any],
    diagnostic_frames: Sequence[DiagnosticFrame],
    publication_deadline: float | None = None,
) -> Path:
    """Validate and exclusively create one complete episode package."""

    validated_arrays = validate_policy_arrays(arrays)
    sample_count = int(validated_arrays["timestamp_ns"].shape[0])
    validated_end = _validate_episode_end(episode_end, sample_count=sample_count)
    _validate_commanded_alignment(validated_arrays, validated_end)
    validated_diagnostics = _validate_diagnostic_frames(
        diagnostic_frames, sample_count=sample_count
    )

    payloads: dict[str, bytes] = {
        f"{POLICY_DIRECTORY}/{key}.npy": _npy_bytes(validated_arrays[key])
        for key in POLICY_ARRAY_KEYS
    }
    payloads[EPISODE_END_FILENAME] = _json_bytes(validated_end)
    payloads[DIAGNOSTIC_SIDECAR_FILENAME] = _json_bytes(
        {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "frames": [
                _diagnostic_to_mapping(frame) for frame in validated_diagnostics
            ],
        }
    )
    manifest_payload = _json_bytes(_payload_manifest(payloads))

    root = Path(episode_dir)
    publication = Publication(root, deadline=publication_deadline)
    try:
        destination_absent(root)
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=str(root.parent)))
        publication.source = staging
        publication.record['source'] = str(staging)
        (staging / POLICY_DIRECTORY).mkdir(exist_ok=False)
        for relative_path in _PAYLOAD_PATHS:
            publication.write_payload(staging.joinpath(*relative_path.split("/")), payloads[relative_path])
        publication.write_payload(staging / MANIFEST_FILENAME, manifest_payload)
        return publication.publish(staging)
    except BaseException as exc:
        publication.failed(exc)
        # Failed staging is evidence, never temporary-directory cleanup material.
        raise


def _reject_json_constant(value: str) -> None:
    raise EpisodeIOError(f"Non-finite JSON constant is prohibited: {value}.")


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EpisodeIOError(f"Duplicate JSON key is prohibited: {key!r}.")
        result[key] = value
    return result


def _parse_json(payload: bytes, *, name: str) -> Any:
    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except EpisodeIOError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EpisodeIOError(f"Malformed JSON in {name}: {exc}") from exc


def _enumerate_package_files(root: Path) -> set[str]:
    if not root.is_dir() or root.is_symlink():
        raise EpisodeIOError(f"Episode path must be one real directory: {root}")
    files: set[str] = set()
    directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise EpisodeIOError(f"Symbolic links are prohibited in an episode: {relative}")
        if path.is_dir():
            directories.add(relative)
        elif path.is_file():
            files.add(relative)
        else:
            raise EpisodeIOError(f"Unexpected filesystem entry in episode: {relative}")
    if directories != {POLICY_DIRECTORY}:
        raise EpisodeIOError(
            f"Episode directories must be exactly ({POLICY_DIRECTORY!r},); "
            f"received {tuple(sorted(directories))}."
        )
    if files != set(_ALL_FILE_PATHS):
        missing = sorted(set(_ALL_FILE_PATHS) - files)
        unexpected = sorted(files - set(_ALL_FILE_PATHS))
        raise EpisodeIOError(
            f"Episode file set mismatch; missing={missing}, unexpected={unexpected}."
        )
    return files


def _read_verified_payloads(episode_dir: str | Path) -> dict[str, bytes]:
    root = Path(episode_dir)
    _enumerate_package_files(root)

    manifest_bytes = (root / MANIFEST_FILENAME).read_bytes()
    manifest = _parse_json(manifest_bytes, name=MANIFEST_FILENAME)
    if not isinstance(manifest, Mapping):
        raise EpisodeIOError("manifest.json must contain one object.")
    _ordered_keys(manifest, name="manifest", expected=_MANIFEST_KEYS)
    if not _is_int(manifest["schema_version"]) or int(
        manifest["schema_version"]
    ) != MANIFEST_SCHEMA_VERSION:
        raise EpisodeIOError(
            f"manifest schema_version must be {MANIFEST_SCHEMA_VERSION}."
        )
    if manifest["policy_fields"] != list(POLICY_ARRAY_KEYS):
        raise EpisodeIOError(
            f"manifest policy_fields must be exactly {list(POLICY_ARRAY_KEYS)}."
        )
    entries = manifest["files"]
    if not isinstance(entries, list) or len(entries) != len(_PAYLOAD_PATHS):
        raise EpisodeIOError("manifest files must list every payload exactly once.")

    payloads: dict[str, bytes] = {}
    for expected_path, entry in zip(_PAYLOAD_PATHS, entries, strict=True):
        if not isinstance(entry, Mapping):
            raise EpisodeIOError("Every manifest file entry must be an object.")
        _ordered_keys(entry, name="manifest file entry", expected=_MANIFEST_FILE_KEYS)
        if entry["path"] != expected_path:
            raise EpisodeIOError(
                f"Manifest payload order/path mismatch: expected {expected_path!r}, "
                f"received {entry['path']!r}."
            )
        if not _is_int(entry["byte_count"]) or int(entry["byte_count"]) < 0:
            raise EpisodeIOError("Manifest byte_count must be a non-negative integer.")
        if not isinstance(entry["sha256"], str) or not _SHA256_PATTERN.fullmatch(
            entry["sha256"]
        ):
            raise EpisodeIOError("Manifest sha256 must be 64 lowercase hexadecimal digits.")

        path = root.joinpath(*expected_path.split("/"))
        payload = path.read_bytes()
        if len(payload) != int(entry["byte_count"]):
            raise EpisodeIOError(
                f"Byte-count mismatch for {expected_path}: expected "
                f"{entry['byte_count']}, received {len(payload)}."
            )
        digest = hashlib.sha256(payload).hexdigest()
        if digest != entry["sha256"]:
            raise EpisodeIOError(
                f"SHA-256 mismatch for {expected_path}: expected "
                f"{entry['sha256']}, received {digest}."
            )
        payloads[expected_path] = payload
    return payloads


def _load_npy(payload: bytes, *, name: str) -> NDArray[Any]:
    stream = io.BytesIO(payload)
    try:
        value = np.load(stream, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise EpisodeIOError(f"Malformed NumPy payload {name}: {exc}") from exc
    if not isinstance(value, np.ndarray):
        raise EpisodeIOError(f"NumPy payload {name} did not contain one array.")
    if stream.tell() != len(payload):
        raise EpisodeIOError(f"NumPy payload {name} contains trailing bytes.")
    return value


def load_policy_episode(episode_dir: str | Path) -> dict[str, NDArray[Any]]:
    """Verify the complete package, then return exactly six policy arrays."""

    payloads = _read_verified_payloads(episode_dir)
    arrays = {
        key: _load_npy(payloads[f"{POLICY_DIRECTORY}/{key}.npy"], name=f"{key}.npy")
        for key in POLICY_ARRAY_KEYS
    }
    validated = validate_policy_arrays(arrays)
    episode_end = _parse_json(payloads[EPISODE_END_FILENAME], name=EPISODE_END_FILENAME)
    if not isinstance(episode_end, Mapping):
        raise EpisodeIOError("episode_end.json must contain one object.")
    _validate_episode_end(
        episode_end, sample_count=int(validated["timestamp_ns"].shape[0])
    )
    _validate_commanded_alignment(validated, episode_end)
    _parse_diagnostic_sidecar(
        payloads[DIAGNOSTIC_SIDECAR_FILENAME],
        sample_count=int(validated["timestamp_ns"].shape[0]),
    )
    return validated


def load_episode_end(episode_dir: str | Path) -> dict[str, Any]:
    """Verify the complete package and return only safe scalar metadata."""

    payloads = _read_verified_payloads(episode_dir)
    arrays = {
        key: _load_npy(payloads[f"{POLICY_DIRECTORY}/{key}.npy"], name=f"{key}.npy")
        for key in POLICY_ARRAY_KEYS
    }
    validated = validate_policy_arrays(arrays)
    episode_end = _parse_json(payloads[EPISODE_END_FILENAME], name=EPISODE_END_FILENAME)
    if not isinstance(episode_end, Mapping):
        raise EpisodeIOError("episode_end.json must contain one object.")
    result = _validate_episode_end(
        episode_end, sample_count=int(validated["timestamp_ns"].shape[0])
    )
    _validate_commanded_alignment(validated, result)
    _parse_diagnostic_sidecar(
        payloads[DIAGNOSTIC_SIDECAR_FILENAME],
        sample_count=int(validated["timestamp_ns"].shape[0]),
    )
    return result


def _diagnostic_from_mapping(value: Any, *, index: int) -> DiagnosticFrame:
    if not isinstance(value, Mapping):
        raise EpisodeIOError(f"diagnostic sidecar frame {index} must be an object.")
    _ordered_keys(
        value,
        name=f"diagnostic sidecar frame {index}",
        expected=_DIAGNOSTIC_FIELD_NAMES,
    )
    converted = dict(value)
    tuple_fields = (
        "requested_tcp_pose_base_wxyz",
        "actual_tcp_pose_base_wxyz",
        "chicken_pose_world_wxyz",
        "chicken_velocity_world",
        "requested_tcp_movement_m",
        "actual_tcp_movement_m",
        "chicken_movement_m",
        "contact_identity",
    )
    for name in tuple_fields:
        item = converted[name]
        if item is not None:
            if not isinstance(item, list):
                raise EpisodeIOError(
                    f"diagnostic sidecar frame {index}.{name} must be a JSON array or null."
                )
            converted[name] = tuple(item)
    try:
        frame = DiagnosticFrame(**converted)
    except TypeError as exc:
        raise EpisodeIOError(f"Malformed diagnostic frame {index}: {exc}") from exc
    return _normalize_diagnostic_frame(frame, index=index)


def _parse_diagnostic_sidecar(
    payload: bytes, *, sample_count: int
) -> tuple[DiagnosticFrame, ...]:
    sidecar = _parse_json(payload, name=DIAGNOSTIC_SIDECAR_FILENAME)
    if not isinstance(sidecar, Mapping):
        raise EpisodeIOError("diagnostic_sidecar.json must contain one object.")
    _ordered_keys(sidecar, name="diagnostic sidecar", expected=_DIAGNOSTIC_ROOT_KEYS)
    if not _is_int(sidecar["schema_version"]) or int(
        sidecar["schema_version"]
    ) != DIAGNOSTIC_SCHEMA_VERSION:
        raise EpisodeIOError(
            f"diagnostic sidecar schema_version must be {DIAGNOSTIC_SCHEMA_VERSION}."
        )
    records = sidecar["frames"]
    if not isinstance(records, list) or len(records) != sample_count:
        raise EpisodeIOError(
            "diagnostic sidecar frame count must equal the policy-array length."
        )
    return tuple(
        _diagnostic_from_mapping(record, index=index)
        for index, record in enumerate(records)
    )


def load_diagnostic_sidecar(episode_dir: str | Path) -> tuple[DiagnosticFrame, ...]:
    """Verify the package and explicitly load privileged diagnostic frames."""

    payloads = _read_verified_payloads(episode_dir)
    arrays = {
        key: _load_npy(payloads[f"{POLICY_DIRECTORY}/{key}.npy"], name=f"{key}.npy")
        for key in POLICY_ARRAY_KEYS
    }
    validated = validate_policy_arrays(arrays)
    episode_end = _parse_json(payloads[EPISODE_END_FILENAME], name=EPISODE_END_FILENAME)
    if not isinstance(episode_end, Mapping):
        raise EpisodeIOError("episode_end.json must contain one object.")
    _validate_episode_end(
        episode_end, sample_count=int(validated["timestamp_ns"].shape[0])
    )
    _validate_commanded_alignment(validated, episode_end)
    return _parse_diagnostic_sidecar(
        payloads[DIAGNOSTIC_SIDECAR_FILENAME],
        sample_count=int(validated["timestamp_ns"].shape[0]),
    )


def _replay_action(value: Any, *, name: str) -> NDArray[np.float32]:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != POLICY_ACTION_DIM or array.shape[0] < 1:
        raise EpisodeIOError(f"{name} must have shape [N,{POLICY_ACTION_DIM}] with N >= 1.")
    if array.dtype != np.dtype(np.float32):
        raise EpisodeIOError(f"{name} must have dtype float32.")
    _require_finite(array, name=name)
    if not np.all((array[:, 3:5] == 0.0) | (array[:, 3:5] == 1.0)):
        raise EpisodeIOError(f"{name} jaw commands must be exactly binary.")
    try:
        validate_policy_action(array)
    except (TypeError, ValueError) as exc:
        raise EpisodeIOError(f"{name} violates the frozen action contract: {exc}") from exc
    return np.ascontiguousarray(array)


def _replay_jaw(value: Any, *, name: str, sample_count: int) -> NDArray[np.uint8]:
    array = np.asarray(value)
    if array.shape != (sample_count, 2) or array.dtype != np.dtype(np.uint8):
        raise EpisodeIOError(f"{name} must have shape [N,2] and dtype uint8.")
    if not np.all((array == 0) | (array == 1)):
        raise EpisodeIOError(f"{name} values must be exactly binary.")
    return np.ascontiguousarray(array)


def _trajectory(value: Any, *, name: str, sample_count: int) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.shape != (sample_count, 3) or array.dtype.kind not in "iuf":
        raise EpisodeIOError(f"{name} must be a finite real array with shape [N,3].")
    _require_finite(array, name=name)
    return np.ascontiguousarray(array, dtype=np.float64)


def _positive_tolerance(value: Any, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result <= 0.0:
        raise EpisodeIOError(f"{name} must be greater than zero.")
    return result


def _movement(trajectory: NDArray[Any]) -> tuple[float, float, float]:
    delta = np.asarray(trajectory[-1, :3], dtype=np.float64) - np.asarray(
        trajectory[0, :3], dtype=np.float64
    )
    return tuple(float(value) for value in delta)


def _movement_from_initial_diagnostic(
    trajectory_endpoints: NDArray[Any],
    diagnostics: Sequence[DiagnosticFrame],
    *,
    pose_field: str,
    name: str,
) -> tuple[float, float, float]:
    """Return initial pre-action pose to final six-tick endpoint movement."""

    initial_pose = getattr(diagnostics[0], pose_field)
    if initial_pose is None:
        raise EpisodeIOError(f"{name} requires {pose_field} in diagnostic frame 0.")
    initial_xyz = np.asarray(initial_pose[:3], dtype=np.float64)
    delta = np.asarray(trajectory_endpoints[-1, :3], dtype=np.float64) - initial_xyz
    return tuple(float(value) for value in delta)


def compare_replay_metrics(
    *,
    recorded_action: Any,
    replayed_action: Any,
    recorded_jaw_state: Any,
    replayed_jaw_state: Any,
    recorded_actual_tcp_xyz_m: Any,
    replayed_actual_tcp_xyz_m: Any,
    recorded_chicken_xyz_m: Any,
    replayed_chicken_xyz_m: Any,
    recorded_terminal_classification: str,
    replayed_terminal_classification: str,
    recorded_diagnostics: Sequence[DiagnosticFrame],
    replayed_diagnostics: Sequence[DiagnosticFrame],
    tcp_tolerance_m: float = DEFAULT_REPLAY_TOLERANCE_M,
    chicken_tolerance_m: float = DEFAULT_REPLAY_TOLERANCE_M,
    require_chicken_within_tolerance: bool = False,
) -> dict[str, Any]:
    """Compare future replay without treating requested, TCP, or object motion as proxies."""

    recorded_actions = _replay_action(recorded_action, name="recorded_action")
    replayed_actions = _replay_action(replayed_action, name="replayed_action")
    if replayed_actions.shape != recorded_actions.shape:
        raise EpisodeIOError("Recorded and replayed action lengths must match.")
    sample_count = int(recorded_actions.shape[0])
    recorded_jaws = _replay_jaw(
        recorded_jaw_state, name="recorded_jaw_state", sample_count=sample_count
    )
    replayed_jaws = _replay_jaw(
        replayed_jaw_state, name="replayed_jaw_state", sample_count=sample_count
    )
    recorded_tcp = _trajectory(
        recorded_actual_tcp_xyz_m,
        name="recorded_actual_tcp_xyz_m",
        sample_count=sample_count,
    )
    replayed_tcp = _trajectory(
        replayed_actual_tcp_xyz_m,
        name="replayed_actual_tcp_xyz_m",
        sample_count=sample_count,
    )
    recorded_chicken = _trajectory(
        recorded_chicken_xyz_m,
        name="recorded_chicken_xyz_m",
        sample_count=sample_count,
    )
    replayed_chicken = _trajectory(
        replayed_chicken_xyz_m,
        name="replayed_chicken_xyz_m",
        sample_count=sample_count,
    )
    checked_recorded_diagnostics = _validate_diagnostic_frames(
        recorded_diagnostics, sample_count=sample_count
    )
    checked_replayed_diagnostics = _validate_diagnostic_frames(
        replayed_diagnostics, sample_count=sample_count
    )

    if (
        not isinstance(recorded_terminal_classification, str)
        or not recorded_terminal_classification.strip()
        or not isinstance(replayed_terminal_classification, str)
        or not replayed_terminal_classification.strip()
    ):
        raise EpisodeIOError("Terminal classifications must be non-empty strings.")
    if checked_recorded_diagnostics[-1].outcome_label != recorded_terminal_classification:
        raise EpisodeIOError(
            "recorded_terminal_classification must match the final recorded diagnostic outcome."
        )
    if checked_replayed_diagnostics[-1].outcome_label != replayed_terminal_classification:
        raise EpisodeIOError(
            "replayed_terminal_classification must match the final replay diagnostic outcome."
        )
    if type(require_chicken_within_tolerance) is not bool:
        raise EpisodeIOError("require_chicken_within_tolerance must be bool.")
    tcp_tolerance = _positive_tolerance(tcp_tolerance_m, name="tcp_tolerance_m")
    chicken_tolerance = _positive_tolerance(
        chicken_tolerance_m, name="chicken_tolerance_m"
    )

    actual_tcp_errors = np.linalg.norm(replayed_tcp - recorded_tcp, axis=1)
    chicken_errors = np.linalg.norm(replayed_chicken - recorded_chicken, axis=1)
    tcp_max_error = float(np.max(actual_tcp_errors))
    chicken_max_error = float(np.max(chicken_errors))
    action_exact = bool(np.array_equal(recorded_actions, replayed_actions))
    jaw_exact = bool(np.array_equal(recorded_jaws, replayed_jaws))
    tcp_within = tcp_max_error <= tcp_tolerance
    chicken_within = chicken_max_error <= chicken_tolerance
    terminal_match = recorded_terminal_classification == replayed_terminal_classification

    return {
        "sample_count": sample_count,
        "length_match": True,
        "action_exact": action_exact,
        "jaw_state_exact": jaw_exact,
        "actual_tcp_max_error_m": tcp_max_error,
        "actual_tcp_within_tolerance": tcp_within,
        "chicken_max_error_m": chicken_max_error,
        "chicken_within_tolerance": chicken_within,
        "chicken_tolerance_required": require_chicken_within_tolerance,
        "terminal_classification_match": terminal_match,
        "recorded_requested_tcp_movement_m": _movement(recorded_actions),
        "replayed_requested_tcp_movement_m": _movement(replayed_actions),
        "recorded_actual_tcp_movement_m": _movement_from_initial_diagnostic(
            recorded_tcp,
            checked_recorded_diagnostics,
            pose_field="actual_tcp_pose_base_wxyz",
            name="recorded actual TCP movement",
        ),
        "replayed_actual_tcp_movement_m": _movement_from_initial_diagnostic(
            replayed_tcp,
            checked_replayed_diagnostics,
            pose_field="actual_tcp_pose_base_wxyz",
            name="replayed actual TCP movement",
        ),
        "recorded_chicken_movement_m": _movement_from_initial_diagnostic(
            recorded_chicken,
            checked_recorded_diagnostics,
            pose_field="chicken_pose_world_wxyz",
            name="recorded chicken movement",
        ),
        "replayed_chicken_movement_m": _movement_from_initial_diagnostic(
            replayed_chicken,
            checked_replayed_diagnostics,
            pose_field="chicken_pose_world_wxyz",
            name="replayed chicken movement",
        ),
        "passed": bool(
            action_exact
            and jaw_exact
            and tcp_within
            and (chicken_within or not require_chicken_within_tolerance)
            and terminal_match
        ),
    }


__all__ = (
    "DEFAULT_REPLAY_TOLERANCE_M",
    "DIAGNOSTIC_SCHEMA_VERSION",
    "DIAGNOSTIC_SIDECAR_FILENAME",
    "EPISODE_END_FILENAME",
    "EPISODE_END_KEYS",
    "EPISODE_SCHEMA_VERSION",
    "COMMANDED_EPISODE_SCHEMA_VERSION",
    "COMMANDED_EPISODE_END_KEYS",
    "EpisodeIOError",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "OBSERVATION_ACTION_ALIGNMENT",
    "POLICY_ARRAY_KEYS",
    "POLICY_DIRECTORY",
    "TIMESTAMP_ORIGIN",
    "TIMESTAMP_PERIOD_NS",
    "TIMESTAMP_TOLERANCE_NS",
    "compare_replay_metrics",
    "load_diagnostic_sidecar",
    "load_episode_end",
    "load_policy_episode",
    "validate_policy_arrays",
    "write_episode",
)
