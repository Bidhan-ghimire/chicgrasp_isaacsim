"""Copied jaw evidence. No simulation imports, state getters, or guard decisions.

Only already selected comparison tensors are copied. Consumer provenance records
the Python call site, not a native phase, successful frame, or candidate start.
Ordinary capture retains failure-only behavior. An explicit scoped startup-only
observer also receives passing evaluations and partial existing-backend values.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import math
import sys
import time

from .startup_backend_diagnostics import observe_existing_backend


_STARTUP_OBSERVER = ContextVar("chicgrasp_cpu_startup_guard_observer", default=None)
_REAL_OBSERVER = ContextVar("chicgrasp_cpu_real_guard_observer", default=None)
_MAX_STARTUP_EVALUATIONS = 64


@contextmanager
def real_guard_observation(sink, phase):
    """One real CPU guard evaluation; retain no campaign-sized in-memory list."""
    if not callable(sink) or type(phase) is not str or not phase:
        raise ValueError("real_guard_requires_sink_and_explicit_phase")
    if _REAL_OBSERVER.get() is not None:
        raise ValueError("real_guard_observation_already_active")
    state = {"sink": sink, "phase": phase, "errors": []}
    token = _REAL_OBSERVER.set(state)
    try:
        yield state
    finally:
        _REAL_OBSERVER.reset(token)


@contextmanager
def startup_guard_observation(sink, metadata, error_sink):
    """Observe only this explicit diagnostic construction; restore context on exit.

    Metadata is ordinary copied data, never an environment/native handle. The
    yielded state retains recording errors so even a broken error sink is visible.
    Nested contexts are rejected rather than silently replacing another observer.
    """
    if metadata.get("diagnostic_mode") != "CPU_STARTUP_SHUTDOWN_ONLY":
        raise ValueError("startup_guard_observation_requires_explicit_diagnostic_mode")
    if not callable(sink) or not callable(error_sink):
        raise TypeError("startup_guard_observation_requires_sinks")
    if _STARTUP_OBSERVER.get() is not None:
        raise ValueError("startup_guard_observation_already_active")
    state = {"sink": sink, "metadata": json.loads(json.dumps(metadata, allow_nan=False)),
             "error_sink": error_sink, "sequence": 0, "errors": [], "overflow_reported": False}
    token = _STARTUP_OBSERVER.set(state)
    try:
        yield state
    finally:
        _STARTUP_OBSERVER.reset(token)


def _type(value):
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _plain(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"nonfinite": "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise TypeError("diagnostic dictionary keys must be strings")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, slice):
        return {"type": "builtins.slice", "start": _plain(value.start),
                "stop": _plain(value.stop), "step": _plain(value.step)}
    raise TypeError(f"unsupported diagnostic value: {_type(value)}")


def _scalar_representations(value):
    if isinstance(value, list):
        return [_scalar_representations(item) for item in value]
    row = {"value": _plain(value), "type": _type(value)}
    if type(value) is float:
        row.update(float_hex=value.hex(), negative_zero=(value == 0.0 and math.copysign(1.0, value) < 0))
    return row


def _tensor(value):
    if value is None:
        return None
    values = value.detach().cpu().tolist()
    return {"type": _type(value), "dtype": str(value.dtype), "device": str(value.device),
            "shape": list(value.shape), "values": _plain(values),
            "scalar_representations": _scalar_representations(values)}


def _consumer():
    frame = sys._getframe(1)
    try:
        # A bounded Python-stack read neither evaluates env fields nor retains frames.
        for _ in range(16):
            if frame is None:
                break
            module = frame.f_globals.get("__name__")
            name = frame.f_code.co_qualname
            if module == "isaaclab.managers.observation_manager":
                kind = {"ObservationManager._prepare_terms": "DIMENSION_DISCOVERY",
                        "ObservationManager.compute_group": "OBSERVATION_COMPUTATION"}.get(name)
                if kind is not None:
                    return {"kind": kind, "module": module, "callable": name,
                            "file": frame.f_code.co_filename, "line": frame.f_lineno,
                            "basis": "observed Python caller; not native-state or candidate proof"}
            frame = frame.f_back
        return {"kind": "UNKNOWN", "module": None, "callable": None, "file": None,
                "line": None, "basis": "no recognized manager consumer in bounded Python stack"}
    finally:
        del frame


def attach_jaw_state_diagnostic(error, *, positions, asset_cfg, threshold, expected_shape,
                                verdict, joint_bits=None, left_disagrees=None, right_disagrees=None,
                                env=None):
    """Attach a copied strict-JSON payload; failure cannot replace the original error.

    The scoped observer may inspect an existing env but cannot reread joint_pos.
    The zero classification threshold is not a pairwise position-difference tolerance.
    """
    observer = _STARTUP_OBSERVER.get()
    real_observer = _REAL_OBSERVER.get()
    if error is None and observer is None and real_observer is None:
        return
    sequence = None
    if observer is not None:
        sequence = observer["sequence"]
        observer["sequence"] += 1

    def recording_error(exc):
        detail = f"{type(exc).__name__}:{exc}"
        if real_observer is not None:
            real_observer["errors"].append(detail)
        if error is not None:
            try:
                error.jaw_state_diagnostic_error = detail
            except Exception:
                pass
        if observer is not None:
            entry = {"sequence": sequence, "error": detail, "phase": "guard_recording"}
            observer["errors"].append(entry)
            try:
                observer["error_sink"](json.loads(json.dumps(entry, allow_nan=False)))
            except Exception as sink_error:
                observer["errors"].append({"sequence": sequence, "phase": "guard_error_sink",
                                           "error": f"{type(sink_error).__name__}:{sink_error}"})

    try:
        record = {
            "schema": "jaw-state-failure/v1" if error is not None else "jaw-state-observation/v1",
            "verdict": verdict,
            "error_type": _type(error) if error is not None else None,
            "error": str(error) if error is not None else None,
            "measurement_source": "env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]",
            "units": "m", "normalization": "none; signed prismatic joint coordinates",
            "asset_name": asset_cfg.name, "joint_names": _plain(asset_cfg.joint_names),
            "joint_ids": _plain(asset_cfg.joint_ids), "joint_ids_type": _type(asset_cfg.joint_ids),
            "preserve_order": asset_cfg.preserve_order, "expected_shape": list(expected_shape),
            "positions": _tensor(positions), "joint_bits": _tensor(joint_bits),
            "left_disagrees": _tensor(left_disagrees), "right_disagrees": _tensor(right_disagrees),
            "threshold_source": "chicgrasp_policy.interface.JAW_OBSERVATION_THRESHOLD_M",
            "threshold_m": _plain(threshold), "threshold_type": _type(threshold),
            "pairwise_position_tolerance_m": None,
            "criterion": "joint_bits=(positions >= threshold).to(uint8); reject if any((bits[:,0] != bits[:,1]) | (bits[:,2] != bits[:,3]))",
            "threshold_inclusive": True, "pair_columns": {"left": [0, 1], "right": [2, 3]},
            "consumer": _consumer(), "policy_frame_emitted": False,
        }
        # These differences explain the captured operands. They do not feed the
        # original tensor comparisons and do not introduce a residual tolerance.
        values = record["positions"]["values"] if record["positions"] is not None else None
        if type(values) is list and all(type(row) is list and len(row) == 4 for row in values):
            record["pairwise_differences_m"] = [
                [_plain(row[0] - row[1]) if all(type(v) in (int, float) for v in row[:2]) else None,
                 _plain(row[2] - row[3]) if all(type(v) in (int, float) for v in row[2:]) else None]
                for row in values]
            record["signed_threshold_distances_m"] = values if threshold == 0.0 else [
                [_plain(v - threshold) if type(v) in (int, float) else None for v in row] for row in values]
        else:
            record["pairwise_differences_m"] = None
            record["signed_threshold_distances_m"] = None
        record["explanatory_arithmetic"] = "copied operands in Python scalar precision; not guard decision inputs"
        if real_observer is not None:
            record.update(phase="REAL_OBSERVATION", observation_phase=real_observer["phase"],
                          monotonic_ns=time.monotonic_ns())
        # Validate and detach every field, including configuration selectors.
        record = json.loads(json.dumps(record, allow_nan=False))
        if error is not None:
            error.jaw_state_diagnostic = record
        if real_observer is not None:
            real_observer["sink"](record)
        if observer is not None:
            if sequence >= _MAX_STARTUP_EVALUATIONS:
                if not observer["overflow_reported"]:
                    observer["overflow_reported"] = True
                    recording_error(ValueError("startup_guard_evaluation_limit_exceeded:64"))
                return
            backend = observe_existing_backend(env, phase="CONSTRUCTOR_GUARD", asset_name=asset_cfg.name)
            observed = {"schema": "cpu-startup-guard-observation/v1", "sequence": sequence,
                        "monotonic_ns": time.monotonic_ns(), "metadata": observer["metadata"],
                        "guard": record, "backend": backend}
            observer["sink"](json.loads(json.dumps(observed, allow_nan=False)))
    except Exception as diagnostic_error:
        recording_error(diagnostic_error)
