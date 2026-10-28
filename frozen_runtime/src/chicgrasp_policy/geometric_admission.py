"""Frozen Cycle50 geometric eligibility. Pure saved-value math, never contact proof.

The fully-closed reference is the executed float32 commanded position, not a
claim that a measured jaw reached its mechanical stop. Positive joint travel
opens all four jaws. Selection is once at 839/1; callers own that one-shot gate.
"""

from __future__ import annotations

import copy
import math

import numpy as np


SOURCE = "geometric-eligibility/v1"
CLOSED_REFERENCE_M = -0.00930000003427267
DELTA_M = 0.001
EPSILON_M = 0.00005
INTERVALS = 60
RADIUS_M = 0.15
PHYSICS_DT_S = 1.0 / 120.0
ADMISSION_CONTEXT = (839, 1)


def parameters():
    """Return a fresh copy of the one frozen configuration, not tunable inputs."""
    return {"delta_m": DELTA_M, "epsilon_m": EPSILON_M,
            "intervals": INTERVALS, "samples": INTERVALS + 1,
            "radius_m": RADIUS_M,
            "closed_reference_m": [CLOSED_REFERENCE_M] * 4,
            "opening_signs": [1, 1, 1, 1],
            "joint_names": [f"PrismaticJoint{i}" for i in range(1, 5)],
            "left_jaw_ids": [1, 2], "right_jaw_ids": [3, 4],
            "reference_kind": "executed_float32_commanded_closed_target",
            "frame": "world", "units": "m", "pending_owner_approval": False}


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _safe_copy(value):
    """Strict-JSON diagnostics retain invalidity without inventing measurements."""
    if isinstance(value, np.ndarray):
        return _safe_copy(value.tolist())
    if isinstance(value, np.generic):
        return _safe_copy(value.item())
    if type(value) is float and not math.isfinite(value):
        return {"invalid_numeric": repr(value)}
    if isinstance(value, dict):
        return {str(k): _safe_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_copy(v) for v in value]
    if value is None or type(value) in (bool, str, int, float):
        return value
    return {"unsupported_type": type(value).__module__ + "." + type(value).__qualname__}


def _vector(value, length):
    return isinstance(value, (list, tuple)) and len(value) == length and all(_number(v) for v in value)


def _jaw_qualifies(shortfall_m, motion_m):
    return shortfall_m >= DELTA_M and motion_m <= EPSILON_M


def sample_from_readings(context, pose, articulation, commands):
    """Copy actual scheduled readings. No pose/clock/command defaults or refresh."""
    poses = pose.get("poses", {})
    clock = pose.get("clock_after")
    context = vars(context) if not isinstance(context, dict) else context
    articulation_clock_ok = (articulation.get("clock_before") == clock
                             and articulation.get("clock_after") == clock)
    valid = (pose.get("valid") is True and articulation.get("valid") is True
             and pose.get("clock_before") == clock and articulation_clock_ok
             and pose.get("units") == "m"
             and pose.get("pose_convention") == "position_xyz_quaternion_wxyz")
    return _safe_copy({
        "episode_id": context.get("episode_id"),
        "control_tick": context.get("control_tick"), "substep": context.get("substep"),
        "physics_step": clock[0] if isinstance(clock, (list, tuple)) and len(clock) == 2 else None,
        "simulation_time_s": clock[1] if isinstance(clock, (list, tuple)) and len(clock) == 2 else None,
        "jaw_positions_m": articulation.get("joint_positions_m"),
        "jaw_commands": commands, "pose_frame": "world",
        "chicken_world_wxyz": poses.get("chicken_actor_world_wxyz"),
        "tcp_world_wxyz": poses.get("tcp_world_wxyz"), "pose_valid": valid})


def evaluate_history(history):
    """Evaluate exactly N+1 consecutive real samples; malformed data fails closed.

    Each side needs one fixed jaw whose *own* endpoint shortfall and complete
    window peak-to-peak motion satisfy both inclusive limits. Endpoint net
    displacement is reported only for audit and never substituted for motion.
    """
    result = {"source": SOURCE, "status": "UNAVAILABLE", "sample": None,
              "parameters": parameters(), "history": _safe_copy(history),
              "jaw_results": [], "left_witness_jaw_ids": [],
              "right_witness_jaw_ids": [], "distance_m": None, "issues": [],
              "interpretation": "geometric eligibility, not measured physical contact"}
    issues = result["issues"]
    if not isinstance(history, (list, tuple)) or len(history) != INTERVALS + 1:
        issues.append("exact_N_plus_one_history_required")
        return result
    if any(not isinstance(row, dict) for row in history):
        issues.append("history_sample_not_mapping")
        return result
    end = history[-1]
    result["sample"] = _safe_copy({key: end.get(key) for key in
        ("episode_id", "control_tick", "substep", "physics_step", "simulation_time_s")})
    for index, row in enumerate(history):
        if not isinstance(row.get("episode_id"), str) or not row["episode_id"]:
            issues.append(f"sample{index}:episode_missing")
        for key in ("control_tick", "substep", "physics_step"):
            if type(row.get(key)) is not int or row[key] < 0:
                issues.append(f"sample{index}:{key}_invalid")
        if row.get("substep") not in (0, 1):
            issues.append(f"sample{index}:substep_invalid")
        if not _number(row.get("simulation_time_s")) or row["simulation_time_s"] < 0:
            issues.append(f"sample{index}:time_invalid")
        if row.get("pose_frame") != "world" or row.get("pose_valid") is not True:
            issues.append(f"sample{index}:required_world_pose_invalid")
        for key, length in (("jaw_positions_m", 4), ("jaw_commands", 2),
                            ("chicken_world_wxyz", 7), ("tcp_world_wxyz", 7)):
            if not _vector(row.get(key), length):
                issues.append(f"sample{index}:{key}_invalid")
        for key in ("chicken_world_wxyz", "tcp_world_wxyz"):
            if _vector(row.get(key), 7):
                norm = math.sqrt(sum(q * q for q in row[key][3:]))
                if not math.isfinite(norm) or norm == 0:
                    issues.append(f"sample{index}:{key}_quaternion_invalid")
        if _vector(row.get("jaw_commands"), 2) and any(v not in (0, 1) for v in row["jaw_commands"]):
            issues.append(f"sample{index}:jaw_commands_not_binary")
    if issues:
        return result
    if (end["control_tick"], end["substep"]) != ADMISSION_CONTEXT:
        issues.append("admission_requires_839_1")
    for index, (before, after) in enumerate(zip(history, history[1:])):
        if before["episode_id"] != after["episode_id"]:
            issues.append(f"interval{index}:episode_changed")
        if 2 * after["control_tick"] + after["substep"] != 2 * before["control_tick"] + before["substep"] + 1:
            issues.append(f"interval{index}:scheduled_history_not_consecutive")
        if after["physics_step"] != before["physics_step"] + 1:
            issues.append(f"interval{index}:native_history_not_consecutive")
        if not math.isclose(after["simulation_time_s"] - before["simulation_time_s"],
                            PHYSICS_DT_S, rel_tol=0, abs_tol=1e-8):
            issues.append(f"interval{index}:native_time_interval_invalid")
    if issues:
        return result
    for jaw in range(4):
        positions = [row["jaw_positions_m"][jaw] for row in history]
        shortfall = positions[-1] - CLOSED_REFERENCE_M
        motion = max(positions) - min(positions)
        net = positions[-1] - positions[0]
        if not all(_number(v) for v in (shortfall, motion, net)):
            issues.append(f"jaw{jaw+1}:derived_value_not_finite")
            return result
        result["jaw_results"].append({"jaw_id": jaw + 1,
            "side": "LEFT" if jaw < 2 else "RIGHT", "opening_sign": 1,
            "closed_reference_m": CLOSED_REFERENCE_M, "position_m": positions[-1],
            "shortfall_m": shortfall, "min_m": min(positions), "max_m": max(positions),
            "motion_m": motion, "net_displacement_m": net,
            "qualifies": _jaw_qualifies(shortfall, motion)})
    result["left_witness_jaw_ids"] = [r["jaw_id"] for r in result["jaw_results"][:2] if r["qualifies"]]
    result["right_witness_jaw_ids"] = [r["jaw_id"] for r in result["jaw_results"][2:] if r["qualifies"]]
    distance = math.dist(end["chicken_world_wxyz"][:3], end["tcp_world_wxyz"][:3])
    if not math.isfinite(distance):
        issues.append("distance_not_finite")
        return result
    result["distance_m"] = distance
    closed = end["jaw_commands"] == [0, 0] or end["jaw_commands"] == (0, 0)
    if not closed:
        issues.append("both_sides_not_commanded_closed_at_admission")
    if not result["left_witness_jaw_ids"]:
        issues.append("no_fixed_left_jaw_witness")
    if not result["right_witness_jaw_ids"]:
        issues.append("no_fixed_right_jaw_witness")
    if distance > RADIUS_M:
        issues.append("chicken_root_measured_TCP_distance_exceeded")
    result["status"] = "FALSE" if issues else "TRUE"
    return copy.deepcopy(result)
