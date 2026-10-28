"""Pure geometry; historical defaults remain PROPOSED_NOT_APPROVED.

No individual-jaw stillness criterion. Another obstruction or partial enclosure
can satisfy this rule without a secure grasp. Inputs use measured metres.
"""
from __future__ import annotations

import math

CLOSED_REFERENCE_M = -0.00930000003427267
ENVELOPE_M = 0.003
PROGRESS_M = 0.001
SHORTFALL_M = 0.001
RADIUS_M = 0.150
START = (809, 1)
END = (839, 1)
SAMPLES = 61


def parameters():
    return dict(label="PROPOSED_NOT_APPROVED", source="side-opening-progress/v1",
        start=list(START), end=list(END), samples=SAMPLES,
        closed_reference_m=[CLOSED_REFERENCE_M]*4, opening_signs=[1]*4,
        side_jaw_ids=[[1, 2], [3, 4]], envelope_m=ENVELOPE_M,
        progress_m=PROGRESS_M, shortfall_m=SHORTFALL_M, radius_m=RADIUS_M,
        reference="accepted original demo at839/1", comparison="inclusive; no epsilon repair",
        integration="NONE", interpretation="geometric admission, not clamp proof")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _vector(value, size):
    return isinstance(value, (list, tuple)) and len(value) == size and all(map(_finite, value))


def _jaw_qualifies(minimum_progress, minimum_shortfall):
    return minimum_progress >= PROGRESS_M and minimum_shortfall >= SHORTFALL_M


def _envelope_qualifies(maximum_absolute_deviation):
    return maximum_absolute_deviation <= ENVELOPE_M


def _extrema(values):
    return dict(min_m=min(values), max_m=max(values), endpoint_m=values[-1])


def evaluate_history(history, reference_opening_m, *, pre_close_records=None, expected_end=END):
    """Evaluate measured history ending839/1 and its current closure transitions.

    Caller authenticates the accepted demo reference. Full consecutive history
    from before the current closure is required by default. The integrated path
    supplies authenticated pre_close_records alongside exactly61 actual samples.
    Neither path substitutes a commanded open endpoint for a measured position.
    Jaw progress and remaining shortfall must hold for the same fixed jaw across
    the entire final61 window. Individual rejected alternatives remain diagnostics.
    """
    result = dict(label="PROPOSED_NOT_APPROVED", status="UNAVAILABLE",
        parameters=parameters(), issues=[], sides=[], distance_m=None,
        window_start=None, window_end=None, sample_count=0)
    issues = result["issues"]
    if not _vector(reference_opening_m, 2):
        issues.append("accepted_demo_reference_unavailable")
    if not isinstance(history, (list, tuple)) or len(history) < SAMPLES:
        issues.append("complete_history_unavailable")
        return result
    for i, row in enumerate(history):
        if not isinstance(row, dict):
            issues.append(f"sample{i}:not_mapping")
            continue
        for key in ("control_tick", "substep", "physics_step"):
            if type(row.get(key)) is not int or row[key] < 0:
                issues.append(f"sample{i}:{key}_invalid")
        if row.get("substep") not in (0, 1):
            issues.append(f"sample{i}:substep_invalid")
        if not isinstance(row.get("episode_id"), str) or not row["episode_id"]:
            issues.append(f"sample{i}:episode_missing")
        if not _finite(row.get("simulation_time_s")) or row["simulation_time_s"] < 0:
            issues.append(f"sample{i}:time_invalid")
        if row.get("pose_valid") is not True or row.get("pose_frame") != "world":
            issues.append(f"sample{i}:required_world_pose_invalid")
        for key, size in (("jaw_positions_m", 4), ("jaw_commands", 2),
                ("chicken_world_wxyz", 7), ("tcp_world_wxyz", 7)):
            if not _vector(row.get(key), size):
                issues.append(f"sample{i}:{key}_invalid")
        if _vector(row.get("jaw_commands"), 2) and any(v not in (0, 1) for v in row["jaw_commands"]):
            issues.append(f"sample{i}:commands_not_binary")
        for key in ("chicken_world_wxyz", "tcp_world_wxyz"):
            if _vector(row.get(key), 7) and math.hypot(*row[key][3:]) == 0:
                issues.append(f"sample{i}:{key}_quaternion_invalid")
    if issues:
        return result
    for i, (a, b) in enumerate(zip(history, history[1:])):
        if a["episode_id"] != b["episode_id"]:
            issues.append(f"interval{i}:episode_changed")
        if 2*b["control_tick"]+b["substep"] != 2*a["control_tick"]+a["substep"]+1:
            issues.append(f"interval{i}:scheduled_gap")
        if b["physics_step"] != a["physics_step"]+1:
            issues.append(f"interval{i}:native_gap")
        if not math.isclose(b["simulation_time_s"]-a["simulation_time_s"], 1/120, rel_tol=0, abs_tol=1e-8):
            issues.append(f"interval{i}:native_time_invalid")
    window = history[-SAMPLES:]
    context = lambda r: [r["control_tick"], r["substep"]]
    result.update(window_start=context(window[0]), window_end=context(window[-1]), sample_count=len(window))
    end_index = 2*expected_end[0]+expected_end[1]
    expected_start = divmod(end_index-60, 2)
    if tuple(result["window_start"]) != expected_start or tuple(result["window_end"]) != tuple(expected_end):
        issues.append("exact_809_1_through_839_1_required")
    if issues:
        return result
    end = window[-1]
    distance = math.dist(end["chicken_world_wxyz"][:3], end["tcp_world_wxyz"][:3])
    if not math.isfinite(distance):
        issues.append("distance_not_finite")
        return result
    result["distance_m"] = distance
    if distance > RADIUS_M:
        issues.append("chicken_root_measured_TCP_distance_exceeded")
    missing = False
    for side_index, side_name in enumerate(("LEFT", "RIGHT")):
        ids = [2*side_index, 2*side_index+1]
        sums = [sum(row["jaw_positions_m"][j]-CLOSED_REFERENCE_M for j in ids) for row in window]
        deviations = [value-reference_opening_m[side_index] for value in sums]
        max_abs = max(map(abs, deviations))
        side = dict(side=side_name, reference_m=reference_opening_m[side_index],
            opening_sum=_extrema(sums), signed_reference_deviation=_extrema(deviations),
            maximum_absolute_deviation_m=max_abs, envelope_qualifies=_envelope_qualifies(max_abs),
            endpoint_command=end["jaw_commands"][side_index], pre_close=None,
            witnesses=[], jaws=[], issues=[])
        result["sides"].append(side)
        if not side["envelope_qualifies"]:
            side["issues"].append("opening_outside_demo_envelope")
        pre = None
        if side["endpoint_command"] != 0:
            side["issues"].append("not_commanded_closed_at_admission")
            side["progress_status"] = "NOT_APPLICABLE_OPEN_COMMAND"
        else:
            if pre_close_records is None:
                transition = len(history)-1
                while transition > 0 and history[transition-1]["jaw_commands"][side_index] == 0:
                    transition -= 1
                record = None if transition == 0 else dict(
                    pre_sample=history[transition-1], first_active=history[transition])
            else:
                record = pre_close_records[side_index]
            if record is None or record.get("pre_sample") is None:
                side["issues"].append("measured_pre_close_history_unavailable")
                side["progress_status"] = "UNAVAILABLE"
                missing = True
            else:
                pre, first = record["pre_sample"], record["first_active"]
                side["progress_status"] = "MEASURED"
                side["pre_close"] = dict(context=context(pre), physics_step=pre["physics_step"],
                    simulation_time_s=pre["simulation_time_s"],
                    positions_m=[pre["jaw_positions_m"][j] for j in ids],
                    transition_context=context(first), transition_time_s=first["simulation_time_s"])
        for j in ids:
            q = [row["jaw_positions_m"][j] for row in window]
            shortfall = [v-CLOSED_REFERENCE_M for v in q]
            progress = [pre["jaw_positions_m"][j]-v for v in q] if pre is not None else None
            jaw = dict(jaw_id=j+1, positions=_extrema(q), peak_to_peak_m=max(q)-min(q),
                shortfall=_extrema(shortfall), progress=_extrema(progress) if progress else None,
                qualifies=False, issues=[])
            if min(shortfall) < SHORTFALL_M:
                jaw["issues"].append("remaining_shortfall_below_1mm_in_window")
            if progress is not None:
                if min(progress) < PROGRESS_M:
                    jaw["issues"].append("closing_progress_below_1mm_in_window")
                jaw["qualifies"] = _jaw_qualifies(min(progress), min(shortfall))
            else:
                jaw["issues"].append("progress_"+side["progress_status"])
            if jaw["qualifies"]:
                side["witnesses"].append(j+1)
            side["jaws"].append(jaw)
        if not side["witnesses"]:
            side["issues"].append("no_fixed_jaw_witness")
        issues.extend(side_name+":"+issue for issue in side["issues"])
    result["status"] = "UNAVAILABLE" if missing else "FALSE" if issues else "TRUE"
    return result
