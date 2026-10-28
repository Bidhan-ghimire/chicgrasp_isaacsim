"""Pure capture policy helpers; no simulation imports, stepping or body writes.

The collector remains a raw diagnostic. ``bilateral_contact`` selects current
header identity/existence, never force meaning. ``evaluate_end_state`` consumes
the saved post-step rows independently of the runner's success decision.
"""

from copy import deepcopy
from dataclasses import asdict
import math
import numbers
import re

import numpy as np

from .interface import validate_policy_action
from .lift_event_pose_evidence import CHICKEN_BODY, Context, ENV_ROOT, GRIPPER_BODY, _json


CONTACT_SOURCE = "accepted_jaw_contact_collector_current_header_identity/v1"
PASS_LABEL = "PIPELINE_VERTICAL_SLICE_PASS_WITH_TEMPORARY_ATTACHMENT"
TARGETS = {
    "A": (0.65, -0.16, 0.2155),
    "B": (0.65, -0.16, 0.11130000000000001),
    "C": (0.650011448690015, -0.11144560745023943, 0.11130000000000001),
    "D": (0.650011448690015, -0.11144560745023943, 0.10041283641442127),
    "L": (0.650011448690015, -0.11144560745023943, 0.20041283641442126),
}
_UNUSED_CONTACT = re.compile(
    r"contact\[\d+\]:(?:malformed_or_nonfinite|unavailable):"
    r"(?:position|normal|impulse|separation|face_index0|face_index1|material0|material1)(?::.*)?\Z")


def frozen_candidate_actions():
    """The unchanged Cycle 2 float64 lerps, then the existing float32 storage."""
    target = {name: np.asarray(value, dtype=np.float64) for name, value in TARGETS.items()}
    rows = []
    for i in range(200):
        if i < 50:
            xyz = target["A"]
        elif i < 90:
            xyz = target["A"] + (target["B"] - target["A"]) * ((i - 49) / 40)
        elif i < 110:
            xyz = target["B"] + (target["C"] - target["B"]) * ((i - 89) / 20)
        elif i < 120:
            xyz = target["C"] + (target["D"] - target["C"]) * ((i - 109) / 10)
        elif i < 140:
            xyz = target["D"]
        elif i < 180:
            xyz = target["D"] + (target["L"] - target["D"]) * ((i - 139) / 40)
        else:
            xyz = target["L"]
        action = np.r_[xyz, (1, 1) if i < 120 else (0, 0)]
        validate_policy_action(action)
        stored = action.astype(np.float32)
        validate_policy_action(stored)
        rows.append(stored)
    return np.asarray(rows, dtype=np.float32)


def _integer(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def _finite(value):
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def _context(value, *, total_actions=200):
    if type(value) is Context:
        value = asdict(value)
    if (not isinstance(value, dict) or set(value) != {"episode_id", "control_tick", "substep"}
            or not isinstance(value["episode_id"], str) or not value["episode_id"].strip()
            or not _integer(value["control_tick"]) or not 0 <= value["control_tick"] < 6*total_actions
            or not _integer(value["substep"]) or value["substep"] not in (0, 1)):
        raise ValueError("invalid_episode_context")
    return dict(value)


def _clock(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or not _integer(value[0]) or value[0] < 0
            or not _finite(value[1]) or value[1] < 0):
        raise ValueError("invalid_native_clock")
    return tuple(value)


def bilateral_contact(sample, coverage, *, episode_id, context, native_clock):
    """Tri-state current bilateral identity, without retaining contact history.

``sample`` is the unchanged JawContactDiagnostic.post_physics_step output.
The independent current pose clock/context must match that sample. Unknown raw
force/vector payloads remain saved diagnostics, not requirements for the header's
contact existence. Header, identity, acquisition and reporting-health failures
remain UNAVAILABLE. No state is carried over from an earlier found event.
"""
    result = {"source": CONTACT_SOURCE, "sample": None, "status": "UNAVAILABLE",
              "left": None, "right": None, "active_jaw_ids": [], "issues": [],
              "ignored_unused_diagnostics": []}
    issues, ignored = result["issues"], result["ignored_unused_diagnostics"]
    try:
        current, clock = _context(context), _clock(native_clock)
        result["sample"] = dict(current, physics_step=clock[0], simulation_time_s=clock[1])
        if current["episode_id"] != episode_id:
            raise ValueError("wrong_current_episode")
        if not isinstance(sample, dict) or not isinstance(sample.get("reading"), dict):
            raise ValueError("missing_collector_sample")
        if _context({key: sample.get(key) for key in current}) != current:
            raise ValueError("stale_or_wrong_collector_context")
        reading = sample["reading"]
        callback = reading.get("collector_transport") == "callback-contact/v1"
        if reading.get("collector_transport") not in (None, "polling-contact/v1", "callback-contact/v1"):
            raise ValueError("unsupported_contact_transport")
        if _clock(reading.get("clock_before")) != clock or _clock(reading.get("clock_after")) != clock:
            raise ValueError("stale_or_changed_contact_clock")
        if reading.get("container_status") != "present":
            raise ValueError("contact_container_unavailable")
        if (not _integer(reading.get("stage_id")) or reading["stage_id"] <= 0
                or not isinstance(reading.get("headers"), list)):
            raise ValueError("invalid_contact_header_container")
        for name in ("local_omitted_headers", "local_omitted_data"):
            if not _integer(reading.get(name)) or reading[name] != 0:
                issues.append("incomplete_contact_copy:" + name)
        for source in (sample, reading):
            if not isinstance(source.get("issues"), list) or any(not isinstance(i, str) for i in source["issues"]):
                raise ValueError("malformed_collector_issues")
            for issue in source["issues"]:
                (ignored if _UNUSED_CONTACT.fullmatch(issue) else issues).append(issue)
        health = reading.get("health")
        if not isinstance(health, dict):
            raise ValueError("missing_contact_health")
        for when in ("before", "after"):
            item = health.get(when)
            if (not isinstance(item, dict) or not isinstance(item.get("issues"), list)
                    or any(not isinstance(i, str) for i in item["issues"])):
                raise ValueError("missing_contact_health:" + when)
            issues.extend(item["issues"])
            domain = item.get("stage_identity_domain")
            if domain == "constructed_managed_USD_stage_cache":
                validation = item.get("managed_identity_validation", {})
                stage_matches = (isinstance(validation, dict)
                    and validation.get("mode") == "isaaclab_usd_context"
                    and validation.get("verdict") == "PASS" and validation.get("phase") == "jaw_health"
                    and _integer(validation.get("monotonic_ns"))
                    and _integer(validation.get("stage_cache_id"))
                    and _integer(item.get("managed_stage_cache_id"))
                    and validation["stage_cache_id"] == item["managed_stage_cache_id"] == reading["stage_id"])
            else:
                # Existing explicit-attachment evidence keeps its original gate;
                # an unknown labelled domain never selects a permissive fallback.
                stage_matches = domain is None and item.get("attached_stage_id") == reading["stage_id"]
            if not stage_matches:
                issues.append("contact_health_stage_mismatch:" + when)
        if (not isinstance(coverage, dict) or coverage.get("chicken_actor") != CHICKEN_BODY
                or not isinstance(coverage.get("jaws"), dict)
                or not isinstance(coverage.get("chicken_colliders"), list)
                or not coverage["chicken_colliders"]):
            raise ValueError("missing_accepted_identity_map")
        chicken_shapes = set(coverage["chicken_colliders"])
        if any(not isinstance(p, str) or not p.startswith(CHICKEN_BODY + "/") for p in chicken_shapes):
            raise ValueError("invalid_chicken_collider_map")
        jaws = {}
        for j in range(1, 5):
            path = ENV_ROOT + f"/Robot/GripperRig/jaw_{j}_link"
            mapped = coverage["jaws"].get(path)
            if (not isinstance(mapped, dict) or mapped.get("jaw_id") != j
                    or not isinstance(mapped.get("enabled_colliders"), list) or not mapped["enabled_colliders"]
                    or any(not isinstance(p, str) or not p.startswith(path + "/") for p in mapped["enabled_colliders"])):
                raise ValueError("invalid_jaw_collider_map:" + str(j))
            jaws[path] = (j, set(mapped["enabled_colliders"]))
        pairs, ended = {}, set()
        for header in reading["headers"]:
            if not isinstance(header, dict) or "error" in header:
                raise ValueError("malformed_contact_header")
            if (not _integer(header.get("stage_id")) or header["stage_id"] != reading["stage_id"]
                    or not _integer(header.get("type")) or header["type"] not in (0, 1, 2)
                    or not _integer(header.get("num_contact_data")) or header["num_contact_data"] < 0
                    or (header["type"] == 1) != (header["num_contact_data"] == 0)):
                raise ValueError("invalid_current_contact_type_count_or_stage")
            for slot in (0, 1):
                if any(not isinstance(header.get(f"{kind}{slot}_path"), str)
                       or not header[f"{kind}{slot}_path"].startswith("/") for kind in ("actor", "collider")):
                    raise ValueError("missing_contact_pair_identity")
            for slot in (0, 1):
                actor, collider = header[f"actor{slot}_path"], header[f"collider{slot}_path"]
                other, other_shape = header[f"actor{1-slot}_path"], header[f"collider{1-slot}_path"]
                if actor not in jaws:
                    continue
                jaw_id, shapes = jaws[actor]
                if collider not in shapes:
                    raise ValueError("unmapped_jaw_collider")
                if other != CHICKEN_BODY:
                    continue
                if other_shape not in chicken_shapes:
                    raise ValueError("unmapped_chicken_collider")
                key = (actor, collider, other, other_shape)
                if header["type"] == 1:
                    ended.add(key)
                else:
                    pairs[key] = jaw_id
        if ended.intersection(pairs):
            issues.append("contradictory_active_and_ended_current_pair")
        active = sorted(set(pairs.values()))
        if not issues:
            result.update(left=bool(set(active) & {1, 2}), right=bool(set(active) & {3, 4}), active_jaw_ids=active)
            result["status"] = "TRUE" if result["left"] and result["right"] else "FALSE"
        if callback:
            from .callback_contact_collector import reduce_boundary
            boundary = reading.get("callback_boundary")
            if not isinstance(boundary, dict) or boundary.get("boundary", {}).get("clock") != list(clock):
                raise ValueError("callback_boundary_clock_not_current")
            awake = {actor: value.get("sleeping") is False for actor, value in
                     health["after"].get("jaws", {}).items()}
            reduction = reduce_boundary(boundary, reading.get("callback_decoded_headers"), coverage, awake=awake)
            # Recompute current witnesses, never trust a persisted TRUE or old
            # event-state latch. The replay checker separately audits history.
            issues.extend(reduction["issues"])
            result.update(collector_transport="callback-contact/v1",
                negative_qualification=reduction["negative_qualification"],
                ending_validation=reduction["ending_validation"],
                status="UNAVAILABLE", left=None, right=None, active_jaw_ids=[])
            if not issues and reduction["status"] == "TRUE":
                result.update(status="TRUE", left=True, right=True,
                              active_jaw_ids=reduction["active_jaw_ids"])
    except (KeyError, TypeError, ValueError) as exc:
        issues.append(str(exc))
        result.update(status="UNAVAILABLE", left=None, right=None, active_jaw_ids=[])
    result["issues"] = list(dict.fromkeys(issues))
    result["ignored_unused_diagnostics"] = list(dict.fromkeys(ignored))
    return _json(result)


def _measured_pose(value):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 7:
        raise ValueError("missing_measured_pose")
    if not all(_finite(v) for v in value):
        raise ValueError("nonfinite_or_invalid_pose")
    pose = np.asarray(value, dtype=np.float64)
    scale = max(abs(float(v)) for v in pose[3:])
    if scale == 0:
        raise ValueError("zero_quaternion")
    q = pose[3:] / scale
    q = q / math.hypot(*q)
    return pose, q


def _angle(q, reference):
    # conjugate(reference) * q; atan2 is stable near zero and q/-q invariant.
    w, u = reference[0], -reference[1:]
    vector = w * q[1:] + q[0] * u + np.cross(u, q[1:])
    scalar = w * q[0] - np.dot(u, q[1:])
    return math.degrees(2 * math.atan2(math.hypot(*vector), abs(float(scalar))))


def evaluate_end_state(rows, *, episode_id, origin_step, origin_time, admission_policy=None, event_close=None, protocol="v2"):
    """Evaluate the complete saved 2400-boundary episode, not a pass flag.

Each row has context (the three Context fields), pose (NativePoseReader.read),
runtime (resetting/truncated/terminated/invalid booleans), and attachment
{status, joint_path, generation, owned_usd_verified, request_context,
    confirmation_context}. The PENDING row also contains request_pose_reading,
    the owner's actual pre-authoring reading (not a different same-clock read).
    PENDING and the following CONFIRMED row must be present;
every later observation must verify the same owned USD description. The runner
stores these rows before reset or cleanup. This evaluator never mutates them.
The explicit geometric admission policy requests at839/1 and confirms840/0;
the legacy contact policy still requires confirmation by839/1. Only that
structural timing differs: baseline, rise and fixed-reference hold math is shared.
"""
    event = event_schedule(event_close, protocol=protocol) if event_close is not None else None
    d, b, h, e = (event[k] for k in ("decision", "confirmation", "room_reference", "score_end")) if event else (1679,1680,2159,2399)
    context_at = lambda value: _context(value, total_actions=event["total_actions"] if event else 200)
    room_pose = "chicken_actor_world_wxyz" if event else "chicken_in_gripper_wxyz"
    result = {"schema": "attachment-capture-end-state/v1", "episode_id": episode_id,
              "status": "FAIL", "label": None, "issues": [], "sample_count": len(rows) if isinstance(rows, list) else None,
              "requested_tcp_lift_m": None if event else TARGETS["L"][2] - TARGETS["D"][2],
              "actual_tcp_lift_end_m": None, "actual_tcp_final_delta_m": None,
              "chicken_baseline_world_z_m": None, "rise_max_m": None, "rise_at_lift_end_m": None,
              "first_5cm_sample": None, "translation_drift_max_m": None, "rotation_drift_max_deg": None,
              "lift_pass": False, "hold_pass": False, "attachment_history_pass": False,
              "confirmation_translation_change_m": None, "confirmation_rotation_change_deg": None,
              "native_origin": [origin_step, origin_time], "native_final_clock": None,
              "native_elapsed_s": None, "nominal_candidate_duration_s": (e+1)/120 if event else 20.0,
              "unassisted_grasp_established": False, "table_clear_established": False}
    issues, poses = result["issues"], []
    try:
        if admission_policy not in (None, "geometric-eligibility/v1", "side-opening-progress/v1"):
            raise ValueError("unsupported_admission_policy")
        geometric = admission_policy in ("geometric-eligibility/v1", "side-opening-progress/v1")
        origin = _clock((origin_step, origin_time))
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("missing_episode_identity")
        if not isinstance(rows, list) or len(rows) != e+1:
            raise ValueError("missing_or_extra_scheduled_samples")
        previous_clock, previous_status = origin, "UNATTACHED"
        pending_index = None
        confirmed = None
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"malformed_row:{i}")
            context = context_at(row.get("context"))
            expected = {"episode_id": episode_id, "control_tick": i // 2, "substep": i % 2}
            if context != expected:
                raise ValueError(f"cross_episode_gap_or_out_of_order:{i}")
            runtime = row.get("runtime")
            if (not isinstance(runtime, dict)
                    or any(runtime.get(flag) is not False for flag in ("resetting", "truncated", "terminated", "invalid"))):
                raise ValueError(f"invalid_runtime_or_reset_row:{i}")
            reading = row.get("pose")
            if not isinstance(reading, dict) or reading.get("valid") is not True or reading.get("issues") != []:
                raise ValueError(f"invalid_required_pose:{i}")
            if (reading.get("identities", {}).get("chicken_actor") != CHICKEN_BODY
                    or reading.get("identities", {}).get("gripper_actor") != GRIPPER_BODY
                    or reading.get("units") != "m"
                    or reading.get("pose_convention") != "position_xyz_quaternion_wxyz"):
                raise ValueError(f"invalid_pose_identity_or_convention:{i}")
            clock = _clock(reading.get("clock_before"))
            if (_clock(reading.get("clock_after")) != clock or clock[0] != origin[0] + i + 1
                    or clock[0] != previous_clock[0] + 1 or clock[1] <= previous_clock[1]):
                raise ValueError(f"invalid_native_clock_history:{i}")
            previous_clock = clock
            measured = reading.get("poses", {})
            values = {key: _measured_pose(measured.get(key)) for key in (
                "chicken_actor_world_wxyz", "gripper_world_wxyz", "tcp_base_wxyz", "chicken_in_gripper_wxyz")}
            poses.append(values)
            attachment = row.get("attachment")
            if not isinstance(attachment, dict):
                raise ValueError(f"missing_attachment_history:{i}")
            status = attachment.get("status")
            if status not in {"UNATTACHED", "PENDING", "CONFIRMED"}:
                raise ValueError(f"failed_or_detached_attachment:{i}")
            if status in {"PENDING", "CONFIRMED"}:
                path, generation = attachment.get("joint_path"), attachment.get("generation")
                if (attachment.get("owned_usd_verified") is not True or not isinstance(path, str)
                        or not path.startswith("/World/ChicGraspAttachment_")
                        or not _integer(generation) or generation <= 0):
                    raise ValueError(f"unverified_owned_joint:{i}")
                request = context_at(attachment.get("request_context"))
                if status == "PENDING":
                    if pending_index is not None or confirmed is not None or previous_status != "UNATTACHED":
                        raise ValueError(f"repeated_or_stale_pending:{i}")
                    if (request != context or
                            (i != d if geometric else
                             (not 732 <= context["control_tick"] <= 839 or i > 1678))):
                        raise ValueError(f"request_outside_closing_opportunity:{i}")
                    pending_index, identity = i, (path, generation)
                    request_identity = request
                    if attachment.get("confirmation_context") is not None:
                        raise ValueError(f"pending_falsely_confirmed:{i}")
                    request_reading = attachment.get("request_pose_reading")
                    if (not isinstance(request_reading, dict) or request_reading.get("valid") is not True
                            or request_reading.get("issues") != []
                            or _clock(request_reading.get("clock_before")) != clock
                            or _clock(request_reading.get("clock_after")) != clock
                            or request_reading.get("identities") != reading.get("identities")
                            or request_reading.get("units") != "m"
                            or request_reading.get("pose_convention") != "position_xyz_quaternion_wxyz"):
                        raise ValueError(f"missing_or_invalid_pre_authoring_measurement:{i}")
                    request_relation, request_q = _measured_pose(request_reading.get("poses", {}).get("chicken_in_gripper_wxyz"))
                else:
                    confirmation = context_at(attachment.get("confirmation_context"))
                    if confirmed is None:
                        if (pending_index is None or i != pending_index + 1 or i > (b if geometric else 1679)
                                or previous_status != "PENDING" or confirmation != context):
                            raise ValueError(f"confirmation_not_immediate_or_before_lift:{i}")
                        relation, relation_q = values["chicken_in_gripper_wxyz"]
                        distance = math.dist(relation[:3], request_relation[:3])
                        angle = _angle(relation_q, request_q)
                        result.update(confirmation_translation_change_m=distance, confirmation_rotation_change_deg=angle)
                        if not math.isfinite(distance) or distance > 0.01 or angle > 5.0:
                            raise ValueError(f"confirmation_relative_change_exceeds_limits:{i}")
                        confirmed = confirmation
                    if (path, generation) != identity or request != request_identity or confirmation != confirmed:
                        raise ValueError(f"attachment_replaced_or_history_changed:{i}")
            elif pending_index is not None:
                raise ValueError(f"unreported_detachment:{i}")
            elif any(attachment.get(key) is not None for key in ("joint_path", "generation", "request_context", "confirmation_context")) or attachment.get("owned_usd_verified") is not False:
                raise ValueError(f"unattached_contradictory_owned_state:{i}")
            previous_status = status
        if confirmed is None:
            raise ValueError("no_genuine_confirmation_before_lift")
        result.update(attachment_history_pass=True, native_final_clock=list(previous_clock),
                      native_elapsed_s=previous_clock[1] - origin[1])
        baseline = float(poses[d]["chicken_actor_world_wxyz"][0][2])
        rise = [float(poses[i]["chicken_actor_world_wxyz"][0][2]) - baseline for i in range(b, h+1)]
        reference, reference_q = poses[h][room_pose]
        translation = [math.dist(poses[i][room_pose][0][:3], reference[:3]) for i in range(h+1, e+1)]
        angles = [_angle(poses[i][room_pose][1], reference_q) for i in range(h+1, e+1)]
        if event:
            result["extrema_indices"] = dict(rise=b+rise.index(max(rise)),
                room_translation=h+1+translation.index(max(translation)), room_rotation=h+1+angles.index(max(angles)))
            result["signed_margins"] = dict(rise_m=max(rise)-0.05, room_translation_m=0.01-max(translation), room_rotation_deg=5.0-max(angles))
            result["event_schedule"] = event
            result["world_reference"] = rows[h]["pose"]["poses"][room_pose]
        first = next((i + b for i, value in enumerate(rise) if value >= 0.05), None)
        result.update(chicken_baseline_world_z_m=baseline, rise_max_m=max(rise), rise_at_lift_end_m=rise[-1],
                      first_5cm_sample=None if first is None else deepcopy(rows[first]["context"]),
                      translation_drift_max_m=max(translation), rotation_drift_max_deg=max(angles),
                      actual_tcp_lift_end_m=float(poses[h]["tcp_base_wxyz"][0][2] - poses[d]["tcp_base_wxyz"][0][2]),
                      actual_tcp_final_delta_m=float(poses[e]["tcp_base_wxyz"][0][2] - poses[d]["tcp_base_wxyz"][0][2]))
        if any(not math.isfinite(result[key]) for key in ("rise_max_m", "rise_at_lift_end_m", "translation_drift_max_m",
                "rotation_drift_max_deg", "actual_tcp_lift_end_m", "actual_tcp_final_delta_m")):
            raise ValueError("nonfinite_derived_measurement")
        if event:
            from .random_collection import world_scores
            wrise,wroom=world_scores(rows,event_close=event_close,protocol=protocol)
            if wrise["status"] not in ("PASS","FAIL") or wroom["status"] not in ("PASS","FAIL"):
                raise ValueError("incomplete_event_world_scores")
            result.update(rise_max_m=wrise["maximum_m"],translation_drift_max_m=wroom["maximum_displacement_m"],rotation_drift_max_deg=wroom["maximum_rotation_deg"], world_rise=wrise, world_room=wroom)
            result["extrema_indices"] = dict(rise=2*wrise["maximum_sample"][0]+wrise["maximum_sample"][1], room_translation=2*wroom["maximum_displacement_sample"][0]+wroom["maximum_displacement_sample"][1], room_rotation=2*wroom["maximum_rotation_sample"][0]+wroom["maximum_rotation_sample"][1])
            result["signed_margins"] = dict(rise_m=wrise["maximum_m"]-.05,room_translation_m=.01-wroom["maximum_displacement_m"],room_rotation_deg=5.-wroom["maximum_rotation_deg"])
        result["lift_pass"] = result["rise_max_m"] >= 0.05
        result["hold_pass"] = result["translation_drift_max_m"] <= 0.01 and result["rotation_drift_max_deg"] <= 5.0
        if not result["lift_pass"]:
            issues.append("measured_root_rise_below_0.05m")
        if not result["hold_pass"]:
            issues.append("measured_fixed_reference_drift_exceeds_limits")
        if not issues:
            result.update(status="PASS", label=PASS_LABEL)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        issues.append(str(exc))
    return _json(result)


def event_schedule(c, *, protocol="v2"):
    """Versioned schedule; default v2 retains the published timetable."""
    from .envelope_schedule import event_schedule as schedule
    return schedule(c, protocol=protocol)


def event_anchor(rows):
    """Reconstruct only from accepted native action-term receipts."""
    from .side_opening_admission import validate_command
    for row in rows:
        t,s=row["context"]["control_tick"],row["context"]["substep"]
        if t%6 == 0 and s == 0 and t//6 <= 199:
            from .geometric_capture import history_sample
            command=row.get("accepted_command")
            if command is None:
                raise ValueError("accepted_anchor_receipt_missing")
            validate_command(history_sample(row),command)
            if command["bits"] == [0,0]:
                return t//6
    return None


def evaluate_envelope_v2(rows, *, episode_id, origin_step, origin_time, close_action,
                         completed=False, primary_errors=(), cleanup_errors=(), lifecycle_events=(), protocol="v2"):
    """Retain the score through e separately from subsequent process behavior."""
    if protocol not in ("v2", "v3c"):
        raise ValueError("unknown_learned_envelope_protocol")
    plan=event_schedule(close_action, protocol=protocol) if close_action is not None else None
    verdict_key = "V3C_VERDICT" if protocol == "v3c" else "V2_VERDICT"
    gate_key = "FIRST_V3C_FAILURE_GATE" if protocol == "v3c" else "FIRST_V2_FAILURE_GATE"
    result=dict(schema="learned-envelope/"+protocol, status="FAIL", label=None,
        V2_VERDICT="NOT_ESTABLISHED", FIRST_V2_FAILURE_GATE=None, SCORE_EVIDENCE_STATUS="NOT_ESTABLISHED",
        TAIL_RELEASE=False, TAIL_TECHNICAL=False, PROCESS_COMPLETION="COMPLETE" if completed else "STOPPED",
        CLEANUP_STATUS="FAIL" if cleanup_errors else "CLEAN", event_schedule=plan, movement=None,
        unassisted_grasp_established=False, table_clear_established=False)
    if protocol == "v3c":
        result[verdict_key] = result.pop("V2_VERDICT")
        result[gate_key] = result.pop("FIRST_V2_FAILURE_GATE")
    if plan is None:
        result[gate_key]="ADMISSION_NO_ANCHOR" if completed and len(rows)==2400 else "TECHNICAL_BEFORE_ANCHOR"
        if completed and len(rows)==2400:
            result.update({verdict_key:"FAIL", "SCORE_EVIDENCE_STATUS":"ESTABLISHED"})
        return result
    d,e=plan["decision"],plan["score_end"]
    admission=rows[d].get("geometric_eligibility") if len(rows)>d else None
    result["admission"]=admission
    if admission and admission.get("status")=="FALSE":
        result.update({verdict_key:"FAIL", gate_key:"ADMISSION", "SCORE_EVIDENCE_STATUS":"ESTABLISHED"})
        return result
    if not admission or admission.get("status") != "TRUE":
        result[gate_key]="ADMISSION_NOT_ESTABLISHED"
        return result
    if len(rows)<=e:
        result[gate_key]="TECHNICAL_BEFORE_SCORE_END"
        return result
    score=evaluate_end_state(rows[:e+1],episode_id=episode_id,origin_step=origin_step,
        origin_time=origin_time,admission_policy="side-opening-progress/v1",event_close=close_action,protocol=protocol)
    result["movement"]=score
    # A runtime event at e is still in the score, even when its row was captured first.
    in_score_events=[x for x in lifecycle_events if 2*x.get("context",{}).get("control_tick",0)+x.get("context",{}).get("substep",0)<=e]
    if not score["attachment_history_pass"] or in_score_events:
        result[gate_key]="TECHNICAL_BEFORE_SCORE_END"
    else:
        result.update({verdict_key:score["status"], "status":score["status"], "label":score["label"],
            "SCORE_EVIDENCE_STATUS":"ESTABLISHED", gate_key:("RISE" if not score["lift_pass"] else "WORLD_ROOM" if not score["hold_pass"] else None)})
    tail=rows[e+1:]
    result["TAIL_RELEASE"]=any(r.get("attachment",{}).get("status")=="RELEASED" and r.get("commands") and 1 in r["commands"] for r in tail)
    result["TAIL_TECHNICAL"]=bool(result["SCORE_EVIDENCE_STATUS"]=="ESTABLISHED" and (primary_errors or lifecycle_events))
    result["tail_rows_observed"]=len(tail)
    return result


def world_event_motion(rows,c, *, protocol="v2"):
    from .random_collection import world_scores
    plan=event_schedule(c, protocol=protocol)
    rise,room=world_scores(rows,event_close=c,protocol=protocol)
    status="NOT_OBSERVED" if "NOT_EVALUABLE" in (rise['status'],room['status']) else "PASS" if rise['status']==room['status']=="PASS" else "FAIL"
    return dict(status=status,rise=rise,WORLD_ROOM=room,required=dict(rise=plan["rise_count"],room=240),
        available=dict(rise=max(0,min(len(rows),plan['rise_end']+1)-plan['confirmation']),room=max(0,min(len(rows),plan['score_end']+1)-plan['room_start'])))
