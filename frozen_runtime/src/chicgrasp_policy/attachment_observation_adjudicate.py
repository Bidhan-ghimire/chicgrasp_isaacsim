"""Read-only saved observation validation, independent of the lift verdict.

No native imports, scene changes, candidate execution or runtime acceptance
thresholds. A complete negative primitive observation can pass this objective.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from .attachment_observation_control import JAW_NAMES, JAW_LINKS, classify_control_observations
from .attachment_observation_evidence import _load, validate as validate_rgb
from .lift_event_pose_evidence import CHICKEN_BODY, GRIPPER_BODY
from .attachment_capture import bilateral_contact, frozen_candidate_actions
from .jaw_contact_diagnostic import HEADER_FIELDS, CONTACT_EVENT_NAMES
from .callback_contact_collector import flatten_boundary, reduce_boundary, verify_boundary, verify_bucket


PRELIFT_STOP = "CaptureFailure:attachment_not_confirmed_before_lift:FALSE"
CALLBACK_PRELIFT_STOP = "CaptureFailure:attachment_not_confirmed_before_lift:UNAVAILABLE"
POSE_KEYS = ("robot_root_world_wxyz", "gripper_world_wxyz", "chicken_actor_world_wxyz",
             "tcp_world_wxyz", "tcp_base_wxyz", "chicken_in_gripper_wxyz")
OPTIONAL_CHANNELS = {
    "default_joint_pos": ("m", 4), "default_joint_vel": ("m/s", 4),
    "joint_pos_target": ("m", 4), "joint_vel_target": ("m/s", 4),
    "joint_effort_target": ("N", 4), "computed_torque": ("N", 4), "applied_torque": ("N", 4),
    "native_actuation_force": ("N", 4), "native_position_target": ("m", 4),
    "measured_generalized_effort": ("N", 4), "incoming_joint_wrench": (["N"]*3 + ["N m"]*3, (4, 6)),
}


def _clock(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or type(value[0]) is not int or value[0] < 0
            or type(value[1]) not in (int, float) or not math.isfinite(value[1]) or value[1] < 0):
        raise ValueError("invalid_native_clock")
    return list(value)


def _array(value, shape, *, poses=False):
    uncoerced = np.asarray(value, dtype=object)
    if any(type(item) not in (int, float) for item in uncoerced.flat):
        raise ValueError("nonfinite_or_invalid_measurement_shape")
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != tuple(shape) or not np.isfinite(arr).all():
        raise ValueError("nonfinite_or_invalid_measurement_shape")
    if poses and np.any(np.linalg.norm(arr[..., 3:], axis=-1) <= 0):
        raise ValueError("zero_measurement_quaternion")
    return arr


def _pose(record, clock):
    if (record.get("valid") is not True or record.get("issues") != []
            or _clock(record.get("clock_before")) != clock or _clock(record.get("clock_after")) != clock
            or record.get("units") != "m" or record.get("pose_convention") != "position_xyz_quaternion_wxyz"
            or record.get("identities", {}).get("chicken_actor") != CHICKEN_BODY
            or record.get("identities", {}).get("gripper_actor") != GRIPPER_BODY
            or record.get("source") != "native_get_root_transforms/get_link_transforms/get_transforms"):
        raise ValueError("required_pose_provenance_invalid")
    for key in POSE_KEYS:
        _array(record["poses"][key], (7,), poses=True)


def _articulation(record, clock, anchor):
    if (record.get("valid") is not True or record.get("issues") != []
            or _clock(record.get("clock_before")) != clock or _clock(record.get("clock_after")) != clock
            or record.get("sample_interval_s") != 1 / 120
            or record.get("frame") != "joint coordinates (prismatic positive axis); link poses world xyz_wxyz"):
        raise ValueError("required_articulation_provenance_invalid")
    joints, links = record["joint_mapping"], record["link_mapping"]
    if (len(joints) != 4 or len(links) != 4 or [j.get("name") for j in joints] != list(JAW_NAMES)
            or [j.get("name") for j in links] != list(JAW_LINKS)
            or any(j.get("type") != "prismatic" or type(j.get("dof_index")) is not int
                   or j["dof_index"] < 0 for j in joints)
            or len({j["dof_index"] for j in joints}) != 4
            or any(type(j.get("link_index")) is not int or j["link_index"] < 0 for j in links)
            or len({j["link_index"] for j in links}) != 4):
        raise ValueError("jaw_joint_or_link_mapping_invalid")
    for link in links:
        if (link.get("path") != "/World/envs/env_0/Robot/GripperRig/" + link["name"]
                or anchor["robot_link_paths"][0][link["link_index"]] != link["path"]):
            raise ValueError("jaw_link_mapping_not_in_managed_articulation")
    _array(record["joint_positions_m"], (4,))
    _array(record["joint_velocities_m_s"], (4,))
    _array(record["jaw_link_world_wxyz"], (4, 7), poses=True)
    channels = record["channels"]
    if set(channels) != set(OPTIONAL_CHANNELS):
        raise ValueError("optional_channels_must_be_available_or_explicitly_unavailable")
    for name, (units, shape) in OPTIONAL_CHANNELS.items():
        channel = channels[name]
        if (channel.get("units") != units or channel.get("native_sample_clock") != clock
                or any(not isinstance(channel.get(k), str) or not channel[k]
                       for k in ("source", "frame", "quantity_status", "timing"))):
            raise ValueError("optional_channel_provenance_invalid:" + name)
        if channel.get("status") == "AVAILABLE":
            _array(channel["values"], (shape,) if isinstance(shape, int) else shape)
        elif (channel.get("status") != "UNAVAILABLE" or channel.get("values") is not None
              or not isinstance(channel.get("reason"), str) or not channel["reason"]):
            raise ValueError("optional_channel_status_invalid:" + name)


def _observation(record, clock, anchor):
    if (record.get("issues") != [] or _clock(record.get("clock_before")) != clock
            or _clock(record.get("clock_after")) != clock):
        raise ValueError("observation_missing_or_clock_invalid")
    _articulation(record["articulation"], clock, anchor)
    body = record["control"]
    if (body.get("valid") is not True or body.get("issues") != []
            or body.get("configuration_valid") is not True or body.get("configuration_issues") != []
            or _clock(body.get("clock_before")) != clock or _clock(body.get("clock_after")) != clock):
        raise ValueError("control_measurement_missing_or_clock_invalid")
    _array(body["body_world_wxyz"], (7,), poses=True)
    _array(body["body_velocity_world"], (6,))
    _array(body["support_static_usd_world_wxyz"], (7,), poses=True)


def _raw_snapshot(raw, coverage, spec):
    """Recheck saved classification/count claims using the retained same snapshot."""
    for key in ("source_header_count", "copied_header_count", "retained_header_count", "source_data_count", "copied_data_count"):
        if type(raw.get(key)) is not int or raw[key] < 0:
            raise ValueError("raw_contact_invalid_count:" + key)
    occupied = set()
    for index, (original, header) in enumerate(zip(raw["raw_headers"], raw["headers"])):
        if (original.get("source_header_index") != index or header.get("source_header_index") != index
                or any(type(original.get(k)) is not int or original[k] != header.get(k) for k in HEADER_FIELDS)
                or header.get("valid") is not True or header.get("issues") != []):
            raise ValueError("copied_and_decoded_header_disagree")
        kind, start, count = header["type"], header["contact_data_offset"], header["num_contact_data"]
        if (header["stage_id"] != raw["stage_id"] or kind not in (0, 1, 2) or start < 0 or count < 0
                or (kind == 1) != (count == 0) or (count and start + count > len(raw["data"]))
                or header["num_friction_anchors_data"] < 0 or header["friction_anchors_offset"] < 0
                or header["proto_index0"] != 0xFFFFFFFF or header["proto_index1"] != 0xFFFFFFFF
                or header.get("event_name") != CONTACT_EVENT_NAMES[kind]):
            raise ValueError("saved_header_lifecycle_or_range_invalid")
        indices = list(range(start, min(start + count, len(raw["data"]))))
        if header.get("copied_data_indices") != indices or occupied.intersection(indices):
            raise ValueError("saved_contact_association_invalid")
        occupied.update(indices)
        actors = [header[f"actor{i}_path"] for i in (0, 1)]
        colliders = [header[f"collider{i}_path"] for i in (0, 1)]
        if any(not isinstance(p, str) or not p.startswith("/") for p in actors + colliders):
            raise ValueError("saved_decoded_identity_invalid")
        if (set(colliders) == {spec["body_collider_path"], spec["support_collider_path"]}
                and any(actors[i] == spec["body_path"] and colliders[i] == spec["body_collider_path"] for i in (0, 1))):
            classification = "CONTROL_PAIR"
        elif any(p in coverage["jaws"] for p in actors):
            classification = "PRODUCTION_CHICKEN_JAW" if coverage["chicken_actor"] in actors else "PRODUCTION_JAW_OTHER"
        else:
            classification = "CHICKEN_OTHER" if coverage["chicken_actor"] in actors else "OTHER"
        if header.get("pair_classification") != classification:
            raise ValueError("saved_pair_classification_disagreement")
    if occupied != set(range(len(raw["data"]))):
        raise ValueError("raw_contact_unassociated_data")
    for index, item in enumerate(raw["data"]):
        if item.get("source_data_index") != index or item.get("issues") != []:
            raise ValueError("saved_contact_data_invalid")
        for name in ("position_raw", "normal_raw", "impulse_raw"):
            _array(item[name], (3,))
        if type(item.get("separation_raw")) not in (int, float) or not math.isfinite(item["separation_raw"]):
            raise ValueError("saved_separation_invalid")
        if any(type(item.get(name)) is not int for name in ("material0_raw", "material1_raw", "face_index0_raw", "face_index1_raw")):
            raise ValueError("saved_contact_metadata_invalid")


def _callback_snapshot(reading, raw, coverage, spec, clock, previous_state):
    """Rebuild callback provenance/currentness from original copied receipts.

    Saved attribution, current pairs, route labels and TRUE are claims to compare,
    never inputs that authorize a positive result.
    """
    envelope = reading.get("callback_boundary")
    if not isinstance(envelope, dict) or envelope != raw.get("callback_boundary"):
        raise ValueError("callback_envelope_missing_or_disagreement")
    check = verify_boundary(envelope)
    if check["integrity_issues"]:
        raise ValueError("callback_envelope_integrity:" + repr(check["integrity_issues"]))
    for records_key, counts_key in (("callbacks", "counts"), ("unassigned_callbacks", "unassigned_counts")):
        bucket = verify_bucket(envelope[records_key], envelope[counts_key],
            in_flight_sequences=envelope["in_flight_sequences"], limits=envelope["limits"])
        if bucket["integrity_issues"]:
            raise ValueError("callback_bucket_integrity:" + repr(bucket["integrity_issues"]))
    if (envelope.get("boundary", {}).get("clock") != clock
            or envelope["boundary"].get("stage_id") != reading.get("stage_id")
            or type(envelope.get("attribution_valid")) is not bool
            or envelope["attribution_valid"] != (not check["admission_issues"])):
        raise ValueError("callback_boundary_or_attribution_disagreement")
    flat = flatten_boundary(envelope)
    if (raw.get("raw_headers") != flat["raw_headers"] or raw.get("data") != flat["data"]
            or any(raw.get(key) != flat[key] for key in
                   ("source_header_count", "source_data_count", "local_omitted_headers", "local_omitted_data"))):
        raise ValueError("callback_flattened_payload_disagreement")
    decoded = reading.get("callback_decoded_headers")
    if not isinstance(decoded, list) or len(decoded) != len(raw["headers"]):
        raise ValueError("callback_decoded_coverage_incomplete")
    decoded_count = sum(all(isinstance(header.get(f"{kind}{side}_path"), str)
                            and header[f"{kind}{side}_path"].startswith("/")
                            for kind in ("actor", "collider") for side in (0, 1)) for header in decoded)
    if raw.get("decoded_header_count") != decoded_count:
        raise ValueError("callback_decoded_count_disagreement")
    for original, header, routed in zip(flat["raw_headers"], decoded, raw["headers"]):
        fields = (*original.keys(), "actor0_path", "actor1_path", "collider0_path", "collider1_path")
        if any(header.get(key) != routed.get(key) for key in fields):
            raise ValueError("callback_decoded_and_routed_fields_disagree")
        if any(header.get(key) != original[key] for key in original):
            raise ValueError("callback_decoded_changed_original_scalar")
    degraded = bool(check["admission_issues"] or reading.get("issues")
                    or any(item.get("issues") for item in raw["data"]))
    if not degraded:
        _raw_snapshot(raw, coverage, spec)
    else:
        # Keep truthful partial copies and decoding failures as observations.
        # They cannot produce a valid header/positive merely through a saved
        # flag. Original copied fields are still compared above, not omitted.
        for original, header in zip(flat["raw_headers"], raw["headers"]):
            if type(header.get("valid")) is not bool or not isinstance(header.get("issues"), list):
                raise ValueError("callback_degraded_header_diagnostic_missing")
            if header["valid"] != (not header["issues"]):
                raise ValueError("callback_degraded_header_validity_disagreement")
            if "error" in original and header["valid"]:
                raise ValueError("callback_failed_copy_claimed_valid")
            if header["valid"]:
                start, count = header["contact_data_offset"], header["num_contact_data"]
                if (type(start) is not int or type(count) is not int or start < 0 or count < 0
                        or (count and start + count > len(raw["data"]))):
                    raise ValueError("callback_degraded_range_claimed_valid")
                selected = deepcopy(raw["data"][start:start+count])
                for index, item in enumerate(selected):
                    item["source_data_index"] = index
                isolated_original, isolated_header = deepcopy(original), deepcopy(header)
                for item in (isolated_original, isolated_header):
                    item.update(source_header_index=0, contact_data_offset=0)
                isolated_header["copied_data_indices"] = list(range(count))
                _raw_snapshot(dict(raw, raw_headers=[isolated_original], headers=[isolated_header], data=selected), coverage, spec)
    awake = {actor: item.get("sleeping") is False
             for actor, item in reading["health"]["after"].get("jaws", {}).items()}
    replay = reduce_boundary(envelope, decoded, coverage, awake=awake, previous_state=previous_state)
    # JSON normalizes the nested canonical tuple pair keys without interpreting
    # any measured value or replacing unknown values with zero.
    replay = json.loads(json.dumps(replay, allow_nan=False))
    if replay != reading.get("callback_reduction") or replay != raw.get("callback_reduction"):
        raise ValueError("callback_currentness_replay_disagreement")
    return replay


def validate_callback_lifecycle(record, *, closed, stage_id=None):
    """Validate persisted setup/holder evidence; no native operations or imports."""
    if not isinstance(record, dict) or record.get("transport") != "callback-contact/v1":
        raise ValueError("callback_lifecycle_transport_missing")
    value = record.get("collector")
    if (not isinstance(value, dict) or value.get("schema") != "gate5-callback-lifecycle/v1"
            or value.get("transport") != "callback-contact/v1" or value.get("started") is not True
            or value.get("closed") is not closed or value.get("subscription_retained") is not (not closed)
            or type(value.get("subscription_generation")) is not int or value["subscription_generation"] <= 0
            or value.get("negative_qualification") != "NOT_ESTABLISHED"
            or value.get("native_drop_accounting") != "UNAVAILABLE"
            or (stage_id is not None and value.get("stage_id") != stage_id)):
        raise ValueError("callback_lifecycle_setup_or_closure_invalid")
    events = value.get("events")
    if (not isinstance(events, list) or not events
            or any(type(e.get("monotonic_ns")) is not int for e in events)
            or any(a["monotonic_ns"] > b["monotonic_ns"] for a, b in zip(events, events[1:]))):
        raise ValueError("callback_lifecycle_events_missing_or_reordered")
    kinds = [event.get("kind") for event in events]
    if (kinds[:2] != ["SUBSCRIBE_REQUESTED", "SUBSCRIBE_RETURNED"]
            or kinds.count("SUBSCRIBE_REQUESTED") != 1 or kinds.count("SUBSCRIBE_RETURNED") != 1
            or "MANAGED_ASSOCIATION" not in kinds or "EPOCH_CHANGED" not in kinds
            or (closed and kinds[-2:] != ["UNSUBSCRIBE_REQUESTED", "UNSUBSCRIBE_RETURNED"])
            or (not closed and any(str(k).startswith("UNSUBSCRIBE") for k in kinds))):
        raise ValueError("callback_lifecycle_setup_epoch_or_unsubscribe_order_invalid")
    for receipt in value["initialization_callbacks"]:
        if (receipt.get("phase") != "initialization" or receipt.get("assignment") not in
                ("INITIALIZATION_UNBOUND", "LATE_OR_UNASSIGNED")):
            raise ValueError("initialization_receipts_relabelled_as_candidate")
    for records_key, counts_key in (("initialization_callbacks", "initialization_counts"),
                                    ("undrained_unassigned_callbacks", "undrained_unassigned_counts")):
        bucket = verify_bucket(value[records_key], value[counts_key],
            in_flight_sequences=value["in_flight_sequences"], limits=value["limits"])
        if bucket["integrity_issues"]:
            raise ValueError("callback_lifecycle_bucket_integrity:" + repr(bucket["integrity_issues"]))
        if (set(value[counts_key]) != set(value["totals"])
                or any(type(amount) is not int or amount < value[counts_key][key]
                       for key, amount in value["totals"].items())):
            raise ValueError("callback_lifecycle_total_count_disagreement")
    expected_association = [dict(sequence=r["sequence"], candidate_current=False,
        stage_matches=None if not r["raw_headers"] else all(h.get("stage_id") == value["stage_id"]
            for h in r["raw_headers"])) for r in value["initialization_callbacks"]]
    association = value.get("initialization_stage_association")
    if (not isinstance(association, list)
            or sorted(association, key=lambda item: item["sequence"]) != expected_association):
        raise ValueError("initialization_stage_association_disagreement")
    return value


def _initialization_routing(record, coverage, spec, stage_id):
    """Verify the frozen non-current decoded subset against original receipts.

    An earlier-sequence in-flight copy may finish after the archive was decoded.
    Receipt order is preserved without inventing a native completion barrier.
    """
    archive = record.get("initialization_routing")
    originals = record["collector"]["initialization_callbacks"]
    if (not isinstance(archive, dict) or archive.get("schema") != "gate5-callback-initialization-routing/v1"
            or archive.get("candidate_current") is not False or archive.get("stage_id") != stage_id
            or not isinstance(archive.get("callbacks"), list)
            or len(archive["callbacks"]) > len(originals)
            or archive.get("source_sequences") != [item.get("sequence") for item in archive["callbacks"]]
            or archive["source_sequences"] != sorted(set(archive["source_sequences"]))
            or not set(archive["source_sequences"]) <= {item["sequence"] for item in originals}):
        raise ValueError("initialization_routing_missing_or_not_original_snapshot")
    control_records = 0
    original_by_sequence = {item["sequence"]: item for item in originals}
    for decoded in archive["callbacks"]:
        original = original_by_sequence[decoded["sequence"]]
        if (decoded.get("candidate_current") is not False or decoded.get("sequence") != original["sequence"]
                or not isinstance(decoded.get("headers"), list)
                or len(decoded["headers"]) != len(original["raw_headers"])):
            raise ValueError("initialization_routing_callback_disagreement")
        clean = True
        for raw_header, header in zip(original["raw_headers"], decoded["headers"]):
            if (any(header.get(key) != value for key, value in raw_header.items() if key != "error")
                    or header.get("production_route") != "INITIALIZATION_DIAGNOSTIC_ONLY"
                    or type(header.get("valid")) is not bool
                    or not isinstance(header.get("issues"), list)
                    or header["valid"] != (not header["issues"])):
                raise ValueError("initialization_routing_changed_copied_fields")
            clean = clean and header["valid"]
        if clean and not original["issues"]:
            raw = dict(stage_id=stage_id, source_header_count=original["raw_header_count"],
                source_data_count=original["raw_data_count"], copied_header_count=len(original["raw_headers"]),
                retained_header_count=len(decoded["headers"]), copied_data_count=len(original["data"]),
                raw_headers=original["raw_headers"], headers=decoded["headers"], data=original["data"])
            _raw_snapshot(raw, coverage, spec)
            control_records += sum(h["pair_classification"] == "CONTROL_PAIR" and h["type"] in (0, 2)
                                   for h in decoded["headers"])
    return dict(decoded_callback_snapshot=len(archive["callbacks"]),
                unarchived_initialization_callbacks=len(originals)-len(archive["callbacks"]),
                control_found_or_persist_records=control_records, candidate_current=False,
                scope="initialization delivery only; not candidate contact or production negative qualification")


def adjudicate_observation(root, result, measurements, candidate, *, compact_require_final=False):
    """Validate experiment-root sidecars and rows without trusting result labels.

    Called before observation_result.json is saved; the outer saved-file checker
    compares its persisted result with this same independent recomputation.
    END STATE and process/capture-integrity verdicts remain separate callers.
    """
    root = Path(root)
    answer = {"schema": "attachment-observation-result/v1", "CONTACT_CONTROL_RESULT": "CONTROL_OBSERVATION_INCOMPLETE",
              "OBSERVATION_RESULT": "INCOMPLETE", "issues": [], "details": {}, "control": None, "rgb": None}
    issues = answer["issues"]
    from .attachment_backend_adjudicate import (
        CPU, PRODUCTION_GPU, check_cpu_backend, interpret_cpu_contact, scoped_candidate_result)
    cpu = result.get("backend_mode", PRODUCTION_GPU) == CPU
    backend = check_cpu_backend(result) if cpu else None
    production_witnesses, bilateral_boundaries = 0, 0
    if cpu:
        answer.update(backend_mode=CPU, backend_diagnostic=backend,
            BACKEND_DIAGNOSTIC_RESULT=backend["BACKEND_DIAGNOSTIC_RESULT"],
            gpu_gate5_acceptance=False, end_state_scope=CPU)
    def load(name):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing_or_linked_observation_file:" + name)
        value = _load(path.read_bytes())
        if not isinstance(value, dict):
            raise ValueError("observation_file_not_mapping:" + name)
        return value
    try:
        if result.get("observation_mode") is not True:
            raise ValueError("observation_mode_not_recorded")
        configuration, pre = load("observation_configuration.json"), load("pre_candidate.json")
        if cpu:
            backend = check_cpu_backend(result, managed_anchor=pre.get("managed_stage"))
            answer.update(backend_diagnostic=backend,
                BACKEND_DIAGNOSTIC_RESULT=backend["BACKEND_DIAGNOSTIC_RESULT"])
            if backend["BACKEND_DIAGNOSTIC_RESULT"] != "CPU_CONFIGURATION_VERIFIED":
                raise ValueError("backend_configuration:" + backend["BACKEND_DIAGNOSTIC_RESULT"])
            if configuration.get("backend_mode") != CPU:
                raise ValueError("backend_observation_configuration_disagreement")
            if (type(pre.get("wall_monotonic_ns")) is not int
                    or result["backend_diagnostic"]["monotonic_ns"] > pre["wall_monotonic_ns"]):
                raise ValueError("backend_configuration_not_observed_before_pre_candidate")
            if (not isinstance(candidate, dict) or candidate.get("backend_mode") != CPU
                    or candidate.get("gpu_gate5_acceptance") is not False
                    or not isinstance(candidate.get("label"), str)
                    or not candidate["label"].startswith(CPU + ":")):
                raise ValueError("backend_candidate_result_scope_missing")
        transport = result.get("contact_transport", "polling-contact/v1")
        if transport not in ("polling-contact/v1", "callback-contact/v1"):
            raise ValueError("unsupported_collector_transport")
        callback = transport == "callback-contact/v1"
        from .compact_contact_replay import CompactReplay, SnapshotStream, storage_selected
        compact = storage_selected(result, measurements)
        compact_replay = (CompactReplay(root / "contact_compact", require_final=compact_require_final)
                          if compact else None)
        if (measurements.get("contact_transport", "polling-contact/v1") != transport
                or configuration.get("contact_transport", "polling-contact/v1") != transport):
            raise ValueError("collector_transport_disagreement")
        geometry = configuration.get("production_geometry")
        if (not isinstance(geometry, dict) or not isinstance(geometry.get("colliders"), list) or not geometry["colliders"]
                or geometry.get("representation") != "authored_USD_not_native_cooked_surface" or geometry.get("issues") != []
                or any(not isinstance(item, dict) or not isinstance(item.get("path"), str)
                       or not item["path"].startswith("/") or not isinstance(item.get("attributes"), dict)
                       or item.get("issues") != [] for item in geometry["colliders"])
                or configuration.get("native_drop_accounting") != "NATIVE_DROP_ACCOUNTING_UNAVAILABLE"
                or configuration.get("cooked_surfaces", {}).get("status") != "UNAVAILABLE"):
            raise ValueError("geometry_or_optional_availability_context_missing")
        rows, sensors, origin = measurements["rows"], measurements["sensor_samples"], _clock(measurements["origin_clock"])
        if compact:
            compact_replay.reader.verify_record_references([row["contact"]["reference"] for row in rows],
                                                           kinds=("boundary",))
        frames = measurements["command_frames"]
        episode = measurements["episode_id"]
        anchor = pre["managed_stage"]
        if (not isinstance(anchor, dict) or type(anchor.get("stage_cache_id")) is not int or anchor["stage_cache_id"] <= 0
                or anchor.get("physics_scene_path") != "/physicsScene"):
            raise ValueError("pre_candidate_managed_stage_missing")
        if (pre.get("kind") != "PRE_CANDIDATE" or pre.get("episode_id") != episode
                or any(pre.get(k, "missing") is not None for k in ("action_frame", "control_tick", "substep"))
                or pre.get("indices_status") != "NOT_APPLICABLE_NO_COMMAND_EXECUTED"
                or _clock(pre.get("native_clock")) != origin
                or pre.get("reset_provenance", {}).get("normal_initial_reset_returned") is not True
                or pre["reset_provenance"].get("candidate_started") is not False
                or pre["reset_provenance"].get("settling_established") is not False
                or type(pre.get("wall_monotonic_ns")) is not int):
            raise ValueError("pre_candidate_order_or_reset_provenance_invalid")
        _pose(pre["pose"], origin)
        _observation(pre["observation"], origin, anchor)
        if (not isinstance(rows, list) or not isinstance(sensors, list) or not sensors
                or sensors[0].get("native_clock") != origin or sensors[0].get("action_frame") != 0
                or sensors[0].get("wall_monotonic_ns", -1) < pre["wall_monotonic_ns"]):
            raise ValueError("first_camera_sample_not_after_pre_candidate_at_origin")
        pre_rgb = pre.get("rgb_metadata")
        if (not isinstance(pre_rgb, dict) or pre_rgb.get("kind") != "PRE_CANDIDATE"
                or any(pre_rgb.get(k, "missing") is not None
                       for k in ("action_frame", "control_tick", "substep", "policy_timestamp_ns"))
                or type(pre_rgb.get("shared_acquisition_action_frame")) is not int
                or pre_rgb["shared_acquisition_action_frame"] != 0
                or pre_rgb.get("native_clock") != origin
                or type(pre_rgb.get("wall_monotonic_ns")) is not int
                or pre_rgb["wall_monotonic_ns"] < pre["wall_monotonic_ns"]):
            raise ValueError("pre_candidate_rgb_provenance_invalid")
        first_sensor = sensors[0]
        if (type(first_sensor.get("shared_acquisition_action_frame")) is not int
                or first_sensor["shared_acquisition_action_frame"] != 0
                or any(pre_rgb.get(k) != first_sensor.get(k)
                       for k in ("native_clock", "sensor_frames", "wall_monotonic_ns", "camera_metadata"))):
            raise ValueError("pre_candidate_rgb_not_same_reset_acquisition_as_action_zero")
        counts, primary = result.get("counts"), result.get("primary_errors")
        stop = (CALLBACK_PRELIFT_STOP if callback and primary == [CALLBACK_PRELIFT_STOP] else PRELIFT_STOP)
        full = len(rows) == 2400
        partial = len(rows) == 1680
        if full:
            if (len(sensors) != 200 or counts != {"completed_control_ticks": 1200, "observed_substeps": 2400,
                                                 "completed_policy_actions": 200} or primary != []
                    or not isinstance(candidate, dict) or candidate.get("sample_count") != 2400):
                raise ValueError("full_observation_execution_coverage_invalid")
        elif partial:
            if (len(sensors) != 140 or counts != {"completed_control_ticks": 839, "observed_substeps": 1680,
                                                 "completed_policy_actions": 139}
                    or primary != [stop] or result.get("attachment_authoring_attempted") is not False
                    or candidate != (scoped_candidate_result({"status": "FAIL"}, incomplete=True)
                                     if cpu else {"status": "FAIL"})):
                raise ValueError("partial_observation_not_preserved_prelift_stop")
            owner = load("owner_before_cleanup.json")
            if (not isinstance(owner.get("operations"), list)
                    or any(op.get("kind") == "scheduled_request" for op in owner["operations"])):
                raise ValueError("partial_owner_authoring_not_independently_absent")
        else:
            raise ValueError("observation_execution_window_incomplete")
        control_samples, snapshots = [], []
        raw_totals = dict(copied_header_count=0, copied_data_count=0,
                          local_omitted_headers=0, local_omitted_data=0)
        valid_snapshot_count = 0
        callback_state, callback_lifecycle, callback_totals = None, None, {}
        initialization_delivery = None
        callback_sequences, callback_intervals, callback_endings = set(), [], 0
        if callback:
            callback_archive = load("contact_transport_before_cleanup.json")
            if compact:
                callback_archive = compact_replay.lifecycle(callback_archive)
            callback_lifecycle = validate_callback_lifecycle(callback_archive,
                closed=False, stage_id=anchor["stage_cache_id"])
            initialization_delivery = _initialization_routing(callback_archive, measurements["coverage"],
                configuration["control"]["spec"], anchor["stage_cache_id"])
            callback_totals = deepcopy(callback_lifecycle["initialization_counts"])
            for record in callback_lifecycle["initialization_callbacks"]:
                callback_sequences.add(record["sequence"])
            if len(callback_sequences) != len(callback_lifecycle["initialization_callbacks"]):
                raise ValueError("duplicate_initialization_callback_sequence")
        previous_time = origin[1]
        if not isinstance(frames, list) or len(frames) != len(sensors):
            raise ValueError("executed_command_frame_count_invalid")
        actions = frozen_candidate_actions()
        for i, row in enumerate(rows):
            try:
                if compact:
                    row = dict(row, contact=compact_replay.read(row["contact"],
                                                              expected_predicate=row["predicate"]))
                context = {"episode_id": episode, "control_tick": i // 2, "substep": i % 2}
                if row.get("context") != context:
                    raise ValueError("row_context_gap_or_wrong_episode")
                clock = _clock(row["pose"]["clock_after"])
                if clock[0] != origin[0] + i + 1 or clock[1] <= previous_time:
                    raise ValueError("nonsequential_native_clock")
                previous_time = clock[1]
                _pose(row["pose"], clock)
                _observation(row["observation"], clock, anchor)
                terminal_stop = (partial and i == 1679 and row.get("issues") == [stop])
                runtime = row.get("runtime", {})
                if (any(runtime.get(k) is not False for k in ("resetting", "truncated", "terminated"))
                        or (runtime.get("invalid") is not False and not (terminal_stop and runtime.get("invalid") is True))
                        or (row.get("issues", []) and not terminal_stop)):
                    raise ValueError("reset_or_invalid_observation_row")
                if partial and row.get("attachment", {}).get("status") != "UNATTACHED":
                    raise ValueError("partial_stop_contains_attachment_execution")
                predicate = bilateral_contact(row.get("contact"), measurements.get("coverage"),
                    episode_id=episode, context=context, native_clock=clock)
                if predicate != row.get("predicate"):
                    raise ValueError("saved_production_predicate_disagreement")
                if cpu:
                    bilateral_boundaries += predicate["status"] == "TRUE"
                if partial and ((not callback and i >= 1440 and predicate["status"] == "UNAVAILABLE")
                                or (i >= 1464 and predicate["status"] not in
                                    (("FALSE", "UNAVAILABLE") if callback else ("FALSE",)))):
                    # Three closed action frames first permit admission at732/0.
                    # Earlier TRUE contact is allowed but cannot be latched.
                    raise ValueError("partial_closing_or_admission_evidence_inconsistent")
                if terminal_stop and stop.rsplit(":", 1)[-1] != predicate["status"]:
                    raise ValueError("partial_stop_reason_disagrees_with_actual_predicate")
                reading = row["contact"]["reading"]
                if (reading.get("clock_before") != clock or reading.get("clock_after") != clock
                        or reading.get("stage_id") != anchor["stage_cache_id"]):
                    raise ValueError("production_contact_clock_or_managed_anchor_disagreement")
                for phase in ("before", "after"):
                    health = reading["health"][phase]
                    validation = health.get("managed_identity_validation", {})
                    if (health.get("stage_identity_domain") != "constructed_managed_USD_stage_cache"
                            or health.get("managed_stage_cache_id") != anchor["stage_cache_id"]
                            or validation.get("stage_cache_id") != anchor["stage_cache_id"]
                            or validation.get("verdict") != "PASS" or validation.get("mode") != "isaaclab_usd_context"
                            or validation.get("phase") != "jaw_health"):
                        raise ValueError("production_contact_health_managed_anchor_disagreement")
                raw = reading["observation_snapshot"]
                if (raw.get("clock_before") != clock or raw.get("clock_after") != clock
                        or raw.get("stage_id") != anchor["stage_cache_id"]
                        or raw.get("native_api_calls") != (0 if callback else 1)
                        or raw.get("native_api") != ("IPhysxSimulation.subscribe_contact_report_events" if callback
                                                    else "IPhysxSimulation.get_contact_report")
                        or (callback and (raw.get("collector_transport") != transport
                                          or reading.get("collector_transport") != transport))
                        or (not callback and (raw.get("observation_issues") != [] or raw.get("local_unclassified_headers") != 0))
                        or raw.get("container_status") != "present"
                        or (not callback and any(raw.get(k) != v for k, v in (
                                ("acquisition_status", "RAW_REPORT_OBSERVED"), ("acquisition_issues", []),
                                ("local_omitted_headers", 0), ("local_omitted_data", 0))))
                        or any(raw.get("context", {}).get(k) != value for k, value in context.items())
                        or raw.get("context", {}).get("issues") != (row["contact"].get("issues") if callback else [])
                        or raw.get("context", {}).get("origin_step") != origin[0]
                        or raw.get("context", {}).get("origin_time_s") != origin[1]):
                    raise ValueError("raw_contact_snapshot_missing_corrupt_or_unaligned")
                if ((not callback and raw.get("source_header_count") != len(raw["raw_headers"]))
                        or raw.get("copied_header_count") != len(raw["raw_headers"])
                        or raw.get("retained_header_count") != len(raw["headers"])
                        or len(raw["raw_headers"]) != len(raw["headers"])
                        or (not callback and raw.get("source_data_count") != len(raw["data"]))
                        or raw.get("copied_data_count") != len(raw["data"])):
                    raise ValueError("raw_contact_local_count_disagreement")
                if callback:
                    if (raw.get("acquisition_status") not in ("CALLBACK_BOUNDARY_FINALIZED", "UNKNOWN")
                            or raw.get("acquisition_issues") != reading.get("issues")
                            or (raw["acquisition_status"] == "UNKNOWN") != bool(reading.get("issues"))):
                        raise ValueError("callback_acquisition_diagnostic_disagreement")
                    replay = _callback_snapshot(reading, raw, measurements["coverage"],
                        configuration["control"]["spec"], clock, callback_state)
                    if cpu and not set(replay["issues"]) - {"negative_contact_qualification_not_established"}:
                        production_witnesses += len(replay["witnesses"])
                    callback_state = replay["state"]
                    callback_endings += replay["ending_validation"] == "REPORTED_PAIR_ENDING_OBSERVED"
                    envelope = reading["callback_boundary"]
                    interval = envelope["interval"]
                    if (interval["clock_before"] != (origin if i == 0 else rows[i-1]["pose"]["clock_after"])
                            or (callback_intervals and interval["id"] != callback_intervals[-1]+1)
                            or envelope["subscription_generation"] != callback_lifecycle["subscription_generation"]
                            or envelope["epoch"] != callback_lifecycle["epoch"]):
                        raise ValueError("callback_interval_history_or_lifecycle_disagreement")
                    callback_intervals.append(interval["id"])
                    for receipt in envelope["callbacks"] + envelope["unassigned_callbacks"]:
                        if receipt["sequence"] in callback_sequences:
                            raise ValueError("callback_receipt_reused_across_boundaries")
                        callback_sequences.add(receipt["sequence"])
                    for bucket_counts in (envelope["counts"], envelope["unassigned_counts"]):
                        if set(bucket_counts) != set(callback_totals):
                            raise ValueError("callback_count_schema_disagreement")
                        for key, value in bucket_counts.items():
                            callback_totals[key] += value
                else:
                    _raw_snapshot(raw, measurements["coverage"], configuration["control"]["spec"])
                drop = raw.get("native_drop_accounting", {})
                if drop.get("status") != "NATIVE_DROP_ACCOUNTING_UNAVAILABLE" or drop.get("count") is not None:
                    raise ValueError("unsupported_native_drop_claim")
                control_samples.append(row["observation"]["control"])
                valid_snapshot_count += 1
                for key in raw_totals:
                    raw_totals[key] += raw[key]
                if not compact:
                    snapshots.append(raw)
            except Exception as exc:
                issues.append(f"row[{i}]:{type(exc).__name__}:{exc}")
        if callback:
            outside = callback_lifecycle["undrained_unassigned_counts"]
            for key in callback_totals:
                callback_totals[key] += outside[key]
            if callback_totals != callback_lifecycle["totals"]:
                issues.append("callback_lifecycle_and_saved_boundary_totals_disagree")
        if compact:
            if compact_require_final:
                compact_replay.lifecycle(load("contact_transport_final.json"))
                compact_replay.reader.verify_consumed_records()
            else:
                compact_replay.reader.verify_consumed_records(kinds=("boundary",))
        for i, sensor in enumerate(sensors):
            prior = origin if i == 0 else rows[12*i-1]["pose"]["clock_after"]
            if sensor.get("action_frame") != i or sensor.get("native_clock") != prior:
                issues.append("camera_sample_alignment_invalid:" + str(i))
            frame = frames[i]
            first = rows[12*i]
            frame_clock = first["pose"]["clock_after"]
            expected = dict(first["context"], physics_step=frame_clock[0], simulation_time_s=frame_clock[1])
            if (frame.get("action_frame") != i or frame.get("commands") != actions[i, 3:].tolist()
                    or frame.get("sample") != expected
                    or frame.get("source") != "executed_candidate_command_first_post_substep/v1"):
                issues.append("executed_command_provenance_disagreement:" + str(i))
        rgb = validate_rgb(root / "rgb", expected_count=len(sensors) + 1,
                           sensors=sensors, pre_candidate=pre)
        answer["rgb"] = rgb
        rgb_manifest = load("rgb/manifest.json")
        if rgb_manifest.get("image_shape") != [240, 320, 3] or rgb_manifest.get("max_samples") != 200:
            issues.append("production_camera_dimensions_or_sample_bound_changed")
        if rgb_manifest.get("primary_errors") != primary:
            issues.append("rgb_manifest_primary_error_disagreement")
        if (type(rgb.get("pre_candidate_bundles")) is not int or rgb["pre_candidate_bundles"] != 1
                or type(rgb.get("action_bundles")) is not int or rgb["action_bundles"] != len(sensors)
                or type(rgb.get("completed_bundles")) is not int or rgb["completed_bundles"] != len(sensors) + 1):
            issues.append("rgb_pre_candidate_and_action_bundle_counts_disagree")
        if rgb["status"] != "PASS" or rgb["state"] != ("COMPLETE" if full else "PARTIAL"):
            issues.extend("rgb:" + x for x in rgb["issues"] or ["candidate_manifest_state_mismatch"])
        if compact and valid_snapshot_count == len(rows):
            snapshots = SnapshotStream(root / "contact_compact", rows, require_final=compact_require_final)
        control = classify_control_observations(control_samples, configuration=configuration["control"],
            raw_observations=snapshots, persistence_complete=not issues and len(control_samples) == len(rows))
        answer["control"] = control
        answer["CONTACT_CONTROL_RESULT"] = control["result"]
        if control["result"] == "CONTROL_OBSERVATION_INCOMPLETE" and (not callback or control["issues"]):
            issues.extend("control:" + x for x in control["issues"] or ["incomplete"])
        answer["details"] = {"candidate_window": "FULL" if full else "PARTIAL_PRELIFT",
            "scheduled_rows": len(rows), "acquired_camera_bundles": len(sensors),
            "pre_candidate_rgb_bundles": rgb.get("pre_candidate_bundles"),
            "persisted_rgb_bundles": rgb.get("completed_bundles"),
            "persisted_rgb_images": 2 * rgb.get("completed_bundles", 0),
            "pre_candidate_rgb_acquisition": "same reset-time acquisition as action zero; no extra render or acquisition",
            "completed_policy_actions": counts["completed_policy_actions"],
            "pre_candidate_clock": origin, "final_clock": rows[-1]["pose"]["clock_after"],
            "persisted_contact_snapshots": valid_snapshot_count,
            "persisted_raw_headers": raw_totals["copied_header_count"],
            "persisted_contact_data": raw_totals["copied_data_count"],
            "local_omitted_headers": raw_totals["local_omitted_headers"],
            "local_omitted_data": raw_totals["local_omitted_data"],
            "persistence_count_scope": "validated saved snapshots, not native generation/delivery/drop counts",
            "native_drop_accounting": "NATIVE_DROP_ACCOUNTING_UNAVAILABLE",
            "end_state_is_separate": True, "negative_or_inconclusive_control_is_not_a_grasp_gate": True}
        if callback:
            answer["details"].update(contact_transport=transport, negative_contact_qualification="NOT_ESTABLISHED",
                ending_validation="REPORTED_PAIR_ENDING_OBSERVED" if callback_endings else "NOT_ESTABLISHED",
                contact_delivery_degraded=bool(control.get("contact_delivery_issues")),
                initialization_delivery=initialization_delivery,
                interpretation="complete saved diagnostics is not complete native contact delivery")
    except Exception as exc:
        issues.append(type(exc).__name__ + ":" + str(exc))
    answer["issues"] = list(dict.fromkeys(issues))
    if not answer["issues"]:
        answer["OBSERVATION_RESULT"] = "PASS"
    if cpu:
        answer["cpu_interpretation"] = interpret_cpu_contact(answer["CONTACT_CONTROL_RESULT"],
            production_witnesses, bilateral_boundaries, qualified=not answer["issues"]
            and backend["BACKEND_DIAGNOSTIC_RESULT"] == "CPU_CONFIGURATION_VERIFIED")
    return deepcopy(answer)
