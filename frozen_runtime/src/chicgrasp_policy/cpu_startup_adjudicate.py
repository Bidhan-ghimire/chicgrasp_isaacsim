"""Read-only adjudication of the explicitly selected CPU startup observation.

This profile never qualifies a candidate, contact delivery, or a shutdown cure.
It deliberately accepts a fully observed, successfully contained failure while
keeping autonomous process completion as an independent result. No native imports.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .attachment_backend_adjudicate import check_cpu_backend


MODE = "CPU_STARTUP_SHUTDOWN_ONLY"
BACKEND = "CPU_BACKEND_DIAGNOSTIC"
FORBIDDEN_COUNTS = ("candidate_actions", "caller_steps", "later_harness_resets",
                    "caller_renders", "owner_bindings", "attachment_authoring",
                    "pre_candidate_acquisitions")
JOINTS = ["PrismaticJoint1", "PrismaticJoint2", "PrismaticJoint3", "PrismaticJoint4"]
BACKEND_FIELDS = frozenset((
    "configured_simulation_device", "environment_device", "simulation_device", "physics_context_type",
    "physics_context_path", "configured_physics_scene_path", "physics_scene_path", "physics_scene_valid",
    "gpu_dynamics", "broadphase", "suppress_readback", "physics_cuda_device", "renderer_active_gpu",
    "cameras_enabled", "rtx_sensors", "offscreen_render", "disable_contact_processing", "fabric_enabled",
    "native_view_valid", "native_view_device", "native_view_device_ordinal", "physics_cached_device",
    "physics_device", "gpu_sim", "gpu_pipeline", "derived_physics_device", "derived_gpu_sim",
    "derived_gpu_pipeline", "native_clock_step", "native_clock_time_s", "environment_step_counter"))
LIMIT = 8 * 1024 * 1024


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key:" + key)
        result[key] = value
    return result


def _parse(data):
    def reject(value):
        raise ValueError("nonfinite JSON token:" + value)
    return json.loads(data, object_pairs_hook=_pairs, parse_constant=reject)


def _read(path, *, bounded=True):
    if path.is_symlink() or not path.is_file() or (bounded and path.stat().st_size > LIMIT):
        raise ValueError("missing, linked, or oversized evidence:" + path.name)
    return path.read_bytes()


def _identity(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _same_identity(actual, expected):
    return (isinstance(expected, dict) and type(expected.get("bytes")) is int
            and {key: expected.get(key) for key in ("bytes", "sha256")} == actual)


def _integer(value):
    return type(value) is int and value >= 0


def _records(data, *, sequence="sequence"):
    if not data or not data.endswith(b"\n"):
        raise ValueError("empty or truncated JSONL")
    answer = [_parse(line) for line in data.decode("utf-8").splitlines()]
    previous = -1
    for index, record in enumerate(answer):
        if (not isinstance(record, dict) or type(record.get(sequence)) is not int
                or record[sequence] != index or not _integer(record.get("monotonic_ns"))
                or record["monotonic_ns"] < previous):
            raise ValueError("invalid sequence or child monotonic clock")
        previous = record["monotonic_ns"]
    return answer


def _scalar(value, representation):
    if not isinstance(representation, dict):
        raise ValueError("missing lossless scalar representation")
    if isinstance(value, dict):
        tag = value.get("nonfinite")
        if (set(value) != {"nonfinite"} or tag not in ("nan", "inf", "-inf")
                or representation.get("value") != value
                or representation.get("float_hex") != tag):
            raise ValueError("invalid nonfinite scalar representation")
        return float(tag)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("invalid numerical position")
    number = float(value)
    saved = representation.get("value")
    if (type(saved) not in (int, float) or float(saved).hex() != number.hex()
            or representation.get("float_hex") != number.hex()
            or representation.get("negative_zero") is not (number == 0 and math.copysign(1, number) < 0)):
        raise ValueError("lossless scalar or signed zero mismatch")
    return number


def _tensor_values(value):
    if not isinstance(value, dict) or "values" not in value:
        raise ValueError("missing comparison tensor")
    return value["values"]


def replay_guard(record):
    """Independent arithmetic on copied operands, not the native/production guard."""
    if record.get("schema") != "cpu-startup-guard-observation/v1":
        raise ValueError("guard observation schema")
    guard = record["guard"]
    if (guard.get("units") != "m" or guard.get("threshold_m") != 0.0
            or type(guard.get("threshold_m")) not in (float, int)
            or guard.get("threshold_inclusive") is not True
            or guard.get("pairwise_position_tolerance_m") is not None
            or guard.get("joint_names") != JOINTS
            or guard.get("pair_columns") != {"left": [0, 1], "right": [2, 3]}):
        raise ValueError("guard criterion or ordered names changed")
    indices = guard.get("joint_ids")
    if (not isinstance(indices, list) or len(indices) != 4 or len(set(indices)) != 4
            or not all(_integer(index) for index in indices) or guard.get("preserve_order") is not True):
        raise ValueError("resolved joint index provenance missing")
    tensor = guard["positions"]
    if (not isinstance(tensor.get("dtype"), str) or not tensor["dtype"]
            or not isinstance(tensor.get("device"), str) or not tensor["device"]):
        raise ValueError("position precision/device missing")
    values, representations = tensor["values"], tensor["scalar_representations"]
    if len(values) != len(representations):
        raise ValueError("lossless position count mismatch")
    positions = []
    for row, encoded in zip(values, representations):
        if len(row) != len(encoded):
            raise ValueError("lossless row width mismatch")
        positions.append([_scalar(value, representation) for value, representation in zip(row, encoded)])
    shape_valid = tensor.get("shape") == [1, 4] and len(positions) == 1 and len(positions[0]) == 4
    finite = all(math.isfinite(value) for row in positions for value in row)
    verdict = guard.get("verdict")
    if not shape_valid:
        if verdict != "REJECT_SHAPE":
            raise ValueError("shape rejection does not replay")
        return {"verdict": verdict, "positions": values, "joint_ids": indices, "bits": None}
    if not finite:
        if verdict != "REJECT_NONFINITE":
            raise ValueError("nonfinite rejection does not replay")
        return {"verdict": verdict, "positions": values, "joint_ids": indices, "bits": None}
    q = positions[0]
    bits = [int(value >= 0.0) for value in q]
    disagree = [bits[0] != bits[1], bits[2] != bits[3]]
    expected_verdict = "REJECT_PAIRED_BITS" if any(disagree) else "PASS"
    actual_bits = _tensor_values(guard.get("joint_bits"))
    if (actual_bits != [bits] or any(type(value) is not int for row in actual_bits for value in row)
            or _tensor_values(guard.get("left_disagrees")) != [disagree[0]]
            or _tensor_values(guard.get("right_disagrees")) != [disagree[1]]
            or verdict != expected_verdict):
        raise ValueError("captured classification or verdict does not replay")
    differences = [q[0] - q[1], q[2] - q[3]]
    explanatory = guard.get("pairwise_differences_m")
    distances = guard.get("signed_threshold_distances_m")
    if (not isinstance(explanatory, list) or len(explanatory) != 1
            or len(explanatory[0]) != 2 or not isinstance(distances, list) or len(distances) != 1
            or len(distances[0]) != 4
            or any(type(a) not in (int, float) or float(a).hex() != b.hex()
                   for a, b in zip(explanatory[0], differences))
            or any(type(a) not in (int, float) or float(a).hex() != b.hex()
                   for a, b in zip(distances[0], q))):
        raise ValueError("explanatory arithmetic does not replay copied operands")
    return {"verdict": verdict, "positions": values, "positions_float_hex": [v.hex() for v in q],
            "joint_names": JOINTS, "joint_ids": indices, "bits": bits,
            "disagreeing_pairs": [name for name, failed in zip(("left", "right"), disagree) if failed],
            "threshold_m": 0.0, "comparison": ">=", "pairwise_differences_m": differences,
            "signed_threshold_distances_m": q, "consumer": guard.get("consumer"),
            "sequence": record.get("sequence"), "monotonic_ns": record.get("monotonic_ns"),
            "native_clock": record.get("backend", {}).get("native_clock"),
            "metadata": record.get("metadata")}


def adjudicate_cpu_startup(attempt_root):
    """Inspect only saved files. No execution, process query, retry or mutation."""
    root = Path(attempt_root).resolve()
    capture, experiment = root / "capture", root / "capture" / "experiment"
    issues, artifacts, replayed = [], {}, []
    snapshot, phases, process, invocation, guards, journal = {}, [], {}, {}, [], []

    def require(condition, message):
        if not condition:
            issues.append(message)

    def load(path, *, lines=False):
        data = _read(path)
        artifacts[str(path.relative_to(root)).replace("\\", "/")] = _identity(data)
        value = _records(data) if lines else _parse(data)
        if not lines and not isinstance(value, dict):
            raise ValueError("JSON object required:" + path.name)
        return value, data

    try:
        invocation, invocation_bytes = load(root / "invocation.json")
        require(invocation.get("kind") == "attachment_capture"
                and invocation.get("execution_approved") is True
                and invocation.get("diagnostic_mode") == MODE
                and invocation.get("backend_mode") == BACKEND
                and invocation.get("campaign") == "cpu-startup-shutdown/v1"
                and invocation.get("observation_mode") is True
                and invocation.get("contact_transport") == "callback-contact/v1"
                and type(invocation.get("timeout_ms")) is int and invocation["timeout_ms"] == 180000
                and type(invocation.get("shutdown_timeout_ms")) is int and invocation["shutdown_timeout_ms"] == 60000
                and Path(invocation.get("output_directory", "")).resolve() == capture,
                "invocation_profile_invalid")
        argv = invocation.get("arguments", [])
        require(isinstance(argv, list), "invocation_arguments_invalid")
        for flag, expected in (("--diagnostic-mode", MODE), ("--backend-mode", BACKEND),
                               ("--contact-transport", "callback-contact/v1"), ("--seed", "42")):
            positions = [i for i, arg in enumerate(argv) if arg == flag]
            require(len(positions) == 1 and positions[0] + 1 < len(argv)
                    and argv[positions[0] + 1] == expected
                    and not any(isinstance(arg, str) and arg.startswith(flag + "=") for arg in argv),
                    "invocation_argument_invalid:" + flag)
        require("--execute" in argv and "--observation-mode" in argv, "diagnostic_execution_not_explicit")
        snapshot, snapshot_bytes = load(experiment / "startup_diagnostic.json")
        require(snapshot.get("schema") == "cpu-startup-shutdown/v1"
                and snapshot.get("diagnostic_mode") == MODE and snapshot.get("backend_mode") == BACKEND,
                "snapshot_profile_invalid")
        for key in ("candidate_executions", "attachment_authoring"):
            require(type(snapshot.get(key)) is int and snapshot[key] == 0, "scope_violation:" + key)
        counts = snapshot.get("forbidden_counts", {})
        require(isinstance(counts, dict) and all(type(counts.get(key)) is int and counts[key] == 0
                                                for key in FORBIDDEN_COUNTS), "forbidden_work_or_missing_counts")
        require(snapshot.get("diagnostic_write_errors") == [], "diagnostic_write_errors_or_missing")
        provenance = snapshot.get("execution_provenance", {})
        commit = provenance.get("executed_commit")
        require(isinstance(commit, str) and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
                and provenance.get("errors") == [], "execution_identity_unavailable")
        guards, guard_bytes = load(capture / "structured" / "guard_observations.jsonl", lines=True)
        phases, phase_bytes = load(capture / "structured" / "failure_phases.jsonl", lines=True)
        require(snapshot.get("guard_observations") == guards, "snapshot_guard_stream_disagreement")
        identities = snapshot.get("identities", {})
        require(_same_identity(_identity(guard_bytes), identities.get("guard_stream")), "guard_stream_identity_mismatch")
        prefix = identities.get("phase_prefix", {})
        prefix_size, prefix_count = prefix.get("bytes"), prefix.get("records")
        require(_integer(prefix_size) and _integer(prefix_count), "phase_prefix_identity_invalid")
        if _integer(prefix_size) and _integer(prefix_count):
            prefix_bytes = phase_bytes[:prefix_size]
            require(_same_identity(_identity(prefix_bytes), prefix)
                    and prefix_bytes.endswith(b"\n") and prefix_bytes.count(b"\n") == prefix_count,
                    "phase_prefix_identity_mismatch")
        begin = [p for p in phases if p.get("phase") == "SHUTDOWN_SEQUENCE_BEGIN" and p.get("event") == "requested"]
        require(len(begin) == 1, "shutdown_begin_missing_or_duplicate")
        if begin and _integer(prefix_count):
            require(begin[0]["sequence"] == prefix_count, "snapshot_not_immediately_before_shutdown_sequence")
        construction = [p for p in phases if p.get("phase") == "environment_construction"]
        requested = [p for p in construction if p.get("event") == "requested"]
        ended = [p for p in construction if p.get("event") in ("returned", "failed")]
        require(len(requested) == 1 and len(ended) == 1, "construction_boundary_not_complete")
        outcome = snapshot.get("constructor_outcome")
        require(outcome in ("RETURNED", "FAILED"), "constructor_outcome_invalid")
        if ended:
            require(ended[0]["event"] == ("returned" if outcome == "RETURNED" else "failed"), "constructor_outcome_disagreement")
            if begin:
                require(ended[0]["sequence"] < begin[0]["sequence"], "cleanup_preceded_constructor_outcome")
        require((snapshot.get("primary_error") is None if outcome == "RETURNED" else
                 isinstance(snapshot.get("primary_error"), str) and bool(snapshot["primary_error"])),
                "constructor_primary_error_disagreement")
        initialization = snapshot.get("initialization", {})
        require(type(initialization.get("application_attempts")) is int
                and initialization["application_attempts"] == 1
                and initialization.get("application_returned") is True
                and initialization.get("environment_returned") is (outcome == "RETURNED")
                and initialization.get("initial_reset_returned") is False
                and type(initialization.get("pre_reset_observations")) is int
                and initialization["pre_reset_observations"] == 0
                and initialization.get("candidate_started") is False,
                "initialization_accounting_invalid")
        require(0 < len(guards) <= 64, "guard_boundary_not_reached_or_unbounded")
        for record in guards:
            try:
                replay = replay_guard(record)
                replayed.append(replay)
                require(record["guard"].get("consumer", {}).get("kind") == "DIMENSION_DISCOVERY",
                        "nonconstructor_guard_evaluation")
                metadata = record.get("metadata", {})
                require(metadata.get("diagnostic_mode") == MODE
                        and type(metadata.get("construction_generation")) is int and metadata["construction_generation"] == 1
                        and type(metadata.get("subscription_generation")) is int and metadata["subscription_generation"] == 1
                        and type(metadata.get("later_harness_resets")) is int and metadata["later_harness_resets"] == 0
                        and isinstance(metadata.get("episode_id"), str) and bool(metadata["episode_id"]),
                        "guard_generation_or_context_invalid")
                if requested and ended:
                    require(requested[0]["monotonic_ns"] <= record["monotonic_ns"] <= ended[0]["monotonic_ns"],
                            "guard_not_inside_construction_clock_bracket")
            except Exception as exc:
                issues.append("guard_invalid:" + type(exc).__name__ + ":" + str(exc))
        if outcome == "RETURNED":
            require(bool(replayed) and all(g.get("verdict") == "PASS" for g in replayed), "constructor_bypassed_rejection")
        if guards and any(g.get("guard", {}).get("verdict", "").startswith("REJECT") for g in guards):
            last_guard = guards[-1].get("guard", {})
            require(outcome == "FAILED" and isinstance(last_guard.get("error"), str)
                    and bool(last_guard["error"]) and snapshot.get("primary_error", "").endswith(last_guard["error"]),
                    "original_guard_rejection_not_preserved")
        for phase in phases:
            require(isinstance(phase.get("phase"), str) and isinstance(phase.get("event"), str), "phase_fields_invalid")
            require(phase.get("phase") not in ("initial_reset", "candidate", "owner_binding", "pre_candidate"),
                    "forbidden_reached_phase:" + str(phase.get("phase")))
            require(not phase.get("phase_evidence_errors"), "phase_write_errors")
            require(not (phase.get("phase") == "required_finalization" and phase.get("event") == "failed"),
                    "required_finalization_failed")
            if outcome == "FAILED":
                require(not (phase.get("phase") == "cleanup" and phase.get("operation") in
                             ("environment", "recorder_manager")), "fabricated_returned_environment_cleanup")
            if phase.get("phase") == "application_close" and phase.get("event") == "requested":
                require(phase.get("wait_for_replicator") is True and phase.get("skip_cleanup") is False,
                        "application_close_contract_changed")
        require(not any(path.suffix.lower() in (".png", ".npz") for path in experiment.rglob("*")),
                "candidate_artifacts_present")
        observations = snapshot.get("backend_observations")
        require(isinstance(observations, list) and bool(observations), "backend_observations_missing")
        require(isinstance(observations, list)
                and observations[:len(guards)] == [record.get("backend") for record in guards]
                and (len(observations) == len(guards) or
                     (outcome == "RETURNED" and len(observations) == len(guards) + 1
                      and observations[-1].get("phase") == "CONSTRUCTOR_RETURNED")),
                "backend_guard_observation_disagreement")
        for backend in observations or []:
            require(isinstance(backend, dict) and backend.get("schema") == "cpu-startup-backend-observation/v1",
                    "backend_schema_invalid")
            require(isinstance(backend.get("fields"), dict) and set(backend["fields"]) == BACKEND_FIELDS,
                    "backend_field_coverage_missing")
            require(backend.get("verification") == "CPU_CONFIGURATION_UNAVAILABLE", "partial_backend_promoted_to_verified")
            for field in backend.get("fields", {}).values():
                require(isinstance(field, dict) and all(key in field for key in
                    ("source", "raw_value", "raw_type", "phase", "kind", "status"))
                    and field.get("kind") in ("observed", "configured", "derived")
                    and field.get("status") in ("OBSERVED", "UNAVAILABLE", "ERROR"), "backend_field_provenance_invalid")
            require(backend.get("native_clock") == {"step": backend.get("fields", {}).get("native_clock_step"),
                    "time_s": backend.get("fields", {}).get("native_clock_time_s")}, "native_clock_provenance_mismatch")
        journal, journal_bytes = load(capture / "structured" / "events.jsonl", lines=True)
        close_reached = any(p.get("phase") in ("runtime_close_app", "application_close") and p.get("event") == "requested" for p in phases)
        if close_reached or (experiment / "result.json").exists() or (experiment / "artifact_manifest.json").exists():
            finished, _ = load(experiment / "result.json")
            manifest, manifest_bytes = load(experiment / "artifact_manifest.json")
            require(manifest.get("schema") == "capture-artifacts/v1" and isinstance(manifest.get("files"), dict), "finished_manifest_invalid")
            for name, expected in manifest.get("files", {}).items():
                target = (experiment / name).resolve()
                if not target.is_relative_to(experiment) or target == experiment:
                    issues.append("finished_manifest_path_escape")
                    continue
                require(_same_identity(_identity(_read(target)), expected), "finished_artifact_identity_mismatch:" + name)
            require(set(manifest.get("files", {})) == {p.relative_to(experiment).as_posix() for p in experiment.rglob("*") if p.is_file()} - {"artifact_manifest.json"}, "finished_manifest_membership_mismatch")
            require(finished.get("recording_errors") == [], "finished_recording_error")
            require(finished.get("diagnostic_mode") == MODE, "finished_mode_disagreement")
            require(len(journal) >= 2 and journal[-2].get("kind") == "PRE_CLOSE_EVIDENCE_COMPLETE"
                    and journal[-1].get("kind") == "CLOSE_REQUESTED"
                    and all(_same_identity(_identity(manifest_bytes), row.get("manifest")) for row in journal[-2:]),
                    "finished_terminal_evidence_invalid")
        process, _ = load(capture / "process_result.json")
        require(process.get("invocation_sha256") == _identity(invocation_bytes)["sha256"], "parent_invocation_identity_mismatch")
        for field, path in (("stdout", capture / "stdout.bin"), ("stderr", capture / "stderr.bin"),
                            ("completed_failure_phases", capture / "structured" / "failure_phases.jsonl"),
                            ("completed_guard_observations", capture / "structured" / "guard_observations.jsonl"),
                            ("completed_startup_diagnostic", experiment / "startup_diagnostic.json"),
                            ("completed_journal", capture / "structured" / "events.jsonl")):
            data = _read(path, bounded=field not in ("stdout", "stderr"))
            expected = process.get(field)
            require(_same_identity(_identity(data), expected) and Path(expected.get("path", "")).resolve() == path,
                    "parent_artifact_identity_mismatch:" + field)
            artifacts[str(path.relative_to(root)).replace("\\", "/")] = _identity(data)
    except Exception as exc:
        issues.append("essential_evidence_unavailable:" + type(exc).__name__ + ":" + str(exc))

    # Parent clocks are compared only to parent clocks; never align them to child/Kit.
    parent_issues = []
    def parent_require(value, message):
        if not value:
            parent_issues.append(message)
    for key, expected in (("diagnostic_mode", MODE), ("timeout_ms", 180000), ("absolute_deadline_ms", 180000),
                          ("parent_launch_boundary_ms", 0), ("shutdown_timeout_ms", 60000),
                          ("termination_and_drain_cap_ms", 5000), ("probe_process_invocations", 1),
                          ("job_assigned_before_resume", True), ("root_resumed", True),
                          ("root_exit_verified", True), ("tree_empty_verified", True),
                          ("capture_complete", True), ("job_disposed_after_observation", True),
                          ("job_members_before_disposal", 0), ("containment_budget_overrun", False)):
        parent_require(type(process.get(key)) is type(expected) and process.get(key) == expected, "parent_contract:" + key)
    parent_require(process.get("containment_errors") == [] and process.get("errors") == [], "parent_errors_or_missing")
    trigger = process.get("shutdown_trigger_observation")
    effective = 180000
    marker_observations = process.get("shutdown_marker_observations")
    parent_require(isinstance(marker_observations, list) and bool(marker_observations)
                   and isinstance(trigger, dict) and marker_observations[0] == trigger,
                   "first_shutdown_receipt_unproven")
    for index, marker in enumerate(marker_observations or []):
        parent_require(isinstance(marker, dict), "shutdown_marker_invalid")
        if not isinstance(marker, dict):
            continue
        parent_require(marker.get("first_trigger") is (index == 0)
                       and (index == 0 or marker.get("deadline_enforced") is False), "shutdown_timer_renewed")
        source = marker.get("source")
        records = phases if source == "failure_phases.jsonl" else journal if source == "events.jsonl" else []
        sequence = marker.get("seq")
        if not _integer(sequence) or sequence >= len(records):
            parent_issues.append("shutdown_receipt_source_missing")
            continue
        record = records[sequence]
        phase_name = record.get("phase") if source == "failure_phases.jsonl" else record.get("kind")
        primary = phase_name == "SHUTDOWN_SEQUENCE_BEGIN"
        fallback = (source == "events.jsonl" and phase_name == "CLOSE_REQUESTED") or (
            source == "failure_phases.jsonl" and record.get("event") == "requested"
            and phase_name in ("scene_cleanup", "cleanup", "writer_finalization", "required_finalization",
                               "runtime_close_app", "application_close"))
        parent_require((primary or fallback) and marker.get("phase") == phase_name
                       and marker.get("child_monotonic_ns") == record.get("monotonic_ns")
                       and marker.get("fallback") is (not primary)
                       and _integer(marker.get("parent_observed_ms")), "shutdown_receipt_source_mismatch")
    if isinstance(trigger, dict):
        parent_require(process.get("shutdown_marker_observed_ms") == trigger.get("parent_observed_ms"),
                       "shutdown_receipt_time_disagreement")
    received_phases = process.get("failure_phase_observations")
    parent_require(isinstance(received_phases, list) and len(received_phases) == len(phases),
                   "parent_phase_receipts_incomplete")
    for index, receipt in enumerate(received_phases or []):
        if not isinstance(receipt, dict) or index >= len(phases):
            parent_issues.append("parent_phase_receipt_invalid")
            continue
        original = phases[index]
        parent_require(receipt.get("seq") == index and receipt.get("lifecycle_phase") == original.get("phase")
                       and receipt.get("event") == original.get("event")
                       and receipt.get("child_monotonic_ns") == original.get("monotonic_ns")
                       and _integer(receipt.get("parent_observed_ms")), "parent_phase_receipt_mismatch")
    if isinstance(trigger, dict) and trigger.get("deadline_enforced") is True:
        receipt = process.get("shutdown_marker_observed_ms")
        parent_require(_integer(receipt), "shutdown_receipt_missing")
        if _integer(receipt):
            effective = min(180000, receipt + 60000)
            parent_require(process.get("shutdown_deadline_ms") == effective, "renewed_or_invalid_shutdown_deadline")
        parent_require(trigger.get("read_phase") == "live_poll", "late_shutdown_receipt_enforced")
    else:
        parent_require(process.get("shutdown_deadline_ms") is None, "unenforced_shutdown_deadline_present")
    parent_require(process.get("effective_deadline_ms") == effective, "effective_deadline_invalid")
    times = ("root_exit_observed_ms", "job_empty_observed_ms", "drain_completed_observed_ms",
             "job_disposed_ms", "stream_files_finalized_ms")
    for key in times:
        parent_require(_integer(process.get(key)), "parent_clock_missing:" + key)
    if all(_integer(process.get(key)) for key in times):
        parent_require(process["job_empty_observed_ms"] <= process["job_disposed_ms"], "job_disposed_before_empty")
        parent_require(process["drain_completed_observed_ms"] <= process["stream_files_finalized_ms"], "streams_finalized_before_drain")
    forced = any(process.get(key) is True for key in ("forced_termination", "parent_termination_requested",
                  "startup_containment_used", "kill_on_job_close_used", "kill_on_job_close_possible", "timed_out"))
    if forced:
        termination = process.get("termination_requested_ms")
        parent_require(_integer(termination) and termination >= effective, "premature_or_unrecorded_termination")
        cap = process.get("containment_deadline_ms")
        parent_require(_integer(cap) and cap <= effective + 5000, "containment_deadline_renewed")
        if _integer(cap):
            parent_require(all(_integer(process.get(key)) and process[key] <= cap for key in times), "containment_or_drain_exceeds_budget")
        parent_require(process.get("termination_reason") ==
                       ("shutdown_sequence_deadline" if effective < 180000 else "absolute_deadline"),
                       "termination_reason_disagrees_with_deadline")
    else:
        parent_require(all(_integer(process.get(key)) and process[key] <= effective for key in times[:3]), "autonomous_observation_exceeds_deadline")
    for live, exited in (("last_root_alive_observed_ms", "root_exit_observed_ms"),
                         ("last_job_nonempty_observed_ms", "job_empty_observed_ms")):
        if process.get(live) is not None:
            parent_require(_integer(process[live]) and _integer(process.get(exited)) and process[live] <= process[exited], "parent_liveness_clock_invalid:" + live)
    parent_require(process.get("failure_phase_complete_records") == len(phases), "parent_phase_record_coverage_mismatch")
    if "capture/structured/failure_phases.jsonl" in artifacts:
        parent_require(process.get("failure_phase_complete_bytes") == artifacts["capture/structured/failure_phases.jsonl"]["bytes"],
                       "parent_phase_byte_coverage_mismatch")
    issues.extend(parent_issues)
    autonomous = not parent_issues and not forced and process.get("autonomous_tree_exit") is True
    process_pass = autonomous and type(process.get("exit_code")) is int and process["exit_code"] == 0
    containment_pass = not parent_issues
    returned = [p for p in phases if p.get("phase") in ("application_close", "CLOSE_RETURNED") and p.get("event") == "returned"]
    entries = [p for p in phases if p.get("event") == "requested" and p.get("phase") in
               ("scene_cleanup", "cleanup", "runtime_close_app", "application_close", "required_finalization", "writer_finalization")]
    shutdown = ("FAILURE_TO_CONTAIN_OWNED_TREE" if not containment_pass else
                "MISSING_OR_CORRUPT_PHASE_EVIDENCE" if not phases else
                "RECORDED_CLOSE_RETURN_THEN_PERSISTENT_PROCESS" if returned and forced else
                "RECORDED_CLOSE_RETURN_THEN_AUTONOMOUS_EXIT" if returned and autonomous else
                "AUTONOMOUS_EXIT_WITHOUT_CLOSE_RETURN" if autonomous else
                "CLEANUP_OR_CLOSE_ENTRY_WITHOUT_RETURN_BEFORE_CONTAINMENT" if entries and forced else
                "SHUTDOWN_OUTCOME_INDETERMINATE")
    backend_result = {"status": "UNAVAILABLE", "contract_result": "CPU_CONFIGURATION_UNAVAILABLE",
                      "observations": snapshot.get("backend_observations", []),
                      "limit": "Partial construction observations do not replace full CPU-profile qualification."}
    contradictions = []
    expected_fields = {"configured_simulation_device": "cpu", "environment_device": "cpu",
                       "simulation_device": "cpu", "gpu_dynamics": False, "broadphase": "MBP",
                       "suppress_readback": False, "native_view_valid": True}
    for index, backend in enumerate(snapshot.get("backend_observations", [])):
        if not isinstance(backend, dict):
            continue
        for name, expected in expected_fields.items():
            field = backend.get("fields", {}).get(name, {})
            if field.get("status") == "OBSERVED" and (type(field.get("raw_value")) is not type(expected)
                                                       or field.get("raw_value") != expected):
                contradictions.append(f"observation{index}.{name}")
    if contradictions:
        backend_result.update(status="CONTRADICTED", contract_result="CPU_CONFIGURATION_CONTRADICTED",
                              contradictions=contradictions)
    # No partial helper may promote itself to VERIFIED. Only the unchanged full
    # saved-file validator can qualify a separately supplied complete profile.
    full = snapshot.get("full_backend_result")
    if isinstance(full, dict):
        checked = check_cpu_backend(full, invocation=invocation, managed_anchor=snapshot.get("managed_anchor"))
        backend_result.update(contract_result=checked["BACKEND_DIAGNOSTIC_RESULT"],
                              status=checked["BACKEND_DIAGNOSTIC_RESULT"].removeprefix("CPU_CONFIGURATION_"),
                              full_contract_check=checked)
    complete = not issues
    return {"schema": "cpu-startup-parent-adjudication/v1", "diagnostic_mode": MODE,
            "status": "PASS" if complete else "FAIL",
            "OBSERVATION_RESULT": "OBSERVATION_COMPLETE" if complete else "OBSERVATION_BLOCKED",
            "GUARD_OBSERVATION": {"status": "OBSERVED" if replayed else "UNAVAILABLE", "evaluations": replayed},
            "BACKEND_OBSERVATION": backend_result,
            "INITIALIZATION_OUTCOME": snapshot.get("constructor_outcome", "UNAVAILABLE"),
            "SHUTDOWN_OBSERVATION": {"classification": shutdown, "last_phase": phases[-1] if phases else None,
                                     "close_return_recorded": bool(returned), "last_requested": entries[-1] if entries else None,
                                     "clock_limit": "Child, Kit and parent clocks are not aligned."},
            "DIAGNOSTIC_EVIDENCE": "COMPLETE" if complete else "INCOMPLETE",
            "PROCESS_COMPLETION": "PASS" if process_pass else "FAIL",
            "CONTAINMENT_RESULT": "PASS" if containment_pass and forced else "NOT_NEEDED" if containment_pass else "FAIL",
            "CANDIDATE_EXECUTIONS": snapshot.get("candidate_executions"),
            "ATTACHMENT_AUTHORING": snapshot.get("attachment_authoring"), "END_STATE_RESULT": "NOT_EVALUABLE",
            "issues": list(dict.fromkeys(issues)), "artifacts": artifacts,
            "authority": "No retry, candidate, contact qualification, cure, or Cycle45 activation."}
