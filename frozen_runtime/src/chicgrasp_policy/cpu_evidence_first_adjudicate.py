"""Independent saved-file checks for the prospective CPU terminal policy.

This module does not run an experiment or infer scientific validity from a
terminal marker. The caller independently replays the ordinary capture artifacts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .attachment_backend_adjudicate import CPU
from .cpu_startup_adjudicate import replay_guard


POLICY = "CPU_EVIDENCE_FIRST_CLOSE/v1"
ABSOLUTE_MS = 1_200_000
TERMINAL_MS = 60_000
DRAIN_MS = 5_000
SCHEMA = "cpu-evidence-first-terminal/v1"
EARLY_PROFILE_SCHEMA = "cpu-early-profile/v1"


def _integer(value):
    return type(value) is int and value >= 0


def _identity(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing_or_linked_evidence:" + str(path))
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _same_identity(actual, expected):
    return (isinstance(expected, dict) and type(expected.get("bytes")) is int
            and {key: expected.get(key) for key in ("bytes", "sha256")} == actual)


def _parse(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_JSON_key:" + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite_JSON:" + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def _records(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("missing_linked_or_oversized_terminal_stream")
    data = path.read_bytes()
    if not data or not data.endswith(b"\n"):
        raise ValueError("terminal_stream_empty_or_truncated")
    records = [_parse(line) for line in data.decode("utf-8").splitlines()]
    prior = -1
    for index, record in enumerate(records):
        if (not isinstance(record, dict) or type(record.get("sequence")) is not int
                or record["sequence"] != index or not _integer(record.get("monotonic_ns"))
                or record["monotonic_ns"] < prior):
            raise ValueError("terminal_sequence_or_clock_invalid")
        prior = record["monotonic_ns"]
    return records


def _early_equal(left, right):
    # JSON value identity distinguishes False, zero, None and absent members.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False)


def check_early_cpu_profile(root, result, invocation):
    """Authenticate the declared pre-reset observation, never qualify CPU backend.

    Declaration absence in both historical records preserves their interpretation.
    Field values are retained as observations, not evaluated as acceptance rules.
    """
    key = "early_cpu_profile_schema"
    if key not in result and key not in invocation:
        return {"declared": False, "status": "HISTORICAL_UNDECLARED", "issues": [], "reference": None}
    issues = []
    summary = result.get("early_cpu_profile")
    summary = summary if isinstance(summary, dict) else {}
    status = summary.get("status")
    answer = {"declared": True, "status": status, "issues": issues,
              "reference": summary.get("reference"), "backend_qualification": "NOT_EVALUATED"}

    def require(value, message):
        if not value:
            issues.append("early_cpu_profile:" + message)

    def read_object(path):
        identity = _identity(path)
        if identity["bytes"] > 8 * 1024 * 1024:
            raise ValueError("oversized_early_evidence:" + path.name)
        value = _parse(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("non_object_early_evidence:" + path.name)
        return value, identity

    require(result.get(key) == EARLY_PROFILE_SCHEMA and invocation.get(key) == EARLY_PROFILE_SCHEMA,
            "declaration_disagreement")
    require(result.get("backend_mode") == CPU and invocation.get("backend_mode") == CPU
            and result.get("observation_mode") is True and invocation.get("observation_mode") is True
            and result.get("diagnostic_mode") in (None, "")
            and invocation.get("diagnostic_mode") in (None, ""), "declared_scope_invalid")
    require(summary.get("schema") == EARLY_PROFILE_SCHEMA
            and summary.get("capture_phase") == "BEFORE_INITIAL_RESET"
            and status in ("NOT_REACHED", "CAPTURED", "READ_UNAVAILABLE", "PERSISTENCE_FAILED")
            and type(summary.get("reader_calls")) is int
            and summary["reader_calls"] == (0 if status == "NOT_REACHED" else 1)
            and "reference" in summary, "summary_invalid")
    initialization = result.get("initialization", {})
    if not isinstance(initialization, dict):
        initialization = {}
    require(_early_equal(initialization.get("early_cpu_profile"), summary), "initialization_summary_disagreement")
    if result.get("primary_errors"):
        failure = result.get("failure_diagnostic", {})
        require(isinstance(failure, dict) and _early_equal(failure.get("early_cpu_profile"), summary),
                "failure_summary_disagreement")
    try:
        root = Path(root).resolve()
        output = root / "capture" / "experiment"
        saved, result_identity = read_object(output / "result.json")
        require(_early_equal(saved, result), "saved_result_disagreement")
        manifest, _ = read_object(output / "artifact_manifest.json")
        files = manifest.get("files", {})
        require(manifest.get("schema") == "capture-artifacts/v1" and isinstance(files, dict), "manifest_invalid")
        if not isinstance(files, dict):
            files = {}
        require(_same_identity(result_identity, files.get("result.json")), "result_manifest_identity_mismatch")
        records = _records(root / "capture" / "structured" / "failure_phases.jsonl")
        phases = {"application_launch", "environment_construction", "early_cpu_profile", "initial_reset"}
        saved_phases = [{key: value for key, value in row.items() if key != "sequence"}
                        for row in records if row.get("phase") in phases]
        initial_phases = initialization.get("phase_events")
        require(isinstance(initial_phases, list) and all(isinstance(row, dict) for row in initial_phases),
                "initialization_phases_invalid")
        if isinstance(initial_phases, list) and all(isinstance(row, dict) for row in initial_phases):
            require(_early_equal([row for row in initial_phases if row.get("phase") in phases], saved_phases),
                    "initialization_phase_disagreement")
        early = [row for row in records if row.get("phase") == "early_cpu_profile"]
        resets = [row for row in records if row.get("phase") == "initial_reset"]
        if status == "NOT_REACHED":
            require(summary.get("reference") is None and not early and not resets
                    and initialization.get("initial_reset_returned") is False, "not_reached_after_boundary")
            require(not (output / "early_cpu_profile.json").exists()
                    and "early_cpu_profile.json" not in files, "not_reached_has_snapshot")
            failure = result.get("failure_diagnostic", {})
            error = failure.get("error") if isinstance(failure, dict) else None
            failed_before_reset = [row for row in records if row.get("event") == "failed"
                                  and row.get("phase") in ("application_launch", "environment_construction", "initialization")]
            require(isinstance(error, str) and bool(error) and error in result.get("primary_errors", [])
                    and any(row.get("error") == error for row in failed_before_reset), "not_reached_failure_unproven")
            require(all(_early_equal(row["early_cpu_profile"], summary)
                        for row in failed_before_reset if "early_cpu_profile" in row), "not_reached_failure_summary_disagreement")
            return answer
        requested = [row for row in early if row.get("event") == "requested"]
        persisted = [row for row in early if row.get("event") == "persisted"]
        failed = [row for row in early if row.get("event") == "persistence_failed"]
        require(len(requested) == 1 and len(early) == 2, "phase_count_invalid")
        construction = [row for row in records if row.get("phase") == "environment_construction"
                        and row.get("event") == "returned"]
        require(len(construction) == 1 and len(requested) == 1
                and construction[0]["sequence"] < requested[0]["sequence"], "construction_order_invalid")
        if status == "PERSISTENCE_FAILED":
            require(summary.get("reference") is None and isinstance(summary.get("error"), dict)
                    and len(failed) == 1 and not persisted
                    and _early_equal(failed[0].get("early_cpu_profile"), summary), "persistence_failure_invalid")
            require(False, "persistence_failed")
            return answer
        require(status in ("CAPTURED", "READ_UNAVAILABLE") and len(persisted) == 1 and not failed,
                "persisted_phase_missing")
        if len(requested) != 1 or len(persisted) != 1:
            return answer
        require(_early_equal(persisted[0].get("early_cpu_profile"), summary), "persisted_summary_disagreement")
        reference = summary.get("reference")
        if not isinstance(reference, dict):
            require(False, "reference_missing")
            return answer
        profile, profile_identity = read_object(output / "early_cpu_profile.json")
        require(reference.get("path") == "early_cpu_profile.json"
                and _same_identity(profile_identity, reference), "reference_identity_mismatch")
        require(_same_identity(profile_identity, files.get("early_cpu_profile.json")), "profile_manifest_identity_mismatch")
        require(profile.get("schema") == EARLY_PROFILE_SCHEMA
                and profile.get("episode_id") == result.get("episode_id")
                and type(profile.get("episode_id")) is str and bool(profile["episode_id"])
                and profile.get("capture_phase") == "BEFORE_INITIAL_RESET"
                and type(profile.get("reader_calls")) is int and profile["reader_calls"] == 1
                and profile.get("status") == status and "snapshot" in profile and "read_error" in profile,
                "envelope_invalid")
        clocks = [requested[0]["monotonic_ns"], profile.get("capture_started_ns"),
                  profile.get("capture_returned_ns"), profile.get("persistence_started_ns"),
                  reference.get("persistence_completed_ns"), persisted[0]["monotonic_ns"]]
        require(all(_integer(clock) for clock in clocks)
                and all(left <= right for left, right in zip(clocks, clocks[1:]))
                and _early_equal(reference.get("persistence_started_ns"), profile.get("persistence_started_ns")),
                "capture_persistence_clock_invalid")
        reset_requests = [row for row in resets if row.get("event") == "requested"]
        require(len(reset_requests) == 1 and requested[0]["sequence"] < persisted[0]["sequence"]
                < reset_requests[0]["sequence"] and persisted[0]["monotonic_ns"] <= reset_requests[0]["monotonic_ns"],
                "reset_before_persistence_or_missing")
        snapshot, error = profile.get("snapshot"), profile.get("read_error")
        if status == "READ_UNAVAILABLE":
            require(snapshot is None and isinstance(error, dict)
                    and isinstance(error.get("type"), str) and bool(error["type"])
                    and isinstance(error.get("message"), str), "read_error_invalid")
            if "error" in summary:
                require(_early_equal(summary["error"], error), "read_error_summary_disagreement")
        else:
            require(error is None and isinstance(snapshot, dict), "captured_snapshot_missing")
            if not isinstance(snapshot, dict):
                return answer
            require(snapshot.get("schema") == "cpu-startup-backend-observation/v1"
                    and snapshot.get("phase") == "BEFORE_INITIAL_RESET"
                    and snapshot.get("verification") == "CPU_CONFIGURATION_UNAVAILABLE", "snapshot_contract_invalid")
            stamp = snapshot.get("monotonic_ns")
            require(_integer(stamp) and _integer(profile.get("capture_started_ns"))
                    and _integer(profile.get("capture_returned_ns"))
                    and profile["capture_started_ns"] <= stamp <= profile["capture_returned_ns"], "snapshot_clock_invalid")
            fields = snapshot.get("fields")
            require(isinstance(fields, dict) and isinstance(snapshot.get("native_clock"), dict)
                    and isinstance(snapshot.get("access_errors"), list), "snapshot_groups_invalid")
            for field in fields.values() if isinstance(fields, dict) else []:
                require(isinstance(field, dict) and all(name in field for name in
                        ("source", "raw_value", "raw_type", "phase", "kind", "status", "error"))
                        and isinstance(field.get("source"), str) and field.get("phase") == "BEFORE_INITIAL_RESET"
                        and field.get("kind") in ("observed", "configured", "derived")
                        and field.get("status") in ("OBSERVED", "UNAVAILABLE", "ERROR"), "field_provenance_invalid")
    except Exception as exc:
        issues.append("early_cpu_profile:evidence_invalid:" + type(exc).__name__ + ":" + str(exc))
    return answer


def failed_reset_ruling(result, phase_records):
    """Replay a real reset rejection (or declared PRE rejection), never a window."""
    issues = []
    def require(value, message):
        if not value:
            issues.append(message)
    init = result.get("initialization", {})
    guard = result.get("failure_diagnostic", {}).get("jaw_state_diagnostic", {})
    early_pre = (result.get("early_cpu_profile_schema") == EARLY_PROFILE_SCHEMA
                 and guard.get("observation_phase") == "PRE_CANDIDATE")
    observation_phase = "PRE_CANDIDATE" if early_pre else "INITIAL_RESET"
    require(result.get("candidate_started") is False
            and result.get("attachment_authoring_attempted") is False
            and init.get("environment_returned") is True
            and init.get("initial_reset_returned") is early_pre,
            "failed_reset_scope_invalid")
    require(result.get("counts") == dict(completed_control_ticks=0,
            observed_substeps=0, completed_policy_actions=0), "failed_reset_counts_nonzero")
    require(guard.get("phase") == "REAL_OBSERVATION"
            and guard.get("observation_phase") == observation_phase, "failed_reset_real_phase_unproven")
    require(guard.get("policy_frame_emitted") is False, "failed_reset_invalid_frame_emitted")
    failure_phases = ("candidate", "pre_candidate") if early_pre else ("initial_reset",)
    failures = [record for record in phase_records if record.get("phase") in failure_phases
                and record.get("event") == "failed"]
    require(len(failures) == 1 and failures[0].get("jaw_state_diagnostic") == guard,
            "failed_reset_phase_guard_disagreement")
    require(early_pre or not any(record.get("phase") in ("candidate", "owner_binding", "pre_candidate")
                    and record.get("event") == "requested" for record in phase_records),
            "failed_reset_followed_by_candidate_work")
    replay = None
    try:
        replay = replay_guard({"schema": "cpu-startup-guard-observation/v1", "guard": guard})
        require(replay["verdict"] == "REJECT_PAIRED_BITS", "failed_reset_not_paired_bit_rejection")
    except Exception as exc:
        issues.append("failed_reset_guard_replay:" + type(exc).__name__ + ":" + str(exc))
    require(isinstance(guard.get("error"), str) and bool(guard["error"])
            and any(str(error).endswith(guard["error"]) for error in result.get("primary_errors", [])),
            "failed_reset_primary_error_not_preserved")
    return {"status": "RULING_NEEDED" if not issues else "BLOCKED", "issues": issues,
            "guard": replay, "retry_classification": "NO_RETRY_REAL_PRE_RULING_NEEDED" if early_pre
            else "NO_RETRY_REAL_RESET_RULING_NEEDED"}


def check_policy_guard_stream(output, result):
    """Reconstruct every saved real comparison and the sole dimension contract."""
    from .observation_metadata import INSTALLED_MANAGER_SHA256
    issues, real = [], []
    def require(value, message):
        if not value:
            issues.append(message)
    try:
        records = _records(Path(output) / "policy_guard_observations.jsonl")
        dimensions = [record for record in records if record.get("phase") == "DIMENSION_DISCOVERY"]
        require(len(dimensions) == 1 and records[0] == dimensions[0], "policy_metadata_boundary_count_or_order_invalid")
        for record in records:
            require(record.get("episode_id") == result.get("episode_id"), "policy_guard_episode_disagreement")
            if record.get("phase") == "DIMENSION_DISCOVERY":
                boundary = record.get("boundary", {})
                require(record.get("schema") == "jaw-state-dimension-discovery/v1"
                        and record.get("observation_phase") == "CONSTRUCTION"
                        and record.get("shape") == [1, 2] and record.get("dtype") in ("torch.uint8", "uint8")
                        and record.get("device") == "cpu" and record.get("physical_jaw_positions_read") is False
                        and record.get("paired_state_validity_applied") is False
                        and record.get("policy_frame_emitted") is False and "positions" not in record
                        and "joint_bits" not in record and boundary.get("sha256") == INSTALLED_MANAGER_SHA256
                        and boundary.get("callable") == "ObservationManager._prepare_terms"
                        and boundary.get("statement") == "obs_dims = tuple(term_cfg.func(self._env, **term_cfg.params).shape)"
                        and boundary.get("consumes") == "shape only; no returned sizing data retained",
                        "policy_metadata_contract_invalid")
            else:
                require(record.get("phase") == "REAL_OBSERVATION"
                        and record.get("schema") in ("jaw-state-observation/v1", "jaw-state-failure/v1"),
                        "policy_guard_unknown_phase_or_schema")
                replay = replay_guard({"schema": "cpu-startup-guard-observation/v1", "guard": record})
                real.append(dict(sequence=record["sequence"], observation_phase=record.get("observation_phase"),
                                 verdict=replay["verdict"]))
        if result.get("candidate_started") is True:
            require(bool(real) and all(record["verdict"] == "PASS" for record in real), "policy_window_contains_rejected_real_frame")
            phase_indices = {phase: next((record["sequence"] for record in real if record["observation_phase"] == phase), None)
                             for phase in ("INITIAL_RESET", "PRE_CANDIDATE", "CANDIDATE")}
            require(all(type(value) is int for value in phase_indices.values())
                    and phase_indices["INITIAL_RESET"] < phase_indices["PRE_CANDIDATE"] < phase_indices["CANDIDATE"],
                    "policy_real_phase_coverage_missing_or_reordered")
        else:
            early_pre = (result.get("early_cpu_profile_schema") == EARLY_PROFILE_SCHEMA
                         and result.get("failure_diagnostic", {}).get("jaw_state_diagnostic", {}).get(
                             "observation_phase") == "PRE_CANDIDATE")
            if early_pre:
                require(len(real) >= 2 and real[0]["observation_phase"] == "INITIAL_RESET"
                        and real[-1]["observation_phase"] == "PRE_CANDIDATE"
                        and real[-1]["verdict"] == "REJECT_PAIRED_BITS"
                        and all(row["observation_phase"] in ("INITIAL_RESET", "PRE_CANDIDATE")
                                and row["verdict"] == "PASS" for row in real[:-1]),
                        "policy_failed_pre_rejection_missing")
            else:
                require(bool(real) and real[-1]["observation_phase"] == "INITIAL_RESET"
                        and real[-1]["verdict"] == "REJECT_PAIRED_BITS", "policy_failed_reset_rejection_missing")
        return {"issues": issues, "dimension_records": len(dimensions), "real_records": len(real),
                "rejected_real_records": sum(record["verdict"] != "PASS" for record in real)}
    except Exception as exc:
        issues.append("policy_guard_stream_invalid:" + type(exc).__name__ + ":" + str(exc))
        return {"issues": issues, "dimension_records": None, "real_records": len(real)}


def check_terminal_policy(root, result, invocation, process):
    """Check finalized bytes, structured receipts and one original parent deadline."""
    root = Path(root).resolve()
    capture, output = root / "capture", root / "capture" / "experiment"
    issues, evidence_issues, records = [], [], []
    finalization, trigger = {}, None
    def require(value, message, *, evidence=False):
        if not value:
            (evidence_issues if evidence else issues).append(message)
    require(invocation.get("terminal_policy") == POLICY and result.get("terminal_policy") == POLICY
            and process.get("terminal_policy") == POLICY, "cpu_terminal_policy_disagreement")
    require(invocation.get("backend_mode") == CPU and result.get("backend_mode") == CPU
            and invocation.get("observation_mode") is True and result.get("observation_mode") is True
            and invocation.get("diagnostic_mode") in (None, ""), "cpu_terminal_profile_invalid")
    require(invocation.get("campaign") == "cpu-phase-observation/v1"
            and invocation.get("timeout_ms") == ABSOLUTE_MS
            and invocation.get("terminal_timeout_ms") == TERMINAL_MS, "cpu_terminal_invocation_budget_invalid")
    argv = invocation.get("arguments", [])
    require(isinstance(argv, list) and argv.count("--terminal-policy") == 1
            and argv.index("--terminal-policy") + 1 < len(argv)
            and argv[argv.index("--terminal-policy") + 1] == POLICY
            and not any(isinstance(arg, str) and arg.startswith("--terminal-policy=") for arg in argv),
            "cpu_terminal_invocation_not_explicit")
    try:
        terminal_path = capture / "structured" / "terminal_phases.jsonl"
        records = _records(terminal_path)
        require(len(records) >= 2, "cpu_terminal_required_records_missing", evidence=True)
        finalization = records[0]
        require(finalization.get("schema") == SCHEMA and finalization.get("terminal_policy") == POLICY
                and finalization.get("phase") == "evidence_finalization"
                and finalization.get("event") == "returned"
                and finalization.get("evidence_finalized") is True
                and finalization.get("writers_completed") is True
                and finalization.get("required_measurements_pending") is False,
                "cpu_terminal_finalization_claim_invalid", evidence=True)
        writers = finalization.get("writer_statuses")
        require(isinstance(writers, dict) and set(writers) == {"application_artifacts", "rgb", "compact_contact",
                    "episode", "policy_guard", "required_journal"}
                and all(value in ("COMPLETE", "NOT_APPLICABLE") for value in writers.values())
                and all(writers.get(key) == "COMPLETE" for key in
                        ("application_artifacts", "policy_guard", "required_journal")),
                "cpu_terminal_writer_statuses_invalid", evidence=True)
        claim = finalization.get("artifacts", {}).get("application_manifest", {})
        manifest_path = output / "artifact_manifest.json"
        require(claim.get("relative_path") == "../experiment/artifact_manifest.json"
                and _same_identity(_identity(manifest_path), claim),
                "cpu_terminal_finalized_manifest_identity_mismatch", evidence=True)
        manifest = _parse(manifest_path.read_text(encoding="utf-8"))
        require(manifest.get("schema") == "capture-artifacts/v1"
                and isinstance(manifest.get("files"), dict), "cpu_terminal_manifest_invalid", evidence=True)
        files = manifest.get("files", {})
        require(set(files) == {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
                - {"artifact_manifest.json"}, "cpu_terminal_manifest_membership_mismatch", evidence=True)
        for relative, expected in files.items():
            target = (output / relative).resolve()
            require(target.is_relative_to(output) and target != output
                    and _same_identity(_identity(target), expected),
                    "cpu_terminal_artifact_identity_mismatch:" + relative, evidence=True)
        scope = finalization.get("evidence_scope")
        require(scope in ("FULL_CANDIDATE", "839_1", "FAILED_START")
                and isinstance(finalization.get("stop_reason"), str) and bool(finalization["stop_reason"])
                and finalization.get("observation_window_complete") is (scope in ("FULL_CANDIDATE", "839_1")),
                "cpu_terminal_scope_invalid", evidence=True)
        required = {"result.json", "measurements.json", "candidate_result.json", "policy_guard_observations.jsonl"}
        if scope in ("FULL_CANDIDATE", "839_1"):
            required |= {"pre_candidate.json", "observation_result.json", "observation_configuration.json",
                         "rgb/manifest.json", "contact_compact/manifest.json", "owner_before_cleanup.json",
                         "contact_transport_before_cleanup.json", "contact_transport_final.json"}
            require(isinstance(writers, dict) and writers.get("rgb") == "COMPLETE"
                    and writers.get("compact_contact") == "COMPLETE", "cpu_terminal_observation_writers_incomplete", evidence=True)
        require(required <= set(files), "cpu_terminal_required_artifacts_missing", evidence=True)
        counts = result.get("counts", {})
        if scope == "FULL_CANDIDATE":
            require(counts == dict(completed_control_ticks=1200, observed_substeps=2400, completed_policy_actions=200)
                    and any(name.startswith("episode/") for name in files)
                    and writers.get("episode") == "COMPLETE"
                    and finalization.get("stop_reason") == "candidate_complete",
                    "cpu_terminal_full_scope_disagreement", evidence=True)
        elif scope == "839_1":
            require(counts == dict(completed_control_ticks=839, observed_substeps=1680, completed_policy_actions=139),
                    "cpu_terminal_839_scope_disagreement", evidence=True)
        elif scope == "FAILED_START":
            require(counts == dict(completed_control_ticks=0, observed_substeps=0, completed_policy_actions=0)
                    and result.get("candidate_started") is False, "cpu_terminal_failed_start_scope_disagreement", evidence=True)
        if scope in ("839_1", "FAILED_START"):
            primary = result.get("primary_errors", [])
            require(isinstance(primary, list) and bool(primary)
                    and finalization.get("stop_reason") == primary[0], "cpu_terminal_stop_reason_disagreement", evidence=True)
        require(result.get("recording_errors") == [] and result.get("cleanup_errors") == []
                and result.get("evidence_integrity") == "COMPLETE", "cpu_terminal_required_writer_or_cleanup_failure", evidence=True)
        close = records[1]
        require(close.get("schema") == SCHEMA and close.get("terminal_policy") == POLICY
                and close.get("phase") == "terminal_close" and close.get("event") == "requested"
                and type(close.get("finalization_sequence")) is int and close["finalization_sequence"] == 0,
                "cpu_terminal_close_reference_invalid", evidence=True)
        require(not any(record.get("event") == "failed" for record in records[2:]),
                "cpu_terminal_close_error", evidence=True)
        require(all(record.get("schema") == SCHEMA and record.get("terminal_policy") == POLICY
                    and record.get("phase") in ("environment_close", "application_close")
                    and record.get("event") in ("returned", "failed") for record in records[2:])
                and len({record.get("phase") for record in records[2:]}) == len(records[2:]),
                "cpu_terminal_optional_bracket_invalid", evidence=True)
        for record in records[2:]:
            if record.get("phase") == "environment_close" and record.get("event") == "returned":
                cleanup = record.get("cleanup", {})
                require(cleanup.get("errors") == []
                        and (cleanup.get("environment_handle_available") is False
                             or cleanup.get("managed_environment_closed") is True),
                        "cpu_terminal_environment_close_not_confirmed", evidence=True)
        receipts = process.get("terminal_phase_observations")
        require(isinstance(receipts, list) and len(receipts) == len(records), "cpu_terminal_parent_receipts_incomplete")
        for index, receipt in enumerate(receipts or []):
            require(index < len(records) and isinstance(receipt, dict)
                    and receipt.get("record") == records[index] and receipt.get("seq") == index
                    and receipt.get("child_monotonic_ns") == records[index]["monotonic_ns"]
                    and _integer(receipt.get("parent_observed_ms"))
                    and receipt.get("source") == "terminal_phases.jsonl", "cpu_terminal_parent_receipt_disagreement")
        require(all(_integer(a.get("parent_observed_ms")) and _integer(b.get("parent_observed_ms"))
                    and a["parent_observed_ms"] <= b["parent_observed_ms"]
                    for a, b in zip(receipts or [], (receipts or [])[1:])),
                "cpu_terminal_parent_receipts_reordered")
        require(process.get("evidence_finalization_observation") == receipts[0] if receipts else False,
                "cpu_terminal_finalization_receipt_disagreement")
        trigger = process.get("terminal_trigger_observation")
        if trigger is not None:
            require(isinstance(trigger, dict) and len(receipts or []) >= 2
                    and all(trigger.get(key) == value for key, value in receipts[1].items())
                    and trigger.get("eligibility_claim_checked") is True, "cpu_terminal_trigger_disagreement")
        require(process.get("terminal_phase_complete_records") == len(records)
                and process.get("terminal_phase_complete_bytes") == terminal_path.stat().st_size,
                "cpu_terminal_parent_stream_coverage_invalid")
        expected = process.get("completed_terminal_phases")
        require(_same_identity(_identity(terminal_path), expected)
                and Path(expected.get("path", "")).resolve() == terminal_path,
                "cpu_terminal_parent_stream_identity_mismatch")
    except Exception as exc:
        evidence_issues.append("cpu_terminal_evidence_unavailable:" + type(exc).__name__ + ":" + str(exc))

    for key, expected in (("absolute_deadline_ms", ABSOLUTE_MS), ("timeout_ms", ABSOLUTE_MS),
            ("parent_launch_boundary_ms", 0), ("terminal_timeout_ms", TERMINAL_MS),
            ("termination_and_drain_cap_ms", DRAIN_MS), ("job_assigned_before_resume", True),
            ("root_resumed", True), ("root_exit_verified", True), ("tree_empty_verified", True),
            ("capture_complete", True), ("job_disposed_after_observation", True),
            ("job_members_before_disposal", 0), ("containment_budget_overrun", False)):
        require(type(process.get(key)) is type(expected) and process[key] == expected,
                "cpu_terminal_parent_contract:" + key)
    require(process.get("containment_errors") == [] and process.get("errors") == [], "cpu_terminal_parent_errors")
    effective = ABSOLUTE_MS
    if isinstance(trigger, dict) and trigger.get("deadline_enforced") is True:
        observed = trigger.get("parent_observed_ms")
        require(_integer(observed) and trigger.get("read_phase") == "live_poll"
                and (trigger.get("root_exit_already_observed") is False or
                     (trigger.get("root_exit_already_observed") is True and process.get("exit_code") == 0
                      and _integer(process.get("root_exit_observed_ms"))
                      and process["root_exit_observed_ms"] <= observed
                      and _integer(process.get("job_empty_observed_ms"))
                      and process["job_empty_observed_ms"] >= observed)), "cpu_terminal_trigger_not_live")
        if _integer(observed):
            effective = min(ABSOLUTE_MS, observed + TERMINAL_MS)
        require(process.get("terminal_deadline_ms") == effective, "cpu_terminal_deadline_renewed")
    else:
        require(process.get("terminal_deadline_ms") is None, "cpu_terminal_late_marker_deadline")
    require(process.get("effective_deadline_ms") == effective, "cpu_terminal_effective_deadline_invalid")
    force_fields = ("forced_termination", "timed_out", "parent_termination_requested",
                    "startup_containment_used", "kill_on_job_close_used", "kill_on_job_close_possible")
    require(all(type(process.get(key)) is bool for key in (*force_fields, "autonomous_tree_exit")),
            "cpu_terminal_forced_status_fields_missing")
    forced = any(process.get(key) is True for key in force_fields)
    times = ("root_exit_observed_ms", "job_empty_observed_ms", "drain_completed_observed_ms",
             "job_disposed_ms", "stream_files_finalized_ms")
    for key in times:
        require(_integer(process.get(key)), "cpu_terminal_parent_clock_missing:" + key)
    if all(_integer(process.get(key)) for key in times):
        require(process["job_empty_observed_ms"] <= process["job_disposed_ms"]
                and process["drain_completed_observed_ms"] <= process["stream_files_finalized_ms"],
                "cpu_terminal_parent_finalization_order_invalid")
    if forced:
        require(isinstance(trigger, dict) and trigger.get("deadline_enforced") is True,
                "cpu_terminal_forced_before_finalization")
        termination, cap = process.get("termination_requested_ms"), process.get("containment_deadline_ms")
        require(_integer(termination) and effective <= termination <= effective + DRAIN_MS,
                "cpu_terminal_termination_time_invalid")
        require(_integer(cap) and cap <= effective + DRAIN_MS, "cpu_terminal_containment_budget_renewed")
        require(all(_integer(process.get(key)) and _integer(cap) and process[key] <= cap for key in times),
                "cpu_terminal_containment_or_drain_overrun")
        require(process.get("termination_reason") == ("cpu_terminal_deadline" if effective < ABSOLUTE_MS else "absolute_deadline"),
                "cpu_terminal_unrelated_termination")
        require(process.get("forced_termination") is True and process.get("parent_termination_requested") is True
                and process.get("autonomous_tree_exit") is False and type(process.get("exit_code")) is int,
                "cpu_terminal_forced_status_not_retained")
        require(not (_integer(process.get("root_exit_observed_ms")) and _integer(termination)
                     and process["root_exit_observed_ms"] < termination and process.get("exit_code") != 0),
                "cpu_terminal_unrelated_root_crash_before_containment")
    else:
        require(process.get("autonomous_tree_exit") is True and type(process.get("exit_code")) is int
                and process["exit_code"] == 0, "cpu_terminal_unrelated_crash_or_exit")
        require(all(_integer(process.get(key)) and process[key] <= effective for key in times[:3]),
                "cpu_terminal_autonomous_deadline_exceeded")
    valid = not issues and not evidence_issues
    return {"terminal_policy": POLICY, "PROCESS_COMPLETION": "FORCED" if not issues and forced else "PASS" if not issues else "FAIL",
            "CONTAINMENT_RESULT": "PASS" if not issues and forced else "NOT_NEEDED" if not issues else "FAIL",
            "autonomous_exit": not forced and not issues, "actual_os_exit_code": process.get("exit_code"),
            "terminal_exception_eligible": valid, "evidence_scope": finalization.get("evidence_scope"),
            "observation_window_complete": finalization.get("observation_window_complete") is True,
            "stop_reason": finalization.get("stop_reason"), "issues": issues, "evidence_issues": evidence_issues,
            "effective_deadline_ms": effective}
