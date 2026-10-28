"""Offline run-scope and daily harness accounting. Never starts a process.

Invocation notes are not reservations. Unknown prior start evidence blocks a
new harness invocation rather than being relabelled as a pre-Kit failure.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

HARNESS = "HARNESS"
EVIDENCE = "EVIDENCE"
CAMPAIGN = "geometric-admission/v1"
DAILY_KIT_LIMIT = 100
ABSOLUTE_MS = 1_200_000
CONTAINMENT_MS = 5_000


def declared_absolute_ms(invocation):
    """Only an explicit Stage0 GUI HARNESS declaration can alter the default."""
    if "stage0_until_quit" in invocation:
        if (invocation["stage0_until_quit"] is not True
                or "stage0_tuning_minutes" in invocation
                or "timeout_ms" not in invocation or invocation["timeout_ms"] is not None
                or not _stage0_gui_budget_scope(invocation)):
            raise ValueError("stage0_until_quit_scope_invalid")
        return None
    if "stage0_tuning_minutes" not in invocation:
        return ABSOLUTE_MS
    if (type(invocation["stage0_tuning_minutes"]) is not int
            or invocation["stage0_tuning_minutes"] != 40
            or not _stage0_gui_budget_scope(invocation)):
        raise ValueError("stage0_tuning_budget_scope_invalid")
    return 2_400_000


def _stage0_gui_budget_scope(invocation):
    return (invocation.get("kind") == "attachment_capture"
            and invocation.get("stage0_mode") == "TUNE_GUI"
            and invocation.get("geometric_mode") is True
            and invocation.get("run_type") == HARNESS
            and invocation.get("campaign") == CAMPAIGN
            and invocation.get("execution_approved") is False
            and isinstance(invocation.get("purpose"), str) and bool(invocation["purpose"].strip())
            and not any(key in invocation for key in ("evidence_authority", "observation_mode",
                "contact_transport", "backend_mode", "diagnostic_mode", "terminal_policy")))


def _time(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timezone_aware_local_timestamp_required")
    return parsed


def _identity(path):
    data = Path(path).read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _random_xy_smoke_accounting_note(note):
    """The named EVIDENCE smoke shares start accounting, never HARNESS acceptance."""
    if (note.get("schema") != "harness-invocation-note/v1"
            or note.get("run_type") != EVIDENCE
            or note.get("task") != "CHICGRASP_RANDOM_XY_RECORDER_SMOKE"
            or note.get("translated_point") != "R000001"
            or note.get("accounting_only") is not True
            or not isinstance(note.get("attempt_directory"), str)):
        return False
    try:
        attempt = Path(note["attempt_directory"])
        if not attempt.is_absolute():
            return False
        attempt = attempt.resolve()
        root = Path("C:/robotics/ChicGrasp_gate6_stage3/outputs/"
                    "random_xy_recorder_smoke/attempt001").resolve()
        # One initial start plus the owner's three pre-action repair cycles.
        return attempt.parent == root and attempt.name in {
            "native001", "native002", "native003", "native004"}
    except (OSError, RuntimeError, ValueError):
        return False


def _random_xy_collection_accounting_note(note):
    """Only the released collection's EVIDENCE notes share daily accounting."""
    point = note.get("translated_point")
    if (note.get("schema") != "harness-invocation-note/v1"
            or note.get("run_type") != EVIDENCE
            or note.get("task") != "CHICGRASP_RANDOM_XY_50_COLLECTION"
            or not isinstance(point, str)
            or re.fullmatch(r"R000(?:00[2-9]|0[1-9][0-9]|10[01])", point) is None
            or note.get("accounting_only") is not True
            or not isinstance(note.get("attempt_directory"), str)):
        return False
    try:
        attempt = Path(note["attempt_directory"])
        if not attempt.is_absolute():
            return False
        attempt = attempt.resolve()
        root = Path("C:/robotics/ChicGrasp_gate6_stage3/outputs/"
                    "random_xy_50_collection/attempt001").resolve()
        return (attempt.parent == root / point and attempt.name in {
            "native001", "native002", "native003", "native004"})
    except (OSError, RuntimeError, ValueError):
        return False


def _scripted_yaw_accounting_note(note):
    """Only this finite reference probe joins the existing actual-start ledger."""
    point=note.get('translated_point','')
    if (note.get('schema')!='harness-invocation-note/v1' or note.get('run_type')!=EVIDENCE
        or note.get('task')!='CHICGRASP_SCALED50_SCRIPTED_YAW_PROBE'
        or note.get('accounting_only') is not True
        or not isinstance(point,str) or re.fullmatch(r'YH0[57]_(?:Z00|[PM](?:05|10|20))',point) is None):
        return False
    try:
        attempt=Path(note['attempt_directory']).resolve()
        root=Path('C:/robotics/ChicGrasp_gate6_stage3/outputs/scaled50_scripted_yaw_probe/attempt001').resolve()
        return attempt.parent==root/point and attempt.name in {'native001','native002','native003','native004'}
    except (KeyError,TypeError,ValueError,OSError):
        return False


def _hard_xy_anchor_accounting_note(note):
    """Only the three named reference anchors join actual-start accounting."""
    point=note.get('translated_point')
    if (note.get('schema')!='harness-invocation-note/v1' or note.get('run_type')!=EVIDENCE
        or note.get('task')!='CHICGRASP_SCALED50_HARD_XY_SCRIPTED_ANCHORS'
        or note.get('accounting_only') is not True or point not in ('H08','H09','H04')):
        return False
    try:
        attempt=Path(note['attempt_directory'])
        if not attempt.is_absolute():
            return False
        attempt=attempt.resolve()
        root=Path('C:/robotics/ChicGrasp_gate6_stage3/outputs/scaled50_hard_xy_scripted_anchors/attempt001').resolve()
        return attempt.parent==root/point and attempt.name in {'native001','native002','native003','native004'}
    except (KeyError,TypeError,ValueError,OSError):
        return False


def harness_budget(notes, now):
    """Count observed starts on their local date; no synthetic credit for failure.

    A non-final note is an unresolved invocation, not a reserved start. A launch
    whose observation interval straddles midnight cannot establish its local
    start date, so it requires a corrected evidence note before another launch.
    """
    local_now = _time(now)
    count, issues, seen = 0, [], set()
    for note in notes:
        identifier = note.get("invocation_id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            issues.append("harness_note_identity_invalid")
        seen.add(identifier)
        if (note.get("schema") != "harness-invocation-note/v1"
                or (note.get("run_type") != HARNESS
                    and not _random_xy_smoke_accounting_note(note)
                    and not _random_xy_collection_accounting_note(note)
                    and not _scripted_yaw_accounting_note(note)
                    and not _hard_xy_anchor_accounting_note(note))):
            issues.append("harness_note_scope_invalid")
        try:
            _time(note["invocation_local_time"])
            commit = note["executed_commit"]
            if (not isinstance(note.get("purpose"), str) or not note["purpose"].strip()
                    or not isinstance(commit, str) or len(commit) != 40
                    or any(char not in "0123456789abcdef" for char in commit)
                    or not isinstance(note.get("outcome"), str) or not note["outcome"]):
                raise ValueError("required_note_fields_invalid")
        except (KeyError, TypeError, ValueError):
            issues.append("harness_note_time_purpose_outcome_commit_invalid")
        if note.get("outcome") in (None, "IN_PROGRESS"):
            issues.append("harness_invocation_unresolved")
        starts = note.get("actual_kit_starts")
        if type(starts) is not int or starts not in (0, 1):
            issues.append("harness_kit_start_unknown")
            continue
        if starts == 0:
            if note.get("kit_start_basis") != "COMPLETE_PRE_KIT_FAILURE":
                issues.append("harness_pre_kit_failure_unproven")
            continue
        try:
            first = _time(note["kit_start_interval_local"][0])
            last = _time(note["kit_start_interval_local"][1])
            if last < first or first.date() != last.date():
                raise ValueError("midnight_or_order_uncertainty")
            if first.date() == local_now.date():
                count += 1
        except (KeyError, TypeError, ValueError, IndexError):
            issues.append("harness_kit_start_local_day_ambiguous")
    return {"local_date": local_now.date().isoformat(), "actual_kit_starts": count,
            "limit": DAILY_KIT_LIMIT, "remaining": max(0, DAILY_KIT_LIMIT-count),
            "allowed": not issues and count < DAILY_KIT_LIMIT,
            "issues": sorted(set(issues)), "reservation_required": False}


def check_run_type(invocation, result, process):
    """All three independently persisted scopes must agree; HARNESS never accepts."""
    scopes = [item.get("run_type") for item in (invocation, result, process)]
    issues = []
    if any(value not in (HARNESS, EVIDENCE) for value in scopes) or len(set(scopes)) != 1:
        issues.append("run_type_missing_or_disagreement")
    if any(value == HARNESS for value in scopes):
        issues.append("harness_output_not_acceptance_evidence")
    for item in (invocation, result, process):
        if item.get("geometric_mode") is not True:
            issues.append("geometric_mode_not_bound")
    try:
        absolute_ms = declared_absolute_ms(invocation)
    except ValueError as exc:
        issues.append(str(exc))
        absolute_ms = ABSOLUTE_MS
    timeout_matches = ("timeout_ms" in invocation and
        (invocation["timeout_ms"] is None if absolute_ms is None else
         type(invocation["timeout_ms"]) is int and invocation["timeout_ms"] == absolute_ms))
    if (not timeout_matches
            or type(process.get("termination_and_drain_cap_ms")) is not int
            or process["termination_and_drain_cap_ms"] != CONTAINMENT_MS):
        issues.append("run_process_budget_disagreement")
    if absolute_ms != ABSOLUTE_MS:
        if (any(item.get("stage0_mode") != "TUNE_GUI" or item.get("run_type") != HARNESS
                for item in (result, process)) or any(key not in process
                    or (process[key] is not None if absolute_ms is None else
                        type(process[key]) is not int or process[key] != absolute_ms)
                    for key in ("timeout_ms", "absolute_deadline_ms", "effective_deadline_ms"))):
            issues.append("stage0_tuning_parent_budget_disagreement")
    if ((absolute_ms is None and process.get("stage0_until_quit") is not True)
            or (absolute_ms is not None and "stage0_until_quit" in process)):
        issues.append("stage0_until_quit_parent_disagreement")
    eligible = not issues and scopes[0] == EVIDENCE
    return {"run_type": scopes[0], "acceptance_eligible": eligible,
            "issues": sorted(set(issues)), "native_launch_authority": "NONE_CONFERRED"}


def verify_evidence_authority(invocation, attempt_root):
    """Authenticate the future grant/reservation inputs, not their human meaning."""
    issues = []
    try:
        authority = invocation["evidence_authority"]
        grant = authority["grant"]
        reservation = authority["reservation"]
        if _identity(grant["path"]) != {key: grant[key] for key in ("bytes", "sha256")}:
            issues.append("evidence_grant_identity_mismatch")
        if _identity(reservation["path"]) != {key: reservation[key] for key in ("bytes", "sha256")}:
            issues.append("evidence_reservation_identity_mismatch")
        saved = json.loads(Path(reservation["path"]).read_text(encoding="utf-8-sig"))
        if (saved.get("schema") != "geometric-evidence-reservation/v1"
                or saved.get("run_type") != EVIDENCE
                or saved.get("grant_sha256") != grant["sha256"]
                or saved.get("executed_commit") != invocation.get("executed_commit")
                or saved.get("executed_tag") != invocation.get("executed_tag")
                or not saved.get("reservation_id")
                or Path(saved["attempt_directory"]).resolve() != Path(attempt_root).resolve()):
            issues.append("evidence_reservation_contract_invalid")
        if (invocation.get("execution_approved") is not True
                or invocation.get("run_type") != EVIDENCE
                or not isinstance(invocation.get("executed_commit"), str)
                or len(invocation["executed_commit"]) != 40
                or not invocation.get("executed_tag")):
            issues.append("evidence_commit_tag_or_approval_missing")
    except (KeyError, TypeError, ValueError, OSError):
        issues.append("evidence_authority_unavailable")
    return {"valid": not issues, "issues": issues}


def complete_harness_note(note, process, *, outcome):
    """Derive start accounting from parent observations, not requested launches."""
    completed = dict(note, outcome=outcome, actual_kit_starts=None,
                     kit_start_basis="UNRESOLVED", kit_start_interval_local=None)
    completed.pop("kit_start_fallback_evidence", None)
    completed.pop("kit_start_fallback_issues", None)
    observed = process.get("kit_start_observation")
    if isinstance(observed, dict) and observed.get("source") == "complete Kit extension startup line":
        completed.update(actual_kit_starts=1, kit_start_basis=observed["source"],
                         kit_start_interval_local=[process["child_start_local"], observed["observed_local"]])
    elif process.get("started") is False:
        completed.update(actual_kit_starts=0, kit_start_basis="COMPLETE_PRE_KIT_FAILURE")
    elif (process.get("capture_complete") is True and process.get("root_exit_verified") is True
            and process.get("tree_empty_verified") is True
            and process.get("kit_start_scan_complete") is True
            and process.get("pre_kit_failure_verified") is True):
        completed.update(actual_kit_starts=0, kit_start_basis="COMPLETE_PRE_KIT_FAILURE")
    return completed


def complete_harness_note_from_files(note, process_path, *, outcome, note_path):
    """Use positive saved startup evidence only when the parent report is absent.

    Ctrl+C may interrupt the parent before its report is persisted. A complete,
    timestamped Kit line can still prove a start on one local day. It cannot
    prove root/tree exit, complete capture, or a successful application outcome.
    Existing parent reports retain the original accounting semantics.
    """
    process_path = Path(process_path)
    completed = complete_harness_note(note, {}, outcome=outcome)
    try:
        attempt = Path(note["attempt_directory"]).resolve()
        capture = attempt / "capture"
        if process_path.resolve() != capture / "process_result.json":
            raise ValueError("harness_fallback_process_path_mismatch")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        completed["kit_start_fallback_issues"] = [f"{type(exc).__name__}:{exc}"]
        return completed
    if process_path.is_file():
        process = json.loads(process_path.read_text(encoding="utf-8-sig"))
        return complete_harness_note(note, process, outcome=outcome)
    completed["kit_start_fallback_issues"] = []
    try:
        if process_path.exists():
            raise ValueError("harness_fallback_parent_report_not_regular_file")
        if note.get("schema") != "harness-invocation-note/v1" or note.get("run_type") != HARNESS:
            raise ValueError("harness_fallback_note_scope_invalid")
        invocation_path = attempt / "invocation.json"
        invocation_bytes = invocation_path.read_bytes()
        invocation = json.loads(invocation_bytes.decode("utf-8-sig"))
        if (not isinstance(invocation, dict)
                or invocation.get("kind") != "attachment_capture"
                or invocation.get("run_type") != HARNESS
                or invocation.get("geometric_mode") is not True
                or invocation.get("campaign") != CAMPAIGN
                or not note.get("invocation_id")
                or invocation.get("harness_invocation_id") != note["invocation_id"]
                or invocation.get("executed_commit") != note.get("executed_commit")
                or invocation.get("purpose") != note.get("purpose")
                or Path(invocation["harness_note_path"]).resolve() != Path(note_path).resolve()
                or Path(invocation["output_directory"]).resolve() != capture):
            raise ValueError("harness_fallback_invocation_binding_invalid")
        first = _time(note["invocation_local_time"])
        raw_path = capture / "stdout.bin"
        before = raw_path.stat()
        digest, size = hashlib.sha256(), 0
        with raw_path.open("rb") as stream:
            prefix = stream.read(min(1_048_576, before.st_size))
            digest.update(prefix)
            size += len(prefix)
            while size < before.st_size:
                chunk = stream.read(min(1_048_576, before.st_size - size))
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
        after = raw_path.stat()
        if (size != before.st_size or size != after.st_size or before.st_mtime_ns != after.st_mtime_ns):
            raise ValueError("harness_fallback_log_changed_during_read")
        # Match the parent's bounded scanner and exact line grammar. Unlike the
        # live parent, this fallback has no receipt clock, so a UTC prefix is
        # mandatory. File modification time is never substituted for start time.
        complete = prefix[:prefix.rfind(b"\n") + 1].decode("utf-8")
        pattern = re.compile(r"^(?:([0-9T:.-]+Z) \[[0-9,]+ms\] \[Info\] "
            r"\[omni\.kit\.app\.plugin\] )?\s*\[[0-9.]+s\]\s*"
            r"\[ext:\s+[^\]]+\]\s+startup\r?$")
        found = None
        for line_number, line in enumerate(complete.split("\n")[:-1], 1):
            match = pattern.fullmatch(line)
            if match and match.group(1):
                found = line_number, line.rstrip("\r"), match.group(1)
                break
        if found is None:
            raise ValueError("harness_fallback_timestamped_startup_line_unavailable")
        line_number, line, timestamp = found
        last = _time(timestamp).astimezone(first.tzinfo)
        if last < first or first.date() != last.date():
            raise ValueError("harness_fallback_start_day_or_order_ambiguous")
        completed.update(actual_kit_starts=1,
            kit_start_basis="complete timestamped saved Kit extension startup line",
            kit_start_interval_local=[note["invocation_local_time"], last.isoformat()],
            kit_start_fallback_evidence={
                "parent_report": str(process_path.resolve()), "parent_report_state": "MISSING",
                "invocation": {"path": str(invocation_path), "bytes": len(invocation_bytes),
                    "sha256": hashlib.sha256(invocation_bytes).hexdigest()},
                "stdout": {"path": str(raw_path), "bytes": size, "sha256": digest.hexdigest()},
                "scanned_prefix_bytes": len(prefix), "line_number": line_number, "line": line,
                "timestamp_source": "UTC timestamp on complete saved Kit startup line",
                "timestamp_utc": timestamp, "interval_start_source": "invocation note local time",
                "process_completion": "UNVERIFIED", "capture_completion": "UNVERIFIED"})
    except (KeyError, TypeError, ValueError, OSError) as exc:
        completed["kit_start_fallback_issues"].append(f"{type(exc).__name__}:{exc}")
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-notes", type=Path)
    parser.add_argument("--local-time")
    parser.add_argument("--exclude-note", type=Path)
    parser.add_argument("--finish-note", type=Path)
    parser.add_argument("--process", type=Path)
    parser.add_argument("--outcome")
    args = parser.parse_args()
    if args.check_notes:
        notes = [json.loads(path.read_text(encoding="utf-8-sig")) for path in sorted(args.check_notes.glob("*.json"))
                 if args.exclude_note is None or path.resolve() != args.exclude_note.resolve()]
        report = harness_budget(notes, args.local_time)
        print(json.dumps(report, allow_nan=False))
        return 0 if report["allowed"] else 1
    if args.finish_note:
        note = json.loads(args.finish_note.read_text(encoding="utf-8-sig"))
        print(json.dumps(complete_harness_note_from_files(note, args.process, outcome=args.outcome,
            note_path=args.finish_note), allow_nan=False))
        return 0
    parser.error("an offline note operation is required")


if __name__ == "__main__":
    raise SystemExit(main())
