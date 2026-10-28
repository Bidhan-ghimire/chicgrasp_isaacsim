"""Frozen attachment-assisted capture. Import/help/adjudication never start Isaac.

Execution is only through an explicitly authorized owned-process launch. Native
boundaries are in attachment_capture_runtime; this is the same controller used
by offline tests, not a second success-only path.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from chicgrasp_policy import attachment_capture as metrics
from chicgrasp_policy import attachment_owner, episode_io, interface
from chicgrasp_policy.attachment_capture_runtime import (RuntimeSession, CPU_STARTUP_SHUTDOWN_ONLY,
                                                        CPU_EVIDENCE_FIRST_CLOSE, EARLY_CPU_PROFILE)
from chicgrasp_policy.attachment_backend_adjudicate import CPU, PRODUCTION_GPU, scoped_candidate_result
from chicgrasp_policy.lift_event_pose_evidence import Context, TASK_ID, _json


class CaptureFailure(RuntimeError):
    pass


def _valid_pose(reading, clock=None):
    if not isinstance(reading, dict) or reading.get("valid") is not True or reading.get("issues") != []:
        raise CaptureFailure("required_pose_unavailable")
    if reading.get("clock_before") != reading.get("clock_after"):
        raise CaptureFailure("pose_clock_changed")
    if clock is not None and tuple(reading["clock_after"]) != tuple(clock):
        raise CaptureFailure("pose_not_current_scheduled_boundary")
    for key in ("robot_root_world_wxyz", "gripper_world_wxyz", "chicken_actor_world_wxyz",
                "tcp_world_wxyz", "tcp_base_wxyz", "chicken_in_gripper_wxyz"):
        value = np.asarray(reading.get("poses", {}).get(key), dtype=float)
        if value.shape != (7,) or not np.isfinite(value).all() or not np.linalg.norm(value[3:]) > 0:
            raise CaptureFailure("invalid_required_pose:" + key)
    return reading


class CandidateCapture:
    """200 real command frames, each followed by the existing six-tick scheduler."""
    def __init__(self, runtime, *, rgb_writer=None, action_source=None):
        self.runtime, self.owner = runtime, runtime.owner
        self.observation_mode = getattr(runtime, "observation_mode", False)
        self.rgb_writer = rgb_writer
        self.contact_writer = getattr(runtime, "contact_writer", None)
        self.contact_persistence_errors = []
        if self.observation_mode and rgb_writer is None:
            raise ValueError("observation_rgb_writer_required")
        if self.owner is None or not self.owner.journal.capture_policy:
            raise ValueError("reviewed_owner_explicit_capture_policy_required")
        self.action_source = action_source
        self.actions = metrics.frozen_candidate_actions() if action_source is None else None
        self.rows, self.command_frames, self.sensor_samples, self.diagnostics = [], [], [], []
        self.observations, self.executed_actions, self.lifecycle_events = [], [], []
        self.tick, self.substep, self.control_ticks = -1, 0, 0
        self.context = None
        self.current_action = None
        self.last_clock = tuple(runtime.origin_clock)
        self.phase = "initialized"
        self.completed = False

    def lifecycle(self, flags):
        if any(flags.values()):
            if self.rows:
                self.rows[-1]["runtime"].update(flags)
            self.lifecycle_events.append({"context": asdict(self.context) if self.context else None,
                                          "flags": deepcopy(flags), "before_automatic_reset": True})
            raise CaptureFailure("unexpected_termination_or_truncation_before_reset")

    def pre_reset(self):
        self.lifecycle_events.append({"context": asdict(self.context) if self.context else None,
                                      "resetting": True, "before_automatic_reset": True})
        raise CaptureFailure("unexpected_reset_before_reset")

    def boundary(self):
        """Called by public recorder hook immediately after each existing sim.step."""
        t, s = self.tick, self.substep
        if not (0 <= t <= 1199 and s in (0, 1)) or len(self.rows) != 2*t+s:
            raise CaptureFailure("extra_or_out_of_order_scheduled_boundary")
        self.context = Context(self.runtime.episode_id, t, s)
        clock = tuple(self.runtime.clock())
        sample = {"episode_id": self.runtime.episode_id, "control_tick": t, "substep": s,
                  "physics_step": clock[0], "simulation_time_s": clock[1]}
        row = {"context": asdict(self.context), "wall_monotonic_ns": time.monotonic_ns(),
               "pose": None, "contact": None, "predicate": None, "attachment": self.owner.snapshot(),
               "runtime": {"resetting": False, "truncated": False, "terminated": False, "invalid": False},
               "issues": []}
        self.rows.append(row)  # failures preserve the actually reached boundary
        boundary_failed = False
        try:
            if (type(clock[0]) is not int or clock[0] != self.runtime.origin_clock[0]+2*t+s+1
                    or isinstance(clock[1], bool) or not math.isfinite(clock[1])
                    or clock[1] <= self.last_clock[1]):
                raise CaptureFailure("missing_stale_or_nonfinite_scheduled_clock")
            self.last_clock = clock
            authored = row["attachment"]["status"] in ("PENDING", "CONFIRMED")
            try:
                row["contact"] = self.runtime.contact_collector.post_physics_step(t, s,
                    episode_id=self.runtime.episode_id)
                if authored:
                    row["contact"] = _json(row["contact"])
                predicate = metrics.bilateral_contact(row["contact"], self.runtime.coverage,
                    episode_id=self.runtime.episode_id, context=self.context, native_clock=clock)
            except Exception as exc:
                if not authored:
                    raise
                diagnostic = {"exception_type": type(exc).__name__}
                try:
                    diagnostic["message"] = str(exc)
                except Exception:
                    diagnostic["message_unavailable"] = True
                row["contact"] = {"diagnostic_only_contact_error": diagnostic}
                predicate = metrics.bilateral_contact(row["contact"], self.runtime.coverage,
                    episode_id=self.runtime.episode_id, context=self.context, native_clock=clock)
            row["predicate"] = predicate
            if t % 6 == 0 and s == 0:
                self.command_frames.append({"action_frame": t//6,
                    "commands": self.current_action[3:].tolist(), "sample": deepcopy(sample),
                    "source": "executed_candidate_command_first_post_substep/v1"})
            eligibility = {"command_frames": deepcopy(self.command_frames[-3:]),
                "bilateral_contact": predicate,
                "runtime": dict(row["runtime"], sample=sample, source="scheduled_pre_reset_boundary/v1")}
            status = row["attachment"]["status"]
            operation = None
            if status == "PENDING":
                operation = self.owner.confirm_pending(self.context)
                row["pose"] = operation.get("pose_reading")
                # Once authored, contact availability cannot gate confirmation.
                # Jaw-open remains an explicit release command.
                if operation.get("status") == "CONFIRMED" and (
                        np.any(self.current_action[3:] == 1)):
                    operation = self.owner.update(self.context, eligibility, self.current_action[3:])
                    row["pose"] = operation.get("pose_reading", row["pose"])
            elif status == "CONFIRMED":
                operation = self.owner.update(self.context, eligibility, self.current_action[3:])
                row["pose"] = operation.get("pose_reading")
            else:
                row["pose"] = self.runtime.measure()
            _valid_pose(row["pose"], clock)
            if self.observation_mode:
                # Same scheduled boundary, no second contact acquisition or
                # extra stepping. Optional diagnostic failure is not admission.
                row["observation"] = self.runtime.read_observation()
            if operation is not None:
                row["owner_operation_id"] = operation["id"]
                if operation.get("status") != "CONFIRMED":
                    raise CaptureFailure("attachment_lost_or_invalid:" + repr(operation.get("issues")))
            if status == "UNATTACHED" and t >= 720:
                if (predicate["status"] == "UNAVAILABLE"
                        and getattr(self.runtime, "contact_transport", "polling-contact/v1") != "callback-contact/v1"):
                    raise CaptureFailure("required_contact_unavailable_in_closing_phase")
                closed_history = len(self.command_frames) >= 3 and all(
                    f["commands"] == [0.0, 0.0] for f in self.command_frames[-3:])
                if predicate["status"] == "TRUE" and closed_history and (t, s) <= (839, 0):
                    operation = self.owner.activate(self.context, eligibility)
                    row["owner_operation_id"] = operation["id"]
                    row["attachment"] = self.owner.snapshot()
                    row["attachment"]["request_pose_reading"] = deepcopy(operation.get("pose_reading"))
                    if operation.get("status") != "PENDING":
                        raise CaptureFailure("attachment_request_failed:" + repr(operation.get("issues")))
            request_pose = row["attachment"].get("request_pose_reading")
            row["attachment"] = self.owner.snapshot()
            if request_pose is not None:
                row["attachment"]["request_pose_reading"] = request_pose
            if (t, s) == (839, 1) and row["attachment"]["status"] != "CONFIRMED":
                raise CaptureFailure("attachment_not_confirmed_before_lift:" + predicate["status"])
            self.substep += 1
        except Exception as exc:
            boundary_failed = True
            row["attachment"] = dict(row["attachment"], **self.owner.snapshot())
            row["runtime"]["invalid"] = True
            row["issues"].append(type(exc).__name__ + ":" + str(exc))
            raise
        finally:
            if self.contact_writer is not None:
                # The original live predicate/owner paths above finish first.
                # This also runs when 839/1 raises the required admission stop.
                # Recording failure is never an optional post-authoring contact
                # diagnostic, and must not replace an already active exception.
                try:
                    self._persist_contact(row)
                except Exception as exc:
                    error = "compact_contact_persistence:" + type(exc).__name__ + ":" + str(exc)
                    self.contact_persistence_errors.append(error)
                    row["runtime"]["invalid"] = True
                    row["issues"].append(error)
                    row["contact"] = {"schema": self.runtime.contact_storage_schema,
                                      "persistence_status": "FAILED", "error": error}
                    if not boundary_failed:
                        raise CaptureFailure(error) from exc

    def _persist_contact(self, row):
        from chicgrasp_policy.compact_contact_replay import pack_sample, compact_target
        from chicgrasp_policy.compact_contact_storage import SCHEMA
        sample = row["contact"]
        reading = sample.get("reading") if isinstance(sample, dict) else None
        diagnostic_error = isinstance(sample, dict) and set(sample) == {"diagnostic_only_contact_error"}
        unavailable = isinstance(reading, dict) and "callback_boundary" not in reading
        if diagnostic_error or unavailable:
            # Existing unavailable acquisition/contact exceptions keep their
            # original admission/release meaning. An absent boundary is not an
            # empty callback: retain the actual failure context separately.
            if unavailable:
                snapshot = reading.get("observation_snapshot", {})
                if (any(reading.get(key) for key in ("headers", "contacts", "unassociated_contacts", "callback_decoded_headers"))
                        or any(snapshot.get(key) for key in ("raw_headers", "data", "headers"))):
                    raise ValueError("unattributed_payload_without_callback_boundary")
            context = {"schema": "compact-contact-unavailable/v1",
                "sample": deepcopy(row["context"]), "native_clock": list(self.last_clock),
                "unavailable_sample": deepcopy(sample),
                "acquisition": "EXCEPTION" if diagnostic_error else "NO_CALLBACK_BOUNDARY_ACQUIRED"}
            reference = self.contact_writer.write_boundary(None, context,
                {"schema": "compact-contact-unavailable-target/v1", "predicate": deepcopy(row["predicate"])})
            row["contact"] = {"schema": SCHEMA, "reference": reference}
            return
        if not isinstance(sample, dict) or not isinstance(sample.get("reading"), dict):
            raise ValueError("required_callback_sample_missing")
        envelope = sample["reading"].get("callback_boundary")
        if not isinstance(envelope, dict):
            raise ValueError("required_callback_boundary_missing")
        context = pack_sample(sample, self.runtime.coverage,
            self.runtime.contact_reader.observation_control, self.runtime.origin_clock,
            writer=self.contact_writer)
        reference = self.contact_writer.write_boundary(envelope, context, compact_target(sample, row["predicate"]))
        row["contact"] = {"schema": SCHEMA, "reference": reference}

    def run(self):
        if self.phase != "initialized":
            raise CaptureFailure("candidate_cannot_retry")
        self.runtime.set_hooks(self.boundary, self.lifecycle, self.pre_reset)
        self.phase = "pre_candidate" if self.observation_mode else "candidate"
        prior_target = None
        index = 0
        while index < (len(self.actions) if self.action_source is None else getattr(self.action_source, "total_actions", 200)):
            observation = interface.validate_policy_observation(self.runtime.observation())
            before = _valid_pose(self.runtime.measure(), self.runtime.clock())
            if self.observation_mode and index == 0:
                pre = self.runtime.pre_candidate_observation(before)
            attached_before = self.owner.snapshot()["status"] == "CONFIRMED"
            sensor = {"action_frame": index, "native_clock": list(self.runtime.clock()),
                "sensor_frames": self.runtime.camera_frames(), "wall_monotonic_ns": time.monotonic_ns(),
                "policy_timestamp_ns": index*episode_io.TIMESTAMP_PERIOD_NS}
            jaw = RuntimeSession.commanded_jaw_metadata(self.runtime, observation.jaw_state)
            if index == 0:
                if jaw["bits"] != [1, 1]:
                    raise CaptureFailure("initial_jaw_state_not_reset_open")
                jaw["preceding_command"] = dict(kind="reset_open", action_frame=None,
                    control_tick=None, substep=None, observed_native_clock=None,
                    observed_wall_monotonic_ns=None)
            else:
                prior = self.rows[-1]
                if jaw["bits"] != self.command_frames[-1]["commands"]:
                    raise CaptureFailure("jaw_state_not_preceding_accepted_command")
                jaw["preceding_command"] = dict(kind="accepted_control_action", action_frame=index-1,
                    control_tick=prior["context"]["control_tick"], substep=prior["context"]["substep"],
                    observed_native_clock=deepcopy(prior["pose"]["clock_after"]),
                    observed_wall_monotonic_ns=prior["wall_monotonic_ns"])
            sensor["jaw_state"] = jaw
            if self.observation_mode:
                sensor.update(control_tick=6*index, camera_metadata=self.runtime.camera_metadata())
                if index == 0:
                    # One existing acquisition, two explicit roles. Neither the
                    # reset bundle nor its action0 alias implies a new frame.
                    sensor["shared_acquisition_action_frame"] = 0
                    pre_metadata = dict(deepcopy(sensor), kind="PRE_CANDIDATE",
                        action_frame=None, control_tick=None, substep=None,
                        policy_timestamp_ns=None)
                    pre["rgb_metadata"] = pre_metadata
                    write_json(self.runtime.output_dir / "pre_candidate.json", pre)
                    self.rgb_writer.write_pre_candidate(observation.as_mapping(), pre_metadata)
            self.sensor_samples.append(sensor)
            if self.observation_mode:
                # Persist even the action whose later boundary raises. Only
                # this already acquired policy observation supplies its pixels.
                self.rgb_writer.write_bundle(index, observation.as_mapping(), sensor)
            action = (self.actions[index] if self.action_source is None else
                      self.action_source.select(index, observation, sensor))
            if self.action_source is not None and tuple(self.runtime.clock()) != tuple(sensor["native_clock"]):
                raise CaptureFailure("policy_inference_advanced_native_physics")
            self.phase = "candidate"
            self.current_action = action.copy()
            for local_tick in range(6):
                self.tick, self.substep = 6*index+local_tick, 0
                if self.action_source is not None and local_tick == 0:
                    self.rgb_writer.command_event("DISPATCH_REQUESTED", index,
                        scheduled_boundary=[self.tick, 0], native_clock=list(self.runtime.clock()),
                        meaning="about_to_call_runtime_step; acceptance_not_yet_observed")
                flags = self.runtime.step(self.current_action)
                self.lifecycle(flags)
                if self.substep != 2:
                    raise CaptureFailure("scheduler_did_not_deliver_exactly_two_substeps")
                self.control_ticks += 1
            after = _valid_pose(self.rows[-1]["pose"])
            a, b = before["poses"], after["poses"]
            previous = np.asarray(a["tcp_base_wxyz"][:3]) if prior_target is None else prior_target
            self.diagnostics.append(interface.DiagnosticFrame(
                requested_tcp_pose_base_wxyz=tuple(action[:3])+interface.FIXED_TCP_QUATERNION_BASE_WXYZ,
                actual_tcp_pose_base_wxyz=tuple(a["tcp_base_wxyz"]),
                chicken_pose_world_wxyz=tuple(a["chicken_actor_world_wxyz"]),
                requested_tcp_movement_m=tuple(action[:3]-previous),
                actual_tcp_movement_m=tuple(np.asarray(b["tcp_base_wxyz"][:3])-a["tcp_base_wxyz"][:3]),
                chicken_movement_m=tuple(np.asarray(b["chicken_actor_world_wxyz"][:3])-a["chicken_actor_world_wxyz"][:3]),
                attachment_state=attached_before))
            self.observations.append(observation)
            self.executed_actions.append(action.copy())
            if self.action_source is not None:
                self.rgb_writer.command_event("COMPLETED", index,
                    scheduled_boundary=[self.tick, 1], native_clock=list(self.runtime.clock()))
            prior_target = action[:3].copy()
            index += 1
        self.completed = True
        self.phase = "candidate_complete"
        if getattr(self.runtime, "envelope_v2", False):
            return metrics.evaluate_envelope_v2(self.rows, episode_id=self.runtime.episode_id,
                origin_step=self.runtime.origin_clock[0], origin_time=self.runtime.origin_clock[1],
                close_action=self.action_source.first_accepted_close, completed=True, lifecycle_events=self.lifecycle_events,
                protocol=getattr(self.runtime, "envelope_protocol", "v2"))
        evaluated = metrics.evaluate_end_state(self.rows, episode_id=self.runtime.episode_id,
            origin_step=self.runtime.origin_clock[0], origin_time=self.runtime.origin_clock[1],
            **({"admission_policy": getattr(self.runtime, "admission_policy", "geometric-eligibility/v1")} if getattr(self.runtime, "geometric_mode", False) else {}))
        if self.action_source is not None:
            from chicgrasp_policy.stage3_policy import with_requested_motion
            evaluated = with_requested_motion(evaluated, self.executed_actions,
                **({"source": "translated_script_commands_robot_base_metres"} if getattr(self.runtime, "translated_point", None) is not None else {}))
        if getattr(self.runtime, "backend_mode", PRODUCTION_GPU) == CPU:
            evaluated = scoped_candidate_result(evaluated)
        self.diagnostics[-1] = replace(self.diagnostics[-1], outcome_label=(evaluated.get("label")
            if evaluated["status"] == "PASS" or getattr(self.runtime, "backend_mode", PRODUCTION_GPU) == CPU
            else "ATTACHMENT_ASSISTED_CANDIDATE_FAIL"))
        return evaluated

    def episode(self):
        if not self.completed or len(self.observations) != 200 or len(self.rows) != 2400:
            raise CaptureFailure("incomplete_candidate_not_a_complete_episode")
        arrays = {"timestamp_ns": np.arange(200, dtype=np.int64)*episode_io.TIMESTAMP_PERIOD_NS}
        for key in interface.POLICY_OBSERVATION_KEYS:
            arrays[key] = np.stack([observation.as_mapping()[key] for observation in self.observations])
        arrays["action"] = np.stack(self.executed_actions)
        metadata = {"schema_version": episode_io.COMMANDED_EPISODE_SCHEMA_VERSION,
            "episode_id": self.runtime.episode_id, "task_id": TASK_ID, "seed": self.runtime.seed,
            "num_policy_steps": 200, "policy_hz": 10, "control_hz": 60,
            "control_ticks_per_policy_action": 6, "timestamp_origin": episode_io.TIMESTAMP_ORIGIN,
            "observation_action_alignment": episode_io.OBSERVATION_ACTION_ALIGNMENT,
            "jaw_state_semantics": interface.JAW_STATE_SEMANTICS}
        return arrays, metadata, self.diagnostics


def write_json(path, value):
    """Fresh strict JSON plus OS-facing flush; never overwrite evidence."""
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(_json(value), stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def identity(path):
    digest, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", encoding="utf-8", newline="\n")
        self.sequence = 0

    def record(self, kind, **details):
        self.stream.write(json.dumps(_json(dict(sequence=self.sequence, kind=kind,
            monotonic_ns=time.monotonic_ns(), **details)), allow_nan=False)+"\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.sequence += 1

    def close(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()


def _candidate_consumed(result):
    counts = result.get("counts", {})
    return bool(result.get("candidate_started") or result.get("initialization", {}).get("candidate_started")
            or result.get("attachment_authoring_attempted")
            or any(counts.get(key, 0) != 0 for key in
                   ("completed_control_ticks", "observed_substeps", "completed_policy_actions"))
            or result.get("experiment_result", {}).get("status") == "PASS")


def campaign_accounting(attempt_root, ledger_path=None, *, current_result=None):
    """Verify saved campaign evidence, never create launch authority.

    The explicit ledger's state and limits are owner assertions, referenced to
    the grant bytes and original reservations, not an interpretation of grant
    prose. Invocation campaign='observation-v1' is a transport mode. The actual
    campaign identity comes from the per-invocation reservation. Storage names
    are used only to bind paths. A durable reservation always charges one slot.
    """
    answer = {"schema": "attachment-campaign-accounting/v1", "status": "UNAVAILABLE",
              "issues": [], "campaign_id": None, "authority_state": None,
              "reserved_invocations": 0, "observed_invocations": 0,
              "application_initializations": 0, "candidate_consumed": False,
              "numerical_unused_slots": None, "active_numerical_slots": 0,
              "processes_safe": False, "native_launch_authority": "NONE_CONFERRED_BY_THIS_RESULT"}
    if ledger_path is None or attempt_root is None:
        answer["issues"].append("campaign_ledger_missing")
        return answer

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def strict_json(text):
        def unique(pairs):
            value = {}
            for key, item in pairs:
                require(key not in value, "duplicate_json_key:" + key)
                value[key] = item
            return value
        def invalid_constant(value):
            raise ValueError("nonfinite_json_constant:" + value)
        return json.loads(text, object_pairs_hook=unique, parse_constant=invalid_constant)

    def referenced(record, expected_path=None, *, parse=True):
        require(isinstance(record, dict) and isinstance(record.get("path"), str), "invalid_evidence_reference")
        path = Path(record["path"])
        require(path.is_absolute() and path.is_file() and not path.is_symlink(), "missing_or_linked_evidence")
        require(expected_path is None or path.resolve() == expected_path.resolve(), "evidence_path_disagreement")
        raw = path.read_bytes()
        actual = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        require(type(record.get("bytes")) is int and record["bytes"] >= 0
                and actual == {key: record.get(key) for key in ("bytes", "sha256")},
                "evidence_identity_disagreement")
        return strict_json(raw.decode("utf-8")) if parse else None

    def natural(value):
        return type(value) is int and value >= 0

    try:
        ledger = strict_json(Path(ledger_path).read_text(encoding="utf-8"))
        require(isinstance(ledger, dict) and ledger.get("schema") == "attachment-campaign-ledger/v1",
                "campaign_ledger_schema_invalid")
        campaign = ledger.get("campaign_id")
        require(isinstance(campaign, str) and campaign and campaign != "observation-v1", "campaign_identity_invalid")
        maximum = ledger.get("maximum_initializations")
        require(type(maximum) is int and maximum > 0
                and type(ledger.get("maximum_candidates")) is int and ledger["maximum_candidates"] == 1,
                "campaign_allowance_invalid")
        state, terminal = ledger.get("authority_state"), ledger.get("terminal_result")
        require(state in ("ACTIVE", "EXPIRED", "TERMINAL_STOP") and "terminal_result" in ledger,
                "campaign_authority_state_missing")
        require((state == "ACTIVE" and terminal is None)
                or (state != "ACTIVE" and isinstance(terminal, str) and bool(terminal)),
                "campaign_authority_state_conflict")
        referenced(ledger["grant"], parse=False)
        attempts = ledger.get("attempts")
        require(isinstance(attempts, list) and 0 < len(attempts) <= maximum, "campaign_reservations_ambiguous")
        ids, paths, invocations, initializations, candidates, all_safe = set(), set(), 0, 0, 0, True
        current_found = False
        for ordinal, entry in enumerate(attempts, 1):
            reservation = referenced(entry["reservation"])
            directory = Path(reservation["attempt_directory"])
            require(directory.is_absolute(), "reservation_directory_not_absolute")
            directory = directory.resolve()
            reservation_id = reservation.get("reservation_id")
            require(isinstance(reservation_id, str) and reservation_id and reservation_id not in ids
                    and directory not in paths, "duplicate_reservation_or_attempt")
            ids.add(reservation_id)
            paths.add(directory)
            require(reservation.get("campaign") == campaign, "reservation_campaign_disagreement")
            expected = {"campaign_invocation_ordinal": ordinal, "maximum_initializations": maximum,
                        "maximum_candidates": 1, "observed_invocations_before_reservation": invocations,
                        "observed_initializations_before_reservation": initializations,
                        "observed_candidates_before_reservation": candidates}
            require(all(type(reservation.get(key)) is int and reservation[key] == value
                        for key, value in expected.items()), "reservation_accounting_disagreement")
            require(reservation.get("grant") == {key: ledger["grant"][key] for key in ("bytes", "sha256")},
                    "reservation_grant_disagreement")
            invocation = referenced(entry["invocation"], directory/"invocation.json")
            saved = referenced(entry["result"], directory/"capture"/"experiment"/"result.json")
            process = referenced(entry["process"], directory/"capture"/"process_result.json")
            require(invocation.get("kind") == "attachment_capture"
                    and invocation.get("execution_approved") is True
                    and Path(invocation["output_directory"]).resolve() == directory/"capture",
                    "invocation_reservation_disagreement")
            require(process.get("started") is True and type(process.get("probe_process_invocations")) is int
                    and process["probe_process_invocations"] == 1
                    and process.get("invocation_sha256") == entry["invocation"]["sha256"],
                    "invocation_observation_ambiguous")
            provenance = saved.get("execution_provenance", {})
            require(provenance.get("attempt_id") == directory.name
                    and isinstance(reservation.get("executed_commit"), str)
                    and provenance.get("executed_commit") == reservation["executed_commit"],
                    "result_reservation_disagreement")
            initialization, counts = saved.get("initialization", {}), saved.get("counts", {})
            require(type(initialization.get("application_attempts")) is int
                    and initialization["application_attempts"] in (0, 1)
                    and all(type(value) is bool for value in
                            (saved.get("candidate_started"), saved.get("attachment_authoring_attempted"),
                             initialization.get("candidate_started")))
                    and all(natural(counts.get(key)) for key in
                            ("completed_control_ticks", "observed_substeps", "completed_policy_actions")),
                    "candidate_or_initialization_evidence_ambiguous")
            consumed = _candidate_consumed(saved)
            answer["candidate_consumed"] = answer["candidate_consumed"] or consumed
            invocations += 1
            initializations += initialization["application_attempts"]
            candidates += int(consumed)
            require(candidates <= 1, "campaign_candidate_allowance_exceeded")
            safe_fields = {"exit_code": 0, "root_exit_verified": True, "tree_empty_verified": True,
                           "capture_complete": True, "forced_termination": False, "timed_out": False,
                           "parent_termination_requested": False, "autonomous_tree_exit": True,
                           "kill_on_job_close_used": False, "kill_on_job_close_possible": False}
            all_safe = all_safe and all(type(process.get(key)) is type(value) and process[key] == value
                                       for key, value in safe_fields.items())
            if directory == Path(attempt_root).resolve():
                require(ordinal == len(attempts) and (current_result is None or saved == current_result),
                        "current_attempt_stale_or_result_disagrees")
                current_found = True
        require(current_found, "current_attempt_not_in_campaign")
        answer.update(status="VERIFIED", campaign_id=campaign, authority_state=state,
                      maximum_initializations=maximum, reserved_invocations=len(attempts),
                      observed_invocations=invocations, application_initializations=initializations,
                      numerical_unused_slots=maximum-len(attempts), processes_safe=all_safe,
                      active_numerical_slots=maximum-len(attempts) if state == "ACTIVE" else 0)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        answer["status"] = "AMBIGUOUS"
        answer["issues"].append(type(exc).__name__ + ":" + str(exc))
    return answer


def retry_classification(result, process_safe=False, *, instrumentation_retry_authorized=False,
                         instrumentation_gap_used=False, remaining_invocations=None,
                         campaign_ledger_path=None, attempt_root=None):
    """Classification only. Caller flags and ledger metadata never grant permission."""
    if _candidate_consumed(result):
        return "NO_RETRY_EXPERIMENTAL_ATTEMPT"
    accounting = campaign_accounting(attempt_root, campaign_ledger_path, current_result=result)
    if accounting["candidate_consumed"]:
        return "NO_RETRY_EXPERIMENTAL_ATTEMPT"
    if accounting["status"] != "VERIFIED":
        return "NO_RETRY_CAMPAIGN_LEDGER_UNAVAILABLE_OR_AMBIGUOUS"
    if accounting["authority_state"] != "ACTIVE":
        return "NO_RETRY_AUTHORITY_EXPIRED"
    if accounting["active_numerical_slots"] == 0:
        return "NO_RETRY_CAMPAIGN_EXHAUSTED"
    if (not process_safe or not accounting["processes_safe"]
            or (remaining_invocations is not None and (type(remaining_invocations) is not int
                or remaining_invocations != accounting["active_numerical_slots"]))):
        return "NO_RETRY_AMBIGUOUS_OR_EVIDENCE_FAILURE"
    counts = result.get("counts", {})
    if instrumentation_retry_authorized:
        if instrumentation_gap_used:
            return "NO_RETRY_INSTRUMENTATION_GAP_ALREADY_USED"
        if (result.get("failure_phase") == "initialization"
                and result.get("candidate_started") is False
                and result.get("attachment_authoring_attempted") is False
                and all(type(counts.get(k)) is int and counts[k] == 0 for k in
                        ("completed_control_ticks", "observed_substeps", "completed_policy_actions"))
                and "ValueError:native_stage_identity_mismatch" in result.get("primary_errors", [])
                and not result.get("initialization", {}).get("stage_identity_diagnostics", {}).get("records")):
            return "STARTUP_FAILURE_REQUIRES_SEPARATE_AUTHORITY"
        return "NO_RETRY_AMBIGUOUS_OR_EVIDENCE_FAILURE"
    if process_safe and result.get("failure_phase") == "initialization":
        return "STARTUP_FAILURE_REQUIRES_SEPARATE_AUTHORITY"
    return "NO_RETRY_AMBIGUOUS_OR_EVIDENCE_FAILURE"


def execution_identity(output):
    """Read local Git provenance before initialization; never alter the checkout."""
    provenance = {"attempt_id": Path(output).parent.parent.name,
                  "executed_commit": None, "commit_source": "git rev-parse HEAD",
                  "errors": []}
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True,
            timeout=10, check=True)
        commit = completed.stdout.strip()
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise ValueError("invalid_git_commit_identity")
        provenance["executed_commit"] = commit
    except Exception as exc:
        provenance["errors"].append(type(exc).__name__ + ":" + str(exc))
    return provenance


def owner_cleanup_issues(close_result):
    """No owned resource is a qualified no-op, never native destruction proof."""
    release = close_result.get("cleanup") or {}
    removed_or_unneeded = (release.get("cleanup_confirmed") is True or
        (release.get("no_owned_resource") is True and release.get("cleanup_required") is False))
    if (close_result.get("issues") or not removed_or_unneeded or release.get("issues")
            or release.get("lifecycle_issues") or release.get("lifecycle_recorded") is not True):
        return ["owner_cleanup_not_qualified"]
    return []


def _phase_sink(runtime, journal_path):
    # Separate from the finalized required journal. Each write closes its file
    # before returning, including the final marker before nonreturning fast close.
    phase_path = Path(journal_path).with_name("failure_phases.jsonl")
    phase_sequence, phase_errors = 0, []
    def persist_phase(record):
        nonlocal phase_sequence
        try:
            line = json.dumps(_json(dict(sequence=phase_sequence, **record)), allow_nan=False) + "\n"
            with phase_path.open("x" if phase_sequence == 0 else "a", encoding="utf-8", newline="\n") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
            phase_sequence += 1
        except Exception as exc:
            phase_errors.append(type(exc).__name__ + ":" + str(exc))
    runtime.phase_record_sink = persist_phase
    def phase_record(phase, event, **details):
        persist_phase(dict(phase=phase, event=event, monotonic_ns=time.monotonic_ns(), **details))
    return phase_path, phase_record, phase_errors


def _finalize_and_close(output, journal_path, journal, runtime, result, phase_record, phase_errors):
    """Shared Cycle44 finalization path; no required evidence relies on close returning."""
    manifest = None
    required_records_complete = False
    cpu_terminal = result.get("terminal_policy") == CPU_EVIDENCE_FIRST_CLOSE
    terminal_sequence = 0
    def terminal_record(phase, event, **details):
        nonlocal terminal_sequence
        path = Path(journal_path).with_name("terminal_phases.jsonl")
        record = dict(schema="cpu-evidence-first-terminal/v1", sequence=terminal_sequence,
                      monotonic_ns=time.monotonic_ns(), terminal_policy=CPU_EVIDENCE_FIRST_CLOSE,
                      phase=phase, event=event, **details)
        with path.open("x" if terminal_sequence == 0 else "a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(_json(record), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        terminal_sequence += 1
    try:
        # This means files are complete and checked, not that the experiment passed.
        result["evidence_integrity"] = "COMPLETE" if not result["recording_errors"] else "INCOMPLETE"
        write_json(output/"result.json", result)
        finished = {p.relative_to(output).as_posix(): identity(p)
                    for p in sorted(output.rglob("*")) if p.is_file()}
        write_json(output/"artifact_manifest.json", {"schema": "capture-artifacts/v1", "files": finished})
        manifest = identity(output/"artifact_manifest.json")
        journal.record("PRE_CLOSE_EVIDENCE_COMPLETE" if result["evidence_integrity"] == "COMPLETE"
                       else "PRE_CLOSE_EVIDENCE_INCOMPLETE", manifest=manifest,
                       script_status=result["script_status"], integrity=result["evidence_integrity"])
        journal.record("CLOSE_REQUESTED", app_handle_available=runtime.app is not None, manifest=manifest)
        required_records_complete = True
    except Exception as exc:
        if cpu_terminal:
            phase_errors.append("required_evidence:" + type(exc).__name__ + ":" + str(exc))
            phase_record("required_finalization", "failed", operation="result_manifest_or_journal_records",
                         **RuntimeSession.failure_details(exc))
        raise
    finally:
        # Finalization diagnostics cannot prevent the supported app close.
        finalization_failure = None
        for name, finalize in (("journal", journal.close), ("stdout", sys.stdout.flush),
                               ("stderr", sys.stderr.flush)):
            try:
                finalize()
            except Exception as exc:
                if finalization_failure is None:
                    finalization_failure = exc
                phase_errors.append(name + ":" + type(exc).__name__ + ":" + str(exc))
                phase_record("required_finalization", "failed", operation=name,
                             **RuntimeSession.failure_details(exc))
        if cpu_terminal:
            try:
                writers = dict(result["writer_statuses"],
                    application_artifacts="COMPLETE" if manifest else "FAILED",
                    required_journal="COMPLETE" if required_records_complete and finalization_failure is None else "FAILED")
                writers_done = all(value in ("COMPLETE", "NOT_APPLICABLE") for value in writers.values())
                counts = result.get("counts", {})
                observed = result.get("observation_result", {}).get("OBSERVATION_RESULT") == "PASS"
                if observed and counts == dict(completed_control_ticks=1200, observed_substeps=2400, completed_policy_actions=200):
                    scope, stop = "FULL_CANDIDATE", "candidate_complete"
                elif observed and counts == dict(completed_control_ticks=839, observed_substeps=1680, completed_policy_actions=139):
                    scope, stop = "839_1", result["primary_errors"][0]
                elif (result.get("candidate_started") is False and result.get("attachment_authoring_attempted") is False
                      and counts == dict(completed_control_ticks=0, observed_substeps=0, completed_policy_actions=0)):
                    scope, stop = "FAILED_START", (result["primary_errors"] or ["failed_start_without_primary_error"])[0]
                else:
                    scope, stop = "INCOMPLETE_WINDOW", (result["primary_errors"] or ["incomplete_observation_window"])[0]
                finalized = (manifest is not None and writers_done and not phase_errors
                             and not result["recording_errors"] and not result["cleanup_errors"])
                terminal_record("evidence_finalization", "returned", evidence_finalized=finalized,
                    writers_completed=writers_done, required_measurements_pending=False,
                    artifacts={"application_manifest": dict(manifest or {}, relative_path="../experiment/artifact_manifest.json")},
                    writer_statuses=writers, evidence_scope=scope, stop_reason=stop,
                    observation_window_complete=scope in ("FULL_CANDIDATE", "839_1"))
                terminal_record("terminal_close", "requested", finalization_sequence=0)
            except Exception as exc:
                phase_errors.append("terminal_evidence:" + type(exc).__name__ + ":" + str(exc))
                phase_record("terminal_evidence", "failed", **RuntimeSession.failure_details(exc))
            try:
                terminal_cleanup = runtime.close_terminal_scene()
                terminal_record("environment_close", "failed" if terminal_cleanup.get("errors") else "returned",
                                cleanup=terminal_cleanup)
            except Exception as exc:
                phase_record("terminal_environment_close", "failed", **RuntimeSession.failure_details(exc))
                try:
                    terminal_record("environment_close", "failed", error=type(exc).__name__ + ":" + str(exc))
                except Exception:
                    pass
        phase_record("runtime_close_app", "requested", app_handle_available=runtime.app is not None,
                     phase_evidence_errors=list(phase_errors))
        def optional_close_record(returned, error=None):
            try:
                write_json(Path(journal_path).with_name("post_close.json"), {
                    "schema": "capture-optional-close/v1", "monotonic_ns": time.monotonic_ns(),
                    "close_returned": returned, "close_error": error,
                    "primary_errors": list(result["primary_errors"]),
                    "cleanup_errors": deepcopy(result["cleanup_errors"]),
                    "phase_evidence_errors": list(phase_errors),
                    "app_handle_available": runtime.app is not None,
                    **({"terminal_scene_cleanup": deepcopy(runtime.terminal_scene_close_result)} if cpu_terminal else {}),
                    "scope": "Python close call outcome only; OS process completion remains parent-owned"})
            except Exception as exc:
                # Remains optional; preserve a write failure where possible
                # without replacing either the primary or close exception.
                phase_record("optional_close_evidence", "failed", **RuntimeSession.failure_details(exc))
        try:
            runtime.close_app()
        except Exception as exc:
            if cpu_terminal:
                try:
                    terminal_record("application_close", "failed", error=type(exc).__name__ + ":" + str(exc))
                except Exception:
                    pass
            optional_close_record(False, type(exc).__name__ + ":" + str(exc))
            raise
        else:
            if cpu_terminal:
                try:
                    terminal_record("application_close", "returned", app_handle_available=runtime.app is not None)
                except Exception as exc:
                    phase_record("terminal_evidence", "failed", **RuntimeSession.failure_details(exc))
            if runtime.app is not None:
                optional_close_record(True)
            if finalization_failure is not None:
                raise finalization_failure


def execute_startup_observation(output_dir, journal_path, *, seed=42, runtime_factory=RuntimeSession):
    """One normal constructor, then existing cleanup. Never enters capture preparation."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(journal_path)
    runtime = runtime_factory(output, seed=seed, observation_mode=True,
        contact_transport="callback-contact/v1", backend_mode=CPU,
        diagnostic_mode=CPU_STARTUP_SHUTDOWN_ONLY)
    runtime.execution_context = execution_identity(output)
    phase_path, phase_record, phase_errors = _phase_sink(runtime, journal_path)
    guard_path = Path(journal_path).with_name("guard_observations.jsonl")
    # Each guard observation closes/fsyncs before construction resumes. Empty
    # means NOT_REACHED, never a valid negative guard observation.
    with guard_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    def persist_guard(record):
        with guard_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(_json(record), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    runtime.startup_guard_sink = persist_guard
    result = {"schema": "cpu-startup-shutdown/v1", "diagnostic_mode": CPU_STARTUP_SHUTDOWN_ONLY,
        "backend_mode": CPU, "observation_mode": True, "contact_transport": "callback-contact/v1",
        "episode_id": runtime.episode_id, "primary_errors": [], "recording_errors": [], "cleanup_errors": [],
        "candidate_started": False, "candidate_executions": 0, "attachment_authoring": 0,
        "attachment_authoring_attempted": False, "END_STATE_RESULT": "NOT_EVALUABLE",
        "failure_phase_evidence": {"schema": "capture-failure-phases/v1", "path": str(phase_path.resolve())},
        "execution_provenance": deepcopy(runtime.execution_context)}
    constructor_outcome = "FAILED"
    journal.record("INITIALIZATION_REQUESTED", diagnostic_mode=CPU_STARTUP_SHUTDOWN_ONLY)
    try:
        runtime.initialize()
        constructor_outcome = "RETURNED"
        journal.record("INITIALIZATION_RETURNED", initialization=runtime.initialization)
    except Exception as exc:
        result["primary_errors"].append(type(exc).__name__ + ":" + str(exc))
        result["failure_diagnostic"] = RuntimeSession.failure_details(exc)
        phase_record("initialization", "failed", **result["failure_diagnostic"])
    finally:
        result["initialization"] = deepcopy(runtime.initialization)
        diagnostic_errors = (deepcopy(runtime.startup_diagnostic_errors) + list(phase_errors)
                             + runtime.initialization.get("phase_evidence_errors", []))
        snapshot = {"schema": "cpu-startup-shutdown/v1", "diagnostic_mode": CPU_STARTUP_SHUTDOWN_ONLY,
            "backend_mode": CPU, "constructor_outcome": constructor_outcome,
            "primary_error": result["primary_errors"][0] if result["primary_errors"] else None,
            "initialization": deepcopy(runtime.initialization),
            "guard_observations": deepcopy(runtime.startup_guard_records),
            "backend_observations": [deepcopy(r["backend"]) for r in runtime.startup_guard_records]
                                    + deepcopy(runtime.startup_backend_records),
            "candidate_executions": 0, "attachment_authoring": 0,
            "forbidden_counts": dict.fromkeys(("candidate_actions", "caller_steps", "later_harness_resets",
                "caller_renders", "owner_bindings", "attachment_authoring", "pre_candidate_acquisitions"), 0),
            "diagnostic_write_errors": diagnostic_errors,
            "execution_provenance": deepcopy(runtime.execution_context)}
        try:
            snapshot["identities"] = {"guard_stream": identity(guard_path),
                "phase_prefix": dict(identity(phase_path), records=len(phase_path.read_bytes().splitlines()))}
            write_json(output / "startup_diagnostic.json", snapshot)
        except Exception as exc:
            diagnostic_errors.append("startup_snapshot:" + type(exc).__name__ + ":" + str(exc))
            phase_record("startup_snapshot", "failed", **RuntimeSession.failure_details(exc))
        result["recording_errors"].extend(diagnostic_errors)
        # Both are pre-cleanup facts, not exit or cleanup success. Failure to
        # write the journal cannot suppress the durable phase attempt or cleanup.
        phase_record("SHUTDOWN_SEQUENCE_BEGIN", "requested")
        try:
            journal.record("SHUTDOWN_SEQUENCE_BEGIN")
        except Exception as exc:
            result["recording_errors"].append("shutdown_marker:" + type(exc).__name__ + ":" + str(exc))
            phase_record("shutdown_marker", "failed", **RuntimeSession.failure_details(exc))
        try:
            phase_record("scene_cleanup", "requested")
            result["cleanup"] = runtime.close_scene(None)
            result["cleanup_errors"].extend(result["cleanup"].get("errors", []))
            phase_record("scene_cleanup", "returned")
        except Exception as exc:
            result["cleanup_errors"].append(type(exc).__name__ + ":" + str(exc))
            phase_record("scene_cleanup", "failed", **RuntimeSession.failure_details(exc))
        result["script_status"] = "FAIL" if any(result[key] for key in
            ("primary_errors", "recording_errors", "cleanup_errors")) else "PASS"
        _finalize_and_close(output, journal_path, journal, runtime, result, phase_record, phase_errors)
    return result


def execute(output_dir, journal_path, *, seed=42, runtime_factory=RuntimeSession,
            bind_owner=attachment_owner.bind_runtime, observation_mode=False,
            contact_transport="polling-contact/v1", backend_mode=PRODUCTION_GPU, diagnostic_mode=None,
            terminal_policy=None, geometric_mode=False, run_type=None, stage0_mode=None, stage1_demo=False,
            stage3_learned=False, stage3_witness=False, translated_point=None, r4_live_point=None, envelope_v2=False, envelope_v3c=False):
    """All required files and terminal intent precede a potentially nonreturning close."""
    if r4_live_point is not None and (not geometric_mode or not stage3_learned or not stage3_witness
            or run_type != "HARNESS" or translated_point is not None or stage0_mode or stage1_demo):
        raise ValueError("r4_requires_exclusive_learned_witness_GPU_HARNESS")
    if translated_point is not None and not geometric_mode:
        raise ValueError("translated_script_requires_geometric_mode")
    if stage3_witness and (not stage3_learned or not geometric_mode or run_type not in ("HARNESS", "EVIDENCE")):
        raise ValueError("witness_requires_Stage3_explicit_run_type")
    if stage0_mode is not None and (not geometric_mode or run_type != "HARNESS"):
        raise ValueError("stage0_is_HARNESS_only")
    if stage1_demo and (not geometric_mode or run_type not in ("EVIDENCE", "HARNESS") or stage0_mode):
        raise ValueError("stage1_requires_geometric_recording_or_startup_check")
    if stage3_learned and (not geometric_mode or run_type not in ("EVIDENCE", "HARNESS") or stage0_mode or stage1_demo):
        raise ValueError("stage3_requires_separate_geometric_EVIDENCE")
    if geometric_mode:
        if observation_mode or contact_transport != "polling-contact/v1" or backend_mode != PRODUCTION_GPU or diagnostic_mode or terminal_policy:
            raise ValueError("geometric_GPU_path_cannot_mix_legacy_contact_or_CPU_modes")
        from chicgrasp_policy.geometric_capture import execute_slim
        from types import SimpleNamespace
        return execute_slim(SimpleNamespace(**globals()),
            output_dir, journal_path, seed=seed, runtime_factory=runtime_factory,
            bind_owner=bind_owner, run_type=run_type, stage0_mode=stage0_mode, stage1_demo=stage1_demo,
            stage3_learned=stage3_learned, stage3_witness=stage3_witness, translated_point=translated_point, r4_live_point=r4_live_point, envelope_v2=envelope_v2, envelope_v3c=envelope_v3c)
    if contact_transport not in ("polling-contact/v1", "callback-contact/v1"):
        raise ValueError("unsupported_contact_transport")
    if contact_transport == "callback-contact/v1" and not observation_mode:
        raise ValueError("callback_transport_requires_observation_mode")
    if backend_mode not in (PRODUCTION_GPU, CPU):
        raise ValueError("unsupported_backend_mode")
    if backend_mode == CPU and (not observation_mode or contact_transport != "callback-contact/v1"):
        raise ValueError("cpu_diagnostic_requires_callback_observation")
    if diagnostic_mode not in (None, CPU_STARTUP_SHUTDOWN_ONLY):
        raise ValueError("unsupported_diagnostic_mode")
    if terminal_policy not in (None, CPU_EVIDENCE_FIRST_CLOSE):
        raise ValueError("unsupported_terminal_policy")
    if terminal_policy is not None and (backend_mode != CPU or diagnostic_mode is not None):
        raise ValueError("cpu_terminal_policy_requires_cpu_campaign_not_startup_trial")
    if diagnostic_mode == CPU_STARTUP_SHUTDOWN_ONLY:
        if backend_mode != CPU:
            raise ValueError("startup_only_requires_cpu_callback_observation")
        return execute_startup_observation(output_dir, journal_path, seed=seed, runtime_factory=runtime_factory)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(journal_path)
    runtime_options = {"observation_mode": True} if observation_mode else {}
    if contact_transport == "callback-contact/v1":
        runtime_options["contact_transport"] = contact_transport
    if backend_mode == CPU:
        runtime_options["backend_mode"] = CPU
    if terminal_policy is not None:
        runtime_options["terminal_policy"] = terminal_policy
    runtime = runtime_factory(output, seed=seed, **runtime_options)
    runtime.execution_context = execution_identity(output)
    runtime.guard_record_sink = lambda record: journal.record("GUARD_COMPARISON", record=record)
    phase_path, phase_record, phase_errors = _phase_sink(runtime, journal_path)
    capture, rgb_writer, contact_writer = None, None, None
    result = {"schema": "attachment-assisted-capture/v1", "experiment_result": {"status": "FAIL"},
        "evidence_integrity": "INCOMPLETE", "process_completion": "PARENT_OBSERVATION_REQUIRED",
        "failure_phase_evidence": {"schema": "capture-failure-phases/v1", "path": str(phase_path.resolve())},
        "primary_errors": [], "recording_errors": [], "cleanup_errors": [], "candidate_started": False,
        "failure_phase": "initialization", "episode_id": runtime.episode_id,
        "observation_mode": observation_mode, "contact_transport": contact_transport,
        "limitations": ["attachment_assisted_not_unassisted_grasp", "not_table_clear_or_drag_free_proof",
            "SHUTDOWN_PROCESS_EXIT_UNRESOLVED", "USDRT_GLOBAL_TIME_WARNING_UNRESOLVED",
            "projection_unresolved_non_gating"]}
    if backend_mode == CPU:
        result.update(backend_mode=CPU, gpu_gate5_acceptance=False, end_state_scope=CPU,
                      experiment_result=scoped_candidate_result({"status": "FAIL"}, incomplete=True),
                      early_cpu_profile_schema=EARLY_CPU_PROFILE)
        def persist_early_profile(record):
            # Use the same exclusive-create, flush/fsync/close writer as other
            # required evidence, before reset. Do not read the native state again.
            captured = json.loads(json.dumps(record, allow_nan=False))
            captured["persistence_started_ns"] = time.monotonic_ns()
            path = output / "early_cpu_profile.json"
            write_json(path, captured)
            return dict(path=path.name, **identity(path),
                persistence_started_ns=captured["persistence_started_ns"],
                persistence_completed_ns=time.monotonic_ns())
        runtime.early_profile_sink = persist_early_profile
    guard_state = {"closed": False, "records": 0}
    if terminal_policy is not None:
        result["terminal_policy"] = terminal_policy
        result["writer_statuses"] = dict(application_artifacts="PENDING", required_journal="PENDING",
            rgb="NOT_APPLICABLE", compact_contact="PENDING", episode="NOT_APPLICABLE", policy_guard="PENDING")
        guard_path = output / "policy_guard_observations.jsonl"
        def persist_policy_guard(record):
            if guard_state["closed"]:
                raise RuntimeError("policy_guard_write_after_finalization")
            with guard_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(_json(record), allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            guard_state["records"] += 1
        runtime.policy_guard_sink = persist_policy_guard
    journal.record("INITIALIZATION_REQUESTED")
    try:
        if terminal_policy is not None:
            with guard_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.flush()
                os.fsync(stream.fileno())
        if contact_transport == "callback-contact/v1":
            from chicgrasp_policy.compact_contact_storage import CompactContactWriter, SCHEMA
            contact_writer = CompactContactWriter(output / "contact_compact")
            runtime.contact_writer = contact_writer
            runtime.contact_storage_schema = SCHEMA
            result["contact_storage_schema"] = SCHEMA
        runtime.initialize()
        if contact_writer is not None:
            runtime.contact_collector.use_external_storage()
        journal.record("INITIALIZATION_RETURNED", initialization=runtime.initialization,
                       effective_configuration=runtime.effective_configuration)
        if observation_mode:
            from chicgrasp_policy.attachment_observation_evidence import IncrementalRGBWriter
            write_json(output / "observation_configuration.json", runtime.observation_configuration)
            rgb_writer = IncrementalRGBWriter(output / "rgb")
            if terminal_policy is not None:
                result["writer_statuses"]["rgb"] = "PENDING"
        binding = {"task_id": TASK_ID, "capture_policy": True, "episode_id": runtime.episode_id,
                   "origin_step": runtime.origin_clock[0], "origin_time": runtime.origin_clock[1]}
        scene_identity = getattr(runtime, "scene_identity", None)
        if scene_identity is not None:
            binding["scene_identity"] = scene_identity
        runtime.owner = bind_owner(runtime.env, runtime.stage, **binding)
        capture = CandidateCapture(runtime, rgb_writer=rgb_writer)
        result["failure_phase"] = "candidate"
        journal.record("CANDIDATE_REQUESTED", origin_clock=runtime.origin_clock)
        result["experiment_result"] = capture.run()
    except Exception as exc:
        result["primary_errors"].append(type(exc).__name__+":"+str(exc))
        result["failure_diagnostic"] = RuntimeSession.failure_details(exc)
        phase_record(result["failure_phase"], "failed", **result["failure_diagnostic"])
    finally:
        result["candidate_started"] = bool(runtime.initialization.get("candidate_started"))
        result["initialization"] = deepcopy(runtime.initialization)
        result["execution_provenance"] = deepcopy(runtime.execution_context)
        # Unknown until the existing export below succeeds; never infer no partial
        # authoring from a false completion flag or a failed owner export.
        result["attachment_authoring_attempted"] = False if runtime.owner is None else None
        result["effective_configuration"] = runtime.effective_configuration
        if backend_mode == CPU:
            result["backend_diagnostic"] = deepcopy(getattr(runtime, "backend_diagnostic", {}))
            early = deepcopy(getattr(runtime, "early_cpu_profile", None))
            result["early_cpu_profile"] = early
            result["initialization"]["early_cpu_profile"] = deepcopy(early)
            if result.get("failure_diagnostic") is not None:
                result["failure_diagnostic"]["early_cpu_profile"] = deepcopy(early)
            result["recording_errors"].extend("early_cpu_profile:" + str(error) for error in
                getattr(runtime, "early_profile_errors", []))
            if not isinstance(early, dict):
                result["recording_errors"].append("early_cpu_profile:missing_runtime_accounting")
        result["counts"] = {"completed_control_ticks": capture.control_ticks if capture else 0,
            "observed_substeps": len(capture.rows) if capture else 0,
            "completed_policy_actions": len(capture.executed_actions) if capture else 0}
        if capture is not None:
            result["recording_errors"].extend(capture.contact_persistence_errors)
        if (observation_mode and capture is not None and capture.phase == "pre_candidate"
                and not result["candidate_started"] and all(value == 0 for value in result["counts"].values())):
            # This includes a failed pre-command encoder/write check, not an
            # experimental retry grant. Saved owner/process evidence must also
            # establish no authoring and qualified owned-tree completion.
            result["failure_phase"] = "initialization"
        # Terminal measurements and decision are frozen BEFORE scene teardown.
        try:
            write_json(output/"candidate_result.json", result["experiment_result"])
            write_json(output/"measurements.json", {"episode_id": runtime.episode_id,
                "origin_clock": runtime.origin_clock, "rows": capture.rows if capture else [],
                "command_frames": capture.command_frames if capture else [],
                "sensor_samples": capture.sensor_samples if capture else [],
                "lifecycle_events": capture.lifecycle_events if capture else [],
                "coverage": runtime.coverage, "contact_transport": contact_transport,
                **({"contact_storage_schema": runtime.contact_storage_schema} if contact_writer is not None else {})})
            if contact_transport == "callback-contact/v1":
                write_json(output/"contact_transport_before_cleanup.json", runtime.export_contact_transport("before_cleanup"))
            if runtime.owner is not None:
                owner_export = runtime.owner.export()
                result["attachment_authoring_attempted"] = any(
                    op.get("kind") == "scheduled_request" for op in owner_export.get("operations", []))
                write_json(output/"owner_before_cleanup.json", owner_export)
            if capture and capture.completed:
                if terminal_policy is not None:
                    result["writer_statuses"]["episode"] = "PENDING"
                arrays, metadata, diagnostics = capture.episode()
                episode_io.write_episode(output/"episode", arrays=arrays,
                                         episode_end=metadata, diagnostic_frames=diagnostics)
                loaded = episode_io.load_policy_episode(output/"episode")
                if not all(np.array_equal(loaded[key], arrays[key]) for key in arrays):
                    raise CaptureFailure("written_episode_readback_mismatch")
                if episode_io.load_episode_end(output/"episode") != metadata:
                    raise CaptureFailure("written_metadata_readback_mismatch")
                journal.record("EPISODE_WRITERS_COMPLETE", policy_steps=200)
                if terminal_policy is not None:
                    result["writer_statuses"]["episode"] = "COMPLETE"
        except Exception as exc:
            result["recording_errors"].append(type(exc).__name__+":"+str(exc))
        if observation_mode:
            try:
                if rgb_writer is not None:
                    phase_record("writer_finalization", "requested", operation="rgb_writer")
                    rgb_writer.finish(bool(capture and capture.completed), result["primary_errors"])
                    phase_record("writer_finalization", "returned", operation="rgb_writer")
                    if terminal_policy is not None:
                        result["writer_statuses"]["rgb"] = "COMPLETE"
                if terminal_policy is not None and capture is None and not result["candidate_started"]:
                    guard = result.get("failure_diagnostic", {}).get("jaw_state_diagnostic", {})
                    real_reset_rejected = (guard.get("phase") == "REAL_OBSERVATION"
                        and guard.get("observation_phase") == "INITIAL_RESET"
                        and guard.get("verdict") == "REJECT_PAIRED_BITS")
                    observation_result = {"OBSERVATION_RESULT": "RULING_NEEDED" if real_reset_rejected else "INCOMPLETE",
                        "END_STATE_RESULT": "NOT_EVALUABLE", "scope": "FAILED_START",
                        "reason": (result["primary_errors"] or ["initialization_incomplete"])[0]}
                else:
                    from chicgrasp_policy.attachment_observation_adjudicate import adjudicate_observation
                    measurement = json.loads((output / "measurements.json").read_text(encoding="utf-8"))
                    observation_result = adjudicate_observation(output, result, measurement, result["experiment_result"])
                result["observation_result"] = observation_result
                write_json(output / "observation_result.json", observation_result)
            except Exception as exc:
                result["recording_errors"].append("observation_finalization:" + type(exc).__name__ + ":" + str(exc))
                phase_record("observation_finalization", "failed", **RuntimeSession.failure_details(exc))
        try:
            phase_record("scene_cleanup", "requested")
            cleanup = runtime.close_scene(capture.context if capture else None,
                **({"defer_terminal": True} if terminal_policy is not None else {}))
            phase_record("scene_cleanup", "returned")
            result["cleanup"] = cleanup
            result["cleanup_errors"].extend(cleanup.get("errors", []))
            owner_cleanup = cleanup.get("owner")
            if owner_cleanup:
                result["cleanup_errors"].extend(owner_cleanup_issues(owner_cleanup))
        except Exception as exc:
            result["cleanup_errors"].append(type(exc).__name__+":"+str(exc))
            phase_record("scene_cleanup", "failed", **RuntimeSession.failure_details(exc))
        if contact_transport == "callback-contact/v1":
            try:
                # Closure and any late teardown diagnostics are separate from
                # the frozen observation epoch; all remain before fast close.
                write_json(output/"contact_transport_final.json", runtime.export_contact_transport("final"))
            except Exception as exc:
                result["recording_errors"].append("contact_transport_final:" + type(exc).__name__ + ":" + str(exc))
        if contact_writer is not None:
            try:
                phase_record("writer_finalization", "requested", operation="compact_contact_writer")
                contact_writer.finish(primary_errors=result["primary_errors"] + result["recording_errors"])
                phase_record("writer_finalization", "returned", operation="compact_contact_writer")
                if terminal_policy is not None:
                    result["writer_statuses"]["compact_contact"] = "COMPLETE"
            except Exception as exc:
                result["recording_errors"].append("compact_contact_finalization:" + type(exc).__name__ + ":" + str(exc))
                phase_record("writer_finalization", "failed", operation="compact_contact_writer",
                             **RuntimeSession.failure_details(exc))
        # Identity checks during owned cleanup are required pre-close evidence.
        # Refresh only this diagnostic group; terminal motion/measurements above
        # remain frozen before cleanup and are never re-read from the scene.
        try:
            if "stage_identity_diagnostics" in runtime.initialization:
                before = result["initialization"].get("stage_identity_diagnostics", {})
                after = deepcopy(runtime.initialization["stage_identity_diagnostics"])
                changed = []
                for key in ("records", "errors"):
                    prefix, current = before.get(key, []), after.get(key, [])
                    if not isinstance(prefix, list) or not isinstance(current, list):
                        raise ValueError("malformed_cleanup_identity_diagnostics:" + key)
                    if current[:len(prefix)] != prefix:
                        changed.append("cleanup_changed_preexisting_identity_" + key)
                    # Never replace already journalled evidence with a mutable
                    # cleanup snapshot; only newly appended observations join it.
                    after[key] = deepcopy(prefix) + current[len(prefix):]
                after["errors"].extend(changed)
                result["initialization"]["stage_identity_diagnostics"] = after
            guard = result["initialization"].get("stage_identity_diagnostics", {})
            guard_errors = guard.get("errors", [])
            result["diagnostic_evidence_status"] = ("INCOMPLETE" if guard_errors else
                "COMPLETE" if guard.get("records") else "NOT_REACHED")
            result["recording_errors"].extend("stage_identity_diagnostic:" + str(e) for e in guard_errors)
        except Exception as exc:
            result["diagnostic_evidence_status"] = "INCOMPLETE"
            result["recording_errors"].append("stage_identity_diagnostic:" + type(exc).__name__ + ":" + str(exc))
        for key in ("phase_events", "phase_evidence_errors"):
            if key in runtime.initialization:
                result["initialization"][key] = deepcopy(runtime.initialization[key])
        result["recording_errors"].extend("failure_phase_evidence:" + error for error in
            phase_errors + runtime.initialization.get("phase_evidence_errors", []))
        if terminal_policy is not None:
            guard_state["closed"] = True
            guard_errors = list(getattr(runtime, "policy_guard_errors", []))
            result["recording_errors"].extend("policy_guard:" + error for error in guard_errors)
            result["policy_guard_observations"] = {"relative_path": "policy_guard_observations.jsonl",
                "records": guard_state["records"], "errors": guard_errors}
            result["writer_statuses"]["policy_guard"] = "COMPLETE" if guard_path.is_file() and not guard_errors else "FAILED"
            result["writer_statuses"] = {key: "FAILED" if value == "PENDING" and key not in
                ("application_artifacts", "required_journal") else value for key, value in result["writer_statuses"].items()}
        result["script_status"] = "PASS" if (result["experiment_result"].get("status") == "PASS"
            and not any(result[key] for key in ("primary_errors", "recording_errors", "cleanup_errors"))) else "FAIL"
        if observation_mode:
            result["script_status"] = "PASS" if (
                result.get("observation_result", {}).get("OBSERVATION_RESULT") == "PASS"
                and not result["recording_errors"] and not result["cleanup_errors"]) else "FAIL"
        _finalize_and_close(output, journal_path, journal, runtime, result, phase_record, phase_errors)
    return result


def _managed_record_issues(record):
    """Independently recompute the managed association from saved observations.

    Deliberately does not import/call the live validator or trust its checks.
    A faithfully recorded rejection is complete diagnostic evidence, not a
    successful scene association. The explicit-attachment getter is diagnostic.
    """
    try:
        anchor, observed = record["anchor"], record["observations"]
        construction, continuity = observed["construction"], observed["continuity"]
        cache, physics, native, reader = (observed[k] for k in
                                         ("cache", "physics_scene", "native", "reader"))
        acquisition_errors = record["acquisition_errors"]
        if (record.get("guard") != "managed_scene_identity"
                or not isinstance(record.get("mode"), str)
                or not isinstance(record.get("phase"), str) or not record["phase"]
                or record.get("context", {}).get("errors") not in (None, [])
                or not isinstance(acquisition_errors, list)
                or any(not isinstance(x, str) for x in acquisition_errors)
                or set(anchor) != {"stage_cache_id", "physics_scene_path", "robot_articulation_path",
                                   "robot_link_paths", "robot_body_names"}
                or set(construction) != {"requested_mode", "create_stage_in_memory", "sim_create_stage_in_memory"}
                or set(continuity) != {"env_sim", "physics_context", "env_scene", "robot_asset", "chicken_asset",
                    "manager_simulation_view", "robot_simulation_view", "chicken_simulation_view",
                    "robot_data_simulation_view", "robot_view", "chicken_view"}
                or set(cache["stage_ids"]) != {"constructed", "simulation_initial", "simulation_retained", "current_usd", "current_managed", "cache_lookup",
                    "retained_physics_scene", "selected_physics_scene", "robot_asset", "chicken_asset", "env_scene"}):
            return ["managed_observation_schema_invalid"]
        stage_id = anchor["stage_cache_id"]
        positive_id = type(stage_id) is int and stage_id > 0
        robot_root = "/World/envs/env_0/Robot"
        required = [robot_root + "/GripperRig/" + name for name in
                    ("gripper_base_link", "jaw_1_link", "jaw_2_link", "jaw_3_link", "jaw_4_link")]
        roots, links, names = (native[k] for k in ("robot_roots", "robot_link_paths", "robot_body_names"))
        mapping = (type(links) is list and len(links) == 1 and type(links[0]) is list
            and type(names) is list and all(type(x) is str for x in names)
            and len(names) == len(links[0]) and len(set(names)) == len(names))
        if mapping:
            mapping = all(links[0].count(path) == 1 and names[links[0].index(path)] == path.rsplit("/", 1)[1]
                          for path in required)
        reader_ok = reader.get("supplied") is False and set(reader) == {"supplied"}
        if reader.get("supplied") is True:
            index = reader.get("body_index")
            reader_ok = (all(reader.get(k) is True for k in
                ("env", "robot_asset", "chicken_asset", "robot_view", "chicken_view"))
                and type(index) is int and index >= 0 and reader.get("selected_path") == required[0]
                and type(links) is list and len(links) == 1 and type(links[0]) is list
                and index < len(links[0]) and links[0][index] == reader["selected_path"])
        checks = {
            "supported_mode": (record["mode"] == "isaaclab_usd_context"
                and construction["requested_mode"] == record["mode"]
                and construction["create_stage_in_memory"] is False
                and construction["sim_create_stage_in_memory"] is False),
            "construction_anchor": positive_id,
            "owned_object_continuity": all(x is True for x in continuity.values()),
            "cache_association": (cache["contains"] is True and positive_id
                and all(type(x) is int and x == stage_id for x in cache["stage_ids"].values())),
            "physics_scene_association": (type(anchor["physics_scene_path"]) is str
                and anchor["physics_scene_path"].startswith("/")
                and all(physics[k] == anchor["physics_scene_path"] for k in
                    ("configured_path", "sim_configured_path", "context_path", "retained_path", "selected_path"))
                and physics["retained_valid"] is True and physics["selected_valid"] is True
                and physics["retained_type"] == "PhysicsScene" and physics["selected_type"] == "PhysicsScene"),
            "native_validity": all(native[k] is True for k in
                ("simulation_valid", "robot_check", "chicken_check", "robot_initialized", "chicken_initialized")),
            "native_counts": type(native["robot_count"]) is int and native["robot_count"] == 1
                and type(native["chicken_count"]) is int and native["chicken_count"] == 1,
            "native_actor_mapping": (type(roots) is list and len(roots) == 1 and type(roots[0]) is str
                and (roots[0] == robot_root or roots[0].startswith(robot_root + "/"))
                and roots[0] == anchor["robot_articulation_path"] and mapping
                and links == anchor["robot_link_paths"] and names == anchor["robot_body_names"]
                and native["chicken_paths"] == ["/World/envs/env_0/Chicken/raw_chicken_visual/raw_chicken_visual_mesh"]),
            "reader_association": reader_ok,
            "required_observations_complete": not acquisition_errors,
        }
        expected_issues = [name for name, passed in checks.items() if not passed]
        errors = []
        if (record.get("checks") != checks or any(type(v) is not bool for v in record.get("checks", {}).values())
                or record.get("issues") != expected_issues
                or record.get("verdict") != ("REJECT" if expected_issues else "PASS")):
            errors.append("managed_association_recomputation_disagreement")
        explicit = observed["explicit_simulation"]
        if (explicit.get("source") != "omni.physx.get_physx_simulation_interface().get_attached_stage()"
                or explicit.get("phase") != record["phase"]
                or explicit.get("status") not in ("observed", "query_exception", "not_sampled_at_continuity_boundary")
                or (explicit.get("status") == "query_exception" and not isinstance(explicit.get("exception"), str))):
            errors.append("managed_explicit_diagnostic_invalid")
        return errors
    except (KeyError, TypeError, AttributeError, ValueError, IndexError):
        return ["managed_observation_schema_invalid"]


def adjudicate(attempt_root, *, campaign_ledger_path=None):
    """Saved files + independent evaluator + parent kernel observations; no native imports."""
    from chicgrasp_policy.attachment_backend_adjudicate import (
        CPU, PRODUCTION_GPU, CPU_EXPECTED, check_cpu_backend, scoped_candidate_result,
        interpret_cpu_contact)
    root = Path(attempt_root)
    from chicgrasp_policy.cpu_evidence_first_adjudicate import (
        POLICY as CPU_TERMINAL_POLICY, check_terminal_policy, failed_reset_ruling, check_policy_guard_stream,
        check_early_cpu_profile)
    saved_invocation = {}
    # Only an explicitly saved diagnostic invocation selects this narrow
    # profile. The full candidate checker and its requirements stay unchanged.
    invocation_path = root / "invocation.json"
    if invocation_path.is_file():
        try:
            saved_invocation = json.loads(invocation_path.read_text(encoding="utf-8-sig"))
        except (ValueError, OSError):
            saved_invocation = {}  # Existing full checker records malformed evidence.
        if isinstance(saved_invocation, dict) and saved_invocation.get("diagnostic_mode") == CPU_STARTUP_SHUTDOWN_ONLY:
            from chicgrasp_policy.cpu_startup_adjudicate import adjudicate_cpu_startup
            return adjudicate_cpu_startup(root)
        if isinstance(saved_invocation, dict) and saved_invocation.get("geometric_mode") is True:
            if saved_invocation.get("r4_live_point"):
                from chicgrasp_policy.r4_live_check import adjudicate as adjudicate_r4
                return adjudicate_r4(root)
            if saved_invocation.get("translated_point"):
                from chicgrasp_policy.translated_script_check import adjudicate as adjudicate_script
                return adjudicate_script(root)
            if saved_invocation.get("stage3_learned"):
                from chicgrasp_policy.stage3_check import adjudicate_stage3
                return adjudicate_stage3(root)
            if saved_invocation.get("stage0_mode"):
                from chicgrasp_policy.stage0_capture_check import adjudicate_stage0
                return adjudicate_stage0(root)
            from chicgrasp_policy.geometric_capture_check import adjudicate_slim
            return adjudicate_slim(root)
    output = root/"capture"/"experiment"
    issues, result, process = [], {}, {}
    experiment_issues = []
    diagnostic_issues = []
    diagnostic_status = "NOT_REACHED"
    end_state_result = "NOT_EVALUABLE"
    observation_result = None
    backend_report, backend_anchor = None, None
    cpu_terminal = saved_invocation.get("terminal_policy") == CPU_TERMINAL_POLICY
    terminal_report, reset_ruling = None, None
    early_profile_report = None

    def parse_json(text):
        def pairs(items):
            value = {}
            for key, item in items:
                if key in value:
                    raise ValueError("duplicate_JSON_key:" + key)
                value[key] = item
            return value
        def invalid(token):
            raise ValueError("nonfinite_JSON_literal:" + token)
        return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)

    def load_json(path):
        return parse_json(Path(path).read_text(encoding="utf-8"))

    def require(ok, issue):
        if not ok:
            issues.append(issue)

    def inspect_evidence():
        nonlocal result, diagnostic_status, end_state_result, observation_result, backend_report, backend_anchor, reset_ruling, early_profile_report
        early_pre_failed_start = False
        result = load_json(output/"result.json")
        if result.get("backend_mode", PRODUCTION_GPU) == CPU:
            # Retain a truthful configuration result even if startup produced no
            # measurements. Full invocation authentication happens below.
            backend_report = check_cpu_backend(result)
        elif result.get("backend_mode", PRODUCTION_GPU) != PRODUCTION_GPU:
            issues.append("backend_mode_unsupported")
        manifest = load_json(output/"artifact_manifest.json")
        actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
        if actual != set(manifest["files"]) | {"artifact_manifest.json"}:
            issues.append("artifact_membership_mismatch")
        for relative, expected in manifest["files"].items():
            path = (output/relative).resolve()
            if not path.is_relative_to(output.resolve()) or identity(path) != expected:
                issues.append("artifact_identity_mismatch:"+relative)
        early_profile_report = check_early_cpu_profile(root, result, saved_invocation)
        issues.extend(early_profile_report["issues"])
        journal_bytes = (root/"capture"/"structured"/"events.jsonl").read_bytes()
        require(journal_bytes.endswith(b"\n"), "terminal_journal_incomplete_line")
        events = [json.loads(line) for line in journal_bytes.decode("utf-8").splitlines()]
        if (len(events) < 2 or [e["sequence"] for e in events] != list(range(len(events)))
                or any(type(e["sequence"]) is not int or type(e.get("monotonic_ns")) is not int for e in events)
                or any(a["monotonic_ns"] > b["monotonic_ns"] for a, b in zip(events, events[1:]))
                or [e["kind"] for e in events[-2:]] != ["PRE_CLOSE_EVIDENCE_COMPLETE", "CLOSE_REQUESTED"]
                or any(e.get("manifest") != identity(output/"artifact_manifest.json") for e in events[-2:])):
            issues.append("pre_close_terminal_evidence_invalid")
        if result.get("script_status") != "PASS":
            issues.append("saved_script_or_integrity_failure")
            experiment_issues.append("saved_script_or_integrity_failure")
        if result.get("evidence_integrity") != "COMPLETE":
            issues.append("saved_integrity_failure")
        require(all(isinstance(result.get(key), list) for key in ("primary_errors", "recording_errors", "cleanup_errors")),
                "saved_error_groups_missing_or_malformed")
        if any(result.get(k) for k in ("primary_errors", "recording_errors", "cleanup_errors")):
            issues.append("saved_errors_not_erased_by_OS_zero")
            experiment_issues.append("saved_errors_not_erased_by_OS_zero")
        if cpu_terminal:
            policy_guards = check_policy_guard_stream(output, result)
            issues.extend(policy_guards["issues"])
        if cpu_terminal and result.get("candidate_started") is False and result.get("failure_phase") == "initialization":
            phases = [parse_json(line) for line in (root/"capture"/"structured"/"failure_phases.jsonl").read_text(encoding="utf-8").splitlines()]
            reset_ruling = failed_reset_ruling(result, phases)
            issues.extend(reset_ruling["issues"])
            saved = load_json(output/"measurements.json")
            require(all(saved.get(key) == [] for key in ("rows", "command_frames", "sensor_samples", "lifecycle_events")),
                    "failed_reset_measurement_or_candidate_present")
            require(load_json(output/"candidate_result.json") == result.get("experiment_result"),
                    "failed_reset_candidate_result_disagreement")
            require(not (output/"pre_candidate.json").exists() and not (output/"episode").exists(),
                    "failed_reset_candidate_artifact_present")
            guard_records = [parse_json(line) for line in (output/"policy_guard_observations.jsonl").read_text(encoding="utf-8").splitlines()]
            failure_guard = result.get("failure_diagnostic", {}).get("jaw_state_diagnostic", {})
            early_pre_failed_start = (early_profile_report.get("declared") is True
                and failure_guard.get("observation_phase") == "PRE_CANDIDATE")
            rejected = [record for record in guard_records if record.get("phase") == "REAL_OBSERVATION"
                        and record.get("observation_phase") == ("PRE_CANDIDATE" if early_pre_failed_start else "INITIAL_RESET")
                        and record.get("verdict") == "REJECT_PAIRED_BITS"]
            require(len(rejected) == 1 and all(rejected[0].get(key) == value for key, value in failure_guard.items()),
                    "failed_reset_guard_stream_disagreement")
            diagnostic_status = "COMPLETE" if not reset_ruling["issues"] else "INCOMPLETE"
            if not early_pre_failed_start:
                return
        guard = result.get("initialization", {}).get("stage_identity_diagnostics")
        managed_anchor = None
        managed = []
        if guard is not None:
            records, errors = guard.get("records"), guard.get("errors")
            journal_records = [e.get("record") for e in events if e.get("kind") == "GUARD_COMPARISON"]
            if not isinstance(records, list) or not isinstance(errors, list):
                diagnostic_issues.append("guard_diagnostic_groups_invalid")
            else:
                if records != journal_records:
                    diagnostic_issues.append("guard_journal_result_disagreement")
                if errors:
                    diagnostic_issues.append("guard_diagnostic_recording_failure")
                for index, record in enumerate(records):
                    operands, context = record.get("operands", {}), record.get("context", {})
                    common_invalid = (type(record.get("sequence")) is not int or record["sequence"] != index
                            or type(record.get("monotonic_ns")) is not int
                            or context != dict(context, executed_commit=result.get("execution_provenance", {}).get("executed_commit"),
                                               attempt_id=result.get("execution_provenance", {}).get("attempt_id")))
                    if record.get("guard") == "managed_scene_identity":
                        failures = _managed_record_issues(record)
                        if common_invalid:
                            failures.append("managed_diagnostic_provenance_invalid")
                        diagnostic_issues.extend(failure + ":" + str(index) for failure in failures)
                    elif (common_invalid or record.get("guard") != "native_stage_identity_mismatch"
                            or record.get("verdict") not in ("PASS", "REJECT", "COMPARISON_ERROR")
                            or set(operands) != {"cache_id_wrapper", "cached_stage_id", "attached_stage_id"}
                            or any(not isinstance(item, dict) or item.get("availability") not in
                                   ("captured", "not_evaluated") or not item.get("source")
                                   for item in operands.values())):
                        diagnostic_issues.append("guard_diagnostic_record_invalid:" + str(index))
                managed = [record for record in records if record.get("guard") == "managed_scene_identity"]
                if managed:
                    managed_anchor = managed[0].get("anchor", {}).get("stage_cache_id")
                if managed and any(record.get("anchor") != managed[0].get("anchor") for record in managed):
                    diagnostic_issues.append("managed_construction_anchor_changed")
                continuity = guard.get("continuity")
                if continuity is not None:
                    counters = [continuity.get(k) for k in ("validations", "passes", "rejections")]
                    if (any(type(v) is not int or v < 0 for v in counters)
                            or counters[0] != counters[1] + counters[2] or counters[0] < len(managed)
                            or not isinstance(continuity.get("latest"), dict)):
                        diagnostic_issues.append("managed_continuity_accounting_invalid")
                    else:
                        latest = continuity["latest"]
                        diagnostic_issues.extend(failure + ":latest" for failure in _managed_record_issues(latest))
                        if managed and latest.get("anchor") != managed[0].get("anchor"):
                            diagnostic_issues.append("managed_latest_anchor_changed")
                        if (latest.get("context", {}).get("executed_commit") != result.get("execution_provenance", {}).get("executed_commit")
                                or latest.get("context", {}).get("attempt_id") != result.get("execution_provenance", {}).get("attempt_id")
                                or type(latest.get("monotonic_ns")) is not int):
                            diagnostic_issues.append("managed_latest_provenance_invalid")
                if ((result.get("execution_provenance", {}).get("attempt_id") == "attempt004" or result.get("observation_mode") is True)
                        and (result.get("candidate_started") is True or result.get("attachment_authoring_attempted") is True)):
                    required_phases = {"managed_bind", "after_initial_reset", "jaw_bind", "owner_bind", "pre_cleanup"}
                    phase_complete = required_phases.issubset({record.get("phase") for record in managed})
                    if not managed or (result.get("script_status") == "PASS" and
                            (not phase_complete or continuity is None)):
                        diagnostic_issues.append("managed_candidate_phase_association_unproven")
                    elif not phase_complete:
                        issues.append("managed_candidate_phase_association_not_completed")
                        experiment_issues.append("managed_candidate_phase_association_not_completed")
                if (any(record.get("verdict") == "REJECT" for record in managed)
                        or (continuity is not None and continuity.get("rejections", 0) != 0)):
                    issues.append("managed_scene_association_rejected")
                    experiment_issues.append("managed_scene_association_rejected")
                if result.get("observation_mode") is True:
                    required_phases = {"managed_bind", "after_initial_reset", "jaw_bind", "owner_bind", "pre_cleanup"}
                    require(bool(managed) and required_phases.issubset({r.get("phase") for r in managed})
                        and all(r.get("verdict") == "PASS" for r in managed)
                        and isinstance(continuity, dict) and continuity.get("rejections") == 0,
                        "observation_managed_association_incomplete_or_rejected")
                if "ValueError:native_stage_identity_mismatch" in result.get("primary_errors", []) and not records:
                    diagnostic_issues.append("guard_failure_missing_comparison_record")
                if (any(str(error).startswith("ValueError:managed_scene_identity_invalid:")
                        for error in result.get("primary_errors", [])) and not managed):
                    diagnostic_issues.append("managed_failure_missing_comparison_record")
            diagnostic_status = "INCOMPLETE" if diagnostic_issues else ("COMPLETE" if records else "NOT_REACHED")
            if result.get("diagnostic_evidence_status") != diagnostic_status:
                diagnostic_issues.append("guard_diagnostic_status_disagreement")
                diagnostic_status = "INCOMPLETE"
            issues.extend(diagnostic_issues)
        elif ((result.get("execution_provenance", {}).get("attempt_id") == "attempt004" or result.get("observation_mode") is True)
                and (result.get("candidate_started") is True or result.get("attachment_authoring_attempted") is True)):
            diagnostic_status = "INCOMPLETE"
            diagnostic_issues.append("managed_candidate_phase_association_unproven")
            issues.extend(diagnostic_issues)
        if result.get("backend_mode", PRODUCTION_GPU) == CPU:
            backend_anchor = managed[0].get("anchor") if managed else None
            backend_report = check_cpu_backend(result, managed_anchor=backend_anchor)
            require(backend_report["BACKEND_DIAGNOSTIC_RESULT"] == "CPU_CONFIGURATION_VERIFIED",
                    "backend_configuration:" + backend_report["BACKEND_DIAGNOSTIC_RESULT"])
            require(backend_anchor is not None, "backend_managed_scene_association_unproven")
        if early_pre_failed_start:
            # PRE is after reset/full-profile verification. The new partial
            # evidence cannot bypass those unchanged managed/backend checks.
            return
        saved = load_json(output/"measurements.json")
        from chicgrasp_policy.compact_contact_replay import CompactReplay, storage_selected
        compact_storage = storage_selected(result, saved)
        compact_replay = CompactReplay(output/"contact_compact", require_final=True) if compact_storage else None
        if compact_storage:
            require(compact_replay.reader.manifest.get("storage_errors") == [],
                    "compact_storage_reported_errors")
            require(compact_replay.reader.manifest.get("primary_errors") ==
                    result.get("primary_errors", []) + result.get("recording_errors", []),
                    "compact_storage_saved_error_disagreement")
            from chicgrasp_policy.compact_contact_storage_budget import validate_output_allowances
            issues.extend(validate_output_allowances(root))
        require(saved["episode_id"] == result.get("episode_id"), "saved_episode_identity_disagreement")
        require(load_json(output/"candidate_result.json") == result.get("experiment_result"),
                "candidate_result_artifact_disagreement")
        if result.get("observation_mode") is True:
            from chicgrasp_policy.attachment_observation_adjudicate import (
                adjudicate_observation, validate_callback_lifecycle)
            observation_result = adjudicate_observation(output, result, saved, result["experiment_result"],
                                                        compact_require_final=compact_storage)
            if result.get("contact_transport") == "callback-contact/v1":
                before_transport = load_json(output/"contact_transport_before_cleanup.json")
                final_transport = load_json(output/"contact_transport_final.json")
                if compact_storage:
                    compact_replay.reader.verify_record_references(
                        [row["contact"]["reference"] for row in saved["rows"]] +
                        [before_transport["reference"], final_transport["reference"]])
                    before_transport = compact_replay.lifecycle(before_transport)
                    final_transport = compact_replay.lifecycle(final_transport)
                before_collector = validate_callback_lifecycle(before_transport, closed=False)
                final_collector = validate_callback_lifecycle(final_transport, closed=True)
                require(all(before_collector.get(key) == final_collector.get(key) for key in
                    ("subscription_generation", "epoch", "stage_id", "association", "limits"))
                    and final_collector["events"][:len(before_collector["events"])] == before_collector["events"],
                    "callback_cleanup_changed_frozen_lifecycle_evidence")
                for key in ("initialization_callbacks", "initialization_stage_association"):
                    original = {item["sequence"]: item for item in before_collector[key]}
                    retained = {item["sequence"]: item for item in final_collector[key]}
                    require(all(retained.get(sequence) == item for sequence, item in original.items()),
                        "callback_cleanup_changed_frozen_initialization_record:" + key)
                require(before_transport.get("initialization_routing") == final_transport.get("initialization_routing"),
                    "callback_cleanup_changed_initialization_routing")
                require(all(type(final_collector["totals"].get(key)) is int
                    and final_collector["totals"][key] >= value
                    and final_collector["totals"][key] - value ==
                        final_collector["undrained_unassigned_counts"].get(key, -1) -
                        before_collector["undrained_unassigned_counts"].get(key, -1)
                        + final_collector["initialization_counts"].get(key, -1) -
                        before_collector["initialization_counts"].get(key, -1)
                    for key, value in before_collector["totals"].items()),
                    "callback_final_counts_regressed_missing_or_unreconciled")
                require(final_transport.get("runtime", {}).get("errors") == []
                    and final_transport["runtime"].get("locally_omitted_metadata") == 0,
                    "callback_runtime_cleanup_or_metadata_failure")
            require(load_json(output/"pre_candidate.json").get("managed_stage") == managed[0].get("anchor")
                if managed else False, "observation_pre_candidate_anchor_disagreement")
            require(observation_result == result.get("observation_result")
                and observation_result == load_json(output/"observation_result.json"),
                "observation_independent_result_disagreement")
            require(observation_result.get("OBSERVATION_RESULT") == "PASS", "observation_incomplete")
            require(result.get("recording_errors") == [] and result.get("cleanup_errors") == [],
                    "observation_recording_or_cleanup_failure")
            require(result.get("candidate_started") is True
                and result.get("initialization", {}).get("initial_reset_returned") is True
                and saved.get("lifecycle_events") == [], "observation_candidate_reset_or_lifecycle_invalid")
            cfg = result.get("effective_configuration", {})
            expected = {"task_id": TASK_ID, "cfg_class": "ChicgraspDynamicChickenEnvCfg", "num_envs": 1,
                "device": "cuda:0", "simulation_device": "cuda:0", "physics_device": "cuda:0",
                "gpu_sim": True, "gpu_pipeline": True, "gpu_dynamics": True, "broadphase": "GPU",
                "decimation": 2, "render_interval": 2, "action_dimension": 5, "cameras_enabled": True,
                "offscreen_render": True, "rtx_sensors": True, "fast_shutdown": True, "has_gui": False,
                "configured_seed": 42}
            if result.get("backend_mode", PRODUCTION_GPU) == CPU:
                expected = CPU_EXPECTED
            require(all(type(cfg.get(k)) is type(v) and cfg[k] == v for k, v in expected.items()),
                    "observation_effective_configuration_invalid")
            require(all(type(cfg.get(k)) in (int, float) and math.isfinite(cfg[k])
                and math.isclose(cfg[k], v, rel_tol=0, abs_tol=1e-12)
                for k, v in (("physics_dt_s", 1/120), ("control_dt_s", 1/60))),
                "observation_scheduled_timing_invalid")
            cleanup = result.get("cleanup", {})
            owner = cleanup.get("owner", {})
            require(cleanup.get("errors") == [] and (cleanup.get("managed_environment_closed") is True
                or (cpu_terminal and cleanup.get("terminal_environment_deferred") is True))
                and not owner_cleanup_issues(owner)
                and (owner.get("observer_unsubscribed") is True or "observer_not_available" in owner),
                "observation_managed_cleanup_unqualified")
        # Complete evidence of a failed/partial experiment is not corruption.
        # Do not fabricate the missing candidate/episode after startup failure.
        full_counts = {"completed_control_ticks":1200, "observed_substeps":2400, "completed_policy_actions":200}
        if ((result.get("script_status") != "PASS" or result.get("observation_mode") is True) and result.get("experiment_result", {}).get("status") != "PASS"
                and result.get("counts") != full_counts):
            experiment_issues.append("experiment_did_not_complete_successfully")
            return
        require(result.get("candidate_started") is True, "candidate_execution_not_recorded")
        require(result.get("counts") == {"completed_control_ticks": 1200, "observed_substeps": 2400,
                                        "completed_policy_actions": 200}, "frozen_execution_counts_disagreement")
        require(saved.get("lifecycle_events") == [], "unexpected_saved_lifecycle_event")
        config = result.get("effective_configuration", {})
        expected_config = {"task_id": TASK_ID, "cfg_class": "ChicgraspDynamicChickenEnvCfg", "num_envs": 1,
            "device": "cuda:0", "simulation_device": "cuda:0", "physics_device": "cuda:0",
            "gpu_sim": True, "gpu_pipeline": True, "gpu_dynamics": True, "broadphase": "GPU",
            "decimation": 2, "render_interval": 2, "action_dimension": 5, "cameras_enabled": True,
            "offscreen_render": True, "rtx_sensors": True, "fast_shutdown": True, "has_gui": False,
            "configured_seed": 42}
        if result.get("backend_mode", PRODUCTION_GPU) == CPU:
            expected_config = CPU_EXPECTED
        require(isinstance(config, dict) and all(type(config.get(key)) is type(value) and config[key] == value
                for key, value in expected_config.items()), "effective_configuration_evidence_invalid")
        require(all(isinstance(config.get(key), (int, float)) and not isinstance(config[key], bool)
                and math.isfinite(config[key]) and math.isclose(config[key], expected, rel_tol=0, abs_tol=1e-12)
                for key, expected in (("physics_dt_s", 1/120), ("control_dt_s", 1/60))),
                "effective_scheduled_timing_evidence_invalid")
        cleanup = result.get("cleanup", {})
        owner_cleanup = cleanup.get("owner", {}) if isinstance(cleanup, dict) else {}
        removal = owner_cleanup.get("cleanup", {}) if isinstance(owner_cleanup, dict) else {}
        require(cleanup.get("errors") == [] and (cleanup.get("managed_environment_closed") is True
                or (cpu_terminal and cleanup.get("terminal_environment_deferred") is True))
                and owner_cleanup.get("issues") == [] and removal.get("cleanup_confirmed") is True
                and removal.get("issues") == [] and removal.get("lifecycle_recorded") is True
                and removal.get("lifecycle_issues") == []
                and (owner_cleanup.get("observer_unsubscribed") is True or "observer_not_available" in owner_cleanup),
                "saved_managed_cleanup_not_qualified")
        evaluated = metrics.evaluate_end_state(saved["rows"], episode_id=saved["episode_id"],
            origin_step=saved["origin_clock"][0], origin_time=saved["origin_clock"][1])
        if result.get("backend_mode", PRODUCTION_GPU) == CPU:
            evaluated = scoped_candidate_result(evaluated)
        end_state_result = evaluated.get("status", "NOT_EVALUABLE")
        if evaluated != result["experiment_result"]:
            issues.append("independent_end_state_failure_or_disagreement")
        if evaluated.get("status") != "PASS":
            issues.append("independent_end_state_failure")
            experiment_issues.append("independent_end_state_failure")
        require(load_json(output/"candidate_result.json") == evaluated, "candidate_result_artifact_disagreement")
        # A valid manifest establishes bytes, not the truth of cached predicate,
        # command-history or success fields. Reproduce them from saved inputs.
        rows, origin = saved["rows"], tuple(saved["origin_clock"])
        actions = metrics.frozen_candidate_actions()
        frames, sensors = saved.get("command_frames"), saved.get("sensor_samples")
        require(isinstance(frames, list) and len(frames) == 200, "command_history_count_invalid")
        require(isinstance(sensors, list) and len(sensors) == 200, "sensor_history_count_invalid")
        if not isinstance(frames, list) or not isinstance(sensors, list) or len(frames) != 200 or len(sensors) != 200:
            raise ValueError("complete_command_and_sensor_history_required")
        previous_sensor_frames, previous_sensor_wall, previous_row_wall = None, None, None
        recomputed, pending_indices = [], []
        for index, row in enumerate(rows):
            if compact_storage:
                row = dict(row, contact=compact_replay.read(row["contact"], expected_predicate=row["predicate"]))
            clock = tuple(row["pose"]["clock_after"])
            context = row["context"]
            predicate = metrics.bilateral_contact(row.get("contact"), saved.get("coverage"),
                episode_id=saved["episode_id"], context=context, native_clock=clock)
            recomputed.append(predicate)
            require(predicate == row.get("predicate"), "raw_contact_predicate_disagreement:" + str(index))
            if ((result.get("execution_provenance", {}).get("attempt_id") == "attempt004" or result.get("observation_mode") is True)
                    and predicate["status"] != "UNAVAILABLE"):
                # Available contact diagnostics must be associated with the
                # same constructed managed stage, not a self-consistent foreign
                # stage ID. Unavailable held-contact remains diagnostic only.
                reading = row["contact"]["reading"]
                require(type(managed_anchor) is int and managed_anchor > 0
                    and reading.get("stage_id") == managed_anchor
                    and all(item.get("stage_identity_domain") == "constructed_managed_USD_stage_cache"
                            and item.get("managed_stage_cache_id") == managed_anchor
                            for item in (reading["health"]["before"], reading["health"]["after"])),
                    "contact_managed_construction_anchor_disagreement:" + str(index))
            state = row["attachment"]["status"]
            if state == "PENDING":
                pending_indices.append(index)
                require(predicate["status"] == "TRUE", "attachment_requested_without_current_bilateral_contact")
                action_index = context["control_tick"] // 6
                require(action_index >= 2 and all(f["commands"] == [0.0, 0.0]
                    for f in frames[action_index-2:action_index+1]), "attachment_request_without_three_closed_frames")
            # Contact remains required before authoring during closing. The
            # PENDING request row above requires TRUE; subsequent confirmation
            # and held rows retain contact diagnostics without making them gates.
            if (context["control_tick"] >= 720 and state == "UNATTACHED"
                    and result.get("contact_transport", "polling-contact/v1") != "callback-contact/v1"):
                require(predicate["status"] != "UNAVAILABLE", "required_contact_unavailable:" + str(index))
            wall = row.get("wall_monotonic_ns")
            require(type(wall) is int and wall >= 0 and (previous_row_wall is None or wall >= previous_row_wall),
                    "scheduled_wall_observation_invalid:" + str(index))
            previous_row_wall = wall
        require(len(pending_indices) == 1, "one_genuine_attachment_request_required")
        qualifying = [index for index, row in enumerate(rows)
            if 732 <= row["context"]["control_tick"] <= 839 and index <= 1678
            and recomputed[index]["status"] == "TRUE"
            and all(frame["commands"] == [0.0, 0.0] for frame in
                    frames[row["context"]["control_tick"]//6-2:row["context"]["control_tick"]//6+1])]
        require(bool(qualifying) and pending_indices == qualifying[:1],
                "request_not_first_qualifying_scheduled_opportunity")
        for index, (frame, sensor) in enumerate(zip(frames, sensors)):
            row = rows[index*12]
            expected_sample = dict(row["context"], physics_step=row["pose"]["clock_after"][0],
                                   simulation_time_s=row["pose"]["clock_after"][1])
            require(type(frame.get("action_frame")) is int and frame["action_frame"] == index
                and frame.get("commands") == actions[index, 3:].tolist()
                and frame.get("sample") == expected_sample
                and frame.get("source") == "executed_candidate_command_first_post_substep/v1",
                "executed_command_provenance_disagreement:" + str(index))
            pre_clock = origin if index == 0 else tuple(rows[index*12-1]["pose"]["clock_after"])
            require(type(sensor.get("action_frame")) is int and sensor["action_frame"] == index
                and tuple(sensor.get("native_clock", ())) == pre_clock
                and type(sensor.get("policy_timestamp_ns")) is int
                and sensor["policy_timestamp_ns"] == index*episode_io.TIMESTAMP_PERIOD_NS,
                "preaction_sensor_clock_disagreement:" + str(index))
            sensor_frames, wall = sensor.get("sensor_frames"), sensor.get("wall_monotonic_ns")
            valid_frames = (isinstance(sensor_frames, dict) and set(sensor_frames) == {"camera_left", "camera_right"}
                and all(type(value) is int and value >= 0 for value in sensor_frames.values()))
            if valid_frames and previous_sensor_frames is not None:
                valid_frames = all(sensor_frames[name] > previous_sensor_frames[name] for name in sensor_frames)
            require(valid_frames, "camera_frame_provenance_invalid:" + str(index))
            require(type(wall) is int and wall >= 0 and (previous_sensor_wall is None or wall > previous_sensor_wall)
                and wall <= row["wall_monotonic_ns"]
                and (index == 0 or wall >= rows[index*12-1]["wall_monotonic_ns"]),
                "preaction_sensor_wall_order_invalid:" + str(index))
            previous_sensor_frames, previous_sensor_wall = sensor_frames, wall
        owner_saved = load_json(output/"owner_before_cleanup.json")
        journal = owner_saved.get("event_evidence", {})
        records = journal.get("records", [])
        require(all(record.get("issues") == [] and record.get("kind") not in
            ("release", "reset", "truncation", "termination", "teardown", "invalid", "attachment_request_failure")
            for record in records), "owner_in_window_event_failure_or_interruption")
        requests = [r for r in records if r.get("kind") == "activation_request"]
        outcomes = [r for r in records if r.get("kind") == "scheduled_attachment_outcome"]
        require(journal.get("provenance_domain") == "capture" and journal.get("episode_id") == saved["episode_id"]
            and journal.get("origin_step") == origin[0] and journal.get("origin_time_s") == origin[1]
            and owner_saved.get("held") is True and len(requests) == len(outcomes) == 1,
            "real_owner_capture_provenance_missing")
        if len(pending_indices) == len(requests) == len(outcomes) == 1:
            index, request, outcome = pending_indices[0], requests[0], outcomes[0]
            row, after = rows[index], rows[index+1]
            action_index = row["context"]["control_tick"]//6
            expected_runtime = dict(row["runtime"], source="scheduled_pre_reset_boundary/v1",
                sample=dict(row["context"], physics_step=row["pose"]["clock_after"][0],
                            simulation_time_s=row["pose"]["clock_after"][1]))
            require(request.get("context") == row["context"] and request.get("issues") == []
                and request.get("eligibility_issues") == []
                and request.get("pose_reading") == row["attachment"].get("request_pose_reading")
                and request.get("eligibility") == {"command_frames": frames[action_index-2:action_index+1],
                    "bilateral_contact": recomputed[index], "runtime": expected_runtime},
                "owner_request_disagrees_with_actual_capture_inputs")
            require(outcome.get("request_id") == request.get("sequence") and outcome.get("issues") == []
                and outcome.get("scheduled_confirmation") is True and outcome.get("confirmed_creation") is False
                and outcome.get("context") == after["context"]
                and outcome.get("native_clock") == after["pose"]["clock_after"],
                "owner_scheduled_outcome_disagreement")
        arrays = episode_io.load_policy_episode(output/"episode")
        if not np.array_equal(arrays["action"], actions):
            issues.append("frozen_candidate_action_disagreement")
        metadata = episode_io.load_episode_end(output/"episode")
        require(metadata["episode_id"] == saved["episode_id"] and metadata["task_id"] == TASK_ID
                and metadata["seed"] == 42 and metadata["num_policy_steps"] == 200,
                "episode_terminal_metadata_disagreement")
        diagnostics = episode_io.load_diagnostic_sidecar(output/"episode")
        terminal_label = (metrics.PASS_LABEL if evaluated.get("status") == "PASS"
                          else "ATTACHMENT_ASSISTED_CANDIDATE_FAIL")
        if result.get("backend_mode", PRODUCTION_GPU) == CPU:
            terminal_label = CPU + ":" + terminal_label
        require(len(diagnostics) == 200 and diagnostics[-1].outcome_label == terminal_label,
                "episode_terminal_classification_disagreement")
        require(np.array_equal(arrays["timestamp_ns"], np.arange(200, dtype=np.int64)*episode_io.TIMESTAMP_PERIOD_NS),
                "policy_timestamp_schedule_disagreement")
        for index, frame in enumerate(diagnostics):
            require(frame.reset is False and frame.automatic_reset is False and frame.invalid_policy_frame is False
                    and frame.truncated is False and frame.terminated is False,
                    "ineligible_episode_diagnostic_frame:" + str(index))
            require(tuple(frame.requested_tcp_pose_base_wxyz[:3]) == tuple(float(v) for v in actions[index,:3]),
                    "diagnostic_requested_target_disagreement:" + str(index))
    try:
        inspect_evidence()
    except Exception as exc:
        issues.append("evidence_unavailable:"+type(exc).__name__+":"+str(exc))
    # Optional for historical packages and never proof of native completion.
    # A saved finalization failure cannot be erased by a finalized PASS package
    # followed by supported nonreturning fast shutdown and OS exit zero.
    phase_path = root/"capture"/"structured"/"failure_phases.jsonl"
    phase_required = isinstance(result, dict) and "failure_phase_evidence" in result
    if phase_required:
        declared = result["failure_phase_evidence"]
        require(isinstance(declared, dict) and declared.get("schema") == "capture-failure-phases/v1"
                and isinstance(declared.get("path"), str)
                and declared["path"] == str(phase_path.resolve()),
                "required_failure_phase_declaration_invalid")
        require(phase_path.exists(), "required_failure_phase_evidence_missing")
    if phase_path.exists() or phase_path.is_symlink():
        try:
            if phase_path.is_symlink() or not phase_path.is_file() or phase_path.stat().st_size > 8388608:
                raise ValueError("missing_linked_or_oversized_phase_file")
            phase_bytes = phase_path.read_bytes()
            if not phase_bytes.endswith(b"\n"):
                raise ValueError("incomplete_phase_line")
            phase_records = [parse_json(line) for line in phase_bytes.decode("utf-8").splitlines()]
            for index, record in enumerate(phase_records):
                if (not isinstance(record, dict) or type(record.get("sequence")) is not int
                        or record["sequence"] != index or type(record.get("monotonic_ns")) is not int
                        or record["monotonic_ns"] < 0 or not isinstance(record.get("phase"), str)
                        or (record.get("event") not in ("requested", "returned", "failed", "available")
                            and not (early_profile_report is not None and early_profile_report.get("declared")
                                and record.get("phase") == "early_cpu_profile"
                                and record.get("event") in ("persisted", "persistence_failed")))):
                    raise ValueError("malformed_phase_record")
                errors = record.get("phase_evidence_errors", [])
                if not isinstance(errors, list) or any(not isinstance(error, str) for error in errors):
                    raise ValueError("malformed_phase_errors")
                if (record["phase"] == "required_finalization" and record["event"] == "failed") or errors:
                    require(False, "saved_pre_close_finalization_failure")
                if record["phase"] in ("application_close", "optional_close_evidence", "terminal_environment_close") and record["event"] == "failed":
                    require(False, "saved_close_or_close_evidence_failure")
            if phase_required:
                requests = [record for record in phase_records
                            if record["phase"] == "runtime_close_app" and record["event"] == "requested"]
                if cpu_terminal and result.get("cleanup", {}).get("terminal_environment_deferred") is True:
                    environment_requests = [record for record in phase_records
                        if record["phase"] == "terminal_environment_close" and record["event"] == "requested"]
                    require(len(environment_requests) == 1, "required_terminal_environment_phase_missing_or_invalid")
                    if requests:
                        require(len(requests) == 1 and type(requests[0].get("app_handle_available")) is bool
                                and isinstance(requests[0].get("phase_evidence_errors"), list),
                                "required_pre_close_phase_missing_or_invalid")
                else:
                    require(len(requests) == 1 and type(requests[0].get("app_handle_available")) is bool
                            and isinstance(requests[0].get("phase_evidence_errors"), list),
                            "required_pre_close_phase_missing_or_invalid")
        except Exception as exc:
            issues.append("failure_phase_evidence_unavailable:" + type(exc).__name__ + ":" + str(exc))
    try:
        process = load_json(root/"capture"/"process_result.json")
        checks = {"exit_code": 0, "root_exit_verified": True, "tree_empty_verified": True,
                  "capture_complete": True, "forced_termination": False, "timed_out": False,
                  "parent_termination_requested": False, "kill_on_job_close_used": False,
                  "kill_on_job_close_possible": False, "startup_containment_used": False,
                  "autonomous_tree_exit": True, "job_assigned_before_resume": True, "root_resumed": True,
                  "job_disposed_after_observation": True, "job_members_before_disposal": 0,
                  "absolute_deadline_ms": 1200000, "parent_launch_boundary_ms": 0}
        if not cpu_terminal:
            for key, value in checks.items():
                if type(process.get(key)) is not type(value) or process[key] != value:
                    issues.append("process_completion_failed:"+key)
            if process.get("containment_errors") != []:
                issues.append("containment_errors_or_missing")
            for key in ("root_exit_observed_ms", "job_empty_observed_ms", "drain_completed_observed_ms"):
                require(type(process.get(key)) is int and 0 <= process[key] <= 1200000,
                        "process_completion_deadline_unproven:" + key)
        for field, relative in (("stdout", "stdout.bin"), ("stderr", "stderr.bin"),
                                ("completed_journal", "structured/events.jsonl")):
            path = root/"capture"/relative
            expected = process.get(field)
            require(isinstance(expected, dict) and Path(expected.get("path", "")).resolve() == path.resolve()
                and {k: expected.get(k) for k in ("bytes", "sha256")} == identity(path),
                "process_raw_capture_identity_mismatch:" + field)
        require(process.get("invocation_sha256") == identity(root/"invocation.json")["sha256"],
                "process_invocation_identity_mismatch")
        invocation = load_json(root/"invocation.json")
        if cpu_terminal:
            terminal_report = check_terminal_policy(root, result, invocation, process)
            issues.extend("process_" + issue for issue in terminal_report["issues"])
            issues.extend(terminal_report["evidence_issues"])
        else:
            require(result.get("terminal_policy") in (None, ""), "process_terminal_policy_not_explicit")
        selected_backend = result.get("backend_mode", PRODUCTION_GPU)
        requested_backend = invocation.get("backend_mode", PRODUCTION_GPU)
        require(selected_backend in (PRODUCTION_GPU, CPU) and requested_backend == selected_backend,
                "backend_request_result_disagreement")
        backend_arguments = invocation.get("arguments", [])
        backend_argument_value = None
        if isinstance(backend_arguments, list):
            indices = [i for i, item in enumerate(backend_arguments) if item == "--backend-mode"]
            if indices and indices[0] + 1 < len(backend_arguments):
                backend_argument_value = backend_arguments[indices[0] + 1]
            require(not any(isinstance(item, str) and item.startswith("--backend-mode=")
                            for item in backend_arguments)
                    and (not indices or (len(indices) == 1 and backend_argument_value == requested_backend)),
                    "backend_invocation_argument_disagreement")
        else:
            require(False, "backend_invocation_arguments_malformed")
        if CPU in (selected_backend, requested_backend, backend_argument_value):
            backend_report = check_cpu_backend(result, invocation=invocation, managed_anchor=backend_anchor)
            if selected_backend != requested_backend or (backend_argument_value is not None
                                                        and backend_argument_value != selected_backend):
                backend_report["contradictions"].append("invocation.result_backend_mode_disagreement")
                backend_report["BACKEND_DIAGNOSTIC_RESULT"] = "CPU_CONFIGURATION_CONTRADICTED"
            if reset_ruling is None:
                require(backend_report["BACKEND_DIAGNOSTIC_RESULT"] == "CPU_CONFIGURATION_VERIFIED",
                        "backend_configuration:" + backend_report["BACKEND_DIAGNOSTIC_RESULT"])
        require(invocation.get("kind") == "attachment_capture" and invocation.get("execution_approved") is True
            and invocation.get("timeout_ms") == 1200000 and Path(invocation["output_directory"]).resolve() == (root/"capture").resolve(),
            "process_invocation_contract_invalid")
        if result.get("observation_mode") is True or invocation.get("observation_mode") is True:
            require(result.get("observation_mode") is True and invocation.get("observation_mode") is True
                and invocation.get("campaign") == ("cpu-phase-observation/v1" if cpu_terminal else "observation-v1")
                and "--observation-mode" in invocation.get("arguments", []),
                "process_observation_mode_contract_invalid")
        selected_transport = result.get("contact_transport", "polling-contact/v1")
        require(selected_transport in ("polling-contact/v1", "callback-contact/v1")
            and invocation.get("contact_transport", "polling-contact/v1") == selected_transport,
            "process_contact_transport_disagreement")
        if selected_transport == "callback-contact/v1":
            arguments = invocation.get("arguments", [])
            require(result.get("observation_mode") is True
                and arguments.count("--contact-transport") == 1
                and arguments.index("--contact-transport") + 1 < len(arguments)
                and arguments[arguments.index("--contact-transport") + 1] == selected_transport,
                "process_callback_transport_not_explicitly_selected")
    except Exception as exc:
        issues.append("process_evidence_unavailable:"+repr(exc))
    process_safe = not any(i.startswith(("process_", "containment_")) for i in issues)
    integrity_issues = [item for item in issues if item not in experiment_issues
        and (not item.startswith(("process_", "containment_"))
             or item.startswith(("process_raw_capture_", "process_invocation_identity_")))]
    observation_pass = (result.get("observation_mode") is True and observation_result is not None
        and observation_result.get("OBSERVATION_RESULT") == "PASS" and not integrity_issues and process_safe)
    if cpu_terminal:
        observation_pass = (observation_pass and terminal_report is not None
            and terminal_report["terminal_exception_eligible"]
            and terminal_report["observation_window_complete"])
    answer = {"schema": "capture-parent-adjudication/v1", "status": "PASS" if observation_pass or (not issues and not experiment_issues) else "FAIL",
        "issues": issues, "experiment_result": result.get("experiment_result"),
        "END_STATE_RESULT": end_state_result,
        "OBSERVATION_RESULT": observation_result.get("OBSERVATION_RESULT") if observation_result else "NOT_REQUESTED",
        "CONTACT_CONTROL_RESULT": observation_result.get("CONTACT_CONTROL_RESULT") if observation_result else "NOT_REQUESTED",
        "observation_result": observation_result,
        "CAPTURE_INTEGRITY": "PASS" if not integrity_issues else "FAIL",
        "PROCESS_COMPLETION": "PASS" if process_safe else "FAIL",
        "diagnostic_evidence_status": diagnostic_status, "diagnostic_issues": diagnostic_issues,
        "evidence_integrity": "PASS" if not integrity_issues else "FAIL",
        "evidence_issues": integrity_issues, "experiment_issues": list(dict.fromkeys(experiment_issues)),
        "process_completion": "PASS" if process_safe else "FAIL",
        "campaign_accounting": campaign_accounting(root, campaign_ledger_path, current_result=result),
        "retry_classification": retry_classification(result, process_safe,
            campaign_ledger_path=campaign_ledger_path, attempt_root=root),
        "native_launch_authority": "NONE_CONFERRED_BY_THIS_RESULT"}
    if cpu_terminal:
        terminal_report = terminal_report or {"PROCESS_COMPLETION": "FAIL", "CONTAINMENT_RESULT": "FAIL",
            "autonomous_exit": False, "terminal_exception_eligible": False, "actual_os_exit_code": process.get("exit_code")}
        answer.update(terminal_policy=CPU_TERMINAL_POLICY, terminal_result=terminal_report,
            PROCESS_COMPLETION=terminal_report["PROCESS_COMPLETION"], process_completion=terminal_report["PROCESS_COMPLETION"],
            CONTAINMENT_RESULT=terminal_report["CONTAINMENT_RESULT"], autonomous_exit=terminal_report["autonomous_exit"],
            actual_os_exit_code=terminal_report["actual_os_exit_code"])
        if reset_ruling is not None:
            ruling_complete = (reset_ruling["status"] == "RULING_NEEDED" and not integrity_issues and process_safe
                and terminal_report.get("terminal_exception_eligible")
                and terminal_report.get("evidence_scope") == "FAILED_START")
            answer.update(status="RULING_NEEDED" if ruling_complete else "FAIL",
                OBSERVATION_RESULT="RULING_NEEDED" if ruling_complete else "INCOMPLETE",
                CONTACT_CONTROL_RESULT="NOT_EVALUABLE", END_STATE_RESULT="NOT_EVALUABLE",
                reset_ruling=reset_ruling, retry_classification=reset_ruling["retry_classification"])
    if backend_report is not None:
        answer.update(backend_mode=CPU, backend_diagnostic=backend_report,
            BACKEND_DIAGNOSTIC_RESULT=backend_report["BACKEND_DIAGNOSTIC_RESULT"],
            gpu_gate5_acceptance=False, end_state_scope=CPU)
        summary = (observation_result or {}).get("cpu_interpretation", {})
        answer["cpu_interpretation"] = interpret_cpu_contact(answer["CONTACT_CONTROL_RESULT"],
            summary.get("production_current_witnesses", 0), summary.get("fresh_bilateral_boundaries", 0),
            qualified=not integrity_issues and backend_report["BACKEND_DIAGNOSTIC_RESULT"] ==
            "CPU_CONFIGURATION_VERIFIED" and answer["OBSERVATION_RESULT"] == "PASS")
    if early_profile_report is not None and early_profile_report.get("declared"):
        answer["early_cpu_profile"] = early_profile_report
        answer["EARLY_PROFILE_RESULT"] = early_profile_report["status"]
    return answer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--adjudicate", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--campaign-ledger", type=Path,
                        help="Optional saved campaign ledger for offline adjudication only; grants no authority")
    parser.add_argument("--seed", type=int, default=42, choices=[42])
    parser.add_argument("--observation-mode", action="store_true", help="Requires a separate observation campaign execution grant")
    parser.add_argument("--contact-transport", default="polling-contact/v1",
                        choices=("polling-contact/v1", "callback-contact/v1"))
    parser.add_argument("--backend-mode", default=PRODUCTION_GPU, choices=(PRODUCTION_GPU, CPU))
    parser.add_argument("--diagnostic-mode", choices=(CPU_STARTUP_SHUTDOWN_ONLY,))
    parser.add_argument("--terminal-policy", choices=(CPU_EVIDENCE_FIRST_CLOSE,))
    parser.add_argument("--geometric-mode", action="store_true")
    parser.add_argument("--stage1-demo", action="store_true")
    parser.add_argument("--stage3-learned", action="store_true")
    parser.add_argument("--stage3-witness", action="store_true")
    from chicgrasp_policy.r4_live_binding import POINTS, HELDOUT_POINTS, HB_POINTS, HC_POINTS
    parser.add_argument("--envelope-v2", action="store_true")
    parser.add_argument("--envelope-v3c", action="store_true")
    parser.add_argument("--r4-live-point", choices=POINTS + HELDOUT_POINTS + HB_POINTS + HC_POINTS)
    parser.add_argument("--translated-point", choices=("ORIGINAL", "H08", "H09", "H04", *(f"G{i:02d}" for i in range(9)), *(f"R{i:06d}" for i in range(1, 102)), *(f"Y{site}_{code}" for site in ("H05","H07") for code in ("Z00","P05","M05","P10","M10","P20","M20"))))
    parser.add_argument("--run-type", choices=("HARNESS", "EVIDENCE"))
    parser.add_argument("--stage0-mode", choices=("TUNE_GUI", "CHECK_HEADLESS"))
    args = parser.parse_args(argv)
    if args.adjudicate:
        if args.r4_live_point or args.translated_point or args.stage3_witness or args.stage3_learned or args.stage1_demo or args.stage0_mode or args.geometric_mode or args.run_type or args.observation_mode or args.contact_transport != "polling-contact/v1" or args.backend_mode != PRODUCTION_GPU or args.diagnostic_mode or args.terminal_policy:
            parser.error("adjudication reads observation mode from authenticated saved evidence")
        result = adjudicate(args.adjudicate, campaign_ledger_path=args.campaign_ledger)
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, allow_nan=False))
        return 0 if result["status"] in ("PASS", "HARNESS_COMPLETE", "GATE6_STAGE3_GPU_END_STATE_PASS") else 1
    if args.campaign_ledger is not None:
        parser.error("--campaign-ledger is only valid with --adjudicate")
    if args.output_dir is None or args.journal is None:
        parser.error("--execute requires --output-dir and --journal")
    if bool(args.geometric_mode) != bool(args.run_type):
        parser.error("--geometric-mode requires explicit --run-type; legacy modes have none")
    if args.stage0_mode and (not args.geometric_mode or args.run_type != "HARNESS"):
        parser.error("--stage0-mode requires geometric HARNESS; it never authorizes EVIDENCE")
    if args.contact_transport == "callback-contact/v1" and not args.observation_mode:
        parser.error("callback transport requires --observation-mode")
    if args.backend_mode == CPU and (not args.observation_mode or args.contact_transport != "callback-contact/v1"):
        parser.error("CPU diagnostic requires --observation-mode and --contact-transport callback-contact/v1")
    if args.diagnostic_mode and args.backend_mode != CPU:
        parser.error("startup-only diagnostic requires explicit CPU backend")
    if args.terminal_policy and (args.backend_mode != CPU or args.diagnostic_mode):
        parser.error("CPU terminal policy requires CPU callback observation and no startup-only mode")
    result = execute(args.output_dir, args.journal, seed=args.seed, observation_mode=args.observation_mode,
                     contact_transport=args.contact_transport, backend_mode=args.backend_mode,
                     diagnostic_mode=args.diagnostic_mode, terminal_policy=args.terminal_policy,
                     geometric_mode=args.geometric_mode, run_type=args.run_type, stage0_mode=args.stage0_mode,
                     stage1_demo=args.stage1_demo, stage3_learned=args.stage3_learned, stage3_witness=args.stage3_witness, translated_point=args.translated_point, r4_live_point=args.r4_live_point, envelope_v2=args.envelope_v2, envelope_v3c=args.envelope_v3c)
    return 0 if result["script_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
