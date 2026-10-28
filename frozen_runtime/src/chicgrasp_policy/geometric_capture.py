"""Slim GPU selection of the existing scheduler/owner/RGB/terminal paths.

Import is inert. No contact reader, control primitive, codec or force channel.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import math
import time
import sys

from . import attachment_capture as metrics
from .geometric_admission import evaluate_history
from .interface import JAW_STATE_SEMANTICS
from .lift_event_pose_evidence import Context, _json


class Stage0TuningStopped(Exception):
    """Owner quit at the frozen pre-confirmation boundary, not a candidate failure."""


class Stage1StartupChecked(Exception):
    """HARNESS stops after the same initial visual checks, before owner or command."""


def make_capture(core, runtime, writer, stage0=None, action_source=None):
    class GeometricCapture(core.CandidateCapture):
        def __init__(self):
            super().__init__(runtime, rgb_writer=writer, action_source=action_source)
            self.observation_mode = True  # RGB persistence only, no control scene.
            self.history = []
            self.witness = None
            self.recording_errors = []
            self.envelope_v2 = bool(getattr(runtime, "envelope_v2", False))
            self.event = None
            if getattr(runtime, "stage3_witness", False) or getattr(runtime, "translated_point", None) is not None:
                from .side_opening_admission import WitnessBuffer
                def persist(event):
                    from .stage1_demo import append
                    append(runtime.output_dir / "witness_transitions.jsonl", event)
                (runtime.output_dir / "witness_transitions.jsonl").touch(exist_ok=False)
                self.witness = WitnessBuffer(persist, run_type=runtime.witness_run_type)

        def boundary(self):
            t, s = self.tick, self.substep
            if not (0 <= t < 6*(action_source.total_actions if self.envelope_v2 else 200) and s in (0, 1)) or len(self.rows) != 2*t+s:
                raise core.CaptureFailure("extra_or_out_of_order_scheduled_boundary")
            self.context = Context(runtime.episode_id, t, s)
            clock = tuple(runtime.clock())
            row = dict(context=asdict(self.context), wall_monotonic_ns=time.monotonic_ns(),
                pose=None, jaws=None, commands=self.current_action[3:].tolist(),
                attachment=self.owner.snapshot(), issues=[],
                runtime=dict(resetting=False, truncated=False, terminated=False, invalid=False))
            self.rows.append(row)
            if action_source is not None:
                row["policy_indices"] = action_source.indices(t//6) if hasattr(action_source, "indices") else dict(global_action=t//6,
                    setup_action=t//6 if t < 300 else None,
                    learned_action=t//6-50 if t >= 300 else None)
            try:
                if (type(clock[0]) is not int or clock[0] != runtime.origin_clock[0]+len(self.rows)
                        or isinstance(clock[1], bool) or not math.isfinite(clock[1])
                        or clock[1] <= self.last_clock[1]):
                    raise core.CaptureFailure("missing_stale_or_nonfinite_scheduled_clock")
                self.last_clock = clock
                if action_source is not None and t % 6 == 0 and s == 0:
                    row["accepted_command"] = core.RuntimeSession.commanded_jaw_metadata(runtime, self.current_action[3:])
                    writer.command_event("ACCEPTANCE_OBSERVED", t//6,
                        scheduled_boundary=[t, s], accepted_command=deepcopy(row["accepted_command"]),
                        meaning="accepted_action_term_observed_after_scheduled_substep; exact_issue_time_unknown")
                status = row["attachment"]["status"]
                operation = None
                if status == "PENDING":
                    operation = self.owner.confirm_pending(self.context)
                    if operation.get("status") == "CONFIRMED" and any(self.current_action[3:] == 1):
                        operation = self.owner.update(self.context, {}, self.current_action[3:])
                elif status == "CONFIRMED":
                    operation = self.owner.update(self.context, {}, self.current_action[3:])
                row["pose"] = operation.get("pose_reading") if operation else runtime.measure()
                core._valid_pose(row["pose"], clock)
                row["jaws"] = runtime.read_jaws()
                if (row["jaws"]["clock_before"] != list(clock)
                        or row["jaws"]["clock_after"] != list(clock)):
                    raise core.CaptureFailure("jaw_measurement_not_current")
                sample = history_sample(row)
                if self.witness is not None:
                    if t % 6 == 0 and s == 0 and "accepted_command" not in row:
                        row["accepted_command"] = core.RuntimeSession.commanded_jaw_metadata(runtime, self.current_action[3:])
                    try:
                        self.witness.append(sample, row.get("accepted_command"))
                        if self.envelope_v2 and t % 6 == 0 and s == 0 and action_source.accepted(sample, row["accepted_command"]):
                            from .attachment_capture import event_schedule
                            self.event = event_schedule(action_source.first_accepted_close, protocol=runtime.envelope_protocol)
                            self.owner.journal.bind_event_close(action_source.first_accepted_close)
                            writer.rgb.set_event_bound(action_source.total_actions)
                            row["event_anchor"] = dict(self.event, accepted_command=deepcopy(row["accepted_command"]))
                    except Exception as exc:
                        if operation and operation.get("status") != "CONFIRMED":
                            self.recording_errors.append(type(exc).__name__+":"+str(exc))
                            raise core.CaptureFailure("attachment_lost_or_invalid:" + repr(operation.get("issues"))) from exc
                        raise
                else:
                    self.history.append(sample)
                if operation:
                    row["owner_operation_id"] = operation["id"]
                    clean_tail_release = (self.envelope_v2 and self.event is not None and 2*t+s > self.event["score_end"]
                        and operation.get("status") == "RELEASED" and not operation.get("issues")
                        and operation.get("release", {}).get("cause") == "jaw_open"
                        and operation["release"].get("cleanup_confirmed") is True)
                    if operation.get("status") != "CONFIRMED" and not clean_tail_release:
                        raise core.CaptureFailure("attachment_lost_or_invalid:" + repr(operation.get("issues")))
                    if clean_tail_release:
                        row["TAIL_RELEASE"] = True
                if t % 6 == 0 and s == 0:
                    self.command_frames.append(dict(action_frame=t//6, commands=row["commands"],
                        sample={k: sample[k] for k in ("episode_id", "control_tick", "substep", "physics_step", "simulation_time_s")},
                        source="executed_candidate_command_first_post_substep/v1"))
                decision = (None if self.event is None else self.event["decision"]) if self.envelope_v2 else 1679
                if 2*t+s == decision:
                    admission = self.witness.evaluate(expected_end=(t,s)) if self.witness is not None else evaluate_history(self.history[-61:])
                    row["geometric_eligibility"] = admission
                    if admission["status"] == "FALSE":
                        raise core.CaptureFailure("geometric_admission_refused:" + admission["status"])
                    if admission["status"] != "TRUE":
                        raise core.CaptureFailure("geometric_admission_unavailable:" + repr(admission["issues"]))
                    eligibility = dict(geometric_eligibility=admission,
                        runtime=dict(row["runtime"], sample=admission["sample"],
                                     source="scheduled_pre_reset_boundary/v1"))
                    operation = self.owner.activate(self.context, eligibility)
                    row["owner_operation_id"] = operation["id"]
                    row["attachment"] = self.owner.snapshot()
                    row["attachment"]["request_pose_reading"] = deepcopy(operation.get("pose_reading"))
                    if operation.get("status") != "PENDING":
                        raise core.CaptureFailure("geometric_attachment_request_failed:" + repr(operation.get("issues")))
                else:
                    row["attachment"] = self.owner.snapshot()
                self.substep += 1
            except Exception as exc:
                row["attachment"].update(self.owner.snapshot())
                row["runtime"]["invalid"] = True
                row["issues"].append(type(exc).__name__ + ":" + str(exc))
                raise
            finally:
                if getattr(runtime, "stage1_demo", False) or action_source is not None:
                    primary = sys.exc_info()[1]
                    try:
                        writer.write_substep(row)
                    except Exception as exc:
                        self.recording_errors.append(type(exc).__name__+":"+str(exc))
                        if primary is None:
                            raise
            # Stage0 observes the completed unchanged boundary. A controlled
            # tuning stop is outside the measurement-failure block above.
            if stage0 is not None and stage0.boundary(self.context, self):
                if (t, s) != (839, 1):
                    raise RuntimeError("stage0_tuning_stop_at_wrong_boundary")
                raise Stage0TuningStopped("owner_quit_before_840_0")
    return GeometricCapture()


def history_sample(row):
    """One measured boundary in a common world frame; no requested TCP target."""
    clock = row["pose"]["clock_after"]
    return _json(dict(row["context"], physics_step=clock[0], simulation_time_s=clock[1], **{
            "jaw_positions_m": deepcopy(row["jaws"]["positions_m"]),
            "jaw_commands": deepcopy(row["commands"]),
            "pose_frame": "world", "pose_valid": row["pose"]["valid"],
            "chicken_world_wxyz": deepcopy(row["pose"]["poses"]["chicken_actor_world_wxyz"]),
            "tcp_world_wxyz": deepcopy(row["pose"]["poses"]["tcp_world_wxyz"])}))


def execute_slim(core, output_dir, journal_path, *, seed, runtime_factory, bind_owner, run_type,
                 stage0_mode=None, stage1_demo=False, stage3_learned=False, stage3_witness=False, translated_point=None, r4_live_point=None, envelope_v2=False, envelope_v3c=False):
    if envelope_v3c and (envelope_v2 or r4_live_point not in ("H05_R2", *(f'HB{i:02d}_R{r}' for r in range(3) for i in range(1,11)), *(f'HC{i:02d}_R{r}' for r in range(3) for i in range(1,11))) or run_type != "HARNESS" or not stage3_learned or not stage3_witness):
        raise ValueError("v3c_requires_exclusive_owner_allocation_learned_witness_HARNESS")
    protocol = "v3c" if envelope_v3c else "v2"
    if r4_live_point is not None and (run_type != "HARNESS" or not stage3_learned or not stage3_witness
            or translated_point is not None or stage0_mode or stage1_demo):
        raise ValueError("r4_requires_exclusive_learned_witness_GPU_HARNESS")
    from .r4_live_binding import HB_POINTS
    if envelope_v2 and (r4_live_point not in ("H03_R0",) + HB_POINTS or run_type != "HARNESS" or not stage3_learned or not stage3_witness):
        raise ValueError("v2_requires_owner_H03_R0_learned_witness_HARNESS")
    verified_r4 = None
    if r4_live_point is not None:
        from .r4_live_binding import verify_binding
        verified_r4 = verify_binding(r4_live_point, envelope_v2=envelope_v2, envelope_v3c=envelope_v3c)
    # Reuse the proven event-horizon machinery; scientific routing is explicit.
    envelope_v2 = envelope_v2 or envelope_v3c
    original_probe = (translated_point == "ORIGINAL" and run_type == "HARNESS"
        and Path(output_dir).resolve() == (Path(__file__).resolve().parents[2] /
            "outputs/current_original_script_probe/attempt001/capture/experiment").resolve())
    if translated_point is not None and ((run_type != "EVIDENCE" and not original_probe) or stage0_mode or stage1_demo or stage3_learned or stage3_witness):
        raise ValueError("translated_script_requires_exclusive_recording")
    if type(stage3_witness) is not bool or (stage3_witness and (not stage3_learned or run_type not in ("HARNESS", "EVIDENCE"))):
        raise ValueError("witness_requires_Stage3_explicit_run_type")
    if type(stage3_learned) is not bool or (stage3_learned and
            (run_type not in ("EVIDENCE", "HARNESS") or stage0_mode or stage1_demo)):
        raise ValueError("stage3_requires_separate_geometric_EVIDENCE")
    if type(stage1_demo) is not bool or (stage1_demo and (run_type not in ("EVIDENCE", "HARNESS") or stage0_mode)):
        raise ValueError("stage1_requires_geometric_recording_or_startup_check")
    if run_type not in ("HARNESS", "EVIDENCE"):
        raise ValueError("explicit_geometric_run_type_required")
    if stage0_mode not in (None, "TUNE_GUI", "CHECK_HEADLESS") or (stage0_mode and run_type != "HARNESS"):
        raise ValueError("stage0_is_explicit_HARNESS_only")
    if run_type == "EVIDENCE":
        import json
        from .translated_script import COLLECTION_POINTS, YAW_POINTS, ANCHOR_POINTS
        config = json.loads((Path(__file__).resolve().parents[2] / "config/policy_camera_calibration.json").read_text())
        if translated_point == "R000001" or translated_point in COLLECTION_POINTS or translated_point in YAW_POINTS or translated_point in ANCHOR_POINTS:
            from .stage1_demo import approved_scaled_profile
            approved_scaled_profile()
        elif config.get("calibration_status") not in ("USER_APPROVED", "STAGE0_OWNER_APPROVED"):
            raise ValueError("camera_owner_approval_required_before_EVIDENCE")
    verified_translation = None
    if translated_point is not None:
        from .translated_script import verified_output_point
        verified_translation = verified_output_point(translated_point, output_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    journal = core.Journal(journal_path)
    runtime = runtime_factory(output, seed=seed, geometric_mode=True,
                              **({"stage0_mode": stage0_mode} if stage0_mode else {}))
    runtime.execution_context = core.execution_identity(output)
    runtime.translated_point = verified_translation
    runtime.r4_live_binding = verified_r4
    runtime.stage1_demo = stage1_demo
    runtime.stage3_learned = stage3_learned
    runtime.stage3_witness = stage3_witness
    runtime.envelope_v2 = envelope_v2
    runtime.envelope_protocol = protocol
    runtime.witness_run_type = run_type
    runtime.admission_policy = "side-opening-progress/v1" if stage3_witness or translated_point is not None else "geometric-eligibility/v1"
    runtime.guard_record_sink = lambda record: journal.record("GUARD_COMPARISON", record=record)
    phase_path, phase_record, phase_errors = core._phase_sink(runtime, journal_path)
    capture = writer = stage0 = controller = None
    result = dict(schema="geometric-slim-capture/v2", jaw_state_semantics=JAW_STATE_SEMANTICS, geometric_mode=True,
        admission_policy=runtime.admission_policy, run_type=run_type, backend_mode="PRODUCTION_GPU",
        primary_errors=[], recording_errors=[], cleanup_errors=[], candidate_started=False,
        failure_phase="initialization", episode_id=runtime.episode_id,
        experiment_result={"status": "FAIL"}, process_completion="PARENT_OBSERVATION_REQUIRED",
        failure_phase_evidence={"schema": "capture-failure-phases/v1", "path": str(phase_path.resolve())})
    journal.record("INITIALIZATION_REQUESTED", run_type=run_type)
    if translated_point is not None:
        result["translated_point"] = translated_point
    if stage0_mode:
        result["stage0_mode"] = stage0_mode
    if stage1_demo:
        result["stage1_demo"] = True
    if stage3_learned:
        result["stage3_learned"] = True
    if stage3_witness:
        result["stage3_witness"] = True
    if r4_live_point is not None:
        result["r4_live_point"] = r4_live_point
        if envelope_v2:
            result["envelope_v2"] = True
        if envelope_v3c:
            result["envelope_protocol"] = "v3c"
        result["acceptance_eligible"] = False
    try:
        if translated_point is not None:
            from .translated_script import check_storage
            storage = check_storage(output)
            core.write_json(output / "script_storage_prelaunch.json", storage)
            core.write_json(output / "script_translation.json", runtime.translated_point)
            reference = runtime.translated_point.get("yaw_probe") or runtime.translated_point.get("hard_xy_anchor")
            if reference:
                core.write_json(output / "reference_labels.json", dict(labels=reference["labels"], training_eligible=False))
            if not storage["passed"]:
                raise ValueError("translated_insufficient_free_space")
        if stage3_learned:
            from .stage3_storage import check_available
            storage = check_available(output, witness=True, envelope_v2=envelope_v2) if stage3_witness else check_available(output)
            core.write_json(output / "stage3_storage_prelaunch.json", storage)
            if not storage["passed"]:
                raise ValueError("stage3_insufficient_free_space")
        runtime.initialize()
        journal.record("INITIALIZATION_RETURNED", initialization=runtime.initialization,
                       effective_configuration=runtime.effective_configuration)
        from .attachment_observation_evidence import IncrementalRGBWriter
        writer = IncrementalRGBWriter(output / "rgb", **(dict(envelope_v2=True) if envelope_v2 else {}))
        if stage3_learned:
            from .stage3_policy import PolicyWriter, LiveController, prepare_predictor
            from .stage1_demo import save_visuals
            writer = PolicyWriter(writer, output)
            save_visuals(runtime, output, "before_candidate",
                **({'scene_profile': True} if verified_r4 and verified_r4.get('heldout_allocation') else {}))
            if r4_live_point is not None:
                from .r4_live_binding import prepare_predictor as prepare_r4, save_preparation
                save_preparation(runtime, output)
                predictor, checkpoint = prepare_r4(r4_live_point, **({"envelope_v3c": True} if envelope_v3c else {"envelope_v2": True} if envelope_v2 else {}))
            else:
                predictor, checkpoint = prepare_predictor()
            core.write_json(output / "stage3_checkpoint.json", checkpoint)
            controller = LiveController(predictor, writer, envelope_v2=envelope_v2,
                evaluation_seed=checkpoint["binding"]["evaluation_seed"],
                **(dict(setup_actions=verified_r4["setup_actions"],
                        checkpoint_sha256=verified_r4["checkpoint_identity"]["sha256"])
                   if verified_r4 is not None else {}))
        if translated_point is not None:
            from .translated_script import ScriptWriter, ScriptController, save_preparation
            writer = ScriptWriter(writer, output)
            save_preparation(runtime, output, "before_candidate")
            controller = ScriptController(runtime.translated_point, writer)
        if stage1_demo:
            from .stage1_demo import DemoWriter, save_visuals
            writer = DemoWriter(writer, output)
            save_visuals(runtime, output, "before_candidate")
            if run_type == "HARNESS":
                raise Stage1StartupChecked()
        if stage0_mode:
            from .stage0_camera_runtime import Stage0CameraRuntime
            stage0 = Stage0CameraRuntime(runtime, output / "stage0", stage0_mode)
        binding = dict(task_id=core.TASK_ID, capture_policy=True, episode_id=runtime.episode_id,
            origin_step=runtime.origin_clock[0], origin_time=runtime.origin_clock[1])
        if getattr(runtime, "scene_identity", None) is not None:
            binding["scene_identity"] = runtime.scene_identity
        runtime.owner = bind_owner(runtime.env, runtime.stage, **binding)
        runtime.owner.journal.select_geometric_admission(runtime.admission_policy, run_type=run_type, envelope_v2=envelope_v2, protocol=protocol)
        capture = make_capture(core, runtime, writer, stage0=stage0, action_source=controller)
        result["failure_phase"] = "candidate"
        journal.record("CANDIDATE_REQUESTED", origin_clock=runtime.origin_clock, run_type=run_type)
        result["experiment_result"] = capture.run()
        if stage1_demo or stage3_learned:
            save_visuals(runtime, output, "after_candidate",
                **({'scene_profile': True} if verified_r4 and verified_r4.get('heldout_allocation') else {}))
        if translated_point is not None:
            save_preparation(runtime, output, "after_candidate")
    except Stage1StartupChecked:
        result["experiment_result"] = dict(status="NOT_EVALUABLE", reason="STAGE1_STARTUP_ONLY",
                                           acceptance_eligible=False)
        journal.record("STAGE1_STARTUP_CHECKED_NO_CANDIDATE")
    except Stage0TuningStopped:
        result["experiment_result"] = dict(status="NOT_EVALUABLE", reason="STAGE0_OWNER_QUIT_AT_839_1",
                                           acceptance_eligible=False)
        journal.record("STAGE0_OWNER_QUIT", control_tick=839, substep=1)
    except Exception as exc:
        result["primary_errors"].append(type(exc).__name__ + ":" + str(exc))
        if str(exc) == "geometric_admission_refused:FALSE" or ((stage3_witness or translated_point is not None) and str(exc).startswith("geometric_admission_unavailable:")):
            result["experiment_result"] = dict(status="NOT_EVALUABLE", decision="TRAJECTORY_DECISION_REQUIRED",
                                               reason="geometric_admission_unavailable" if str(exc).startswith("geometric_admission_unavailable:") else "geometric_admission_refused")
        phase_record(result["failure_phase"], "failed", **core.RuntimeSession.failure_details(exc))
    finally:
        if stage0 is not None:
            try:
                result["stage0_report"] = stage0.finish()
            except Exception as exc:
                result["recording_errors"].append("stage0_finalize:" + type(exc).__name__ + ":" + str(exc))
        result.update(initialization=deepcopy(runtime.initialization),
            execution_provenance=deepcopy(runtime.execution_context),
            effective_configuration=runtime.effective_configuration,
            candidate_started=bool(runtime.initialization.get("candidate_started")),
            counts=dict(completed_control_ticks=capture.control_ticks if capture else 0,
                observed_substeps=len(capture.rows) if capture else 0,
                completed_policy_actions=len(capture.executed_actions) if capture else 0),
            attachment_authoring_attempted=False if runtime.owner is None else None)
        if stage3_learned or translated_point is not None:
            result["command_provenance_schema"] = "script-command-lifecycle/v1" if translated_point is not None else "stage3-command-lifecycle/v1"
        if capture:
            result["recording_errors"].extend(capture.recording_errors)
        try:
            if controller is not None:
                result["script" if translated_point is not None else "policy"] = controller.summary(len(capture.executed_actions) if capture else 0)
            if envelope_v2 and capture:
                from .attachment_capture import evaluate_envelope_v2
                result["experiment_result"] = evaluate_envelope_v2(capture.rows, episode_id=runtime.episode_id,
                    origin_step=runtime.origin_clock[0], origin_time=runtime.origin_clock[1], close_action=controller.first_accepted_close,
                    completed=capture.completed, primary_errors=result["primary_errors"], lifecycle_events=capture.lifecycle_events, protocol=protocol)
            core.write_json(output / "candidate_result.json", result["experiment_result"])
            core.write_json(output / "measurements.json", dict(schema="geometric-measurements/v2",
                jaw_state_semantics=JAW_STATE_SEMANTICS,
                episode_id=runtime.episode_id, origin_clock=runtime.origin_clock,
                rows=capture.rows if capture else [],
                sensor_samples=capture.sensor_samples if capture else [],
                lifecycle_events=capture.lifecycle_events if capture else []))
            if runtime.owner is not None:
                owner = runtime.owner.export()
                result["attachment_authoring_attempted"] = any(
                    o.get("kind") == "scheduled_request" for o in owner["operations"])
                core.write_json(output / "owner_before_cleanup.json", owner)
            if writer:
                writer.finish(bool(capture and capture.completed), result["primary_errors"])
            if (stage1_demo or translated_point is not None) and capture and capture.completed:
                from .stage1_demo import finalize_episode
                result["demo"] = finalize_episode(capture, output)
        except Exception as exc:
            result["recording_errors"].append(type(exc).__name__ + ":" + str(exc))
            if hasattr(exc, "publication_diagnostics"):
                result["episode_publication_failure"] = dict(
                    diagnostic_path=exc.publication_diagnostic_path,
                    diagnostics=exc.publication_diagnostics)
        try:
            cleanup = runtime.close_scene(capture.context if capture else None)
            result["cleanup"] = cleanup
            result["cleanup_errors"].extend(cleanup.get("errors", []))
            if cleanup.get("owner"):
                result["cleanup_errors"].extend(core.owner_cleanup_issues(cleanup["owner"]))
        except Exception as exc:
            result["cleanup_errors"].append(type(exc).__name__ + ":" + str(exc))
        if envelope_v2:
            result["v2_cleanup_status"] = "FAIL" if result["cleanup_errors"] else "CLEAN"
        result["recording_errors"].extend(phase_errors)
        result["script_status"] = "PASS" if (capture and capture.completed
            and result["experiment_result"].get("status") == "PASS"
            and not any(result[k] for k in ("primary_errors", "recording_errors", "cleanup_errors"))) else "FAIL"
        if stage3_learned or translated_point is not None:
            # A complete physical miss must remain independently adjudicable
            # as END STATE FAIL, not become an application/process failure.
            result["script_status"] = "PASS" if capture and capture.completed and not any(
                result[k] for k in ("primary_errors", "recording_errors", "cleanup_errors")) else "FAIL"
        if stage0_mode:
            diagnostic_done = result.get("stage0_report", {}).get("status") in (
                "GUI_QUIT_EXPORTED", "HEADLESS_CHECKPOINTS_COMPLETE")
            result["script_status"] = "PASS" if diagnostic_done and not any(
                result[k] for k in ("primary_errors", "recording_errors", "cleanup_errors")) else "FAIL"
            # The candidate evaluator still ran unmodified for headless; retain
            # its diagnostic outcome but never turn this HARNESS into acceptance.
            result["acceptance_eligible"] = False
        if stage1_demo and run_type == "HARNESS":
            result["acceptance_eligible"] = False
            result["script_status"] = "PASS" if result["experiment_result"].get("reason") == "STAGE1_STARTUP_ONLY" and not any(
                result[k] for k in ("primary_errors", "recording_errors", "cleanup_errors")) else "FAIL"
        core._finalize_and_close(output, journal_path, journal, runtime, result, phase_record, phase_errors)
    return result
