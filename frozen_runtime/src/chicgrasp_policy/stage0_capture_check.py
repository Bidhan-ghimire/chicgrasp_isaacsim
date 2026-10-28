"""Independent Stage0 HARNESS checks. Never qualifies a demo or END STATE."""
from pathlib import Path
import hashlib
import math

import numpy as np

from .attachment_capture import frozen_candidate_actions, _measured_pose, CHICKEN_BODY, GRIPPER_BODY
from .attachment_observation_evidence import validate as validate_rgb
from .attachment_run_policy import check_run_type, declared_absolute_ms
from .geometric_admission import evaluate_history
from .geometric_capture import history_sample
from .geometric_capture_check import identity, load, check_commanded_sensors, check_measurements
from .interface import JAW_STATE_SEMANTICS


def validate_report(directory, report, measurement):
    """Reopen pixels and transforms; do not trust the adapter's status alone."""
    from PIL import Image
    from .stage0_visuals import validate_config, validate_export
    from .render_profile import validate_renderer
    directory = Path(directory).resolve()
    issues = []
    def artifact(descriptor):
        path = (directory / descriptor["path"]).resolve()
        if not path.is_relative_to(directory) or path == directory:
            raise ValueError("visual_path_escape")
        if identity(path) != {k: descriptor[k] for k in ("bytes", "sha256")}:
            raise ValueError("visual_file_identity:" + descriptor["path"])
        return path
    try:
        if (report["schema"] != "gate6-stage0-camera-runtime/v2" or report["run_type"] != "HARNESS"
                or report["approval_status"] != "NOT_APPROVED" or report["errors"]):
            raise ValueError("visual_scope_or_errors")
        # Old reports remain historical; they cannot prove the new profile check.
        validate_renderer(report["renderer"])
        baseline = validate_config(report["baseline_configuration"])
        initial = load(artifact(report["initial_snapshot"]))
        validate_export(initial, baseline, initial_snapshot=initial)
        # Historical reports predate the exact owner-exported rig. They do not
        # qualify that rig; any new supplied check is independently reopened.
        if "stage0_owner_dump_input" in baseline and "grey_studio_check" not in report:
            raise ValueError("grey_studio_scene_check_missing")
        if "grey_studio_check" in report:
            from .grey_studio_check import validate_report as validate_scene
            scene_check = load(artifact(report["grey_studio_check"]))
            if scene_check.get("phase") != "BEFORE_INITIAL_RESET_AND_POLICY_RGB":
                raise ValueError("grey_studio_early_check_phase")
            validate_scene(scene_check, baseline)
            checks = [g for g in report["guard_samples"] if g["operation"] == "grey_studio_scene_check"]
            if (len(checks) != 1 or checks[0]["before"] != checks[0]["after"]
                    or checks[0]["before"]["native_clock"] != measurement["rows"][1]["pose"]["clock_after"]):
                raise ValueError("grey_studio_scene_check_boundary")
        rows = measurement["rows"]
        allowed_clocks = [rows[1]["pose"]["clock_after"], rows[1679]["pose"]["clock_after"]]
        if len(rows) == 2400:
            allowed_clocks.append(rows[2399]["pose"]["clock_after"])
        guards = report["guard_samples"]
        if not guards:
            raise ValueError("frozen_physics_guard_records_missing")
        for sample in guards:
            state = sample["before"]
            if state != sample["after"] or state["native_clock"] not in allowed_clocks:
                raise ValueError("physics_or_scheduled_counter_advanced")
            expected_rows = state["native_clock"][0] - measurement["origin_clock"][0]
            if (state["capture_rows"] != expected_rows or state["scheduled_substeps"] != expected_rows
                    or state["capture_substep"] != 2
                    or state["capture_control_ticks"] != expected_rows//2-1
                    or state["runtime_control_ticks"] != expected_rows//2-1):
                raise ValueError("frozen_counter_identity")
        def guarded_state(state):
            return any(g["before"] == state == g["after"] and g["operation"] == "render_only" for g in guards)
        # Older reports have no such field and prove no expanded diagnostic.
        # Authenticate any present records without treating read errors as a
        # light-setting value or promoting capture integrity to appearance PASS.
        if "render_diagnostics" in report:
            diagnostics = report["render_diagnostics"]
            expected_diagnostics = [(tick, phase) for tick in (0, 839, 1199)
                                    for phase in ("before_render", "after_render")] if report["mode"] == "CHECK_HEADLESS" else []
            if len(diagnostics) != len(expected_diagnostics):
                raise ValueError("render_diagnostic_count")
            for entry, (tick, phase) in zip(diagnostics, expected_diagnostics):
                context = rows[2*tick+1]["context"]
                operation = f"render_diagnostic:tick{tick:04d}_substep1:{phase}"
                payload = load(artifact(entry["artifact"]))
                observed = payload["observation"]
                if (entry["context"] != context or entry["phase"] != phase
                        or payload["context"] != context or payload["phase"] != phase
                        or payload["frozen_state"] != entry["frozen_state"]
                        or entry["operation"] != operation
                        or entry["frozen_state"]["native_clock"] != rows[2*tick+1]["pose"]["clock_after"]
                        or not any(g["operation"] == operation and g["before"] == g["after"] == entry["frozen_state"]
                                   for g in guards)
                        or observed["schema"] != "stage0-render-state-diagnostic/v1"
                        or observed["read_only"] is not True or not isinstance(observed["errors"], list)
                        or observed["status"] != ("PARTIAL" if observed["errors"] else "COMPLETE")):
                    raise ValueError("render_diagnostic_provenance")
        if report["mode"] == "TUNE_GUI":
            pause = report["pause"]
            if (report["status"] != "GUI_QUIT_EXPORTED" or not report["exports"]
                    or pause["context"] != rows[-1]["context"] or pause["start"] != pause["end"]
                    or pause["start"]["native_clock"] != rows[-1]["pose"]["clock_after"]
                    or pause["explicit_quit"] is not True or not guarded_state(pause["start"])):
                raise ValueError("tuning_quit_export_or_pause_missing")
            export_guards = [(i, g) for i, g in enumerate(guards)
                             if g["operation"] == "export_visual_configuration"]
            preview_guards = [(i, g) for i, g in enumerate(guards)
                              if g["operation"] == "sensor_RGB_preview"]
            for export_index, export in enumerate(report["exports"]):
                snapshot = load(artifact(export["snapshot"]))
                configuration = load(artifact(export["configuration"]))
                if (export["approval_status"] != "NOT_APPROVED" or export["frozen_state"] != pause["start"]
                        or configuration != validate_export(snapshot, baseline, initial_snapshot=initial)):
                    raise ValueError("export_roundtrip_or_scope")
                # Absent fields remain compatible with older reports; their
                # absence never proves a displayed RGB or detailed light state.
                if "lights_at_export" in export or "displayed_preview" in export:
                    guard_index, export_guard = export_guards[export_index]
                    export_time = export["wall_monotonic_ns"]
                    if (pause["context"]["control_tick"] != 839 or pause["context"]["substep"] != 1
                            or export_guard["before"] != pause["start"]
                            or type(export_time) is not int or export_time < 0
                            or export_guard["wall_monotonic_ns"] > export_time):
                        raise ValueError("GUI_export_diagnostic_boundary")
                if "lights_at_export" in export:
                    lights = load(artifact(export["lights_at_export"]))
                    observation = lights["observation"]
                    observed_time = lights["wall_monotonic_ns"]
                    if (lights["timing"] != "export_time_readback_not_prior_display_acquisition"
                            or lights["context"] != pause["context"] or lights["frozen_state"] != pause["start"]
                            or type(observed_time) is not int
                            or not export_guard["wall_monotonic_ns"] <= observed_time <= export_time
                            or observation["schema"] != "stage0-early-light-inventory/v1"
                            or observation["phase"] != "GUI_EXPORT_READBACK"
                            or observation["read_only"] is not True
                            or not isinstance(observation["errors"], list)
                            or observation["status"] != ("PARTIAL" if observation["errors"] else "COMPLETE")):
                        raise ValueError("GUI_export_light_provenance")
                if "displayed_preview" in export:
                    preview = export["displayed_preview"]
                    completed = [(i, g) for i, g in preview_guards if i < guard_index]
                    if preview["status"] == "UNAVAILABLE":
                        if (preview != dict(status="UNAVAILABLE",
                                reason="no_completed_GUI_sensor_preview_before_export", images={}) or completed):
                            raise ValueError("GUI_unavailable_preview_claim")
                        continue
                    iteration = preview["preview_iteration"]
                    requested = preview["display_requested_monotonic_ns"]
                    returned = preview["display_returned_monotonic_ns"]
                    if (preview["status"] != "AVAILABLE" or not completed
                            or type(iteration) is not int or iteration != len(completed)
                            or iteration > pause["iterations"]
                            or preview["source"] != "stage0_last_displayed_GUI_sensor_RGB/v1"
                            or preview["export_snapshot_alignment"] != "last_displayed_acquisition_not_necessarily_export_snapshot"
                            or preview["context"] != pause["context"] or preview["frozen_state"] != pause["start"]
                            or preview["native_clock"] != pause["start"]["native_clock"]
                            or completed[-1][1]["before"] != pause["start"]
                            or type(requested) is not int or type(returned) is not int
                            or not 0 <= requested <= completed[-1][1]["wall_monotonic_ns"] <= returned <= export_guard["wall_monotonic_ns"] <= export_time):
                        raise ValueError("GUI_displayed_preview_provenance")
                    validate_renderer(preview["renderer"], report["renderer"])
                    if (set(preview["images"]) != {"camera_left", "camera_right"}
                            or set(preview["sensor_metadata"]) != {"camera_left", "camera_right"}):
                        raise ValueError("GUI_both_displayed_cameras_required")
                    for side, name in (("camera_left", "PolicyCameraLeft"), ("camera_right", "PolicyCameraRight")):
                        descriptor, sensor = preview["images"][side], preview["sensor_metadata"][side]
                        if (sensor["prim_path"] != "/World/envs/env_0/" + name
                                or type(sensor.get("frame")) is not int or sensor["frame"] < 0
                                or sensor.get("frame_source") != "Camera.frame"
                                or sensor.get("source") != "stage0_render_only_sensor_update/v1"
                                or type(sensor.get("refresh_dt_s")) not in (int, float) or sensor["refresh_dt_s"] != 0
                                or sensor.get("sensor_time_domain") != "seconds_since_sensor_reset"):
                            raise ValueError("GUI_preview_sensor_provenance")
                        if sensor.get("sensor_time_status") == "OBSERVED":
                            value = sensor.get("sensor_time_s")
                            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                                raise ValueError("GUI_preview_sensor_timestamp")
                        elif (sensor.get("sensor_time_status") != "UNAVAILABLE" or sensor.get("sensor_time_s") is not None
                                or not sensor.get("sensor_time_unavailable_reason")):
                            raise ValueError("GUI_preview_sensor_timestamp")
                        with Image.open(artifact(descriptor)) as image:
                            pixels = np.asarray(image)
                            if (image.format != "PNG" or image.mode != "RGB" or pixels.shape != (240, 320, 3)
                                    or pixels.dtype != np.uint8 or descriptor["shape"] != [240, 320, 3]
                                    or descriptor["dtype"] != "uint8" or descriptor["encoding"] != "PNG_lossless"
                                    or hashlib.sha256(pixels.tobytes()).hexdigest() != descriptor["pixel_sha256"]):
                                raise ValueError("GUI_preview_pixels_or_encoding")
        elif report["mode"] == "CHECK_HEADLESS":
            expected = [(0, 1), (839, 1), (1199, 1)]
            checkpoints = report["checkpoints"]
            if report["status"] != "HEADLESS_CHECKPOINTS_COMPLETE" or len(checkpoints) != 3:
                raise ValueError("six_exact_boundary_images_required")
            for record, (tick, substep) in zip(checkpoints, expected):
                validate_renderer(record["renderer"], report["renderer"])
                row = rows[2*tick+substep]
                if record["context"] != row["context"] or record["native_clock"] != row["pose"]["clock_after"]:
                    raise ValueError("checkpoint_not_actual_boundary")
                snapshot = load(artifact(record["visual_snapshot"]))
                validate_export(snapshot, baseline, initial_snapshot=initial)
                if (snapshot != initial or record["frozen_state"]["native_clock"] != record["native_clock"]
                        or not guarded_state(record["frozen_state"])):
                    raise ValueError("headless_configuration_or_clock_changed")
                if set(record["images"]) != {"camera_left", "camera_right"}:
                    raise ValueError("both_camera_images_required")
                for side, name in (("camera_left", "PolicyCameraLeft"), ("camera_right", "PolicyCameraRight")):
                    descriptor, sensor = record["images"][side], record["sensor_metadata"][side]
                    if sensor["prim_path"] != "/World/envs/env_0/" + name:
                        raise ValueError("checkpoint_camera_identity")
                    if type(sensor.get("frame")) is not int or sensor["frame"] < 0 or sensor.get("frame_source") != "Camera.frame":
                        raise ValueError("checkpoint_sensor_frame")
                    if (sensor.get("source") != "stage0_render_only_sensor_update/v1"
                            or type(sensor.get("refresh_dt_s")) not in (int, float) or sensor["refresh_dt_s"] != 0
                            or sensor.get("sensor_time_domain") != "seconds_since_sensor_reset"):
                        raise ValueError("checkpoint_sensor_acquisition_source")
                    if sensor.get("sensor_time_status") == "OBSERVED":
                        value = sensor.get("sensor_time_s")
                        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                            raise ValueError("checkpoint_invalid_sensor_timestamp")
                    elif (sensor.get("sensor_time_status") != "UNAVAILABLE" or sensor.get("sensor_time_s") is not None
                            or not sensor.get("sensor_time_unavailable_reason")):
                        raise ValueError("checkpoint_sensor_timestamp_provenance")
                    with Image.open(artifact(descriptor)) as image:
                        pixels = np.asarray(image)
                        if (image.format != "PNG" or image.mode != "RGB" or pixels.shape != (240, 320, 3)
                                or pixels.dtype != np.uint8 or descriptor["shape"] != [240, 320, 3]
                                or descriptor["dtype"] != "uint8" or descriptor["encoding"] != "PNG_lossless"
                                or hashlib.sha256(pixels.tobytes()).hexdigest() != descriptor["pixel_sha256"]):
                            raise ValueError("checkpoint_pixels_or_encoding")
        else:
            raise ValueError("visual_mode_invalid")
        from .dome_recovery import check_comparison
        check_comparison(directory, report)
    except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
        issues.append("visual:" + str(exc))
    return issues


def check_tuning_rows(measurement, owner):
    """Exact pre-confirmation window, without relaxing the full-run evaluator."""
    issues = []
    try:
        rows = measurement["rows"]
        origin = measurement["origin_clock"]
        commands = frozen_candidate_actions()
        if len(rows) != 1680 or measurement["lifecycle_events"]:
            raise ValueError("tuning_requires_exact_839_1_window")
        previous_time = origin[1]
        for i, row in enumerate(rows):
            context = dict(episode_id=measurement["episode_id"], control_tick=i//2, substep=i%2)
            if row["context"] != context or row["commands"] != commands[i//12, 3:].tolist():
                raise ValueError("tuning_row_order_or_command")
            pose, jaws = row["pose"], row["jaws"]
            clock = pose["clock_after"]
            if (pose["valid"] is not True or pose["issues"] or pose["clock_before"] != clock
                    or pose.get("units") != "m" or pose.get("pose_convention") != "position_xyz_quaternion_wxyz"
                    or pose.get("identities", {}).get("chicken_actor") != CHICKEN_BODY
                    or pose.get("identities", {}).get("gripper_actor") != GRIPPER_BODY
                    or type(clock[0]) is not int or clock[0] != origin[0]+i+1
                    or type(clock[1]) not in (float, int) or not math.isfinite(clock[1])
                    or clock[1] <= previous_time or row["issues"]
                    or row["runtime"] != dict(resetting=False, truncated=False, terminated=False, invalid=False)):
                raise ValueError("tuning_invalid_pose_clock_or_lifecycle")
            previous_time = clock[1]
            for name in ("robot_root_world_wxyz", "gripper_world_wxyz", "chicken_actor_world_wxyz",
                         "tcp_world_wxyz", "tcp_base_wxyz", "chicken_in_gripper_wxyz"):
                _measured_pose(pose["poses"][name])
            if (jaws["clock_before"] != clock or jaws["clock_after"] != clock or jaws["units"] != "m"
                    or jaws.get("source") != "robot.root_physx_view.get_dof_positions"
                    or jaws["names"] != [f"PrismaticJoint{n}" for n in range(1, 5)]
                    or len(jaws["positions_m"]) != 4 or any(type(v) not in (int, float)
                    or not math.isfinite(v) for v in jaws["positions_m"])):
                raise ValueError("tuning_invalid_measured_jaws")
            if row["attachment"]["status"] != ("PENDING" if i == 1679 else "UNATTACHED"):
                raise ValueError("tuning_early_or_forced_confirmation")
            if ("geometric_eligibility" in row) != (i == 1679):
                raise ValueError("tuning_admission_not_once_at_839_1")
        gate = evaluate_history([history_sample(r) for r in rows[1619:1680]])
        if gate["status"] != "TRUE" or rows[-1]["geometric_eligibility"] != gate:
            raise ValueError("tuning_unchanged_admission_rejected")
        records = owner["event_evidence"]["records"]
        requests = [r for r in records if r["kind"] == "activation_request"]
        if (len(requests) != 1 or requests[0]["context"] != rows[-1]["context"]
                or requests[0]["native_clock"] != rows[-1]["pose"]["clock_after"]
                or requests[0]["eligibility"]["geometric_eligibility"] != gate
                or requests[0]["issues"] or requests[0]["eligibility_issues"]):
            raise ValueError("tuning_request_not_correlated")
        if any(r["context"]["control_tick"] >= 840 for r in records if isinstance(r.get("context"), dict)):
            raise ValueError("tuning_reached_confirmation_boundary")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        issues.append(str(exc))
    return issues


def adjudicate_stage0(attempt_root):
    root = Path(attempt_root).resolve()
    output = root / "capture/experiment"
    issues, process_issues = [], []
    observed_process = False
    mode, visual = None, {}
    def require(condition, label, group=issues):
        if not condition:
            group.append(label)
    try:
        invocation = load(root / "invocation.json")
        result = load(output / "result.json")
        process = load(root / "capture/process_result.json")
        mode = invocation.get("stage0_mode")
        absolute_ms = declared_absolute_ms(invocation)
        require(mode in ("TUNE_GUI", "CHECK_HEADLESS"), "stage0_mode_required")
        require(result.get("stage0_mode") == mode, "stage0_mode_disagreement")
        require(process.get("stage0_mode") == mode, "parent_stage0_mode_disagreement")
        scope = check_run_type(invocation, result, process)
        require(scope["run_type"] == "HARNESS" and not scope["acceptance_eligible"], "stage0_HARNESS_only")
        issues.extend(i for i in scope["issues"] if i != "harness_output_not_acceptance_evidence")
        require(result.get("acceptance_eligible") is False, "stage0_not_acceptance")
        require(result.get("schema") == "geometric-slim-capture/v2"
                and result.get("jaw_state_semantics") == JAW_STATE_SEMANTICS, "stage0_commanded_capture_version")
        manifest = load(output / "artifact_manifest.json")
        actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
        require(actual == set(manifest["files"]) | {"artifact_manifest.json"}, "artifact_membership")
        for relative, expected in manifest["files"].items():
            path = (output / relative).resolve()
            if not path.is_relative_to(output) or path == output:
                raise ValueError("artifact_path_escape")
            require(identity(path) == expected, "artifact_identity:" + relative)
        for field in ("primary_errors", "recording_errors", "cleanup_errors"):
            require(result.get(field) == [], "saved_" + field)
        measurement = load(output / "measurements.json")
        pre = load(output / "pre_candidate.json")
        owner = load(output / "owner_before_cleanup.json")
        gui = mode == "TUNE_GUI"
        issues.extend(check_tuning_rows(measurement, owner) if gui else check_measurements(measurement, owner)["issues"])
        issues.extend(check_commanded_sensors(measurement, pre))
        require(result["counts"] == dict(completed_control_ticks=839 if gui else 1200,
            observed_substeps=1680 if gui else 2400, completed_policy_actions=139 if gui else 200), "stage0_counts")
        sensors = measurement["sensor_samples"]
        for i, sensor in enumerate(sensors):
            require(sensor["action_frame"] == i and sensor["control_tick"] == i*6
                and sensor["native_clock"] == (measurement["origin_clock"] if i == 0
                    else measurement["rows"][i*12-1]["pose"]["clock_after"]), "acquisition_clock:" + str(i))
            require(load(output / "rgb" / f"{i:06d}.json")["metadata"].get("jaw_state") == sensor.get("jaw_state"),
                    "rgb_commanded_jaw_metadata:" + str(i))
        rgb = validate_rgb(output / "rgb", expected_count=141 if gui else 201, sensors=sensors, pre_candidate=pre)
        require(rgb["status"] == "PASS", "action_rgb:" + repr(rgb["issues"]))
        visual = load(output / "stage0/report.json")
        require(visual.get("mode") == mode, "visual_mode_disagreement")
        require(result["stage0_report"] == visual, "visual_report_disagreement")
        issues.extend(validate_report(output / "stage0", visual, measurement))
        configuration = result["effective_configuration"]
        for key, wanted in dict(device="cuda:0", simulation_device="cuda:0", physics_device="cuda:0", gpu_sim=True,
                gpu_pipeline=True, gpu_dynamics=True, broadphase="GPU", num_envs=1, decimation=2, render_interval=2,
                cameras_enabled=True, fast_shutdown=True, has_gui=gui, offscreen_render=not gui,
                configured_seed=42, action_dimension=5).items():
            require(type(configuration.get(key)) is type(wanted) and configuration[key] == wanted, "configuration:"+key)
        require(configuration["physics_dt_s"] == 1/120 and configuration["control_dt_s"] == 1/60, "physics_rates")
        for key, wanted in dict(exit_code=0, root_exit_verified=True, tree_empty_verified=True, capture_complete=True,
                forced_termination=False, timed_out=False, parent_termination_requested=False, kill_on_job_close_used=False,
                kill_on_job_close_possible=False, startup_containment_used=False, autonomous_tree_exit=True,
                job_assigned_before_resume=True, root_resumed=True, job_disposed_after_observation=True,
                job_members_before_disposal=0, absolute_deadline_ms=absolute_ms,
                timeout_ms=absolute_ms, effective_deadline_ms=absolute_ms,
                termination_and_drain_cap_ms=5000, parent_launch_boundary_ms=0).items():
            require(key in process and type(process[key]) is type(wanted) and process[key] == wanted,
                    "process:"+key, process_issues)
        if absolute_ms is None:
            require(process.get("stage0_until_quit") is True, "process:stage0_until_quit", process_issues)
        require(process.get("containment_errors") == [], "containment_errors", process_issues)
        for key in ("root_exit_observed_ms", "job_empty_observed_ms", "drain_completed_observed_ms"):
            require(type(process.get(key)) is int and process[key] >= 0
                    and (absolute_ms is None or process[key] <= absolute_ms), "deadline:"+key, process_issues)
        observed_process = True
        for key, relative in (("stdout", "stdout.bin"), ("stderr", "stderr.bin"), ("completed_journal", "structured/events.jsonl")):
            path = root / "capture" / relative
            expected = process[key]
            require({k: expected.get(k) for k in ("bytes", "sha256")} == identity(path)
                    and Path(expected["path"]).resolve() == path, "raw_capture:"+key)
        require(process["invocation_sha256"] == identity(root / "invocation.json")["sha256"], "invocation_identity")
        require(Path(invocation["output_directory"]).resolve() == root / "capture", "output_path")
        from .attachment_observation_evidence import _load
        events = [_load(line) for line in (root / "capture/structured/events.jsonl").read_bytes().splitlines()]
        require([r["sequence"] for r in events] == list(range(len(events))), "journal_sequence")
        require([e["kind"] for e in events[-2:]] == ["PRE_CLOSE_EVIDENCE_COMPLETE", "CLOSE_REQUESTED"], "preclose_intent")
        require(all(e["manifest"] == identity(output / "artifact_manifest.json") for e in events[-2:]), "terminal_manifest")
        require(result["execution_provenance"]["executed_commit"] == invocation["executed_commit"], "executed_commit")
    except Exception as exc:
        issues.append("saved_evidence:" + type(exc).__name__ + ":" + str(exc))
    return dict(schema="gate6-stage0-adjudication/v1",
        status="HARNESS_COMPLETE" if observed_process and not issues and not process_issues else "FAIL",
        mode=mode, run_type="HARNESS", acceptance_eligible=False, approval_status="NOT_APPROVED",
        END_STATE_RESULT="NOT_EVALUABLE_AS_ACCEPTANCE",
        CAPTURE_INTEGRITY="PASS" if not issues else "FAIL",
        PROCESS_COMPLETION="PASS" if observed_process and not process_issues else "NOT_CONFIRMED",
        issues=issues, process_issues=process_issues, native_launch_authority="NONE")
