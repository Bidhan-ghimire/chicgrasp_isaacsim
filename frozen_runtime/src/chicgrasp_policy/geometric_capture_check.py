"""Independent saved-file slim GPU adjudication. No contact artifacts required."""
from pathlib import Path
import hashlib
import math

from .attachment_observation_evidence import _load, validate as validate_rgb
from .attachment_capture import evaluate_end_state, frozen_candidate_actions, _measured_pose, CHICKEN_BODY, GRIPPER_BODY
from .geometric_admission import SOURCE, evaluate_history
from .geometric_capture import history_sample
from .interface import JAW_STATE_SEMANTICS


def identity(path):
    data = Path(path).read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def load(path):
    return _load(Path(path).read_bytes())


def check_commanded_sensors(measurement, pre, *, allow_terminal_observation=False):
    """V2 labels are checked against prior saved command rows, never the next action."""
    issues = []
    try:
        if measurement.get("schema") != "geometric-measurements/v2" or measurement.get("jaw_state_semantics") != JAW_STATE_SEMANTICS:
            raise ValueError("commanded_measurement_semantics_required")
        if pre.get("reset_provenance", {}).get("normal_initial_reset_returned") is not True:
            raise ValueError("commanded_reset_provenance_required")
        rows, sensors = measurement["rows"], measurement["sensor_samples"]
        # R4 controlled stops may retain an acquired terminal observation before
        # dispatch, or one partially executed action. No future row is invented.
        expected_counts = {len(rows)//12}
        if allow_terminal_observation:
            expected_counts = {(len(rows)+11)//12}
            if len(rows) % 12 == 0:
                expected_counts.add(len(rows)//12+1)
        if len(sensors) not in expected_counts or not sensors:
            raise ValueError("commanded_sensor_count")
        for index, sensor in enumerate(sensors):
            if allow_terminal_observation:
                prior_clock = measurement['origin_clock'] if index == 0 else rows[12*index-1]['pose']['clock_after']
                if sensor['native_clock'] != prior_clock or sensor['control_tick'] != 6*index:
                    raise ValueError('commanded_sensor_terminal_clock')
            jaw = sensor["jaw_state"]
            bits = jaw.get("bits")
            if (not isinstance(bits, list) or len(bits) != 2
                    or any(type(bit) is not int or bit not in (0, 1) for bit in bits)
                    or jaw.get("semantics") != JAW_STATE_SEMANTICS
                    or jaw.get("dtype") != "uint8" or jaw.get("order") != ["LEFT", "RIGHT"]
                    or jaw.get("source") != "env.action_manager.get_term('policy_action').commanded_jaw_state"
                    or jaw.get("acquisition_native_clock") != sensor["native_clock"]
                    or jaw.get("command_issue_time_status") != "NOT_PERSISTED"
                    or "command_issue_time_s" not in jaw or jaw["command_issue_time_s"] is not None):
                raise ValueError("commanded_sensor_definition:" + str(index))
            if index == 0:
                expected = dict(kind="reset_open", action_frame=None, control_tick=None,
                    substep=None, observed_native_clock=None, observed_wall_monotonic_ns=None)
                expected_bits = [1, 1]
            else:
                previous = rows[12*index-1]
                expected = dict(kind="accepted_control_action", action_frame=index-1,
                    control_tick=previous["context"]["control_tick"], substep=previous["context"]["substep"],
                    observed_native_clock=previous["pose"]["clock_after"],
                    observed_wall_monotonic_ns=previous["wall_monotonic_ns"])
                expected_bits = previous["commands"]
                # Coarse native-platform clocks may repeat at consecutive calls.
                # Causality is established by the exact row/native-clock join.
                if sensor["wall_monotonic_ns"] < previous["wall_monotonic_ns"]:
                    raise ValueError("commanded_sensor_precedes_source:" + str(index))
            if bits != expected_bits or jaw.get("preceding_command") != expected:
                raise ValueError("commanded_sensor_prior_command_disagreement:" + str(index))
            if 12*index < len(rows) and sensor["wall_monotonic_ns"] > rows[12*index]["wall_monotonic_ns"]:
                raise ValueError("commanded_sensor_not_before_next_action:" + str(index))
        if pre["rgb_metadata"].get("jaw_state") != sensors[0]["jaw_state"]:
            raise ValueError("commanded_reset_shared_acquisition_disagreement")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        issues.append("commanded_jaw:" + str(exc))
    return issues


def check_measurements(measurement, owner, *, expected_actions=None, witness_transitions=None, run_type="HARNESS", requested_source=None):
    """Recompute gate from saved measured rows, never trust the admission bit."""
    issues = []
    rows = measurement.get("rows", [])
    witness = witness_transitions is not None
    outcome = {"status": "FAIL", "issues": ["incomplete_candidate"]}
    try:
        episode = measurement["episode_id"]
        origin = measurement["origin_clock"]
        partial = len(rows) == 1680
        if len(rows) not in (1680, 2400) or measurement.get("lifecycle_events") != []:
            raise ValueError("full_or_exact_geometric_refusal_window_required")
        commands = frozen_candidate_actions() if expected_actions is None else expected_actions
        if len(commands) < (len(rows)+11)//12:
            raise ValueError("selected_command_evidence_incomplete")
        gate = None
        if witness:
            from .side_opening_admission import replay
            gate = replay(rows, witness_transitions, run_type=run_type)
        previous_clock = origin
        for i, row in enumerate(rows):
            if row["context"] != dict(episode_id=episode, control_tick=i//2, substep=i%2):
                raise ValueError("row_order_or_episode")
            if row["commands"] != commands[i//12, 3:].tolist():
                raise ValueError("frozen_jaw_commands")
            jaws = row["jaws"]
            clock = row["pose"]["clock_after"]
            pose = row["pose"]
            if (pose.get("valid") is not True or pose.get("issues") != [] or pose.get("units") != "m"
                    or pose.get("pose_convention") != "position_xyz_quaternion_wxyz"
                    or pose.get("identities", {}).get("chicken_actor") != CHICKEN_BODY
                    or pose.get("identities", {}).get("gripper_actor") != GRIPPER_BODY
                    or pose.get("clock_before") != clock or type(clock[0]) is not int
                    or clock[0] != origin[0]+i+1 or type(clock[1]) not in (int, float)
                    or not math.isfinite(clock[1]) or clock[1] <= previous_clock[1]):
                raise ValueError("pose_or_native_clock_invalid")
            previous_clock = clock
            for name in ("robot_root_world_wxyz", "gripper_world_wxyz", "chicken_actor_world_wxyz",
                         "tcp_world_wxyz", "tcp_base_wxyz", "chicken_in_gripper_wxyz"):
                _measured_pose(pose["poses"][name])
            if (jaws["clock_before"] != clock or jaws["clock_after"] != clock
                    or jaws["names"] != [f"PrismaticJoint{j}" for j in range(1, 5)]
                    or jaws["units"] != "m" or jaws["source"] != "robot.root_physx_view.get_dof_positions"
                    or len(jaws["positions_m"]) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in jaws["positions_m"])):
                raise ValueError("required_jaw_measurement_invalid")
            last_refusal = partial and i == 1679
            refusal_error = "CaptureFailure:geometric_admission_refused:FALSE"
            if witness and gate and gate["status"] == "UNAVAILABLE":
                refusal_error = "CaptureFailure:geometric_admission_unavailable:" + repr(gate["issues"])
            expected_issues = [refusal_error] if last_refusal else []
            if row.get("issues") != expected_issues or type(row.get("wall_monotonic_ns")) is not int:
                raise ValueError("row_errors_or_wall_provenance_missing")
            if row.get("runtime") != dict(resetting=False, truncated=False, terminated=False, invalid=last_refusal):
                raise ValueError("runtime_invalidation")
            if any(k in row for k in ("contact", "predicate", "observation")):
                raise ValueError("not_slim_geometric_rows")
            if ("geometric_eligibility" in row) != (i == 1679):
                raise ValueError("admission_not_once_at_839_1")
        if not witness:
            gate = evaluate_history([history_sample(r) for r in rows[1619:1680]])
        if gate["status"] not in (("FALSE", "UNAVAILABLE") if partial and witness else ("FALSE",) if partial else ("TRUE",)) or rows[1679].get("geometric_eligibility") != gate:
            raise ValueError("geometric_admission_replay_mismatch_or_refused")
        records = owner["event_evidence"]["records"]
        requests = [r for r in records if r["kind"] == "activation_request"]
        if partial:
            if requests or any(r["attachment"]["status"] != "UNATTACHED" for r in rows):
                raise ValueError("refused_geometry_cannot_author")
            return dict(issues=[], end_state=dict(status="NOT_EVALUABLE", decision="TRAJECTORY_DECISION_REQUIRED",
                                                  reason="geometric_admission_unavailable" if gate["status"] == "UNAVAILABLE" else "geometric_admission_refused"))
        if (len(requests) != 1 or requests[0]["context"] != rows[1679]["context"]
                or requests[0]["native_clock"] != rows[1679]["pose"]["clock_after"]
                or requests[0]["eligibility"]["geometric_eligibility"] != gate
                or requests[0]["issues"] or requests[0]["eligibility_issues"]):
            raise ValueError("owner_request_not_correlated_to_geometric_gate")
        outcome = evaluate_end_state(rows, episode_id=episode, origin_step=origin[0], origin_time=origin[1],
                                     admission_policy="side-opening-progress/v1" if witness else SOURCE)
        if expected_actions is not None:
            from .stage3_policy import with_requested_motion
            outcome = with_requested_motion(outcome, expected_actions, **({"source": requested_source} if requested_source else {}))
        if not outcome.get("attachment_history_pass"):
            issues.extend(outcome["issues"])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        issues.append(str(exc))
    return dict(issues=issues, end_state=outcome)


def adjudicate_slim(attempt_root, *, policy_check=None):
    root = Path(attempt_root).resolve()
    output = root / "capture" / "experiment"
    issues, process_issues = [], []
    outcome = {"status": "NOT_EVALUABLE"}
    scope = {"acceptance_eligible": False}
    process_checked = False
    demo = {"status": "NOT_SELECTED"}
    def require(condition, message, group=issues):
        if not condition:
            group.append(message)
    try:
        result, manifest = load(output / "result.json"), load(output / "artifact_manifest.json")
        invocation, process = load(root / "invocation.json"), load(root / "capture" / "process_result.json")
        from .attachment_run_policy import check_run_type, verify_evidence_authority
        scope = check_run_type(invocation, result, process)
        # Harness can have valid diagnostics, never an acceptance PASS.
        issues.extend(i for i in scope["issues"] if i != "harness_output_not_acceptance_evidence")
        if invocation.get("run_type") == "EVIDENCE":
            authority = verify_evidence_authority(invocation, root)
            if isinstance(authority, dict):
                issues.extend(authority.get("issues", []))
        version = result.get("schema")
        witness = invocation.get("stage3_witness") is True
        if witness:
            require(invocation.get("run_type") in ("HARNESS", "EVIDENCE") and invocation.get("stage3_learned") is True
                    and result.get("stage3_witness") is True and process.get("stage3_witness") is True,
                    "witness_scope_disagreement")
        else:
            require(not result.get("stage3_witness") and not process.get("stage3_witness"), "unselected_witness")
        require(version in ("geometric-slim-capture/v1", "geometric-slim-capture/v2") and result.get("admission_policy") == ("side-opening-progress/v1" if witness else SOURCE),
                "explicit_slim_profile_required")
        actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
        require(actual == set(manifest["files"]) | {"artifact_manifest.json"}, "artifact_membership_mismatch")
        for relative, expected in manifest["files"].items():
            path = (output / relative).resolve()
            require(path.is_relative_to(output) and path != output, "artifact_path_escape")
            if path.is_relative_to(output):
                require(identity(path) == expected, "artifact_identity:" + relative)
        stage1 = invocation.get("stage1_demo") is True
        stage3 = invocation.get("stage3_learned") is True
        require(not stage3 or policy_check is not None, "stage3_independent_policy_check_required")
        require(result.get("stage1_demo", False) is stage1, "stage1_selection_disagreement")
        if stage1 and invocation.get("run_type") == "HARNESS":
            from .stage1_demo import check_startup_attempt
            return check_startup_attempt(root, result, process, invocation, issues)
        if stage1:
            require(invocation.get("run_type") == "EVIDENCE" and process.get("stage1_demo") is True,
                    "stage1_evidence_process_scope")
            from .stage1_demo import check_demo
            demo = check_demo(output)
            require(demo["status"] == "PASS", "demo:" + repr(demo["issues"]))
        require(not any("contact" in p.lower() or (p.startswith("episode/") and not stage1) for p in actual), "non_slim_artifact")
        for key in ("recording_errors", "cleanup_errors"):
            require(result.get(key) == [], "saved_" + key)
        measurement = load(output / "measurements.json")
        if version == "geometric-slim-capture/v2":
            require(result.get("jaw_state_semantics") == JAW_STATE_SEMANTICS, "commanded_result_semantics_required")
            require(measurement.get("schema") == "geometric-measurements/v2", "commanded_measurement_version_required")
        else:
            require(measurement.get("schema") == "geometric-measurements/v1"
                    and "jaw_state_semantics" not in measurement and "jaw_state_semantics" not in result
                    and all("jaw_state" not in sensor for sensor in measurement.get("sensor_samples", [])),
                    "historical_capture_definition_changed")
        checked = check_measurements(measurement, load(output / "owner_before_cleanup.json"),
            expected_actions=policy_check["actions"] if policy_check is not None else None,
            witness_transitions=[_load(line) for line in (output / "witness_transitions.jsonl").read_bytes().splitlines()] if witness else None, run_type=invocation["run_type"])
        issues.extend(checked["issues"])
        outcome = checked["end_state"]
        require(result["experiment_result"] == outcome == load(output / "candidate_result.json"), "end_state_recomputation_mismatch")
        partial = outcome.get("decision") == "TRAJECTORY_DECISION_REQUIRED"
        require(result.get("counts") == (dict(completed_control_ticks=839, observed_substeps=1680, completed_policy_actions=139)
            if partial else dict(completed_control_ticks=1200, observed_substeps=2400, completed_policy_actions=200)),
                "incomplete_counts")
        sensors = measurement["sensor_samples"]
        require(len(sensors) == (140 if partial else 200), "camera_action_count")
        for i, sensor in enumerate(sensors):
            require(sensor.get("action_frame") == i and sensor.get("control_tick") == 6*i
                    and sensor.get("native_clock") == (measurement["origin_clock"] if i == 0 else measurement["rows"][12*i-1]["pose"]["clock_after"]),
                    "camera_acquisition_not_precommand:" + str(i))
        pre = load(output / "pre_candidate.json")
        require(pre["native_clock"] == measurement["origin_clock"] == pre["pose"]["clock_after"], "pre_candidate_clock")
        if version == "geometric-slim-capture/v2":
            issues.extend(check_commanded_sensors(measurement, pre))
            for i, sensor in enumerate(sensors):
                bundle = load(output / "rgb" / f"{i:06d}.json")
                require(bundle["metadata"].get("jaw_state") == sensor.get("jaw_state"),
                        "commanded_rgb_metadata_disagreement:" + str(i))
        rgb = validate_rgb(output / "rgb", expected_count=141 if partial else 201, sensors=sensors, pre_candidate=pre)
        require(rgb["status"] == "PASS", "rgb:" + repr(rgb["issues"]))
        configuration = result["effective_configuration"]
        for key, wanted in dict(device="cuda:0", simulation_device="cuda:0", physics_device="cuda:0", gpu_sim=True,
                gpu_pipeline=True, gpu_dynamics=True, broadphase="GPU", num_envs=1, decimation=2, render_interval=2,
                cameras_enabled=True, fast_shutdown=True, has_gui=False, configured_seed=42, action_dimension=5).items():
            require(type(configuration.get(key)) is type(wanted) and configuration[key] == wanted, "configuration:"+key)
        require(configuration.get("physics_dt_s") == 1/120 and configuration.get("control_dt_s") == 1/60, "physics_rates")
        for key, wanted in dict(exit_code=0, root_exit_verified=True, tree_empty_verified=True, capture_complete=True,
                forced_termination=False, timed_out=False, parent_termination_requested=False, kill_on_job_close_used=False,
                kill_on_job_close_possible=False, startup_containment_used=False, autonomous_tree_exit=True,
                job_assigned_before_resume=True, root_resumed=True, job_disposed_after_observation=True,
                job_members_before_disposal=0, absolute_deadline_ms=1200000, parent_launch_boundary_ms=0).items():
            require(type(process.get(key)) is type(wanted) and process[key] == wanted, "process:"+key, process_issues)
        require(process.get("containment_errors") == [], "containment_errors", process_issues)
        for key in ("root_exit_observed_ms", "job_empty_observed_ms", "drain_completed_observed_ms"):
            require(type(process.get(key)) is int and 0 <= process[key] <= 1200000, "deadline:"+key, process_issues)
        process_checked = True
        for key, relative in (("stdout", "stdout.bin"), ("stderr", "stderr.bin"), ("completed_journal", "structured/events.jsonl")):
            path = root / "capture" / relative
            expected = process[key]
            require({k:expected.get(k) for k in ("bytes", "sha256")} == identity(path)
                    and Path(expected["path"]).resolve() == path, "raw_capture:"+key)
        require(process["invocation_sha256"] == identity(root / "invocation.json")["sha256"], "invocation_identity")
        require(invocation.get("timeout_ms") == 1200000 and Path(invocation["output_directory"]).resolve() == root / "capture",
                "invocation_contract")
        events = [ _load(line) for line in (root / "capture" / "structured" / "events.jsonl").read_bytes().splitlines() ]
        require([r["sequence"] for r in events] == list(range(len(events))), "journal_sequence")
        require([e["kind"] for e in events[-2:]] == ["PRE_CLOSE_EVIDENCE_COMPLETE", "CLOSE_REQUESTED"], "preclose_terminal_intent")
        require(all(e["manifest"] == identity(output / "artifact_manifest.json") for e in events[-2:]), "terminal_manifest_identity")
        require(result.get("execution_provenance", {}).get("executed_commit") == invocation.get("executed_commit"), "executed_commit_disagreement")
        # Even OS zero and valid data cannot erase an application failure.
        require(result.get("primary_errors") == (measurement["rows"][-1]["issues"] if partial and witness else ["CaptureFailure:geometric_admission_refused:FALSE"] if partial else []),
                "saved_primary_errors")
    except Exception as exc:
        issues.append("saved_evidence:" + type(exc).__name__ + ":" + str(exc))
    accepted = process_checked and not issues and not process_issues and outcome.get("status") == "PASS" and scope.get("acceptance_eligible")
    harness_complete = process_checked and not issues and not process_issues and scope.get("run_type") == "HARNESS"
    return dict(schema="geometric-slim-adjudication/v1", status="PASS" if accepted else "HARNESS_COMPLETE" if harness_complete else "FAIL",
        CAPTURE_INTEGRITY="PASS" if not issues else "FAIL", PROCESS_COMPLETION="PASS" if process_checked and not process_issues else "NOT_CONFIRMED",
        END_STATE_RESULT=outcome.get("status"), end_state=outcome, run_scope=scope,
        DEMO_INTEGRITY=demo["status"], demo=demo,
        issues=issues, process_issues=process_issues, native_launch_authority="NONE")
