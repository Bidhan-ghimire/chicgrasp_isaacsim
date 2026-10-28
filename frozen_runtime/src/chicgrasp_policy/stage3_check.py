"""Independent saved-file Stage3 joins and outcome classification; no inference."""
import json
import math
from pathlib import Path

import numpy as np

from . import interface as api
from .geometric_capture_check import adjudicate_slim, identity, load
from .stage1_demo import validate_visuals
from .stage3_policy import BINDING, CHECKPOINT_SHA256, array_identity, binding


def expected_chunks(total_actions=200):
    """Independent authorized schedule, including variable final slices."""
    if type(total_actions) is not int or not 200 <= total_actions <= 279:
        raise ValueError("invalid_authorized_bound")
    out=[]
    for q in range(50,total_actions,12):
        n=min(12,total_actions-q);k=(q-50)//12
        out.append(dict(chunk_index=k,learned_start=q-50,global_start=q,
            global_observations=[q-1,q],execute_count=n,full_prediction_slots=list(range(1,n+1))))
    return out


def recompute_selection(raw, count):
    require(raw.shape == (16, 5) and raw.dtype.kind == "f" and np.isfinite(raw).all(), "raw_prediction_contract")
    api.validate_policy_action(raw[1:13])
    selected = raw[1:count+1].astype(np.float64).copy()
    selected[:, 3:] = api.threshold_jaw_commands(selected[:, 3:])
    dispatched = selected.astype(np.float32)
    api.validate_policy_action(dispatched)
    return dispatched


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def check_command_provenance(output, *, scripted=False):
    """Forward lifecycle receipts; never infer dispatch or completion from selection."""
    root = Path(output)
    issues, counts = [], dict(selected=0, dispatch_requested=0, acceptance_observed=0, completed=0)
    try:
        result = load(root/'result.json')
        if 'command_provenance_schema' not in result:
            return dict(status='HISTORICAL_NOT_RECORDED')
        prefix = 'script' if scripted else 'stage3'
        require(result['command_provenance_schema'] == prefix+'-command-lifecycle/v1', 'command_schema')
        decisions = lines(root/(prefix+'_actions.jsonl'))
        rows = load(root/'measurements.json')['rows']
        by_context = {(r['context']['control_tick'], r['context']['substep']):r for r in rows}
        phases = ['SELECTED_VALIDATED', 'DISPATCH_REQUESTED', 'ACCEPTANCE_OBSERVED', 'COMPLETED']
        index, phase, previous_time = 0, 0, -1
        for e in lines(root/(prefix+'_command_events.jsonl')):
            require(e['schema'] == result['command_provenance_schema'] and e['global_action'] == index
                and e['event'] == phases[phase] and e['wall_monotonic_ns'] >= previous_time, 'command_event_order')
            previous_time = e['wall_monotonic_ns']
            require(e['scheduled_boundary'] == [6*index+(5 if phase == 3 else 0), 1 if phase == 3 else 0], 'command_boundary')
            if phase == 0:
                require(e['decision'] == decisions[index], 'selected_receipt_decision')
                histories = [index] if scripted or index < 50 else expected_chunks(279 if result.get("envelope_v2") else 200)[(index-50)//12]['global_observations']
                require(e['observation_references'] == [f'rgb/{i:06d}.json' for i in histories], 'command_observation_references')
            elif phase == 1:
                require(e['native_clock'] == decisions[index]['native_clock'], 'dispatch_request_clock')
            elif phase == 2:
                row = by_context[(6*index,0)]
                from .side_opening_admission import validate_command
                from .geometric_capture import history_sample
                validate_command(history_sample(row), e['accepted_command'])
                require(row['accepted_command'] == e['accepted_command'], 'accepted_receipt_row')
            else:
                action_rows = rows[index*12:(index+1)*12]
                require(len(action_rows) == 12 and not any(r['issues'] or any(r['runtime'].values()) for r in action_rows), 'completion_without_valid_substeps')
                require(e['native_clock'] == action_rows[-1]['pose']['clock_after'], 'completion_clock')
            counts[list(counts)[phase]] += 1
            phase += 1
            if phase == 4:
                phase, index = 0, index+1
        require(counts['selected'] == len(decisions), 'missing_selected_receipt')
        require(counts['completed'] == result['counts']['completed_policy_actions'], 'completion_receipt_count')
        require(counts['acceptance_observed'] == sum('accepted_command' in r for r in rows), 'acceptance_receipt_count')
        require(not result['recording_errors'], 'recording_errors')
    except Exception as exc:
        issues.append(type(exc).__name__+':'+str(exc))
    return dict(status='PASS' if not issues else 'FAIL', issues=issues, counts=counts,
        meaning='Dispatch request is not acceptance; acceptance is not action completion. No physical acceptance conferred.')


def check_policy(output, *, development_binding=None):
    """Rebuild selected commands from raw outputs and inputs from saved RGB/state."""
    from PIL import Image
    root = Path(output)
    issues, actions, details = [], [], {}
    try:
        observations = lines(root / "stage3_observations.jsonl")
        decisions = lines(root / "stage3_actions.jsonl")
        predictions = lines(root / "stage3_inferences.jsonl")
        measurement = load(root / "measurements.json")
        result = load(root / "result.json")
        v2 = result.get("envelope_v2") is True
        from .attachment_capture import event_anchor
        close = event_anchor(measurement["rows"]) if v2 else None
        total = max(200,close+80) if close is not None else 200
        forecast = expected_chunks(279 if v2 else 200)
        details.update(actual_inferences=sum(p.get("status") in ("VALID", "REJECTED") for p in predictions),
            inference_attempts=len(predictions), selected_setup_actions=min(len(decisions), 50),
            selected_learned_actions=max(0, len(decisions)-50),
            completed_setup_actions=min(result["counts"]["completed_policy_actions"], 50),
            completed_learned_actions=max(0, result["counts"]["completed_policy_actions"]-50))
        value = binding() if development_binding is None else development_binding
        checkpoint_sha256 = CHECKPOINT_SHA256 if development_binding is None else value["checkpoint_identity"]["sha256"]
        from .stage3_storage import ESTIMATE, WITNESS_ESTIMATE
        if result.get("stage3_witness"):
            ESTIMATE = WITNESS_ESTIMATE
        storage = load(root / "stage3_storage_prelaunch.json")
        from .stage3_storage import envelope_estimate
        require(storage["estimate"] == (envelope_estimate(load(ESTIMATE)) if v2 else load(ESTIMATE)) and storage["estimate_sha256"] == identity(ESTIMATE)["sha256"]
            and type(storage["free_bytes"]) is int and storage["free_bytes"] >= storage["estimate"]["required_bytes"]
            and storage["passed"] is True, "storage_prelaunch_evidence")
        checkpoint = load(root / "stage3_checkpoint.json")
        require(checkpoint["binding"] == value and checkpoint["device"] == "cuda:0", "checkpoint_binding_or_device")
        if development_binding is not None:
            from .r4_live_binding import validate_checkpoint_record
            validate_checkpoint_record(checkpoint, value)
        else:
            metadata = checkpoint["checkpoint_metadata"]
            for key, wanted in dict(source=value["source"], normalization=value["normalization"],
                    training_configuration=value["training_configuration"], model_configuration=value["model_configuration"],
                    optimizer_step=10000, weights_kind="EMA", evaluation=value["offline_metrics"],
                    executed_commit=value["stage2_executed_commit"]).items():
                require(metadata.get(key) == wanted, "checkpoint_metadata:"+key)
        require(lines(root / "stage3_substeps.jsonl") == measurement["rows"], "incremental_substeps_disagree")
        sensors = measurement["sensor_samples"]
        require(len(observations) == len(sensors) and 0 < len(observations) <= total, "live_acquisition_count")
        require(len(decisions) <= len(observations) and len(decisions) >= len(measurement["rows"])//12,
                "decisions_vs_reached_physics")
        histories = {}
        needed = {index for spec in forecast for index in spec["global_observations"]}
        for index, row in enumerate(observations):
            require(row["global_action"] == index and row["learned_action"] == (index-50 if index >= 50 else None)
                and row["rgb_bundle"] == f"rgb/{index:06d}.json" and row["acquisition"] == sensors[index],
                "observation_indices_or_acquisition:"+str(index))
            sensor = sensors[index]
            expected_clock = measurement["origin_clock"] if index == 0 else measurement["rows"][12*index-1]["pose"]["clock_after"]
            require(sensor["native_clock"] == expected_clock and sensor["control_tick"] == 6*index,
                    "live_observation_native_clock")
            low = {"arm_joint_pos": np.asarray(row["arm_joint_pos"], dtype=np.float32),
                   "jaw_state": np.asarray(row["jaw_state"], dtype=np.uint8)}
            require(row["jaw_state"] == ([1, 1] if index == 0 else decisions[index-1]["action"][3:]),
                    "future_or_noncausal_jaw_label")
            mapping = {}
            for camera in ("camera_left", "camera_right"):
                with Image.open(root / "rgb" / f"{index:06d}_{camera}.png") as image:
                    mapping[camera+"_rgb"] = np.asarray(image).copy()
            mapping.update(low)
            observation = api.validate_policy_observation(mapping)
            require(row["input_identities"] == {k: array_identity(v) for k, v in mapping.items()},
                    "live_input_bytes_disagree:"+str(index))
            if index in needed:
                histories[index] = observation
        from .attachment_capture import frozen_candidate_actions
        setup = frozen_candidate_actions() if development_binding is None else np.asarray(value["setup_actions"], dtype=np.float32)
        actions.extend(setup[:min(50, len(decisions))])
        for index, prediction in enumerate(predictions):
            spec = forecast[index]
            for key, wanted in spec.items():
                require(prediction[key] == wanted, "inference_schedule:"+key)
            if v2:
                prior = event_anchor(measurement["rows"][:12*spec["global_start"]])
                authorized = max(200,prior+80) if prior is not None else 200
                require(prediction["authorized_bound_at_prediction"] == authorized and prediction["forecast_retained_for_possible_extension"] is True, "provisional_forecast_bound")
                require(spec["global_start"] < authorized and index < 20, "prediction_outside_authorized_horizon")
            require(prediction["seed"] == value["evaluation_seed"]+12*index
                and prediction["checkpoint_sha256"] == checkpoint_sha256, "prediction_identity_seed")
            require(prediction["native_clock"] == sensors[spec["global_start"]]["native_clock"], "inference_native_clock")
            packed = api.pack_model_observation(tuple(histories[i] for i in spec["global_observations"]))
            require(prediction["input_identities"] == {k: array_identity(v) for k, v in packed.items()},
                    "model_inputs_not_exact_live_history")
            path = root / "predictions" / f"{index:02d}.npy"
            require(prediction["raw_file"] == path.relative_to(root).as_posix()
                and prediction["raw_file_sha256"] == identity(path)["sha256"], "raw_prediction_file_identity")
            require(type(prediction["start_monotonic_ns"]) is int and
                prediction["end_monotonic_ns"] >= prediction["start_monotonic_ns"] >= sensors[spec["global_start"]]["wall_monotonic_ns"],
                "inference_acquisition_order")
            raw = np.load(path, allow_pickle=False)
            if development_binding is not None and prediction["status"] == "REJECTED":
                require(index == len(predictions)-1 and len(decisions) == spec["global_start"], "rejection_must_be_terminal_before_selection")
                from .stage3_policy import selected_actions
                try:
                    selected_actions(raw, spec["execute_count"])
                except (ValueError, TypeError) as exc:
                    require(prediction.get("error") == type(exc).__name__+":"+str(exc), "rejection_guard_reason_disagreement")
                else:
                    raise ValueError("saved_rejection_without_guard_failure")
                continue
            selected = recompute_selection(raw, spec["execute_count"])
            require(prediction["status"] == "VALID" and np.array_equal(selected, prediction["selected"]), "raw_selected_disagreement")
            actions.extend(selected)
        # Only dispatched/selected decisions, not an unexecuted chunk tail.
        actions = np.asarray(actions[:len(decisions)], dtype=np.float32)
        require(len(actions) == len(decisions), "missing_prediction_for_selected_action")
        for index, decision in enumerate(decisions):
            learned = index-50 if index >= 50 else None
            expected = dict(global_action=index, learned_action=learned, setup_action=index if index < 50 else None,
                control_tick_start=6*index, native_clock=sensors[index]["native_clock"],
                source="approved_setup_hold" if index < 50 else "checkpoint_prediction",
                chunk_index=None if index < 50 else learned//12,
                full_prediction_slot=None if index < 50 else learned%12+1,
                checkpoint_sha256=None if index < 50 else checkpoint_sha256, action=actions[index].tolist())
            require(decision == expected, "action_not_selected_checkpoint_or_setup:"+str(index))
        for i, row in enumerate(measurement["rows"]):
            index = i//12
            require(row["policy_indices"] == dict(global_action=index, setup_action=index if index < 50 else None,
                learned_action=index-50 if index >= 50 else None), "substep_policy_indices")
        require(result["policy"] == dict(details, schedule=expected_chunks(total), checkpoint_sha256=checkpoint_sha256,
            **(dict(envelope_v2=True,first_accepted_close=close,total_actions=total) if v2 else {})), "policy_count_summary")
        validate_visuals(load(root / "stage1_visuals_before_candidate.json"))
        if len(measurement["rows"]) == 12*total:
            require(len(decisions) == total and len(predictions) == len(expected_chunks(total)), "full_learned_schedule_required")
            validate_visuals(load(root / "stage1_visuals_after_candidate.json"))
        else:
            reached = len(observations) if development_binding is not None else len(decisions)
            require(len(predictions) == sum(spec["global_start"] < reached for spec in expected_chunks(total)), "partial_inference_count")
    except Exception as exc:
        issues.append(type(exc).__name__+":"+str(exc))
    return dict(status="PASS" if not issues else "FAIL", issues=issues, actions=actions, counts=details)


def check_process(process, *, absolute_ms=1200000):
    """Independent GPU qualification even when capture files are incomplete."""
    issues = []
    if type(absolute_ms) is not int or not 0 < absolute_ms <= 1200000:
        return ['process:absolute_limit_invalid']
    expected = dict(exit_code=0, root_exit_verified=True, tree_empty_verified=True, capture_complete=True,
        forced_termination=False, timed_out=False, parent_termination_requested=False, kill_on_job_close_used=False,
        kill_on_job_close_possible=False, startup_containment_used=False, autonomous_tree_exit=True,
        job_assigned_before_resume=True, root_resumed=True, job_disposed_after_observation=True,
        job_members_before_disposal=0, absolute_deadline_ms=absolute_ms, parent_launch_boundary_ms=0,
        termination_and_drain_cap_ms=5000, containment_budget_overrun=False,
        effective_deadline_ms=absolute_ms, timeout_ms=absolute_ms)
    for key, wanted in expected.items():
        if type(process.get(key)) is not type(wanted) or process[key] != wanted:
            issues.append("process:"+key)
    if process.get("containment_errors") != []:
        issues.append("process:containment_errors")
    for key in ("root_exit_observed_ms", "job_empty_observed_ms", "drain_completed_observed_ms",
                "final_journal_read_completed_ms", "stream_files_finalized_ms", "job_disposed_ms"):
        if type(process.get(key)) is not int or not 0 <= process[key] <= absolute_ms:
            issues.append("deadline:"+key)
    if (type(process.get("cleanup_elapsed_ms")) not in (int, float)
            or not 0 <= process["cleanup_elapsed_ms"] <= 5000):
        issues.append("combined_containment_drain")
    duration = process.get("duration_seconds")
    if type(duration) not in (int, float) or not math.isfinite(duration) or not 0 <= duration <= absolute_ms/1000:
        issues.append("absolute_total_duration")
    return issues


def adjudicate_stage3(attempt_root):
    root = Path(attempt_root).resolve()
    output = root / "capture/experiment"
    policy = check_policy(output)
    checked = adjudicate_slim(root, policy_check=policy)
    provenance = check_command_provenance(output)
    if provenance['status'] != 'HISTORICAL_NOT_RECORDED':
        checked['COMMAND_PROVENANCE'] = provenance
        checked['issues'].extend('command_provenance:'+i for i in provenance['issues'])
    checked["issues"].extend("policy:"+i for i in policy["issues"])
    process_issues = []
    try:
        process = load(root / "capture/process_result.json")
        process_issues = check_process(process)
        invocation, result = load(root / "invocation.json"), load(output / "result.json")
        run_type = invocation.get("run_type")
        require(run_type in ("EVIDENCE", "HARNESS"), "stage3_run_type")
        require(all(item.get("stage3_learned") is True and item.get("run_type") == run_type
                    and not item.get("stage1_demo") and not item.get("stage0_mode")
                    for item in (invocation, result, process)), "stage3_scope")
        for item in (invocation, process):
            require(item.get("checkpoint_sha256") == CHECKPOINT_SHA256
                and item.get("stage3_binding_sha256") == identity(BINDING)["sha256"], "stage3_invocation_binding")
        if run_type == "EVIDENCE":
            reservation = load(invocation["evidence_authority"]["reservation"]["path"])
            require(reservation.get("stage3_learned") is True and
                reservation.get("checkpoint_sha256") == CHECKPOINT_SHA256 and
                reservation.get("stage3_binding_sha256") == identity(BINDING)["sha256"], "stage3_reservation_binding")
            if invocation.get("stage3_witness"):
                from .side_opening_admission import RULE_SHA256
                require(all(item.get("stage3_witness") is True and item.get("witness_rule_sha256") == RULE_SHA256
                            for item in (invocation, reservation, process)), "witness_reservation_binding")
        else:
            require(not invocation.get("evidence_authority") and not invocation.get("execution_approved"),
                    "harness_cannot_carry_evidence_authority")
    except Exception as exc:
        checked["issues"].append("stage3_scope:"+type(exc).__name__+":"+str(exc))
        if not process_issues and "process" not in locals():
            process_issues.append("process_record_unavailable")
    checked["process_issues"] = sorted(set(checked["process_issues"] + process_issues))
    checked["PROCESS_COMPLETION"] = "PASS" if not checked["process_issues"] else "NOT_CONFIRMED"
    checked["CAPTURE_INTEGRITY"] = "PASS" if not checked["issues"] else "FAIL"
    checked["STAGE2_MAX_ERROR_WAIVER"] = dict(applied=True, original_verdict=binding()["stage2_verdict"],
        maximum_m=0.00627686083316803, unchanged_limit_m=0.005, checkpoint_sha256=CHECKPOINT_SHA256)
    checked["policy"] = {k: v for k, v in policy.items() if k != "actions"}
    checked["admission"] = dict(status="NOT_REACHED")
    checked["attachment_confirmation"] = None
    try:
        from .geometric_capture import history_sample
        from .geometric_admission import evaluate_history
        rows = load(output / "measurements.json")["rows"]
        checked["runtime_violations"] = [dict(context=r.get("context"), issues=r.get("issues"),
            runtime=r.get("runtime")) for r in rows if r.get("issues") or any(r.get("runtime", {}).values())]
        if len(rows) >= 1680:
            if result.get("stage3_witness"):
                from .side_opening_admission import replay
                transitions = [json.loads(line) for line in (output / "witness_transitions.jsonl").read_text().splitlines()]
                checked["admission"] = replay(rows, transitions, run_type=run_type)
            else:
                checked["admission"] = evaluate_history([history_sample(r) for r in rows[1619:1680]])
        if len(rows) >= 1681:
            attachment = rows[1680]["attachment"]
            checked["attachment_confirmation"] = dict(status=attachment.get("status"),
                request_context=attachment.get("request_context"),
                confirmation_context=attachment.get("confirmation_context"))
    except Exception as exc:
        checked["admission"] = dict(status="UNAVAILABLE", reason=type(exc).__name__+":"+str(exc))
    valid = not checked["issues"] and not checked["process_issues"] and checked["run_scope"].get("acceptance_eligible")
    result = checked["END_STATE_RESULT"]
    suffix = "PASS" if valid and result == "PASS" else "FAIL" if valid and result == "FAIL" else "BLOCKED"
    checked["schema"] = "gate6-stage3-independent-adjudication/v1"
    checked["status"] = "GATE6_STAGE3_GPU_" + ("EVIDENCE_BLOCKED" if suffix == "BLOCKED" else "END_STATE_"+suffix)
    if checked["run_scope"].get("run_type") == "HARNESS":
        checked["status"] = "HARNESS_COMPLETE" if not checked["issues"] and not checked["process_issues"] else "HARNESS_INCOMPLETE"
        checked["acceptance_eligible"] = False
    if checked["CAPTURE_INTEGRITY"] != "PASS":
        checked["END_STATE_RESULT"] = "NOT_EVALUABLE"
    return checked
