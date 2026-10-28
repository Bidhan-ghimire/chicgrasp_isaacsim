"""Saved R4 HARNESS diagnostics. No predictions, native imports, or acceptance."""
from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from . import stage3_check as stage3
from . import geometric_capture_check as geometric
from .attachment_capture import _measured_pose, _angle, CHICKEN_BODY, GRIPPER_BODY
from .attachment_observation_evidence import validate as validate_rgb
from .interface import JAW_STATE_SEMANTICS
from .side_opening_admission import replay

load, identity, require = geometric.load, geometric.identity, stage3.require


def motion_metrics(rows, actions, issued_count):
    """Fixed historical windows, reporting partial data without filling any gap."""
    out = dict(baseline_context=[839, 1], rise_window=[840, 1079], hold_reference=[1079, 1],
        hold_window=[1080, 1199], rise_window_complete=len(rows) >= 2160,
        hold_window_complete=len(rows) == 2400, chicken_baseline_world_z_m=None,
        chicken_rise_max_m=None, measured_tcp_rise_max_m=None, requested_tcp_rise_max_m=None,
        hold_translation_drift_max_m=None, hold_rotation_drift_max_deg=None,
        rise_status='NOT_EVALUABLE', hold_status='NOT_EVALUABLE')
    if len(rows) > 1679:
        baseline = rows[1679]['pose']['poses']
        out['chicken_baseline_world_z_m'] = baseline['chicken_actor_world_wxyz'][2]
        rise = rows[1680:min(len(rows), 2160)]
        if rise:
            out['chicken_rise_max_m'] = max(r['pose']['poses']['chicken_actor_world_wxyz'][2] - out['chicken_baseline_world_z_m'] for r in rise)
            out['measured_tcp_rise_max_m'] = max(r['pose']['poses']['tcp_base_wxyz'][2] - baseline['tcp_base_wxyz'][2] for r in rise)
        if issued_count > 140:
            out['requested_tcp_rise_max_m'] = float(np.max(actions[140:min(issued_count, 180), 2]) - actions[139, 2])
        if out['rise_window_complete']:
            out['rise_status'] = 'PASS' if out['chicken_rise_max_m'] >= .05 else 'FAIL'
    if len(rows) > 2160:
        reference, reference_q = _measured_pose(rows[2159]['pose']['poses']['chicken_in_gripper_wxyz'])
        poses = [_measured_pose(r['pose']['poses']['chicken_in_gripper_wxyz']) for r in rows[2160:2400]]
        out['hold_translation_drift_max_m'] = max(math.dist(p[:3], reference[:3]) for p, q in poses)
        out['hold_rotation_drift_max_deg'] = max(_angle(q, reference_q) for p, q in poses)
        if out['hold_window_complete']:
            out['hold_status'] = 'PASS' if out['hold_translation_drift_max_m'] <= .01 and out['hold_rotation_drift_max_deg'] <= 5 else 'FAIL'
    return out


def check_rows(measurement, actions, primary, *, total_actions=200, decision=1679):
    """Validate every actually reached row; only exact terminal stop is allowed."""
    rows = measurement['rows']; origin = measurement['origin_clock']; previous = origin
    require(0 < len(rows) <= 12*total_actions and measurement['lifecycle_events'] == [], 'row_count_or_reset')
    for i, row in enumerate(rows):
        require(row['context'] == dict(episode_id=measurement['episode_id'], control_tick=i//2, substep=i%2), 'row_sequence')
        require(i//12 < len(actions) and row['commands'] == actions[i//12, 3:].tolist(), 'active_commands')
        pose, jaws = row['pose'], row['jaws']; clock = pose['clock_after']
        require(pose['valid'] is True and pose['issues'] == [] and pose['units'] == 'm'
            and pose['pose_convention'] == 'position_xyz_quaternion_wxyz'
            and pose['identities']['chicken_actor'] == CHICKEN_BODY
            and pose['identities']['gripper_actor'] == GRIPPER_BODY, 'required_pose_invalid')
        require(pose['clock_before'] == clock and type(clock[0]) is int and clock[0] == origin[0]+i+1
            and type(clock[1]) in (int, float) and math.isfinite(clock[1]) and clock[1] > previous[1], 'native_clock')
        previous = clock
        for key in ('robot_root_world_wxyz', 'gripper_world_wxyz', 'chicken_actor_world_wxyz',
                    'tcp_world_wxyz', 'tcp_base_wxyz', 'chicken_in_gripper_wxyz'):
            _measured_pose(pose['poses'][key])
        require(jaws['clock_before'] == jaws['clock_after'] == clock and jaws['units'] == 'm'
            and jaws['source'] == 'robot.root_physx_view.get_dof_positions'
            and jaws['names'] == ['PrismaticJoint'+str(j) for j in range(1, 5)]
            and len(jaws['positions_m']) == 4 and all(type(v) in (int,float) and math.isfinite(v) for v in jaws['positions_m']), 'jaws_invalid')
        terminal = i == len(rows)-1 and bool(row['issues'])
        require(row['issues'] == (primary if terminal else []) and type(row['wall_monotonic_ns']) is int,
                'row_error_provenance')
        require(row['runtime'] == dict(resetting=False, truncated=False, terminated=False, invalid=terminal), 'runtime_flags')
        require(not any(k in row for k in ('contact', 'predicate', 'observation')), 'non_slim_row')
        require(('geometric_eligibility' in row) == (i == decision), 'admission_boundary')


def reopen_and_release(rows, owner):
    first = None; closed = [False, False]
    for row in rows:
        index = row['context']['control_tick']//6
        if row['context']['control_tick'] % 6 or row['context']['substep'] or 'accepted_command' not in row:
            continue
        bits = row['commands']
        if index >= 50 and first is None and any(closed[j] and bits[j] == 1 for j in range(2)):
            first = dict(global_action=index, learned_action=index-50, context=row['context'],
                         native_clock=row['pose']['clock_after'], commands=bits)
        closed = [closed[j] or bits[j] == 0 for j in range(2)]
    releases = [r for r in owner['event_evidence']['records'] if r['kind'] == 'release']
    return first, releases


def classify_terminal(result, rows, predictions, owner, admission, policy):
    primary = result['primary_errors']
    reopen, releases = reopen_and_release(rows, owner)
    if not primary and len(rows) == 2400:
        return 'FULL_WINDOW_OBSERVED', None, reopen, releases
    if primary == ['CaptureFailure:geometric_admission_refused:FALSE'] and len(rows) == 1680 and admission.get('status') == 'FALSE':
        return 'ADMISSION_REFUSED', primary[0], reopen, releases
    if predictions and predictions[-1]['status'] == 'REJECTED' and policy.get('status') == 'PASS':
        rejected = predictions[-1]
        if primary == [rejected['error']] and len(rows) == rejected['global_start']*12:
            return 'POLICY_ACTION_REJECTED', primary[0], reopen, releases
    if primary == ['CaptureFailure:attachment_lost_or_invalid:[]'] and rows[-1]['attachment']['status'] == 'RELEASED':
        final = rows[-1]
        matched = [r for r in releases if r.get('cause') == 'jaw_open' and r.get('context') == final['context']
                   and r.get('native_clock') == final['pose']['clock_after'] and r.get('issues') == []]
        operations = [o for o in owner['operations'] if o.get('kind') == 'release' and o.get('cause') == 'jaw_open'
            and o.get('checkpoint_clock') == final['pose']['clock_after'] and o.get('usd_removed') is True
            and o.get('cleanup_confirmed') is True and o.get('issues') == []]
        if matched and operations and any(final['commands']) and final.get('accepted_command') and reopen:
            return 'POLICY_REOPEN_RELEASE', primary[0], reopen, releases
    return 'HARNESS_ERROR', primary[0] if primary else 'unexplained_incomplete_episode', reopen, releases


def check_attachment_prefix(rows, owner, admission, *, decision=1679, score_end=None):
    requests = [r for r in owner['event_evidence']['records'] if r['kind'] == 'activation_request']
    if (decision is None or len(rows) < decision+1) or admission.get('status') != 'TRUE':
        require(not requests and all(r['attachment']['status'] == 'UNATTACHED' for r in rows), 'unexpected_attachment')
        return
    pending = rows[decision]['attachment']; context = rows[decision]['context']; clock = rows[decision]['pose']['clock_after']
    require(len(requests) == 1 and requests[0]['context'] == context and requests[0]['native_clock'] == clock
        and requests[0]['eligibility']['geometric_eligibility'] == admission and requests[0]['issues'] == []
        and requests[0]['eligibility_issues'] == [], 'owner_request')
    require(pending['status'] == 'PENDING' and pending['request_context'] == context
        and pending['confirmation_context'] is None and pending['owned_usd_verified'] is True, 'pending_attachment')
    request_pose = pending['request_pose_reading']
    require(request_pose['valid'] is True and request_pose['issues'] == []
        and request_pose['clock_before'] == request_pose['clock_after'] == clock, 'preauthor_pose')
    reference, reference_q = _measured_pose(request_pose['poses']['chicken_in_gripper_wxyz'])
    for i, row in enumerate(rows):
        attachment = row['attachment']
        if i < decision:
            require(attachment['status'] == 'UNATTACHED', 'early_attachment')
        elif i > decision:
            released = attachment['status'] == 'RELEASED' and (i == len(rows)-1 or (score_end is not None and i > score_end))
            require(attachment['status'] == ('RELEASED' if released else 'CONFIRMED')
                and attachment['joint_path'] == pending['joint_path'] and attachment['generation'] == pending['generation']
                and attachment['request_context'] == context
                and attachment['confirmation_context'] == rows[decision+1]['context']
                and attachment['owned_usd_verified'] is (not released), 'attachment_history')
            if i == decision+1:
                pose, q = _measured_pose(row['pose']['poses']['chicken_in_gripper_wxyz'])
                require(math.dist(pose[:3], reference[:3]) <= .01 and _angle(q, reference_q) <= 5., 'confirmation_pose_change')


def _package(root, output, result, process, invocation):
    manifest = load(output/'artifact_manifest.json')
    actual = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    require(actual == set(manifest['files']) | {'artifact_manifest.json'}, 'artifact_membership')
    for relative, expected in manifest['files'].items():
        path = (output/relative).resolve()
        require(path.is_relative_to(output) and identity(path) == expected, 'artifact_identity:'+relative)
    for key, relative in (('stdout','stdout.bin'), ('stderr','stderr.bin'), ('completed_journal','structured/events.jsonl')):
        path = root/'capture'/relative; expected = process[key]
        require(Path(expected['path']).resolve() == path and {k:expected[k] for k in ('bytes','sha256')} == identity(path), 'stream_identity:'+key)
    require(process['invocation_sha256'] == identity(root/'invocation.json')['sha256'], 'invocation_identity')
    events = stage3.lines(root/'capture/structured/events.jsonl')
    require([r['sequence'] for r in events] == list(range(len(events))), 'journal_sequence')
    require([r['kind'] for r in events[-2:]] == ['PRE_CLOSE_EVIDENCE_COMPLETE', 'CLOSE_REQUESTED'], 'terminal_journal')
    require(all(r['manifest'] == identity(output/'artifact_manifest.json') for r in events[-2:]), 'terminal_manifest')
    require(result['execution_provenance']['executed_commit'] == invocation['executed_commit'], 'executed_commit')
    require(invocation['timeout_ms'] == 1200000 and Path(invocation['output_directory']).resolve() == root/'capture', 'invocation_contract')


def check_preparation_visual(output, preparation):
    """Authenticate both the writer's absolute path and its exact visual bytes."""
    output = Path(output).resolve()
    visual_path = (output/preparation['visual_file']).resolve()
    expected = preparation['visual_identity']
    require(visual_path.is_relative_to(output) and Path(expected['path']).resolve() == visual_path,
            'preparation_visual_path')
    require(identity(visual_path) == {key: expected[key] for key in ('bytes', 'sha256')},
            'preparation_visual_identity')
    from .stage1_demo import validate_visuals
    validate_visuals(load(visual_path))
    return True


def adjudicate(attempt_root):
    """Interpret one bounded development episode without modifying its verdicts."""
    root = Path(attempt_root).resolve(); output = root/'capture/experiment'
    answer = dict(schema='chicgrasp-r4-live-development/v1', status='HARNESS_INCOMPLETE', technical_clean=False,
        category='HARNESS_ERROR', acceptance_eligible=False, issues=[], process_issues=[],
        CAPTURE_INTEGRITY='FAIL', PROCESS_COMPLETION='NOT_CONFIRMED', admission={'status':'NOT_REACHED'},
        attachment_confirmation=None, native_launch_authority='NONE')
    try:
        from .r4_live_binding import verify_binding, validate_preparation
        invocation, process, result = load(root/'invocation.json'), load(root/'capture/process_result.json'), load(output/'result.json')
        point = invocation['r4_live_point']; value = verify_binding(point, **({'envelope_v3c':True} if invocation.get('envelope_protocol') == 'v3c' else {'envelope_v2':invocation.get('envelope_v2',False)}))
        from .attachment_run_policy import complete_harness_note
        observed_start = complete_harness_note({}, process, outcome='READ_ONLY_ACCOUNTING')
        answer.update(point=point, checkpoint_sha256=value['checkpoint_identity']['sha256'],
            primary_errors=result['primary_errors'], recording_errors=result['recording_errors'], cleanup_errors=result['cleanup_errors'],
            actual_kit_starts=observed_start['actual_kit_starts'], kit_start_basis=observed_start['kit_start_basis'],
            executed_commit=invocation['executed_commit'])
        _package(root, output, result, process, invocation)
        for item in (invocation, process, result):
            require(item.get('r4_live_point') == point and item.get('run_type') == 'HARNESS'
                and item.get('stage3_learned') is True and item.get('stage3_witness') is True, 'r4_harness_scope')
        require(not invocation.get('evidence_authority') and not invocation.get('execution_approved'), 'no_evidence_authority')
        for item in (invocation, process):
            require(item.get('checkpoint_sha256') == value['checkpoint_identity']['sha256']
                and item.get('r4_live_binding') == value, 'parent_binding')
        require(observed_start['actual_kit_starts'] == 1, 'one_actual_kit_start')
        preparation = load(output/'r4_live_preparation.json'); validate_preparation(preparation, value)
        check_preparation_visual(output, preparation)
        answer['preparation'] = preparation
        require(result['recording_errors'] == result['cleanup_errors'] == [], 'recording_or_cleanup_errors')
        configuration = result['effective_configuration']
        for key, wanted in dict(device='cuda:0', simulation_device='cuda:0', physics_device='cuda:0', gpu_sim=True,
                gpu_pipeline=True, gpu_dynamics=True, broadphase='GPU', num_envs=1, decimation=2, render_interval=2,
                cameras_enabled=True, fast_shutdown=True, has_gui=False, configured_seed=42, action_dimension=5).items():
            require(type(configuration.get(key)) is type(wanted) and configuration[key] == wanted, 'configuration:'+key)
        require(configuration['physics_dt_s'] == 1/120 and configuration['control_dt_s'] == 1/60, 'physics_rates')
        policy = stage3.check_policy(output, development_binding=value)
        provenance = stage3.check_command_provenance(output)
        answer.update(policy={k:v for k,v in policy.items() if k != 'actions'}, command_provenance=provenance)
        require(policy['status'] == provenance['status'] == 'PASS', 'policy_or_command_integrity')
        measurement = load(output/'measurements.json'); rows = measurement['rows']; owner = load(output/'owner_before_cleanup.json')
        require(measurement['schema'] == 'geometric-measurements/v2' and measurement['jaw_state_semantics'] == JAW_STATE_SEMANTICS, 'measurement_schema')
        if invocation.get('envelope_v2'):
            return adjudicate_envelope_v2(root, output, invocation, process, result, answer, value, policy, provenance, measurement, owner)
        check_rows(measurement, policy['actions'], result['primary_errors'])
        successful_rows = len(rows) - bool(rows[-1]['issues'])
        require(result['counts'] == dict(observed_substeps=len(rows), completed_control_ticks=successful_rows//2,
            completed_policy_actions=successful_rows//12), 'reached_boundary_counts')
        answer['terminal_boundary'] = [rows[-1]['context']['control_tick'], rows[-1]['context']['substep']]
        answer['counts'] = result['counts']
        pre = load(output/'pre_candidate.json')
        require(pre['native_clock'] == measurement['origin_clock'] == pre['pose']['clock_after'], 'pre_clock')
        sensor_issues = geometric.check_commanded_sensors(measurement, pre, allow_terminal_observation=True)
        require(not sensor_issues, 'causal_sensors:'+repr(sensor_issues))
        rgb = validate_rgb(output/'rgb', expected_count=len(measurement['sensor_samples'])+1,
            sensors=measurement['sensor_samples'], pre_candidate=pre)
        require(rgb['status'] == 'PASS', 'rgb:'+repr(rgb['issues']))
        predictions = stage3.lines(output/'stage3_inferences.jsonl')
        transitions = stage3.lines(output/'witness_transitions.jsonl')
        if len(rows) >= 1680:
            answer['admission'] = replay(rows, transitions, run_type='HARNESS')
            require(answer['admission'] == rows[1679]['geometric_eligibility'], 'admission_replay')
        check_attachment_prefix(rows, owner, answer['admission'])
        answer['motion'] = motion_metrics(rows, policy['actions'], provenance['counts']['dispatch_requested'])
        category, reason, reopen, releases = classify_terminal(result, rows, predictions, owner, answer['admission'], policy)
        answer.update(category=category, primary_reason=reason, first_policy_reopen=reopen, release_events=releases)
        require(category != 'HARNESS_ERROR', 'unexplained_terminal:'+str(reason))
        if len(rows) >= 1681:
            attachment = rows[1680]['attachment']
            answer['attachment_confirmation'] = {k:attachment.get(k) for k in ('status','request_context','confirmation_context')}
            require(attachment['status'] in ('CONFIRMED', 'RELEASED') and attachment['confirmation_context'] == rows[1680]['context'], 'attachment_confirmation')
        if len(rows) in (1680,2400):
            geometric_check = geometric.check_measurements(measurement, owner, expected_actions=policy['actions'],
                witness_transitions=transitions, run_type='HARNESS')
            require(not geometric_check['issues'], 'geometric:'+repr(geometric_check['issues']))
            answer['end_state'] = geometric_check['end_state']
            require(result['experiment_result'] == geometric_check['end_state'] == load(output/'candidate_result.json'), 'end_state_recomputation')
        else:
            answer['end_state'] = dict(status='NOT_EVALUABLE', reason='incomplete_fixed_windows')
        if category == 'FULL_WINDOW_OBSERVED':
            if answer['motion']['rise_status'] == 'FAIL': answer['category'] = 'RISE_SHORTFALL'
            elif answer['motion']['hold_status'] == 'FAIL': answer['category'] = 'HOLD_DRIFT'
        answer['CAPTURE_INTEGRITY'] = 'PASS'
        qualified_process = deepcopy(process)
        if category in ('ADMISSION_REFUSED','POLICY_ACTION_REJECTED','POLICY_REOPEN_RELEASE') and process.get('exit_code') == 1:
            qualified_process['exit_code'] = 0
            answer['controlled_stop_exit_code'] = 1
        answer['process_issues'] = stage3.check_process(qualified_process)
        require(not answer['process_issues'], 'process_qualification')
        answer.update(status='HARNESS_COMPLETE', technical_clean=True, CAPTURE_INTEGRITY='PASS', PROCESS_COMPLETION='PASS')
    except Exception as exc:
        answer['issues'].append(type(exc).__name__+':'+str(exc))
        answer['observed_category'] = answer['category']
        answer['category'] = 'HARNESS_ERROR'
        if 'process' in locals() and not answer['process_issues']:
            answer['process_issues'] = stage3.check_process(process)
        if 'process' in locals() and not answer['process_issues']:
            answer['PROCESS_COMPLETION'] = 'PASS'
    return answer


def v1_shadow(rows, transitions, *, dynamics_protocol="v2"):
    """Fixed-window WORLD scores on the named dynamics; no control authority."""
    require(dynamics_protocol in ("v2", "v3c"), "shadow_dynamics_protocol")
    answer=dict(label="V1_SHADOW_ON_"+dynamics_protocol.upper()+"_DYNAMICS", status="NOT_OBSERVED", admission=None, movement=None)
    if len(rows)>=1680:
        # Only transitions observed by the fixed endpoint participate in its witness.
        prefix=rows[:1680]
        prior=[r for r in transitions if r["first_active"]["physics_step"] <= prefix[-1]["pose"]["clock_after"][0]]
        answer["admission"]=replay(prefix,prior,run_type="HARNESS")
    if len(rows)>=2400:
        from .attachment_capture import world_event_motion
        answer["movement"]=world_event_motion(rows,120)
        answer["status"]="PASS" if answer["admission"]["status"]=="TRUE" and answer["movement"]["status"]=="PASS" else "FAIL"
    return answer



def event_motion_metrics(rows, actions, issued_count, close, *, protocol="v2"):
    """Separate requested TCP, measured TCP and object motion; no verdict authority."""
    from .attachment_capture import event_schedule,world_event_motion
    out=dict(requested_tcp_rise_max_m=None,measured_tcp_rise_max_m=None,chicken_rise_max_m=None,
             frame='robot_base_for_TCP_world_for_chicken',acceptance_authority=False)
    if close is None:return out
    plan=event_schedule(close, protocol=protocol);d=plan['decision'];b=plan['confirmation'];h=plan['rise_end']
    out['event_schedule']=plan
    if len(rows)>d:
        base=rows[d]['pose']['poses'];rise=rows[b:min(len(rows),h+1)]
        if rise:
            out['measured_tcp_rise_max_m']=max(r['pose']['poses']['tcp_base_wxyz'][2]-base['tcp_base_wxyz'][2] for r in rise)
            out['chicken_rise_max_m']=max(r['pose']['poses']['chicken_actor_world_wxyz'][2]-base['chicken_actor_world_wxyz'][2] for r in rise)
        start=b//12;end=min(issued_count,h//12+1)
        if end>start:out['requested_tcp_rise_max_m']=float(np.max(actions[start:end,2])-actions[d//12,2])
    out['WORLD']=world_event_motion(rows,close,protocol=protocol)
    return out


# C1 owner-approved checker agreement only; native scoring is unchanged.
V2_ANGLE_ABS_TOL_DEG = 1e-6
V2_DISTANCE_ABS_TOL_M = 1e-7
V2_DIMENSIONLESS_REL_TOL = 1e-9


def v2_metric_tolerance(native, checker, unit, *, angle_abs_tol_deg=V2_ANGLE_ABS_TOL_DEG,
                        distance_abs_tol_m=V2_DISTANCE_ABS_TOL_M,
                        dimensionless_rel_tol=V2_DIMENSIONLESS_REL_TOL):
    """No implicit relative tolerance, absolute floor, or nonfinite shortcut."""
    require(type(native) in (int, float) and type(checker) in (int, float)
            and math.isfinite(native) and math.isfinite(checker), 'v2_nonfinite_metric')
    if unit == 'deg': return angle_abs_tol_deg
    if unit == 'm': return distance_abs_tol_m
    if unit == 'mm': return 1000 * distance_abs_tol_m
    if unit == 'dimensionless': return dimensionless_rel_tol * max(abs(native), abs(checker))
    raise ValueError('v2_unmapped_metric_unit:' + unit)


def v2_score_agreement(native, checker, *, angle_abs_tol_deg=V2_ANGLE_ABS_TOL_DEG,
                       distance_abs_tol_m=V2_DISTANCE_ABS_TOL_M,
                       dimensionless_rel_tol=V2_DIMENSIONLESS_REL_TOL):
    """Only named derived metrics tolerate arithmetic differences; other leaves are exact."""
    fields = {}
    def metric(path, unit, decisions=()): fields[path] = (unit, decisions)
    for name in ('requested_tcp_lift_m', 'actual_tcp_lift_end_m', 'actual_tcp_final_delta_m',
                 'chicken_baseline_world_z_m', 'rise_at_lift_end_m'):
        metric('/movement/' + name, 'm')
    for path in ('/movement/rise_max_m', '/movement/world_rise/maximum_m'):
        metric(path, 'm', ((.05, 'lower'),))
    for path in ('/movement/translation_drift_max_m', '/movement/world_room/maximum_displacement_m',
                 '/movement/confirmation_translation_change_m'):
        metric(path, 'm', ((.01, 'upper'),))
    for path in ('/movement/rotation_drift_max_deg', '/movement/world_room/maximum_rotation_deg',
                 '/movement/confirmation_rotation_change_deg'):
        metric(path, 'deg', ((5., 'upper'),))
    metric('/movement/world_rise/baseline_world_z_m', 'm')
    for name, unit in (('rise_m', 'm'), ('room_translation_m', 'm'), ('room_rotation_deg', 'deg')):
        metric('/movement/signed_margins/' + name, unit)
    metric('/admission/distance_m', 'm', ((.15, 'upper'),))
    for side in range(2):
        base = '/admission/sides/' + str(side)
        metric(base + '/maximum_absolute_deviation_m', 'm', ((.003, 'upper'),))
        for stat in ('min_m', 'max_m', 'endpoint_m'):
            metric(base + '/opening_sum/' + stat, 'm')
            metric(base + '/signed_reference_deviation/' + stat, 'm', ((-.003, 'lower'), (.003, 'upper')))
        for jaw in range(2):
            jbase = base + '/jaws/' + str(jaw)
            metric(jbase + '/peak_to_peak_m', 'm')
            for family in ('positions', 'shortfall', 'progress'):
                for stat in ('min_m', 'max_m', 'endpoint_m'):
                    decisions = ((.001, 'lower'),) if family in ('shortfall', 'progress') and stat == 'min_m' else ()
                    metric(jbase + '/' + family + '/' + stat, 'm', decisions)
    answer = dict(status='PASS', parameter_bindings=dict(angle_abs_tol_deg=angle_abs_tol_deg,
        distance_abs_tol_m=distance_abs_tol_m, dimensionless_rel_tol=dimensionless_rel_tol),
        field_units={p:u for p,(u,_) in fields.items()}, numeric_rows=[], boundary_rows=[],
        exact_mismatches=[], verdict_mismatches=[], nonfinite_fields=[])
    def visit(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys(): answer['exact_mismatches'].append(path + ':keys')
            for key in a.keys() & b.keys(): visit(a[key], b[key], path + '/' + key)
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b): answer['exact_mismatches'].append(path + ':shape')
            for i,(x,y) in enumerate(zip(a,b)): visit(x,y,path + '/' + str(i))
            return
        if any(type(x) in (int,float) and not math.isfinite(x) for x in (a,b)):
            answer['nonfinite_fields'].append(path); return
        if path in fields and a is not None and b is not None:
            unit, decisions = fields[path]
            if type(a) not in (int,float) or type(b) not in (int,float):
                answer['exact_mismatches'].append(path + ':numeric_type'); return
            tol=v2_metric_tolerance(a,b,unit,angle_abs_tol_deg=angle_abs_tol_deg,
                distance_abs_tol_m=distance_abs_tol_m,dimensionless_rel_tol=dimensionless_rel_tol)
            row=dict(metric=path,unit=unit,native=a,checker=b,difference=b-a,
                     absolute_difference=abs(b-a),agreement_tolerance=tol,agrees=abs(b-a)<=tol)
            answer['numeric_rows'].append(row)
            for threshold, sense in decisions:
                boundary_tol=v2_metric_tolerance(a,threshold,unit,angle_abs_tol_deg=angle_abs_tol_deg,
                    distance_abs_tol_m=distance_abs_tol_m,dimensionless_rel_tol=dimensionless_rel_tol)
                answer['boundary_rows'].append(dict(row,decision_line=threshold,
                    decision_sense=sense,signed_native_margin=a-threshold if sense=='lower' else threshold-a,
                    boundary_tolerance=boundary_tol,BOUNDARY=abs(a-threshold)<=boundary_tol))
            return
        if type(a) is not type(b) or a != b:
            answer['exact_mismatches'].append(path)
            key=path.rsplit('/',1)[-1]
            if key in ('status','V2_VERDICT','V3C_VERDICT','FIRST_V2_FAILURE_GATE','FIRST_V3C_FAILURE_GATE','SCORE_EVIDENCE_STATUS','label',
                       'lift_pass','hold_pass','attachment_history_pass','qualifies','envelope_qualifies'):
                answer['verdict_mismatches'].append(dict(field=path,native=a,checker=b))
    visit(native, checker, '')
    answer['differences']=[r for r in answer['numeric_rows'] if r['difference'] != 0]
    answer['BOUNDARY_ROW_COUNT']=sum(r['BOUNDARY'] for r in answer['boundary_rows'])
    if answer['exact_mismatches'] or answer['nonfinite_fields'] or any(not r['agrees'] for r in answer['numeric_rows']):
        answer['status']='FAIL'
    return answer


def adjudicate_envelope_v2(root,output,invocation,process,result,answer,value,policy,provenance,measurement,owner):
    from .attachment_capture import event_anchor,event_schedule,evaluate_envelope_v2
    protocol=invocation.get('envelope_protocol','v2')
    require(protocol in ('v2','v3c') and all(x.get('envelope_protocol','v2') == protocol for x in (process,result,value)), 'envelope_protocol_binding')
    rows=measurement['rows'];close=event_anchor(rows);plan=event_schedule(close,protocol=protocol) if close is not None else None
    total=plan['total_actions'] if plan else 200;d=plan['decision'] if plan else None;e=plan['score_end'] if plan else None
    require(all(x.get('envelope_v2') is True for x in (invocation,process,result)), 'v2_explicit_scope')
    check_rows(measurement,policy['actions'],result['primary_errors'],total_actions=total,decision=d)
    successful=len(rows)-bool(rows[-1]['issues'])
    require(result['counts']==dict(observed_substeps=len(rows),completed_control_ticks=successful//2,completed_policy_actions=successful//12),'v2_completed_counts')
    pre=load(output/'pre_candidate.json')
    require(pre['native_clock']==measurement['origin_clock']==pre['pose']['clock_after'],'pre_clock')
    require(not geometric.check_commanded_sensors(measurement,pre,allow_terminal_observation=True),'causal_sensors')
    rgb=validate_rgb(output/'rgb',expected_count=len(measurement['sensor_samples'])+1,sensors=measurement['sensor_samples'],pre_candidate=pre,envelope_v2=True)
    require(rgb['status']=='PASS','rgb:'+repr(rgb['issues']))
    transitions=stage3.lines(output/'witness_transitions.jsonl')
    admission=replay(rows,transitions,run_type='HARNESS',envelope_v2=True,protocol=protocol)
    require(admission==(rows[d].get('geometric_eligibility') if d is not None and len(rows)>d else None),'v2_admission_replay')
    check_attachment_prefix(rows,owner,admission or {},decision=d,score_end=e)
    complete=not result['primary_errors'] and len(rows)==12*total
    refused=result['primary_errors']==['CaptureFailure:geometric_admission_refused:FALSE'] and admission and admission['status']=='FALSE' and len(rows)==d+1
    predictions=stage3.lines(output/'stage3_inferences.jsonl')
    rejected=bool(predictions and predictions[-1]['status']=='REJECTED' and result['primary_errors']==[predictions[-1]['error']] and len(rows)==predictions[-1]['global_start']*12)
    category='FULL_WINDOW_OBSERVED' if complete else 'ADMISSION_REFUSED' if refused else 'POLICY_ACTION_REJECTED' if rejected else 'HARNESS_ERROR'
    if category=='HARNESS_ERROR':
        legacy=classify_terminal(result,rows,predictions,owner,admission or {},policy)
        if legacy[0]=='POLICY_REOPEN_RELEASE':category=legacy[0]
    # These are independent expected terminal counts, never inferred from output length.
    if complete:
        require(len(policy['actions'])==total and provenance['counts']['completed']==total and len(predictions)==len(stage3.expected_chunks(total)), 'v2_full_expected_counts')
    if refused:
        require(provenance['counts']==dict(selected=d//12+1,dispatch_requested=d//12+1,acceptance_observed=d//12+1,completed=d//12), 'v2_refusal_expected_counts')
        require(len(predictions)==sum(x['global_start']<=d//12 for x in stage3.expected_chunks(total)), 'v2_refusal_expected_predictions')
    score=evaluate_envelope_v2(rows,episode_id=measurement['episode_id'],origin_step=measurement['origin_clock'][0],origin_time=measurement['origin_clock'][1],close_action=close,completed=complete,primary_errors=result['primary_errors'],lifecycle_events=measurement['lifecycle_events'],protocol=protocol)
    native_score=result['experiment_result']
    answer['score_agreement']=v2_score_agreement(native_score,score)
    require(native_score==load(output/'candidate_result.json'),'v2_native_score_identity')
    require(answer['score_agreement']['status']=='PASS','v2_score_reproduction')
    require(result.get('v2_cleanup_status')=='CLEAN','v2_cleanup')
    answer.update(motion=event_motion_metrics(rows,policy['actions'],provenance['counts']['dispatch_requested'],close,protocol=protocol),category=category,counts=result['counts'],first_accepted_close=close,event_schedule=plan,admission=admission,end_state=score,v1_shadow=v1_shadow(rows,transitions,dynamics_protocol=protocol),terminal_boundary=[rows[-1]['context']['control_tick'],rows[-1]['context']['substep']])
    require(category!='HARNESS_ERROR','unexplained_v2_terminal')
    qualified=deepcopy(process)
    if category in ('ADMISSION_REFUSED','POLICY_ACTION_REJECTED','POLICY_REOPEN_RELEASE') and process.get('exit_code')==1:
        qualified['exit_code']=0;answer['controlled_stop_exit_code']=1
    answer['process_issues']=stage3.check_process(qualified)
    require(not answer['process_issues'],'process_qualification')
    answer.update(status='HARNESS_COMPLETE',technical_clean=True,CAPTURE_INTEGRITY='PASS',PROCESS_COMPLETION='PASS')
    return answer
