"""Saved-file collection readout. No simulator or policy construction."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def world_scores(rows, *, event_close=None, protocol="v2"):
    """Owner WORLD ROOM definition, separate from historical relative hold."""
    if event_close is not None:
        if type(event_close) is not int or not 0 <= event_close <= 199:
            raise ValueError('eligible_event_close_required')
        shift=12*(event_close-120)
    else:
        shift=0
    d,b,h,e=1679+shift,1680+shift,2159+shift,2399+shift
    if protocol == "v3c":
        from .envelope_schedule import event_schedule
        plan=event_schedule(event_close, protocol=protocol)
        d,b,h,e=(plan[k] for k in ("decision","confirmation","room_reference","score_end"))
    elif protocol != "v2":
        raise ValueError("unknown_learned_envelope_protocol")
    indexed = {2*r['context']['control_tick']+r['context']['substep']: r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError('duplicate_measurement_boundary')
    def pose(i):
        r = indexed[i]
        p = np.asarray(r['pose']['poses']['chicken_actor_world_wxyz'], float)
        if (p.shape != (7,) or not np.isfinite(p).all() or np.linalg.norm(p[3:]) == 0
                or r['pose']['valid'] is not True or r['pose'].get('issues')
                or any(r['runtime'].values())):
            raise ValueError('invalid_required_world_pose:'+str(i))
        return p
    rise = dict(status='NOT_EVALUABLE', maximum_m=None)
    room = dict(status='NOT_EVALUABLE', reference=[h//2,h%2], required_samples=240,
                samples=0, maximum_displacement_m=None, maximum_rotation_deg=None,
                limits=dict(displacement_m=.01, rotation_deg=5))
    if all(i in indexed for i in range(d,h+1)):
        try:
            base = pose(d)
            values = [(float(pose(i)[2]-base[2]), i) for i in range(b,h+1)]
            maximum, i = max(values)
            rise.update(status='PASS' if maximum >= .05 else 'FAIL', maximum_m=maximum,
                        baseline_world_z_m=float(base[2]), maximum_sample=[i//2, i%2])
        except ValueError as exc:
            rise['issue'] = str(exc)
    if all(i in indexed for i in range(h,e+1)):
        try:
            ref = pose(h)
            qref = ref[3:]/np.linalg.norm(ref[3:])
            scored = []
            for i in range(h+1,e+1):
                p = pose(i)
                q = p[3:]/np.linalg.norm(p[3:])
                angle = math.degrees(2*math.acos(float(np.clip(abs(q@qref), 0, 1))))
                scored.append((float(np.linalg.norm(p[:3]-ref[:3])), angle, i))
            distance = max(scored, key=lambda v:v[0])
            angle = max(scored, key=lambda v:v[1])
            room.update(status='PASS' if distance[0] <= .01 and angle[1] <= 5 else 'FAIL',
                samples=len(scored), reference_world_wxyz=ref.tolist(),
                maximum_displacement_m=distance[0], maximum_rotation_deg=angle[1],
                maximum_displacement_sample=[distance[2]//2,distance[2]%2],
                maximum_rotation_sample=[angle[2]//2,angle[2]%2])
        except ValueError as exc:
            room['issue'] = str(exc)
    return rise, room


def technical_continuation(result, native, rows):
    """Only a complete record or the existing scheduled refusal can continue."""
    if result.get('CAPTURE_INTEGRITY') != 'PASS' or result.get('PROCESS_COMPLETION') != 'PASS':
        return False
    errors = native.get('primary_errors')
    if result.get('episode_complete') is True:
        return errors == []
    if (len(rows) != 1680 or not rows or
            [rows[-1]['context'][k] for k in ('control_tick','substep')] != [839,1]
            or not isinstance(errors,list) or len(errors) != 1):
        return False
    status = result.get('admission',{}).get('status')
    return ((status == 'FALSE' and errors == ['CaptureFailure:geometric_admission_refused:FALSE'])
        or (status == 'UNAVAILABLE' and errors[0].startswith('CaptureFailure:geometric_admission_unavailable:')))


def assess(attempt):
    """Read the existing independent verdict and required measured WORLD windows."""
    attempt = Path(attempt).resolve()
    cap = attempt/'capture/experiment'
    result = load(attempt/'attempt_result.json')
    native = load(cap/'result.json')
    measurements = load(cap/'measurements.json')
    translated = load(cap/'script_translation.json')
    rows = measurements['rows']
    rise, room = world_scores(rows)
    gate = result.get('admission', {})
    confirmation = result.get('attachment_confirmation') or {}
    technical = technical_continuation(result, native, rows)
    confirmation_ok = (confirmation.get('status') == 'CONFIRMED'
        and [confirmation.get('confirmation_context', {}).get(k) for k in ('control_tick','substep')] == [840,0])
    usable = (technical and result.get('episode_complete') is True
        and bool(result.get('loader_compatibility')) and gate.get('status') == 'TRUE'
        and confirmation_ok and rise['status'] == 'PASS' and room['status'] == 'PASS')
    tracking = dict(status='NOT_REACHED_OR_NOT_PERSISTED')
    indexed = {(r['context']['control_tick'],r['context']['substep']):r for r in rows}
    if (720,1) in indexed:
        r = indexed[720,1]
        actions = {a['global_action']:a for a in lines(cap/'script_actions.jsonl')}
        events = lines(cap/'script_command_events.jsonl')
        command = actions[120]['action']
        if r['policy_indices']['global_action'] != 120 or r['commands'] != command[3:]:
            raise ValueError('tracking_active_command_join')
        if not any(e['event']=='ACCEPTANCE_OBSERVED' and e['global_action']==120 for e in events):
            raise ValueError('tracking_command_acceptance_missing')
        measured = r['pose']['poses']['tcp_base_wxyz'][:3]
        tracking = dict(status='ADVISORY', boundary=[720,1], frame='robot_base',
            measured_tcp_xyz_m=measured, active_command_xyz_m=command[:3],
            error_xy_mm=(1000*(np.asarray(measured)-command[:3]))[:2].tolist(),
            source='measurements.json row1441; script_actions.jsonl action120; ACCEPTANCE_OBSERVED',
            native_xyz_target_readback='NOT_PERSISTED; validated dispatched XYZ and unchanged scheduler')
    margins = []
    for side in gate.get('sides', []):
        margins.append(dict(side=side['side'],
            envelope_signed_margin_mm=1000*(.003-side['maximum_absolute_deviation_m']),
            deviation_min_mm=1000*side['signed_reference_deviation']['min_m'],
            deviation_max_mm=1000*side['signed_reference_deviation']['max_m'],
            witnesses=side['witnesses'], jaws=[dict(jaw_id=j['jaw_id'],
                progress_margin_mm=1000*(j['progress']['min_m']-.001),
                remaining_shortfall_margin_mm=1000*(j['shortfall']['min_m']-.001)) for j in side['jaws']]))
    reason = ('CAPTURE_OR_PROCESS' if not technical else
              'ADMISSION' if gate.get('status') != 'TRUE' else
              'INCOMPLETE_EPISODE' if not result.get('episode_complete') else
              'CONFIRMATION' if not confirmation_ok else
              'RISE' if rise['status'] != 'PASS' else
              'WORLD_ROOM' if room['status'] != 'PASS' else None)
    out = dict(schema='random-xy-collection-result/v1', point=translated['point'],
        attempt_directory=str(attempt), attempt_result_sha256=sha(attempt/'attempt_result.json'),
        measurements_sha256=sha(cap/'measurements.json'),
        authored_xy_m=translated['authored_world_xyz_m'][:2],
        episode_path=str(cap/'episode') if (cap/'episode').is_dir() else None,
        usable=bool(usable), technical_integrity=technical,
        selection_visibility_condition='Campaign first-admitted visibility review must PASS before selection; usable here is the saved recording-criteria result.',
        CAPTURE_INTEGRITY=result.get('CAPTURE_INTEGRITY'), PROCESS_COMPLETION=result.get('PROCESS_COMPLETION'),
        episode_complete=result.get('episode_complete',False),
        admission_status=gate.get('status','NOT_REACHED'), admission_issues=gate.get('issues',[]),
        admission_signed_margins=margins, admission_distance_m=gate.get('distance_m'),
        confirmation_status=confirmation.get('status','NOT_REACHED'),
        rise_m=rise['maximum_m'], rise=rise, WORLD_ROOM=room,
        relative_END_STATE=result.get('END_STATE_RESULT'),
        relative_hold_diagnostic_only=result.get('end_state'),
        first_failure_gate=reason, tracking_720_1=tracking,
        counts=native.get('counts'), command_counts=result.get('COMMAND_PROVENANCE',{}).get('counts'),
        primary_errors=native.get('primary_errors'), scene_profile=load(cap/'stage1_visuals_before_candidate.json').get('scene_profile'),
        GRANT5_OWNER_RELEASE='PENDING')
    if (299,1) in indexed:
        out['settled_after_setup_world_wxyz']=indexed[299,1]['pose']['poses']['chicken_actor_world_wxyz']
    write_new(attempt/'collection_result.json',out)
    return out




