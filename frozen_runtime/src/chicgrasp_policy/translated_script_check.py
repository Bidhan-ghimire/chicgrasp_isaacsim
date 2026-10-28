"""Saved-file checks for translated scripted recordings. Never runs a model/Kit."""
from pathlib import Path
import math
import numpy as np
from PIL import Image

from . import translated_script as script
from . import geometric_capture_check as geometric
from .attachment_capture import _measured_pose
from .attachment_observation_evidence import validate as validate_rgb
from .episode_io import load_policy_episode
from .interface import validate_policy_observation
from .overfit_data import make_training_windows
from .side_opening_admission import replay, validate_command
from .stage1_demo import load, validate_visuals
from .stage3_check import lines, require, check_command_provenance, check_process
from .stage3_policy import array_identity


def check_capture(output):
    root=Path(output).resolve()
    answer=dict(schema='translated-script-check/v1',CAPTURE_INTEGRITY='FAIL',END_STATE_RESULT='NOT_EVALUABLE',
        episode_complete=False,training_eligible=False,admission={'status':'NOT_REACHED'},
        attachment_confirmation=None,issues=[])
    try:
        r=load(root/'result.json');m=load(root/'measurements.json');rows=m['rows'];sensors=m['sensor_samples']
        require(m['schema']=='geometric-measurements/v2' and m['jaw_state_semantics']=='commanded-jaw-state/v1','measurement_schema')
        point=script.verified_output_point(r['translated_point'],root)
        require(load(root/'script_translation.json')==point,'translation_binding')
        if point.get('collection', {}).get('schema') == 'random-xy-collection-candidate/v1':
            answer['collection_candidate']=dict(candidate_id=point['point'],
                candidate_order=point['collection']['candidate_order'], draw_index=point['placement']['draw_index'],
                authored_world_xyz_m=point['authored_world_xyz_m'], candidate_sha256=point['candidate_sha256'],
                attempt_id=point['attempt_id'], physical_attempt_number=point['physical_attempt_number'])
            answer['collection_usability']='NOT_ASSESSED_WORLD_ROOM_AND_OWNER_SELECTION_REQUIRED'
        original_probe=(r['translated_point']=='ORIGINAL' and r['run_type']=='HARNESS' and
            root==(script.ROOT/'outputs/current_original_script_probe/attempt001/capture/experiment').resolve())
        require((r['run_type']=='EVIDENCE' or original_probe) and r['admission_policy']=='side-opening-progress/v1','script_scope')
        require(not any(r.get(k) for k in ('stage3_learned','stage1_demo','stage3_witness','stage0_mode')),'exclusive_script')
        manifest=load(root/'artifact_manifest.json')
        actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
        require(actual==set(manifest['files'])|{'artifact_manifest.json'},'artifact_membership')
        for name,want in manifest['files'].items():
            path=(root/name).resolve()
            require(path.is_relative_to(root) and geometric.identity(path)==want,'artifact_identity:'+name)
        events=check_command_provenance(root,scripted=True)
        answer['COMMAND_PROVENANCE']=events
        answer['recording_errors']=r['recording_errors'];answer['cleanup_errors']=r['cleanup_errors']
        answer['issues'].extend('cleanup:'+str(e) for e in r['cleanup_errors'])
        require(not r['recording_errors'],'recording_errors')
        from .pilot_cleanup import classify_cleanup
        owner=load(root/'owner_before_cleanup.json')
        cleanup=classify_cleanup(owner,r['cleanup']['owner'])
        if cleanup['status']=='ERROR':answer['issues'].append('owner_cleanup:'+repr(cleanup))
        answer['cleanup']=cleanup
        require(lines(root/'script_substeps.jsonl')==rows,'incremental_measurements')
        require(events['status']=='PASS','terminal_provenance:'+repr(events))
        completed=events['counts']['completed']
        require(r['counts']['observed_substeps']==len(rows) and completed*12<=len(rows)<=(completed+1)*12
            and completed*6<=r['counts']['completed_control_ticks']<=min(1200,completed*6+5),'partial_counts')
        decisions=lines(root/'script_actions.jsonl');observations=lines(root/'script_observations.jsonl')
        require(lines(root/'script_inferences.jsonl')==[],'script_has_no_inferences')
        require(len(observations)==len(sensors) and len(decisions)<=len(sensors)<=len(decisions)+1,'acquisition_counts')
        actions=np.asarray(point['actions'],np.float32)
        for i,decision in enumerate(decisions):
            expected=dict(global_action=i,learned_action=None,setup_action=i if i<50 else None,script_action=i,
                training_action=i-50 if i>=50 else None,action=actions[i].tolist(),native_clock=sensors[i]['native_clock'],
                control_tick_start=6*i,source='translated_accepted_Stage1_script',source_action=i,
                source_action_sha256=script.ACTION_SHA256,point=point['point'],point_sha256=point['point_sha256'],
                manifest_sha256=point['manifest_sha256'])
            require(decision==expected,'script_selection:'+str(i))
        require(len(rows)<=2400 and (len(rows)+11)//12<=len(decisions),'schedule_count')
        previous=m['origin_clock']
        for i,row in enumerate(rows):
            clock=row['pose']['clock_after']
            require(row['context']==dict(episode_id=m['episode_id'],control_tick=i//2,substep=i%2),'substep_order')
            require(clock[0]==m['origin_clock'][0]+i+1 and math.isclose(clock[1]-previous[1],1/120,abs_tol=1e-8),'native_clock')
            require(row['pose']['valid'] is True and row['pose']['clock_before']==clock,'valid_current_pose')
            for name in ('robot_root_world_wxyz','chicken_actor_world_wxyz','gripper_world_wxyz','tcp_world_wxyz','tcp_base_wxyz','chicken_in_gripper_wxyz'):
                _measured_pose(row['pose']['poses'][name])
            require(row['jaws']['clock_before']==clock==row['jaws']['clock_after'] and
                    np.asarray(row['jaws']['positions_m']).shape==(4,) and np.isfinite(row['jaws']['positions_m']).all(),'jaw_pose')
            require(row['commands']==actions[i//12,3:].tolist(),'active_script_commands')
            require(row['policy_indices']==dict(global_action=i//12,script_action=i//12,learned_action=None,
                setup_action=i//12 if i<600 else None,training_action=i//12-50 if i>=600 else None),'script_indices')
            require(not any(k in row for k in ('contact','predicate','observation')),'slim_fields')
            require(not row['issues'] and not any(row['runtime'].values()) or i==len(rows)-1,'nonterminal_invalid_row')
            previous=clock
        transitions=lines(root/'witness_transitions.jsonl')
        gate=replay(rows,transitions,run_type=r['run_type'])
        answer['admission']=gate or {'status':'NOT_REACHED'}
        if gate is not None:
            require(rows[1679]['geometric_eligibility']==gate,'saved_witness_recomputed')
        requests=[op for op in owner['event_evidence']['records'] if op['kind']=='activation_request']
        if gate and gate['status']=='TRUE':
            require(len(requests)==1 and requests[0]['context']==rows[1679]['context'] and
                requests[0]['eligibility']['geometric_eligibility']==gate,'owner_request_join')
        else:
            require(not requests and all(row['attachment']['status']=='UNATTACHED' for row in rows),'no_unapproved_attachment')
        if len(rows)>1680:
            answer['attachment_confirmation']=rows[1680]['attachment']
        pre=load(root/'pre_candidate.json')
        require(len(sensors)>0,'no_observations')
        require(pre['reset_provenance']['normal_initial_reset_returned'] is True and
            pre['rgb_metadata']['jaw_state']==sensors[0]['jaw_state'],'PRE_reset_and_shared_acquisition')
        for i,(obs,sensor) in enumerate(zip(observations,sensors)):
            require(obs['global_action']==i and obs['acquisition']==sensor,'sensor_join')
            require(sensor['native_clock']==(m['origin_clock'] if i==0 else rows[i*12-1]['pose']['clock_after']),'pre_action_clock')
            bits=[1,1] if i==0 else rows[i*12-1]['commands']
            require(obs['jaw_state']==sensor['jaw_state']['bits']==bits,'causal_commanded_jaws')
            validate_command(dict(jaw_commands=bits,physics_step=sensor['native_clock'][0],
                simulation_time_s=sensor['native_clock'][1]),sensor['jaw_state'])
            previous_row=rows[i*12-1] if i else None
            preceding=dict(kind='accepted_control_action' if i else 'reset_open',action_frame=i-1 if i else None,
                control_tick=previous_row['context']['control_tick'] if i else None,
                substep=previous_row['context']['substep'] if i else None,
                observed_native_clock=previous_row['pose']['clock_after'] if i else None,
                observed_wall_monotonic_ns=previous_row['wall_monotonic_ns'] if i else None)
            require(sensor['jaw_state']['preceding_command']==preceding,'causal_command_source')
            require(obs['learned_action'] is None and obs['script_action']==i,'observation_script_indices')
            mapping={}
            for camera in ('camera_left','camera_right'):
                with Image.open(root/'rgb'/f'{i:06d}_{camera}.png') as image:
                    mapping[camera+'_rgb']=np.asarray(image).copy()
            mapping.update({k:np.asarray(obs[k],dtype=dtype) for k,dtype in [('arm_joint_pos',np.float32),('jaw_state',np.uint8)]})
            validate_policy_observation(mapping)
            require(obs['input_identities']=={k:array_identity(v) for k,v in mapping.items()},'saved_observation_bytes')
        rgb=validate_rgb(root/'rgb',expected_count=len(sensors)+1,sensors=sensors,pre_candidate=pre)
        require(rgb['status']=='PASS','RGB:'+repr(rgb['issues']))
        validate_visuals(load(root/'stage1_visuals_before_candidate.json'))
        preparation=load(root/'script_preparation.json')
        require(preparation['authored_point']==point and preparation['configuration']['after_xyz_m']==point['authored_world_xyz_m'], 'authored_preparation')
        if point.get('yaw_probe'):
            require(preparation['configuration']['rotation_wxyz']==point['authored_quaternion_wxyz'],'authored_yaw_preparation')
            require(load(root/'reference_labels.json')==dict(labels=script.YAW_LABELS,training_eligible=False),'yaw_reference_only_labels')
            answer['reference_labels']=script.YAW_LABELS
            answer['training_eligible']=False
        if point.get('hard_xy_anchor'):
            require(point['authored_quaternion_wxyz']==[1,0,0,0] and preparation['configuration']['rotation_wxyz']==[1,0,0,0],'authored_anchor_identity_preparation')
            require(load(root/'reference_labels.json')==dict(labels=script.ANCHOR_LABELS,training_eligible=False),'anchor_reference_only_labels')
            answer['reference_labels']=script.ANCHOR_LABELS
            answer['training_eligible']=False
        storage=load(root/'script_storage_prelaunch.json')
        estimate=storage['estimate']
        require(storage['passed'] is True and estimate['fixed_margin_bytes']==1<<30 and
            estimate['required_bytes']==estimate['estimated_payload_bytes']+(1<<30) and storage['free_bytes']>=estimate['required_bytes'],'prelaunch_storage')
        require(preparation['origin_clock']==m['origin_clock'] and preparation['measured']['valid'] is True,'preparation_clock')
        if point.get('placement', {}).get('schema') == 'random-xy-placement/v1':
            from .stage1_demo import approved_scaled_profile
            expected_configuration = approved_scaled_profile()['physical_configuration']
            require(load(root/'stage1_visuals_before_candidate.json').get('scene_profile') is not None,
                    'random_scene_profile_required')
        else:
            expected_configuration = load(script.ROOT/'outputs/generalization_domain_previews_task1c_r2/effective_configuration.json')
        require(r['effective_configuration']==expected_configuration,'effective_GPU_configuration')
        complete=len(rows)==2400 and events['counts']['completed']==200 and not r['primary_errors']
        answer['episode_complete']=complete
        require(r['script']==dict(selected=len(decisions),completed=events['counts']['completed'],
            setup_completed=min(events['counts']['completed'],50),scripted_after_setup_completed=max(0,events['counts']['completed']-50),
            learned_actions=0,actual_inferences=0),'script_counts')
        if complete:
            physical=geometric.check_measurements(m,owner,expected_actions=actions,witness_transitions=transitions,run_type=r['run_type'],requested_source='translated_script_commands_robot_base_metres')
            require(not physical['issues'],'physical_history:'+repr(physical['issues']))
            require(physical['end_state']==r['experiment_result']==load(root/'candidate_result.json'),'end_state_recomputed')
            answer['end_state']=physical['end_state'];answer['END_STATE_RESULT']=physical['end_state']['status']
            validate_visuals(load(root/'stage1_visuals_after_candidate.json'))
            arrays=load_policy_episode(root/'episode')
            require(np.array_equal(arrays['action'],actions),'episode_script_actions')
            for i,obs in enumerate(observations):
                for key in ('camera_left_rgb','camera_right_rgb','arm_joint_pos','jaw_state'):
                    require(array_identity(arrays[key][i])==obs['input_identities'][key],'episode_observation:'+key)
            windows=make_training_windows(arrays)
            answer['loader_compatibility']=dict(loader='load_policy_episode',window_builder='make_training_windows',
                full_episode_rows=200,full_windows=len(windows['action']),setup_rows=50,future_training_rows=150,
                training_executed=False,free_space_validator='UNCHANGED_NOT_APPLICABLE')
            del windows
            training={k:v[50:] for k,v in arrays.items()}
            training['timestamp_ns']=training['timestamp_ns']-training['timestamp_ns'][0]
            windows=make_training_windows(training)
            answer['loader_compatibility']['future_150_row_windows']=len(windows['action'])
            del windows
        else:
            require(not (root/'episode').exists(),'partial_not_complete_episode')
            require(bool(r['primary_errors']),'partial_requires_stop_reason')
            answer['stop_reason']=r['primary_errors']
        answer['CAPTURE_INTEGRITY']='FAIL' if answer['issues'] else 'PASS'
    except Exception as exc:
        answer['issues'].append(type(exc).__name__+':'+str(exc))
    return answer


def recording_attempt_path(point):
    return (script.ROOT/point['output_directory']).resolve()


def adjudicate(attempt):
    root=Path(attempt).resolve();output=root/'capture/experiment'
    answer=check_capture(output);issues=[];r={}
    try:
        inv=load(root/'invocation.json');process=load(root/'capture/process_result.json');r=load(output/'result.json')
        point=script.verified_output_point(r['translated_point'],output)
        absolute_ms=1200000
        if point.get('hard_xy_anchor'):
            used=inv.get('gpu_seconds_used')
            require(type(used) in (int,float) and math.isfinite(used) and 0<=used<1800,'anchor_prior_native_budget')
            absolute_ms=min(1200000,math.floor((1800-used)*1000)-5000)
            require(0<absolute_ms<=1200000 and inv.get('hard_xy_anchor') is True and
                inv.get('task')=='CHICGRASP_SCALED50_HARD_XY_SCRIPTED_ANCHORS','anchor_process_scope')
        issues.extend(check_process(process, absolute_ms=absolute_ms))
        original_probe=(r['translated_point']=='ORIGINAL' and r['run_type']=='HARNESS' and
            root==(script.ROOT/'outputs/current_original_script_probe/attempt001').resolve())
        for v in (inv,process):
            require(v['translated_point']==point['point'] and v['translation_manifest_sha256']==point['manifest_sha256'] and
                v['run_type']==r['run_type'] and (r['run_type']=='EVIDENCE' or original_probe),'process_translation_binding')
        require(original_probe or root==recording_attempt_path(point),'one_fixed_attempt_per_point')
        require(inv['timeout_ms']==absolute_ms and process['invocation_sha256']==geometric.identity(root/'invocation.json')['sha256'],'invocation_identity')
        require(inv['executed_commit']==r['execution_provenance']['executed_commit'],'executed_commit')
        if original_probe:
            from .attachment_run_policy import check_run_type
            scope=check_run_type(inv,r,process)
            require(scope['issues']==['harness_output_not_acceptance_evidence'] and
                not scope['acceptance_eligible'] and inv['execution_approved'] is False and
                'evidence_authority' not in inv,'original_probe_harness_scope')
        else:
            for kind in ('grant','reservation'):
                item=inv['evidence_authority'][kind]
                require(geometric.identity(item['path'])=={k:item[k] for k in ('bytes','sha256')},'authority_identity')
            reservation=load(inv['evidence_authority']['reservation']['path'])
            require(reservation['translated_point']==point['point'] and reservation['translation_manifest_sha256']==point['manifest_sha256']
                and reservation['grant_sha256']==inv['evidence_authority']['grant']['sha256']
                and reservation['executed_commit']==inv['executed_commit'] and reservation['executed_tag']==inv['executed_tag']
                and Path(reservation['attempt_directory']).resolve()==root,'recording_grant_reservation')
        for key,relative in [('stdout','stdout.bin'),('stderr','stderr.bin'),('completed_journal','structured/events.jsonl')]:
            require(geometric.identity(root/'capture'/relative)=={k:process[key][k] for k in ('bytes','sha256')},'stream_identity')
        journal=lines(root/'capture/structured/events.jsonl')
        require([x['kind'] for x in journal[-2:]]==['PRE_CLOSE_EVIDENCE_COMPLETE','CLOSE_REQUESTED'],'before_fast_close')
        require(all(x['manifest']==geometric.identity(output/'artifact_manifest.json') for x in journal[-2:]),'terminal_manifest')
    except Exception as exc:
        issues.append(type(exc).__name__+':'+str(exc))
    answer['process_issues']=issues;answer['PROCESS_COMPLETION']='FAIL' if issues else 'PASS'
    good=not issues and answer['CAPTURE_INTEGRITY']=='PASS' and answer['episode_complete']
    answer['status']='PASS' if good and answer['END_STATE_RESULT']=='PASS' else 'FAIL' if good else 'BLOCKED'
    answer['training_eligible']=answer['status']=='PASS' and r.get('run_type')=='EVIDENCE' and r.get('translated_point') not in (*script.YAW_POINTS, *script.ANCHOR_POINTS)
    if r.get('run_type')=='HARNESS':
        answer['acceptance_eligible']=False
        answer['meaning']='HARNESS diagnostic only; never acceptance evidence or a training demonstration.'
    if r.get('run_type')!='HARNESS':
        answer['meaning']='Attachment-assisted scripted recording only; no unassisted grasp or generalization claim.'
    return answer
