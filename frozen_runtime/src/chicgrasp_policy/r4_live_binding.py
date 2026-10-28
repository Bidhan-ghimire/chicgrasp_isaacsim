"""Exact R4 EMA development binding; import and metadata inspection are offline."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from . import interface as api, overfit_policy
from .attachment_capture import _measured_pose
from .overfit_data import sha256_file
from .stage1_demo import load, write
from .stage3_policy import array_identity, make_predictor

ROOT = Path(__file__).resolve().parents[2]
POINTS = ('ORIGINAL', 'G00', 'G02', 'G04', 'G06', 'G07', 'G08')
HELDOUT_POINTS = tuple(f'H{i:02d}_R{r}' for r in range(3) for i in range(1, 11)) + tuple(f'T{i:02d}_R0' for i in range(1, 8))
HB_POINTS = tuple(f'HB{i:02d}_R{r}' for r in range(3) for i in range(1, 11))
HC_POINTS = tuple(f'HC{i:02d}_R{r}' for r in range(3) for i in range(1, 11))
HC_DOC = ROOT / 'docs/plans/CHICGRASP_SCALED50_HC_V3C_BATCH'
HC_RELEASE_SHA256 = 'bc1308181c750d8ebec95cfa4766489bf03ec699c924773cf69fce84237e0bb1'
HB_DOC = ROOT / 'docs/plans/CHICGRASP_SCALED50_HB_V2_BATCH'
HB_CONTINUATION_RELEASE_SHA256 = 'acca2b32470a9bede3b8c5ea6a52b1b1dbc6f3c668aaa13420e4b464dd8f7c89'
HB_RELEASE_SHA256 = '5ecabec9917715ac249e73c97a99eec6b3941069cce04a52dc830779d6126593'
HELDOUT_DOC = ROOT / 'docs/plans/CHICGRASP_SCALED50_HELDOUT_BATCH'
HELDOUT_POSITIONS_SHA256 = '89be3c48b3f1b82723db97711cd7fa3b984d76ae39da8f4b57c3d0f22c585eeb'
HELDOUT_RELEASE_SHA256 = '3d0e0aee389f0a315ce5822bbeb6da1de93668cce0a1e8b29c702da16ac20f8e'
CHECKPOINT_SHA256 = '5f40b5cb2e409fe34f9623134db9ded44b18b94fc7470dc480f34319228a290d'
NORMALIZER_SHA256 = 'cbd85e6dacf3926e2e5d31c8201fd9b194c77d167ae2090f62d4a0278b28af29'
GRANT_SHA256 = '90968d69dd794664e4d777f2714f93c3ffa2ab3a687fc8e51eb4ba5b01886da8'
RESULT_SHA256 = '77a5703887b6e34862fba1357522fec2afa48be9fd08651a5573f808b60c1f3e'
RELOAD_PROCESS_SHA256 = 'a2ca795fc62998021d1a401cacdf5b6bed566cf0ca34b0742b11fa08becf1e36'


def require(condition, reason):
    if not condition:
        raise ValueError('r4_live_' + reason)


def identity(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha256_file(path))


def authenticate(path, expected, *, hash_bytes=True):
    path = Path(path).resolve(strict=True)
    require(path.stat().st_size == expected['bytes'], 'file_size:' + str(path))
    if hash_bytes:
        require(sha256_file(path) == expected['sha256'], 'file_identity:' + str(path))
    return dict(path=str(path), bytes=expected['bytes'], sha256=expected['sha256'])


def qualified_source(point, dataset):
    """Read only the selected qualified pose/configuration and existing setup."""
    require(point in POINTS, 'excluded_point')
    manifest_ref = authenticate(dataset['manifest']['path'], dataset['manifest'])
    qualified = load(manifest_ref['path'])['recordings']
    require(tuple(r['point'] for r in qualified) == POINTS, 'qualified_point_set')
    source = next(r for r in qualified if r['point'] == point)
    row = next(r for r in dataset['records'] if r['point'] == point)
    require(source['attempt_id'] == row['attempt_id'] and
            (point != 'G00' or source['attempt_id'] == 'G00_R2'), 'qualified_attempt')
    require(source['training_eligible'] is True and all(source[k] == 'PASS' for k in
            ('END_STATE_RESULT', 'CAPTURE_INTEGRITY', 'PROCESS_COMPLETION')), 'source_qualification')
    require(source['setup_indices'] == [0, 49] and source['future_training_indices'] == [50, 199],
            'source_setup_indices')
    episode = Path(source['episode_directory']).resolve(strict=True)
    require(episode == Path(row['episode_directory']).resolve(strict=True), 'source_episode')
    manifest = authenticate(source['artifact_manifest']['path'], source['artifact_manifest'])
    files = load(manifest['path'])['files']
    capture = episode.parent
    wanted = ('script_translation.json', 'script_preparation.json', 'result.json',
              'stage1_visuals_before_candidate.json', 'episode/policy/action.npy')
    refs = {name: authenticate(capture / name, files[name]) for name in wanted}
    translated, prepared = (load(refs[name]['path']) for name in wanted[:2])
    pose = translated['authored_world_xyz_m']
    rotation = translated['authored_quaternion_wxyz']
    require(translated['point'] == point and prepared['authored_point'] == translated,
            'authored_capture_binding')
    require(prepared['configuration']['after_xyz_m'] == pose and
            prepared['configuration']['rotation_wxyz'] == rotation and
            translated['original_authored_world_xyz_m'] == [.65, 0., .8155] and
            pose[2] == .8155 and rotation == [1., 0., 0., 0.], 'authored_pose')
    require(translated['fixed_orientation_base_wxyz'] == list(api.FIXED_TCP_QUATERNION_BASE_WXYZ),
            'fixed_wrist_orientation')
    actions = np.load(refs['episode/policy/action.npy']['path'], allow_pickle=False)
    require(actions.dtype == np.float32 and actions.shape == (200, 5), 'source_action_array')
    authenticate(episode / 'policy/action.npy', {Path(k).as_posix(): v for k, v in row['episode_files'].items()}['policy/action.npy'])
    setup = actions[:50].copy()
    require(np.array_equal(setup, np.asarray(translated['actions'][:50], np.float32)), 'setup_capture_disagreement')
    api.validate_policy_action(setup)
    require(np.all(setup[:, 3:] == 1), 'setup_open_commands')
    return dict(qualified_attempt_id=source['attempt_id'], source_episode_directory=str(episode),
        qualified_manifest=manifest_ref, artifact_manifest=manifest, source_files=refs,
        authored_world_xyz_m=pose, authored_quaternion_wxyz=rotation,
        original_authored_world_xyz_m=translated['original_authored_world_xyz_m'],
        setup_actions=setup.tolist(), setup_action_identity=array_identity(setup),
        setup_source_indices=[0, 49], production_configuration=load(refs['result.json']['path'])['effective_configuration'],
        qualified_settled_measurement=prepared['measured'])


def verify_binding(point, *, verify_checkpoint=True, envelope_v2=False, envelope_v3c=False):
    """Authenticate exact R4 files and the selected seven-point source mapping.

    PlanOnly may omit the 5.27 GB checkpoint hash; that result cannot load a model.
    """
    if point in HC_POINTS:
        require(envelope_v3c and not envelope_v2, 'hc_requires_explicit_v3c')
        value=verify_heldout_binding(point, verify_checkpoint=verify_checkpoint, hc_v3c=True)
        release=load(HC_DOC/'RELEASE.json')
        value.update(schema='scaled50-hc-v3c-binding/v1', envelope_v2=True, envelope_protocol='v3c',
            task=release['task'], grant=identity(HC_DOC/'RELEASE.json'), task_clock=release['clock'],
            approved_protocol=release['approved_protocol'], definition_id=release['definition_id'],
            hc_manifest=release['hc_manifest'], allocation_order=release['allocation_order'],
            admin_cycles_used=2, maximum_extra_actual_starts=release['maximum_extra_actual_starts'],
            daily_actual_at_entry=release['daily_actual_at_entry'], grant_allocation='THIRTY_PREREGISTERED_HC_V3C_ALLOCATIONS')
        return value
    if envelope_v3c:
        is_hb = point in HB_POINTS
        require(not envelope_v2 and (point == "H05_R2" or is_hb), "v3c_owner_allocation_only")
        value=verify_heldout_binding(point, verify_checkpoint=verify_checkpoint, hb_v2=is_hb, hb_v3c=is_hb)
        report=ROOT/('docs/plans/CHICGRASP_SCALED50_HB_V3C_BATCH' if is_hb else 'docs/plans/CHICGRASP_SCALED50_V3C_HARNESS_SMOKE')
        release=load(report/"RELEASE.json")
        for key in ("owner_task","clock","approved_protocol","predictions","prediction_manifest"):
            require(identity(release[key]["path"]) == release[key], "v3c_release_identity:"+key)
        if is_hb:
            require(identity(report/'RELEASE.json')['sha256']=='2cba469262beaf177dee21ccb44ceab6d6368446c314151d0353e74a1cbf231b', 'hb_v3c_release_identity')
            for key in ('positions','hb_manifest','prediction_freeze','smoke_result','smoke_process','v2_result'):
                require(identity(release[key]['path'])==release[key], 'hb_v3c_release_identity:'+key)
            require(release['status']=='RELEASED' and release['consumed_starts']==30 and release['policy_predictions']==600, 'hb_v3c_owner_release')
            expected_order=['HB01_R0','HB01_R1']+[f'HB{i:02d}_R{r}' for r in (0,1) for i in range(2,11)]+[f'HB{i:02d}_R2' for i in range(1,11)]
            positions=load(release['positions']['path'])
            require(release['allocation_order']==positions['allocation_order']==expected_order and positions['attempts'][expected_order.index(point)]==value['heldout_allocation'], 'hb_v3c_position_order')
            smoke=load(release['smoke_result']['path']); process=load(release['smoke_process']['path'])
            require(smoke['V3C_VERDICT']=='PASS' and smoke['HB_TECHNICAL_READY']=='YES' and process['tree_empty_verified'] and process['process_exit_verified'], 'hb_v3c_smoke_release_gate')
            value.update(hb_manifest=release['hb_manifest'], prediction_freeze=release['prediction_freeze'], allocation_order=expected_order, admin_cycles_used=release['admin_cycles_used'])
        else:
            require(release["status"] == "RELEASED" and release["point"] == point and release["consumed_starts"] == 1 and release["policy_predictions"] == 20, "v3c_owner_release")
            require(value["authored_world_xyz_m"] == [0.6468198848126357,0.037499547104199664,0.8155] and value["evaluation_seed"] == 14345, "v3c_exact_pose_seed")
        require(value["scene_profile"]["sha256"] == release["scene_sha256"], "v3c_approved_scene")
        value.update(schema="scaled50-envelope-v3c-development-binding/v1", envelope_v2=True, envelope_protocol="v3c",
            task=release["task"], grant=identity(report/"RELEASE.json"), task_clock=release["clock"],
            approved_protocol=release["approved_protocol"], predictions=release["predictions"],
            prediction_manifest=release["prediction_manifest"], definition_id=release["definition_id"],
            grant_allocation='THIRTY_FROZEN_HB_V3C_ALLOCATIONS' if is_hb else 'ONE_OLD_H05_R2_SMOKE_EXCLUDED_FROM_HB')
        return value
    if point in HB_POINTS:
        require(envelope_v2, "hb_requires_v2")
        return verify_heldout_binding(point, verify_checkpoint=verify_checkpoint, hb_v2=True)
    if envelope_v2:
        require(point == "H03_R0", "v2_only_H03_R0")
        value=verify_heldout_binding(point, verify_checkpoint=verify_checkpoint)
        report=ROOT/'docs/plans/CHICGRASP_SCALED50_ENVELOPE_V2_HARNESS_SMOKE'
        release=load(report/'RELEASE.json')
        for key in ("owner_task", "clock", "approved_protocol"):
            ref=release[key]
            require(identity(ref["path"])==ref, "v2_release_identity:"+key)
        require(release["status"]=="RELEASED" and release["point"]==point, "v2_owner_release")
        require(value["authored_world_xyz_m"]==[0.6708960874942572,0.006105818367496813,0.8155], "v2_exact_authored_pose")
        value.update(schema="scaled50-envelope-v2-development-binding/v1", envelope_v2=True,
            task="CHICGRASP_SCALED50_ENVELOPE_V2_HARNESS_SMOKE", grant=identity(report/'RELEASE.json'),
            task_clock=release["clock"], approved_protocol=release["approved_protocol"],
            grant_allocation="ONE_H03_R0_SMOKE_EXCLUDED_FROM_HB")
        return value
    if point in HELDOUT_POINTS:
        return verify_heldout_binding(point, verify_checkpoint=verify_checkpoint)
    require(point in POINTS, 'excluded_point')
    attempt = ROOT / 'outputs/stride1_d1_training/attempt001'
    frozen = attempt / 'preflight001'
    grant = identity(ROOT / 'docs/plans/CHICGRASP_STRIDE1_D1_LIVE_BATCH/OWNER_GRANT.txt')
    require(grant['sha256'] == GRANT_SHA256, 'owner_grant')
    result_ref = identity(attempt / 'step2/counted001/reload_result.json')
    require(result_ref['sha256'] == RESULT_SHA256, 'historical_r4_result')
    result = load(result_ref['path'])
    require(result['status'] == 'PASS' and result['reproduced'], 'historical_result_or_reload')
    process_path = attempt / 'step2/counted001/reload_process.json'
    process_ref = identity(process_path)
    require(process_ref['sha256'] == RELOAD_PROCESS_SHA256, 'reload_command_record')
    process = load(process_path)
    command = process['command']
    require(process['success'] and process['job_drained'] and not process['forced_termination'] and
            'reload' in command and '--development-r4' in command and
            command[command.index('--step') + 1] == '2', 'successful_r4_reload')
    dispatch = Path(command[command.index('--dispatch-dir') + 1]).resolve(strict=True)
    require(dispatch == process_path.parent.resolve(), 'selected_dispatch')
    require(Path(command[command.index('--frozen') + 1]).resolve() == frozen.resolve(), 'reload_frozen_inputs')
    selection = load(dispatch / 'selection.json')
    checkpoint = authenticate(dispatch / 'BEST.pt', selection['checkpoint'], hash_bytes=verify_checkpoint)
    require(checkpoint['sha256'] == CHECKPOINT_SHA256, 'exact_checkpoint')
    normalizer = identity(ROOT / 'outputs/exposure_control_and_training/preflight001/normalization_step2.json')
    require(normalizer['sha256'] == NORMALIZER_SHA256, 'exact_normalizer')
    authenticate(dispatch / 'normalization.json', normalizer)
    metadata = selection['metadata']
    recipe_ref = authenticate(frozen / 'recipe.json', metadata['recipe_identity'])
    source_ref = authenticate(frozen / 'sources/dataset.json', metadata['source_identity'])
    recipe = load(recipe_ref['path'])
    require(recipe['model_configuration'] == overfit_policy.DEFAULT_POLICY_CONFIG, 'model_configuration')
    require(selection['checkpoint_name'] == 'BEST.pt' and selection['optimizer_step'] == 20000,
            'selection_record')
    source = qualified_source(point, load(source_ref['path']))
    visual = load(source['source_files']['stage1_visuals_before_candidate.json']['path'])
    camera = identity(ROOT / 'config/policy_camera_calibration.json')
    approval = identity(ROOT / 'docs/grants/GATE6_STAGE0_VISUAL_APPROVAL_20261012.json')
    require(camera['sha256'] == visual['config_sha256'] and
            load(camera['path']) == visual['calibration'] and load(approval['path']) == visual['approval'],
            'approved_visual_config')
    value = dict(schema='r4-live-development-binding/v1', point=point, **source,
        checkpoint=checkpoint, checkpoint_identity=checkpoint, checkpoint_verified=bool(verify_checkpoint),
        normalizer=normalizer, normalization=load(normalizer['path']), grant=grant,
        r4_result=result_ref, r4_reload_process=process_ref, recipe_identity=metadata['recipe_identity'],
        source_identity=metadata['source_identity'], schedule_identity=metadata['schedule_identity'],
        model_configuration=recipe['model_configuration'], visual_config=camera, visual_approval=approval,
        evaluation_seed=12345, seed=42, device='cuda:0', weights_kind='EMA', optimizer_step=20000,
        policy_support_path=load(ROOT / 'docs/checkpoints/gate6_stage3/binding.json')['policy_support_path'],
        offline_fit_advisory=dict(authority='CHICGRASP_STRIDE1_D1_LIVE_BATCH',
            historical_status=result['historical_strict_5mm_result'], readiness=result['readiness_scores_result'],
            historical_strict_5mm=result['historical_strict_5mm_result'], run_type='HARNESS'))
    validate_checkpoint_record(dict(binding=value, checkpoint_metadata=selection['metadata'], device='cuda:0'), value)
    return value


def heldout_position(point):
    """Frozen owner lists through the existing translated setup parameter."""
    require(point in HELDOUT_POINTS, 'excluded_heldout_point')
    frozen = HELDOUT_DOC / 'POSITIONS.json'
    require(identity(frozen)['sha256'] == HELDOUT_POSITIONS_SHA256, 'heldout_positions_identity')
    positions = load(frozen)
    require(tuple(r['allocation'] for r in positions['attempts']) == HELDOUT_POINTS, 'heldout_order')
    for key in ('scene_profile', 'heldout_manifest', 'training_diagnostic_manifest',
                'owner_task', 'owner_release', 'clock'):
        authenticate(positions[key]['path'], positions[key])
    row = positions['attempts'][HELDOUT_POINTS.index(point)]
    from .translated_script import point_record
    translated = point_record(f"R{row['order']+1:06d}", authored_xy=row['position']['authored_xy_m'])
    require(translated['actions'][:50] == row['setup_actions'] and
            translated['authored_world_xyz_m'] == row['authored_world_xyz_m'] and
            row['evaluation_seed'] == 12345 + 1000*row['round'], 'heldout_setup_or_seed')
    return positions, row


def hb_position(point):
    """Owner-frozen HB coordinates through the unchanged setup and model path."""
    require(point in HB_POINTS, 'excluded_hb_point')
    require(identity(HB_DOC/'RELEASE.json')['sha256'] == HB_RELEASE_SHA256, 'hb_release_identity')
    release=load(HB_DOC/'RELEASE.json')
    require(release['status']=='RELEASED' and release['consumed_starts']==30 and release['policy_predictions']==600, 'hb_release')
    for key in ('owner_task','clock','approved_protocol','positions','hb_manifest','smoke_result','smoke_process'):
        authenticate(release[key]['path'], release[key])
    require(release['hb_manifest']['sha256']=='c76525600edef6126ebec8bc5fa19209e2eeb2d26fe7b79e560113359233ee17', 'hb_manifest')
    require(release['approved_protocol']['sha256']=='67fc3caaa0b2eebdf5eb2bfe2a2a8e7c74255fb5ffe6e229da279627c74bd25a', 'hb_protocol')
    smoke=load(release['smoke_result']['path']); process=load(release['smoke_process']['path'])
    require(smoke['HB_TECHNICAL_READY']=='YES' and process['tree_empty_verified'] and process['process_exit_verified'], 'hb_smoke_prerequisite')
    positions=load(release['positions']['path'])
    require(tuple(r['allocation'] for r in positions['attempts'])==HB_POINTS, 'hb_round_major_order')
    row=positions['attempts'][HB_POINTS.index(point)]
    manifest=load(release['hb_manifest']['path'])
    frozen=manifest['accepted'][row['order']%10]
    require(row['position']==frozen and row['round']==row['order']//10, 'hb_exact_position_round')
    from .translated_script import point_record
    translated=point_record(f"R{row['order']+1:06d}",authored_xy=frozen['authored_xy_m'])
    require(row['setup_actions']==translated['actions'][:50] and row['authored_world_xyz_m']==frozen['authored_world_xyz_m'] and
            row['authored_world_xyz_m']==translated['authored_world_xyz_m'] and row['evaluation_seed']==12345+1000*row['round'], 'hb_setup_pose_seed')
    authenticate(positions['scene_profile']['path'],positions['scene_profile'])
    require(positions['scene_profile']['sha256']=='4f40d26e122c99ea5cb980edea4a391b5335a99fe060b84cacd40305374c0d5f', 'hb_scene')
    return positions,row


def hc_position(point):
    """Unchanged supplied coordinates and owner-continued seeds; no candidate selection."""
    require(point in HC_POINTS, 'excluded_hc_point')
    release=load(HC_DOC/'RELEASE.json')
    require(identity(HC_DOC/'RELEASE.json')['sha256']==HC_RELEASE_SHA256, 'hc_release_identity')
    for key in ('owner_task','clock','positions','hc_manifest','offline_freeze_checks','daily_ledger','hb_result','approved_protocol'):
        authenticate(release[key]['path'], release[key])
    require(release['status']=='RELEASED' and release['consumed_starts']==30 and release['policy_predictions']==600, 'hc_release')
    require(release['hc_manifest']['sha256']=='6b72af4c8f822d4db52d20c89516c0dfb7a0ee36d0512c6bcd6e0132c8b50cee', 'hc_manifest')
    require(release['approved_protocol']['sha256']=='b52b11581d20ee7bbe97a9f0079caada7c5d3b577a00e72e63cbd2b9505d7603', 'hc_frozen_v3c')
    require(load(release['offline_freeze_checks']['path'])['status']=='PASS', 'hc_offline_gate')
    hb=load(release['hb_result']['path'])
    require(hb['BATCH_STATUS']=='COMPLETE' and hb['PRIMARY_VERDICTS_ESTABLISHED']==30, 'hc_completed_HB')
    positions=load(release['positions']['path']); index=HC_POINTS.index(point)
    require(tuple(positions['allocation_order'])==tuple(release['allocation_order'])==HC_POINTS and tuple(x['allocation'] for x in positions['attempts'])==HC_POINTS, 'hc_round_major_order')
    row=positions['attempts'][index]; frozen=load(release['hc_manifest']['path'])['positions'][index%10]
    require(row['position']==frozen and row['order']==index and row['round']==row['local_round']==index//10 and row['continued_round']==3+index//10, 'hc_exact_position_round')
    from .translated_script import point_record
    translated=point_record(f'R{index+1:06d}',authored_xy=frozen['authored_xy_m'])
    require(row['setup_actions']==translated['actions'][:50] and row['authored_world_xyz_m']==frozen['authored_world_xyz_m']==translated['authored_world_xyz_m'] and row['evaluation_seed']==12345+1000*row['continued_round'], 'hc_setup_pose_seed')
    authenticate(positions['scene_profile']['path'],positions['scene_profile'])
    require(positions['scene_profile']['sha256']==release['scene_sha256']=='4f40d26e122c99ea5cb980edea4a391b5335a99fe060b84cacd40305374c0d5f', 'hc_scene')
    return positions,row


def verify_heldout_binding(point, *, verify_checkpoint=True, hb_v2=False, hb_v3c=False, hc_v3c=False):
    """Exact released BEST, successful fresh reload, and attempt005 identity."""
    positions, row = hc_position(point) if hc_v3c else hb_position(point) if hb_v2 else heldout_position(point)
    release_path = HELDOUT_DOC / 'RELEASE.json'
    require(identity(release_path)['sha256'] == HELDOUT_RELEASE_SHA256, 'heldout_release_identity')
    release = load(release_path)
    require(release['OWNER_REVIEW_GRANT5_FOR_HELDOUT'] == 'RELEASED', 'heldout_owner_release')
    for key in ('publication', 'reload_result', 'reload_process'):
        authenticate(release[key]['path'], release[key])
    publication = load(release['publication']['path'])
    require(publication['status'] == 'CHICGRASP_SCALED50_TRAINING_COMPLETE' and
            publication['BEST']['sha256'] == release['checkpoint']['sha256'] and
            publication['selected_step'] == 120000, 'heldout_completed_publication')
    reload = load(release['reload_result']['path'])
    process = load(release['reload_process']['path'])
    require(reload['status'] == 'PASS' and reload['reproduced'] and process['success'] and
            process['job_drained'] and not process['forced_termination'], 'heldout_fresh_reload')
    checkpoint = authenticate(release['checkpoint']['path'], release['checkpoint'], hash_bytes=verify_checkpoint)
    require(checkpoint['sha256'] == '1e91f1db31fb9b2856845abc2266a9078d9c97c1b291bc1e1faaa3c29a3855fe', 'heldout_best120000')
    normalizer = authenticate(release['normalizer']['path'], release['normalizer'])
    require(normalizer['sha256'] == 'b15853dc2619416f2ff89a04a3da2f476557305b083aefe73dc9a8ccddea8141', 'heldout_normalizer')
    dispatch = Path(checkpoint['path']).parent
    authenticate(dispatch/'normalization.json', normalizer)
    selection = load(dispatch/'selection.json')
    require(selection['optimizer_step'] == 120000 and selection['checkpoint_name'] == 'BEST.pt', 'heldout_selection')
    metadata = selection['metadata']
    frozen = ROOT/'outputs/scaled50_aligned_training/attempt001/preflight002'
    recipe = load(authenticate(frozen/'recipe.json', metadata['recipe_identity'])['path'])
    authenticate(frozen/'sources/dataset.json', metadata['source_identity'])
    require(recipe['model_configuration'] == overfit_policy.DEFAULT_POLICY_CONFIG, 'heldout_model_configuration')
    from .stage1_demo import approved_scaled_profile
    profile = approved_scaled_profile()
    setup = np.asarray(row['setup_actions'], np.float32)
    api.validate_policy_action(setup)
    require(setup.shape == (50, 5) and np.all(setup[:, 3:] == 1), 'heldout_setup')
    value = dict(schema='scaled50-heldout-development-binding/v1', point=point,
        heldout_allocation=row, publication=release['publication'], task='CHICGRASP_SCALED50_HELDOUT_BATCH',
        task_clock=positions['clock'], checkpoint=checkpoint, checkpoint_identity=checkpoint,
        checkpoint_verified=bool(verify_checkpoint), normalizer=normalizer,
        normalization=load(normalizer['path']), grant=identity(release_path),
        r4_result=release['reload_result'], r4_reload_process=release['reload_process'],
        recipe_identity=metadata['recipe_identity'], source_identity=metadata['source_identity'],
        schedule_identity=metadata['schedule_identity'], model_configuration=recipe['model_configuration'],
        authored_world_xyz_m=row['authored_world_xyz_m'], authored_quaternion_wxyz=[1.,0.,0.,0.],
        original_authored_world_xyz_m=[.65,0.,.8155], setup_actions=setup.tolist(),
        setup_action_identity=array_identity(setup), setup_source_indices=[0,49],
        production_configuration=profile['physical_configuration'], scene_profile=positions['scene_profile'],
        evaluation_seed=row['evaluation_seed'], seed=42, device='cuda:0', weights_kind='EMA', optimizer_step=120000,
        policy_support_path=load(ROOT/'docs/checkpoints/gate6_stage3/binding.json')['policy_support_path'],
        offline_fit_advisory=dict(authority='CHICGRASP_SCALED50_HELDOUT_BATCH', historical_status='FAIL',
            readiness='NOT_DEFINED_FOR_SCALED50', historical_strict_5mm='FAIL', run_type='HARNESS'))
    if hb_v2:
        current=load(HB_DOC/'RELEASE.json')
        value.update(schema='scaled50-hb-v2-development-binding/v1', envelope_v2=True,
            task='CHICGRASP_SCALED50_HB_V2_BATCH', grant=identity(HB_DOC/'RELEASE.json'),
            task_clock=current['clock'], approved_protocol=current['approved_protocol'],
            hb_manifest=current['hb_manifest'], grant_allocation='THIRTY_HB_V2_TRIALS')
    if hb_v2 and not hb_v3c and point != 'HB01_R0':
        continuation_path=HB_DOC/'continuation001/RELEASE.json'
        require(identity(continuation_path)['sha256']==HB_CONTINUATION_RELEASE_SHA256, 'hb_continuation_release_identity')
        continuation=load(continuation_path)
        require(continuation['status']=='RELEASED' and point in continuation['allocation_order'][1:], 'hb_continuation_allocation')
        for key in ('original_grant','owner_task','clock','approved_protocol','positions','hb_manifest','readjudication'):
            authenticate(continuation[key]['path'],continuation[key])
        adjudication=load(continuation['readjudication']['path'])
        require(adjudication['technical_clean'] and adjudication['CAPTURE_INTEGRITY']=='PASS'
                and adjudication['PROCESS_COMPLETION']=='PASS' and adjudication['end_state']['V2_VERDICT']=='PASS', 'hb_readjudication_required')
        require(continuation['clock']==current['clock'] and continuation['approved_protocol']==current['approved_protocol']
                and continuation['hb_manifest']==current['hb_manifest'], 'hb_continuation_frozen_inputs')
        value.update(task=continuation['task'],grant=identity(continuation_path),
            grant_allocation='TWENTY_NINE_REMAINING_HB_ALLOCATIONS',
            continuation=dict(original_grant=continuation['original_grant'],readjudication=continuation['readjudication'],
                original_attempt=continuation['original_attempt'],allocation_order=continuation['allocation_order'],
                admin_cycles_used=2))
    validate_checkpoint_record(dict(binding=value, checkpoint_metadata=metadata, device='cuda:0'), value)
    return value


def validate_checkpoint_record(checkpoint, value):
    """Validate the saved deployment receipt without deserializing model weights."""
    require(checkpoint.get('binding') == value and checkpoint.get('device') == 'cuda:0', 'deployment_record_binding')
    metadata = checkpoint['checkpoint_metadata']
    expected = dict(step=2, seed=42, optimizer_step=value['optimizer_step'], weights_kind='EMA',
        normalization=value['normalization'], recipe_identity=value['recipe_identity'],
        source_identity=value['source_identity'], schedule_identity=value['schedule_identity'])
    for key, wanted in expected.items():
        require(metadata.get(key) == wanted, 'checkpoint_metadata:' + key)
    return True


def load_bound_policy(value, device='cuda:0'):
    """Same recovery-format EMA loader as the successful fresh R4 reload."""
    from .exposure_cuda import cuda_constructor
    from .exposure_recovery import load_ema
    require(device == 'cuda:0' and value['checkpoint_verified'], 'verified_cuda_checkpoint_required')
    receipts = []
    with cuda_constructor(receipts):
        policy, metadata = load_ema(value['checkpoint']['path'], device=device)
    validate_checkpoint_record(dict(binding=value, checkpoint_metadata=metadata, device=str(policy.device)), value)
    require(policy.gate3_config == value['model_configuration'], 'loaded_architecture')
    for name, field in value['normalization']['fields'].items():
        for key in ('scale', 'offset'):
            require(policy.normalizer[name].params_dict[key].detach().cpu().tolist() == field[key],
                    'loaded_normalizer:' + name + ':' + key)
    policy.eval()
    policy.requires_grad_(False)
    return policy, metadata, receipts


def prepare_predictor(point, *, envelope_v2=False, envelope_v3c=False):
    """Called only by the authorized CUDA deployment process, never metadata CLI."""
    import torch
    import torchvision
    native = (torch.__file__, torchvision.__file__, np.__file__)
    value = verify_binding(point, envelope_v2=envelope_v2, envelope_v3c=envelope_v3c)
    support = Path(value['policy_support_path']).resolve(strict=True)
    if str(support) not in sys.path:
        sys.path.append(str(support))
    require(native == (torch.__file__, torchvision.__file__, np.__file__), 'support_replaced_native_core')
    overfit_policy.set_deterministic(42)
    policy, metadata, receipts = load_bound_policy(value)
    return make_predictor(policy, value['evaluation_seed']), dict(binding=value, checkpoint_metadata=metadata,
        libraries=dict(torch=torch.__version__, numpy=np.__version__, torchvision=torchvision.__version__),
        core_package_paths=list(native), device=str(policy.device), constructor_receipts=receipts,
        model_eval=True, parameter_gradients_enabled=False)


def configure_start(cfg, session):
    value = session.r4_live_binding
    before, rotation = list(cfg.scene.chicken.init_state.pos), list(cfg.scene.chicken.init_state.rot)
    require(before == value['original_authored_world_xyz_m'] and rotation == value['authored_quaternion_wxyz'],
            'original_authored_pose_changed')
    cfg.scene.chicken.init_state.pos = tuple(value['authored_world_xyz_m'])
    session.r4_live_start_configuration = dict(before_xyz_m=before,
        after_xyz_m=list(cfg.scene.chicken.init_state.pos), rotation_wxyz=rotation,
        changed_fields=['chicken.init_state.pos.x', 'chicken.init_state.pos.y'],
        robot_reset_configuration_changed=False)


def _array(tensor):
    return np.asarray(tensor.detach().cpu().numpy())


def validate_preparation(record, value):
    require(record['point'] == value['point'] and record['checkpoint_identity'] == value['checkpoint_identity']
            and record['normalizer'] == value['normalizer'] and
            record['setup_action_identity'] == value['setup_action_identity'], 'preparation_binding')
    cfg = record['configuration']
    require(cfg['before_xyz_m'] == value['original_authored_world_xyz_m'] and
            cfg['after_xyz_m'] == value['authored_world_xyz_m'] and
            cfg['rotation_wxyz'] == value['authored_quaternion_wxyz'] and
            record['configured_pose_after_reset'] == cfg['after_xyz_m'] + cfg['rotation_wxyz'], 'reset_configuration')
    expected = np.asarray(value['authored_world_xyz_m'] + value['authored_quaternion_wxyz'], np.float32)
    require(np.array_equal(np.asarray(record['default_root_state_world_wxyz'], np.float32), expected),
            'reset_default_root_pose')
    require(record['production_configuration'] == value['production_configuration'], 'production_configuration')
    measured = record['measured']
    require(tuple(record['clock_before']) == tuple(record['clock_after']) ==
            tuple(measured['clock_before']) == tuple(measured['clock_after'])
            and measured['valid'] is True and not measured['issues'], 'preparation_measurement')
    pose, quaternion = _measured_pose(measured['poses']['chicken_actor_world_wxyz'])
    if value.get('heldout_allocation'):
        # A new position has no historical measured settled pose. Authenticate
        # authored/default reset state above; retain measured startup separately.
        require(np.isfinite(pose).all() and np.isfinite(quaternion).all(), 'startup_pose_finite')
        return True
    reference, reference_q = _measured_pose(value['qualified_settled_measurement']['poses']['chicken_actor_world_wxyz'])
    # Startup readback precision only; neither a command tolerance nor physical scoring.
    require(np.max(np.abs(pose[:3] - reference[:3])) <= 1e-6, 'wrong_measured_settled_position')
    require(min(np.linalg.norm(quaternion-reference_q), np.linalg.norm(quaternion+reference_q)) <= 1e-6,
            'wrong_measured_settled_orientation')
    return True


def save_preparation(runtime, output):
    """Read existing reset state once; no step, reset, render or sensor getter."""
    from .stage1_demo import save_visuals
    value = runtime.r4_live_binding
    phase = 'r4_live_preparation'
    if value.get('heldout_allocation'):
        visual = save_visuals(runtime, output, phase, scene_profile=True)
    else:
        expected_visual = load(value['source_files']['stage1_visuals_before_candidate.json']['path'])
        visual = save_visuals(runtime, output, phase)
        for key in ('config_sha256', 'calibration', 'approval', 'startup_profile', 'visuals', 'renderer'):
            require(visual[key] == expected_visual[key], 'qualified_visual:' + key)
    before = list(runtime.clock())
    state = _array(runtime.env.scene['chicken'].data.default_root_state)
    origins = _array(runtime.env.scene.env_origins)
    require(state.shape == (1, 13) and origins.shape == (1, 3) and np.all(origins == 0), 'one_env_default_root')
    root = state[0, :7].copy()
    root[:3] += origins[0]
    cfg = runtime.env.cfg.scene.chicken.init_state
    measured = runtime.measure()
    record = dict(point=value['point'], configuration=deepcopy(runtime.r4_live_start_configuration),
        configured_pose_after_reset=list(cfg.pos) + list(cfg.rot), default_root_state_world_wxyz=root.tolist(),
        clock_before=before, clock_after=list(runtime.clock()), origin_clock=list(runtime.origin_clock),
        measured=measured, production_configuration=runtime.effective_configuration,
        checkpoint_identity=value['checkpoint_identity'], normalizer=value['normalizer'],
        setup_action_identity=value['setup_action_identity'],
        visual_file='stage1_visuals_' + phase + '.json',
        visual_identity=identity(Path(output) / ('stage1_visuals_' + phase + '.json')),
        initialization_physics='Existing constructor/reset advancement; not claimed zero.',
        robot_configuration='unchanged production reset')
    # Preserve the actual observation even when subsequent validation refuses startup.
    write(Path(output) / 'r4_live_preparation.json', record)
    validate_preparation(record, value)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--point', required=True, choices=POINTS + HELDOUT_POINTS + HB_POINTS + HC_POINTS)
    parser.add_argument('--envelope-v2', action='store_true')
    parser.add_argument('--envelope-v3c', action='store_true')
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(verify_binding(args.point, verify_checkpoint=not args.metadata_only, envelope_v2=args.envelope_v2, envelope_v3c=args.envelope_v3c), allow_nan=False))


if __name__ == '__main__':
    main()
