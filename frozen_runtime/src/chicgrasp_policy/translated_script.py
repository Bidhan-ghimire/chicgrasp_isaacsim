"""Constant translations of the accepted Stage1 script. Import is offline/inert."""
from copy import deepcopy
import hashlib
import math
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

import numpy as np

from .attachment_capture import frozen_candidate_actions
from .domain_previews import rotation
from .geometric_capture_check import identity
from .interface import validate_policy_action, FIXED_TCP_QUATERNION_BASE_WXYZ
from .side_opening_admission import RULE, RULE_SHA256, frozen_rule
from .stage1_demo import load, write
from .stage3_policy import PolicyWriter

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/plans/CHICGRASP_TRANSLATED_SCRIPT_RECORDER_BUILD'
BUILD_MANIFEST = DOC / 'SWEEP.json'
EXECUTION_DOC = ROOT / 'docs/plans/CHICGRASP_TRANSLATED_SCRIPT_RECORDING_SWEEP'
MANIFEST = EXECUTION_DOC / 'EXECUTION.json'
CONTINUATION_DOC = ROOT / 'docs/plans/CHICGRASP_TRANSLATED_SCRIPT_RECORDING_SWEEP_R2'
CONTINUATION = CONTINUATION_DOC / 'CONTINUATION.json'
CONTINUATION_OUTPUT = ROOT / 'outputs/translated_script_sweep_002'
FREEZE = ROOT / 'docs/plans/CHICGRASP_EXPERT_PILOT_BUILD/CAMPAIGN.json'
BASIS = ROOT / 'docs/plans/CHICGRASP_GENERALIZATION_DESIGN/DESIGN_BASIS.json'
SOURCE = Path('C:/robotics/ChicGrasp_gate6_stage1/outputs/gate6_stage1_gpu_demo/evidence_20261012_002')
ACTION_FILE = SOURCE / 'capture/experiment/episode/policy/action.npy'
ACTION_SHA256 = '6f6d2142186d485e8e09d5d2d77736c52383daa3cc1c8e53c44b691f08d85152'
FREEZE_SHA256 = '650acba6af2cd5e365c26ddce3699117d7c15487528ec49bc3de28e2b327c0c4'
POINTS = tuple(f'G{i:02d}' for i in range(9))
SMOKE_DOC = ROOT / 'docs/plans/CHICGRASP_RANDOM_XY_RECORDER_SMOKE'
SMOKE_MANIFEST = SMOKE_DOC / 'SMOKE.json'
SMOKE_OUTPUT = ROOT / 'outputs/random_xy_recorder_smoke/attempt001'
SMOKE_PROFILE = ROOT / 'docs/plans/CHICGRASP_SCALED_TABLE_AND_COVERAGE/attempt005/SCENE_PROFILE.json'
SMOKE_XY_BOUNDS = ((0.6400000393390657, 0.699988579750061), (-0.009999997913837447, 0.295))
PLACEMENT_SEED = 42001
COLLECTION_DOC = ROOT / 'docs/plans/CHICGRASP_RANDOM_XY_50_COLLECTION'
COLLECTION_MANIFEST = COLLECTION_DOC / 'COLLECTION.json'
COLLECTION_OUTPUT = ROOT / 'outputs/random_xy_50_collection/attempt001'
COLLECTION_POINTS = tuple(f'R{i:06d}' for i in range(2, 102))
COLLECTION_NATIVE_NAMES = tuple(f'native{i:03d}' for i in range(1, 5))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def approved_freeze():
    if identity(FREEZE)['sha256'] != FREEZE_SHA256:
        raise ValueError('translated_freeze_changed')
    c = load(FREEZE)
    # Identity is pinned in the sweep as well as each input file. No expert runs.
    if c['grid_revision'] != 'task1c-r2-placement-2mm-clearance-1mm/v1' or c['CAMERA_DECISION'] != 'CURRENT_POSES_STAY':
        raise ValueError('translated_grid_revision')
    for path, want in {**c['visual_identity_files'], **c['configuration_hashes']}.items():
        if identity(ROOT/path) != want:
            raise ValueError('translated_visual_identity:' + path)
    for p in c['start_images'].values():
        if p['status'] != 'OWNER_APPROVED_START_IMAGES_ONLY':
            raise ValueError('translated_start_image_approval')
        for item in [p['preview_record'], *p['images'].values()]:
            if identity(ROOT/item['path']) != {k:item[k] for k in ('bytes','sha256')}:
                raise ValueError('translated_start_image_identity')
    if identity(BASIS) != {k:c['basis'][k] for k in ('bytes','sha256')}:
        raise ValueError('translated_calibration_identity')
    return c


def original_actions():
    if identity(ACTION_FILE) != dict(bytes=4128, sha256=ACTION_SHA256):
        raise ValueError('translated_original_candidate_identity')
    original = np.load(ACTION_FILE, allow_pickle=False)
    if original.dtype != np.float32 or not np.array_equal(original, frozen_candidate_actions()):
        raise ValueError('translated_original_generator_disagreement')
    return original


def translate(original, world_delta, robot_quaternion):
    delta = rotation(robot_quaternion).T @ np.asarray(world_delta, dtype=np.float64)
    if delta.shape != (3,) or not np.isfinite(delta).all() or delta[2] != 0:
        raise ValueError('translated_requires_aligned_XY_preserving_Z')
    precast = original.astype(np.float64).copy()
    precast[:, :3] += delta
    stored = precast.astype(np.float32)
    for row in precast:
        validate_policy_action(row)
    for row in stored:
        validate_policy_action(row)
    if not np.array_equal(stored[:, 2:], original[:, 2:]):
        raise ValueError('translated_Z_or_jaws_changed')
    return stored, precast, delta


def point_record(point, freeze=None, *, authored_xy=None):
    if authored_xy is None:
        c = freeze or approved_freeze()
        if point not in ('ORIGINAL', *POINTS):
            raise ValueError('translated_permanent_point_required')
        authored = np.asarray(c['start_images'][point]['authored_world_xyz_m'])
        origin = np.asarray(c['start_images']['ORIGINAL']['authored_world_xyz_m'])
        quaternion = c['start_images'][point]['authored_quaternion_wxyz']
    else:
        if not isinstance(point, str) or re.fullmatch(r'R[0-9]{6}', point) is None or int(point[1:]) == 0:
            raise ValueError('translated_random_candidate_id_required')
        xy = np.asarray(authored_xy, dtype=np.float64)
        bounds = np.asarray(SMOKE_XY_BOUNDS)
        if xy.shape != (2,) or not np.isfinite(xy).all() or np.any(xy < bounds[:, 0]) or np.any(xy > bounds[:, 1]):
            raise ValueError('translated_authored_xy_outside_rectangle')
        authored = np.array([*xy, .8155], dtype=np.float64)
        origin = np.array([.65, 0., .8155], dtype=np.float64)
        quaternion = [1, 0, 0, 0]
    original = original_actions()
    if origin.tolist() != [.65, 0., .8155] or authored[2] != origin[2]:
        raise ValueError('translated_authored_origin_not_measured_root')
    base = load(BASIS)['nominal_object_waypoint_anchor']['robot_root_world_wxyz']
    actions, precast, delta = translate(original, authored-origin, base[3:])
    value = dict(point=point, authored_world_xyz_m=authored.tolist(),
        authored_quaternion_wxyz=quaternion,
        original_authored_world_xyz_m=origin.tolist(), world_displacement_m=(authored-origin).tolist(),
        base_displacement_m=delta.tolist(), robot_root_world_wxyz=base,
        original_action_sha256=ACTION_SHA256, translation_arithmetic='saved_float32_to_float64_plus_constant_then_float32',
        actions=actions.tolist(), precast_actions=precast.tolist(),
        sequence_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
        setup_indices=[0,49], training_indices=[50,199],
        fixed_orientation_base_wxyz=list(FIXED_TCP_QUATERNION_BASE_WXYZ),
        seed=42, output_id='script_sweep_001_'+point,
        output_directory='outputs/translated_script_sweep_001/'+point,
        status='NOT_EXECUTED', training_eligible=False)
    if authored_xy is not None:
        value.update(output_id='random_xy_recorder_smoke_'+point,
            output_directory=SMOKE_OUTPUT.relative_to(ROOT).as_posix()+'/native001',
            placement=dict(schema='random-xy-placement/v1', candidate_id=point,
                source='EXPLICIT_AUTHORED_XY', frame='WORLD', authored_xy_m=xy.tolist(),
                drawn_xy_m=None, bounds_xy_m=[list(b) for b in SMOKE_XY_BOUNDS],
                settled_pose_status='NOT_RECORDED', environment_seed=42))
    return value


YAW_DOC = ROOT / 'docs/plans/CHICGRASP_SCALED50_SCRIPTED_YAW_PROBE'
YAW_MANIFEST = YAW_DOC / 'YAW.json'
YAW_OUTPUT = ROOT / 'outputs/scaled50_scripted_yaw_probe/attempt001'
YAW_ANGLES = (0, 5, -5, 10, -10, 20, -20)
YAW_CODES = ('Z00', 'P05', 'M05', 'P10', 'M10', 'P20', 'M20')
YAW_POINTS = tuple(f'Y{site}_{code}' for site in ('H05', 'H07') for code in YAW_CODES)
YAW_LABELS = ['REFERENCE_ONLY_YAW_PROBE', 'NOT_SELECTED_FOR_TRAINING']


def yaw_manifest():
    """Finite owner-amended probe; unchanged translated scripts and scene bytes."""
    from .stage1_demo import approved_scaled_profile
    profile = approved_scaled_profile()
    if identity(SMOKE_PROFILE)['sha256'] != '4f40d26e122c99ea5cb980edea4a391b5335a99fe060b84cacd40305374c0d5f':
        raise ValueError('yaw_base_scene_changed')
    frozen_rule()
    positions_path = ROOT/'docs/plans/CHICGRASP_SCALED50_HELDOUT_BATCH/POSITIONS.json'
    positions = load(positions_path)
    attempts = []
    for site, xy in [('H05', [.6468198848126357, .037499547104199664]),
                     ('H07', [.6554754588227653, .13946365008643113])]:
        source = next(p for p in positions['attempts'] if p['allocation'] == site+'_R0')
        # The exact frozen position is bound below, independent of outcomes.
        if source['authored_world_xyz_m'] != [*xy, .8155]:
            raise ValueError('yaw_frozen_position_changed')
        for angle, code in zip(YAW_ANGLES, YAW_CODES):
            record = point_record('R000001', authored_xy=xy)
            point = 'Y'+site+'_'+code
            record.update(point=point, authored_quaternion_wxyz=[math.cos(math.radians(angle)/2),0,0,math.sin(math.radians(angle)/2)],
                yaw_probe=dict(site=site, authored_yaw_deg=angle, labels=YAW_LABELS,
                    classification='ADMIN: owner quaternion through existing init_state.rot',
                    training_eligible=False))
            record['placement']['candidate_id'] = point
            record['placement']['source'] = 'OWNER_FROZEN_HELDOUT_XY_AND_AUTHORED_YAW'
            for number in range(1,5):
                item = deepcopy(record)
                item.update(attempt_id=f'{point}_native{number:03d}', physical_attempt_number=number,
                    output_id=f'yaw_probe_{point}_native{number:03d}',
                    output_directory=YAW_OUTPUT.relative_to(ROOT).as_posix()+f'/{point}/native{number:03d}')
                attempts.append(item)
    return dict(schema='scripted-yaw-probe/v1', authority='CHICGRASP_SCALED50_SCRIPTED_YAW_PROBE',
        consumed_start_limit=8, shared_pre_action_repair_limit=3, angles_deg=list(YAW_ANGLES),
        labels=YAW_LABELS, training_eligible=False, automatic_retry=False,
        scene_profile=dict(path=SMOKE_PROFILE.relative_to(ROOT).as_posix(), **identity(SMOKE_PROFILE)),
        positions=dict(path=positions_path.relative_to(ROOT).as_posix(), **identity(positions_path)),
        owner_amendment=identity(YAW_DOC/'OWNER_AMENDMENT.txt'), clock=load(YAW_DOC/'CLOCK.json'),
        scene_clause=profile['scene_clause'], attempts=attempts)






ANCHOR_DOC = ROOT / 'docs/plans/CHICGRASP_SCALED50_HARD_XY_SCRIPTED_ANCHORS'
ANCHOR_MANIFEST = ANCHOR_DOC / 'ANCHORS.json'
ANCHOR_OUTPUT = ROOT / 'outputs/scaled50_hard_xy_scripted_anchors/attempt001'
ANCHOR_POINTS = ('H08', 'H09', 'H04')
ANCHOR_LABELS = ['REFERENCE_ONLY_XY_ANCHOR', 'CONTACT_MAP_EXTENSION', 'NOT_SELECTED_FOR_TRAINING']


def anchor_manifest():
    """Three owner-named identity-pose records using the existing XY translator."""
    from .stage1_demo import approved_scaled_profile
    profile = approved_scaled_profile()
    if identity(SMOKE_PROFILE)['sha256'] != '4f40d26e122c99ea5cb980edea4a391b5335a99fe060b84cacd40305374c0d5f':
        raise ValueError('anchor_base_scene_changed')
    frozen_rule()
    positions_path = ROOT/'docs/plans/CHICGRASP_SCALED50_HELDOUT_BATCH/POSITIONS.json'
    if identity(positions_path)['sha256'] != '89be3c48b3f1b82723db97711cd7fa3b984d76ae39da8f4b57c3d0f22c585eeb':
        raise ValueError('anchor_frozen_positions_changed')
    positions = load(positions_path)
    coordinates = [(0.679301627862094, 0.126151432777097),
                   (0.6886473523633403, 0.029999791581856033),
                   (0.6990706706885993, 0.02152576056590709)]
    attempts = []
    for site, xy in zip(ANCHOR_POINTS, coordinates):
        source = next(p for p in positions['attempts'] if p['allocation'] == site+'_R0')
        if source['authored_world_xyz_m'] != [*xy, .8155]:
            raise ValueError('anchor_frozen_position_changed')
        record = point_record('R000001', authored_xy=xy)
        record.update(point=site, authored_quaternion_wxyz=[1,0,0,0],
            hard_xy_anchor=dict(site=site, authored_yaw_deg=0, labels=ANCHOR_LABELS,
                classification='ADMIN: owner point through existing explicit-XY route',
                training_eligible=False))
        record['placement']['candidate_id'] = site
        record['placement']['source'] = 'OWNER_FROZEN_DEVELOPMENT_XY_IDENTITY_ROTATION'
        for number in range(1,5):
            item = deepcopy(record)
            item.update(attempt_id=f'{site}_native{number:03d}', physical_attempt_number=number,
                output_id=f'hard_xy_anchor_{site}_native{number:03d}',
                output_directory=ANCHOR_OUTPUT.relative_to(ROOT).as_posix()+f'/{site}/native{number:03d}')
            attempts.append(item)
    return dict(schema='hard-xy-scripted-anchors/v1', authority='CHICGRASP_SCALED50_HARD_XY_SCRIPTED_ANCHORS',
        consumed_start_limit=3, shared_pre_action_repair_limit=3, scheduled_points=list(ANCHOR_POINTS),
        cumulative_native_seconds_limit=1800, labels=ANCHOR_LABELS, training_eligible=False, automatic_retry=False,
        scene_profile=dict(path=SMOKE_PROFILE.relative_to(ROOT).as_posix(), **identity(SMOKE_PROFILE)),
        positions=dict(path=positions_path.relative_to(ROOT).as_posix(), **identity(positions_path)),
        owner_task=identity(ANCHOR_DOC/'OWNER_TASK.txt'), clock=load(ANCHOR_DOC/'CLOCK.json'),
        scene_clause=profile['scene_clause'], attempts=attempts)






def placement_draw(draw_index=0, *, seed=PLACEMENT_SEED):
    """Reproduce the indexed draw without using the environment random stream."""
    if type(draw_index) is not int or not 0 <= draw_index < 999999 or type(seed) is not int or seed < 0:
        raise ValueError('translated_placement_seed_or_index')
    generator = np.random.Generator(np.random.PCG64(seed))
    bounds = np.asarray(SMOKE_XY_BOUNDS, dtype=np.float64)
    for _ in range(draw_index + 1):
        xy = generator.uniform(bounds[:, 0], bounds[:, 1])
    return dict(schema='random-xy-placement/v1', candidate_id=f'R{draw_index+1:06d}',
        source='UNIFORM_PCG64', frame='WORLD', placement_seed=seed,
        bit_generator='PCG64', draw_index=draw_index,
        draw_order='X_then_Y_float64_one_pair_per_candidate', drawn_xy_m=xy.tolist(),
        authored_xy_m=xy.tolist(), bounds_xy_m=bounds.tolist(),
        settled_pose_status='NOT_RECORDED', environment_seed=42)


def smoke_manifest():
    """Freeze the first draw and all-command corner checks for this one smoke."""
    from .stage1_demo import approved_scaled_profile
    profile = approved_scaled_profile()
    if profile.get('owner_review') != 'APPROVED' or profile.get('profile_id') != 'scaled-table-path-b-attempt005':
        raise ValueError('translated_smoke_scene_owner_approval')
    frozen_rule()
    draw = placement_draw()
    candidate = point_record(draw['candidate_id'], authored_xy=draw['authored_xy_m'])
    candidate['placement'] = draw
    attempts = []
    for number in range(1, 5):
        record = deepcopy(candidate)
        record.update(attempt_id=f'native{number:03d}', physical_attempt_number=number,
            output_id=f'random_xy_recorder_smoke_native{number:03d}',
            output_directory=SMOKE_OUTPUT.relative_to(ROOT).as_posix()+f'/native{number:03d}')
        attempts.append(record)
    corners = [point_record(f'R{i+1:06d}', authored_xy=xy) for i, xy in enumerate(
        [(x, y) for x in SMOKE_XY_BOUNDS[0] for y in SMOKE_XY_BOUNDS[1]])]
    corner_audit = command_audit(dict(attempts=corners))
    for result, corner in zip(corner_audit['points'], corners):
        result['authored_world_xyz_m'] = corner['authored_world_xyz_m']
        result['setup_actions_checked'] = [0, 49]
        result['precast_and_stored_validated'] = True
    return dict(schema='translated-script-random-xy-smoke/v1',
        authority='CHICGRASP_RANDOM_XY_RECORDER_SMOKE', run_type='EVIDENCE',
        source=dict(action_file=str(ACTION_FILE), action_identity=identity(ACTION_FILE)),
        basis=dict(path=BASIS.relative_to(ROOT).as_posix(), **identity(BASIS)),
        scene_profile=dict(path=SMOKE_PROFILE.relative_to(ROOT).as_posix(), **identity(SMOKE_PROFILE)),
        scene_clause=profile['scene_clause'],
        scene_identity_requirement='Verify this identity at every launch. Any recording or evaluation on a different scene profile is ineligible. All pre-change recordings, including the seven grid demos and the ORIGINAL probe, are reference only.',
        placement=draw, four_corner_audit=corner_audit,
        witness=dict(source='side-opening-progress/v1', rule_sha256=RULE_SHA256),
        schedule=dict(actions=200, setup=50, scripted_after_setup=150, learned=0,
            control_ticks=1200, substeps=2400, admission=[839, 1], confirmation=[840, 0],
            low_level_target='same_float32_action_repeated_six_ticks'),
        absolute_deadline_ms=1200000, combined_containment_drain_ms=5000,
        invocation_limit=4, scripted_episode_limit=1, eligible_pre_action_repair_limit=3,
        automatic_retry=False, attempts=attempts)


def collection_manifest():
    """The next 100 draws, before outcomes; the smoke draw is never replayed."""
    from .stage1_demo import approved_scaled_profile
    profile = approved_scaled_profile()
    smoke = load(SMOKE_MANIFEST)
    if smoke['placement'] != placement_draw() or smoke['scene_profile'] != dict(
            path=SMOKE_PROFILE.relative_to(ROOT).as_posix(), **identity(SMOKE_PROFILE)):
        raise ValueError('translated_collection_smoke_stream_or_scene_changed')
    frozen_rule()
    candidates = []
    for order, point in enumerate(COLLECTION_POINTS):
        draw = placement_draw(order+1)
        if draw['candidate_id'] != point:
            raise ValueError('translated_collection_draw_order')
        record = point_record(point, authored_xy=draw['authored_xy_m'])
        record.update(placement=draw, output_id='random_xy_collection_'+point,
            output_directory=COLLECTION_OUTPUT.relative_to(ROOT).as_posix()+'/'+point,
            collection=dict(schema='random-xy-collection-candidate/v1', candidate_order=order,
                preceding_smoke_candidate='R000001'))
        candidates.append(record)
    release = COLLECTION_DOC/'OWNER_RELEASE.txt'
    return dict(schema='translated-script-random-xy-collection/v1',
        authority='CHICGRASP_RANDOM_XY_50_COLLECTION', run_type='EVIDENCE',
        owner_release=dict(path=release.relative_to(ROOT).as_posix(), **identity(release)),
        source=dict(action_file=str(ACTION_FILE), action_identity=identity(ACTION_FILE)),
        basis=dict(path=BASIS.relative_to(ROOT).as_posix(), **identity(BASIS)),
        source_smoke_manifest=dict(path=SMOKE_MANIFEST.relative_to(ROOT).as_posix(), **identity(SMOKE_MANIFEST)),
        scene_profile=dict(path=SMOKE_PROFILE.relative_to(ROOT).as_posix(), **identity(SMOKE_PROFILE)),
        scene_clause=profile['scene_clause'], scene_identity_requirement=smoke['scene_identity_requirement'],
        placement_stream=dict(bit_generator='PCG64', seed=PLACEMENT_SEED,
            first_draw_index=1, last_draw_index=100, smoke_draw_index=0,
            bounds_xy_m=[list(x) for x in SMOKE_XY_BOUNDS], environment_seed=42),
        witness=deepcopy(smoke['witness']), schedule=deepcopy(smoke['schedule']),
        absolute_deadline_ms=1200000, combined_containment_drain_ms=5000,
        invocation_limit=103, consumed_recording_limit=100, target_usable_records=50,
        shared_pre_action_repair_limit=3, automatic_retry=False,
        selection='FIRST_50_USABLE_IN_CANDIDATE_ORDER', smoke_counts_toward_selected=False,
        native_attempt_names=list(COLLECTION_NATIVE_NAMES), candidates=candidates)


def collection_output_record(candidate, native_name):
    """Bind one frozen candidate to a permitted native child destination."""
    if candidate['point'] not in COLLECTION_POINTS or native_name not in COLLECTION_NATIVE_NAMES:
        raise ValueError('translated_collection_destination')
    expected = COLLECTION_OUTPUT.relative_to(ROOT).as_posix()+'/'+candidate['point']
    if candidate['output_directory'] != expected:
        raise ValueError('translated_collection_candidate_root')
    record = deepcopy(candidate)
    record.update(attempt_id=candidate['point']+'_'+native_name,
        physical_attempt_number=int(native_name[-3:]), candidate_sha256=digest(candidate),
        output_id='random_xy_collection_'+candidate['point']+'_'+native_name,
        output_directory=expected+'/'+native_name)
    return record


def build_manifest():
    c = approved_freeze()
    frozen_rule()
    source_result = load(SOURCE/'attempt_result.json')
    if any(source_result.get(k) != 'PASS' for k in ('CAPTURE_INTEGRITY','END_STATE_RESULT','PROCESS_COMPLETION','DEMO_INTEGRITY')):
        raise ValueError('accepted_Stage1_source_required')
    return dict(schema='translated-script-sweep/v1', authority='OFFLINE_BUILD_ONLY_NO_LAUNCH',
        source=dict(action_file=str(ACTION_FILE), action_identity=identity(ACTION_FILE),
            executed_commit='7074ebfe3af2637b5e586b73d9b3d731a997aa65',
            executed_tag='gate6-stage1-executable-r2-20261012',
            accepted_result=dict(path=str(SOURCE/'attempt_result.json'), **identity(SOURCE/'attempt_result.json'))),
        freeze=dict(path=FREEZE.relative_to(ROOT).as_posix(), **identity(FREEZE)),
        basis=dict(path=BASIS.relative_to(ROOT).as_posix(), **identity(BASIS)),
        visual_identities={**c['visual_identity_files'], **c['configuration_hashes']},
        witness=dict(source='side-opening-progress/v1', rule_sha256=RULE_SHA256),
        schedule=dict(actions=200,setup=50,scripted_after_setup=150,learned=0,control_ticks=1200,substeps=2400,
            admission=[839,1],confirmation=[840,0],low_level_target='same_float32_action_repeated_six_ticks'),
        absolute_deadline_ms=1200000,combined_containment_drain_ms=5000,
        attempts=[point_record(p,c) for p in POINTS],
        failed_points_remain_in_evaluation=True, automatic_retry=False, expert_starts_parked=4,
        learned_evaluation='Same translated 50-action setup; after handover absolute predictions only, no point offset or fallback.')


def execution_manifest():
    """Link the frozen nine-point build without rewriting any grid record."""
    original_build = load(BUILD_MANIFEST)
    if original_build != build_manifest():
        raise ValueError('translated_build_manifest_changed')
    value = deepcopy(original_build)
    value.update(schema='translated-script-execution/v2',
        authority='OWNER_FINITE_SWEEP_GRANT_REQUIRED',
        build_manifest=dict(path=BUILD_MANIFEST.relative_to(ROOT).as_posix(), **identity(BUILD_MANIFEST)),
        invocation_limit=11, scripted_episode_limit=10, shared_startup_retry_limit=1,
        original_must_qualify_first=True)
    value['attempts'] = [point_record('ORIGINAL'), *value['attempts']]
    return value


def verified_point(point, manifest_path=MANIFEST):
    manifest = load(manifest_path)
    expected = execution_manifest() if manifest.get('schema') == 'translated-script-execution/v2' else build_manifest()
    if manifest != expected:
        raise ValueError('translated_manifest_binding_changed')
    found = next((p for p in manifest['attempts'] if p['point']==point), None)
    if found is None:
        raise ValueError('translated_permanent_point_required')
    return dict(deepcopy(found), manifest_sha256=identity(manifest_path)['sha256'],
                point_sha256=digest(found))


def continuation_manifest():
    """New destinations only. Keep the frozen commands and historical manifest."""
    previous = load(MANIFEST)
    if previous != execution_manifest():
        raise ValueError('translated_previous_manifest_changed')
    value = deepcopy(previous)
    value.update(schema='translated-script-continuation/v1',
        purpose='DEVELOPMENT_TRAINING_RECORDINGS',
        previous_manifest=dict(path=MANIFEST.relative_to(ROOT).as_posix(), **identity(MANIFEST)),
        invocation_limit=18, cumulative_invocation_limit=20, historical_starts=2,
        scripted_episode_limit=17, shared_startup_retry_limit=1,
        original_must_qualify_first=False, original_already_qualified_no_rerun=True,
        packaging_rerecord_limit_per_point=1)
    value['attempts'] = []
    for point in POINTS:
        for number in ([2] if point=='G00' else [1,2]):
            record = deepcopy(next(p for p in previous['attempts'] if p['point']==point))
            attempt_id=f'{point}_R{number}'
            record.update(attempt_id=attempt_id, physical_attempt_number=number,
                retry_reason='PACKAGING_RERECORD_ONLY_IF_QUALIFIED' if number==2 else None,
                output_id='script_sweep_002_'+attempt_id,
                output_directory='outputs/translated_script_sweep_002/'+attempt_id)
            value['attempts'].append(record)
    return value


def verified_output_point(point, output):
    """Resolve the committed attempt by destination, with no new native option."""
    attempt = Path(output).resolve().parent.parent
    if point in ANCHOR_POINTS or attempt.is_relative_to(ANCHOR_OUTPUT.resolve()):
        manifest=load(ANCHOR_MANIFEST)
        if manifest != anchor_manifest():
            raise ValueError('anchor_manifest_changed')
        found=next((p for p in manifest['attempts'] if p['point']==point and
                    (ROOT/p['output_directory']).resolve()==attempt),None)
        if found is None or Path(output).resolve()!=attempt/'capture/experiment':
            raise ValueError('anchor_destination')
        return dict(deepcopy(found), manifest_sha256=identity(ANCHOR_MANIFEST)['sha256'], point_sha256=digest(found))
    if point in YAW_POINTS or attempt.is_relative_to(YAW_OUTPUT.resolve()):
        manifest=load(YAW_MANIFEST)
        if manifest != yaw_manifest():
            raise ValueError('yaw_manifest_changed')
        found=next((p for p in manifest['attempts'] if p['point']==point and
                    (ROOT/p['output_directory']).resolve()==attempt),None)
        if found is None or Path(output).resolve()!=attempt/'capture/experiment':
            raise ValueError('yaw_destination')
        return dict(deepcopy(found), manifest_sha256=identity(YAW_MANIFEST)['sha256'], point_sha256=digest(found))
    if attempt.is_relative_to(COLLECTION_OUTPUT.resolve()):
        relative = attempt.relative_to(COLLECTION_OUTPUT.resolve())
        if (len(relative.parts) != 2 or relative.parts[0] != point or
                relative.parts[1] not in COLLECTION_NATIVE_NAMES or
                Path(output).resolve() != attempt/'capture/experiment'):
            raise ValueError('translated_collection_destination')
        manifest = load(COLLECTION_MANIFEST)
        if manifest != collection_manifest():
            raise ValueError('translated_collection_manifest_changed')
        candidate = next((p for p in manifest['candidates'] if p['point'] == point), None)
        if candidate is None:
            raise ValueError('translated_collection_candidate')
        found = collection_output_record(candidate, relative.parts[1])
        return dict(found, manifest_sha256=identity(COLLECTION_MANIFEST)['sha256'],
                    point_sha256=digest(found))
    if attempt.is_relative_to(SMOKE_OUTPUT.resolve()):
        manifest = load(SMOKE_MANIFEST)
        if manifest != smoke_manifest():
            raise ValueError('translated_smoke_manifest_changed')
        found = next((p for p in manifest['attempts'] if p['point'] == point and
                      (ROOT/p['output_directory']).resolve() == attempt), None)
        if found is None:
            raise ValueError('translated_smoke_destination')
        return dict(deepcopy(found), manifest_sha256=identity(SMOKE_MANIFEST)['sha256'],
                    point_sha256=digest(found))
    if not attempt.is_relative_to(CONTINUATION_OUTPUT.resolve()):
        return verified_point(point)
    manifest = load(CONTINUATION)
    if manifest != continuation_manifest():
        raise ValueError('translated_continuation_manifest_changed')
    found = next((p for p in manifest['attempts'] if p['point']==point and
                  (ROOT/p['output_directory']).resolve()==attempt), None)
    if found is None:
        raise ValueError('translated_continuation_destination')
    return dict(deepcopy(found), manifest_sha256=identity(CONTINUATION)['sha256'],
                point_sha256=digest(found))


def command_audit(manifest):
    result=[]
    limits=np.array([[.64,.70],[-.17,.19],[.10,.57]])
    for point in manifest['attempts']:
        actions=np.asarray(point['actions'],np.float32)
        # CandidateCapture dispatches six identical targets, RuntimeSession casts
        # to float32 again. There is no intervening target interpolation.
        dispatched=np.repeat(actions,6,axis=0).astype(np.float32)
        for a in dispatched:
            validate_policy_action(a)
        margins=np.minimum(dispatched[:,:3].astype(float)-limits[:,0],limits[:,1]-dispatched[:,:3].astype(float))
        where=np.unravel_index(np.argmin(margins),margins.shape)
        result.append(dict(point=point['point'],actions=200,dispatches=1200,substeps=2400,status='PASS',
            min_margins_xyz_m=margins.min(axis=0).tolist(),limiting_control_tick=int(where[0]),
            limiting_axis='XYZ'[where[1]],limiting_margin_m=float(margins[where]),
            sequence_sha256=point['sequence_sha256'], violations=[]))
    return dict(points=result,actions=200*len(result),dispatches=1200*len(result),substeps=2400*len(result),status='PASS')


class ScriptWriter(PolicyWriter):
    prefix='script'
    command_schema='script-command-lifecycle/v1'

    def observation_indices(self, index):
        return dict(learned_action=None, setup_action=index if index<50 else None,
                    script_action=index, training_action=index-50 if index>=50 else None)


class ScriptController:
    def __init__(self, point, writer):
        self.point, self.writer, self.next_index = point, writer, 0
        self.actions=np.asarray(point['actions'],np.float32)

    def indices(self,index):
        return dict(global_action=index,**self.writer.observation_indices(index))

    def select(self,index,observation,sensor):
        if index != self.next_index or not 0<=index<200:
            raise ValueError('translated_script_sequence')
        action=self.actions[index].copy();validate_policy_action(action)
        record=dict(self.indices(index),action=action.tolist(),native_clock=deepcopy(sensor['native_clock']),
            control_tick_start=6*index,source='translated_accepted_Stage1_script',
            source_action=index,source_action_sha256=ACTION_SHA256,
            point=self.point['point'],point_sha256=self.point['point_sha256'],
            manifest_sha256=self.point['manifest_sha256'])
        self.writer.action(record)
        self.writer.command_event('SELECTED_VALIDATED',index,decision=record,
            observation_references=[f'rgb/{index:06d}.json'],scheduled_boundary=[6*index,0])
        self.next_index+=1
        return action

    def summary(self,completed_actions):
        return dict(selected=self.next_index,completed=completed_actions,
            setup_completed=min(completed_actions,50),scripted_after_setup_completed=max(0,completed_actions-50),
            learned_actions=0,actual_inferences=0)


def configure_start(cfg, session):
    """Task-local initial pose before construction/reset; owner yaw route only."""
    point=session.translated_point
    before=list(cfg.scene.chicken.init_state.pos)
    expected_rotation=[1,0,0,0] if point.get('yaw_probe') else point['authored_quaternion_wxyz']
    if before != point['original_authored_world_xyz_m'] or list(cfg.scene.chicken.init_state.rot)!=expected_rotation:
        raise ValueError('translated_original_authored_pose_changed')
    cfg.scene.chicken.init_state.pos=tuple(point['authored_world_xyz_m'])
    if point.get('yaw_probe'):
        cfg.scene.chicken.init_state.rot=tuple(point['authored_quaternion_wxyz'])
    session.translated_start_configuration=dict(before_xyz_m=before,
        after_xyz_m=list(cfg.scene.chicken.init_state.pos),rotation_wxyz=list(cfg.scene.chicken.init_state.rot),
        changed_fields=['chicken.init_state.pos.x','chicken.init_state.pos.y']+(['chicken.init_state.rot'] if point.get('yaw_probe') else []),
        robot_reset_configuration_changed=False)


def save_preparation(runtime,output,phase):
    from .stage1_demo import save_visuals, approved_lights_match
    smoke = runtime.translated_point.get('placement', {}).get('schema') == 'random-xy-placement/v1'
    if smoke:
        from .stage1_demo import approved_scaled_profile
        profile = approved_scaled_profile()
        save_visuals(runtime, output, phase, scene_profile=True)
        if runtime.effective_configuration != profile['physical_configuration']:
            raise ValueError('translated_frozen_visual_changed:effective_configuration.json')
    else:
        c=approved_freeze()
        visual=save_visuals(runtime,output,phase)
        for key,observed in [('visual_readback.json',visual['visuals']),('renderer_readback.json',visual['renderer']),
                             ('effective_configuration.json',runtime.effective_configuration)]:
            if observed != load(ROOT/'outputs/generalization_domain_previews_task1c_r2'/key):
                raise ValueError('translated_frozen_visual_changed:'+key)
        if not (approved_lights_match(visual['lights']['lights'], visual['approval'])
                and approved_lights_match(c['composed_lights']['lights'], visual['approval'])):
            raise ValueError('translated_frozen_lights_changed')
    if phase=='before_candidate':
        write(Path(output)/'script_preparation.json',dict(configuration=runtime.translated_start_configuration,
            origin_clock=list(runtime.origin_clock),measured=runtime.measure(),
            initialization=deepcopy(runtime.initialization),robot_configuration='unchanged production reset',
            initialization_physics='existing constructor and reset advancement; not claimed zero',
            authored_point=runtime.translated_point,visual_freeze=identity(SMOKE_PROFILE if smoke else FREEZE),
            **({'measured_boundary':'constructor/reset before scripted setup; not a settled pose'} if smoke else {})))


def check_storage(output):
    continuation = Path(output).resolve().is_relative_to(CONTINUATION_OUTPUT.resolve())
    smoke = Path(output).resolve().is_relative_to(SMOKE_OUTPUT.resolve())
    collection = Path(output).resolve().is_relative_to(COLLECTION_OUTPUT.resolve())
    yaw = Path(output).resolve().is_relative_to(YAW_OUTPUT.resolve())
    anchor = Path(output).resolve().is_relative_to(ANCHOR_OUTPUT.resolve())
    value=load((ANCHOR_DOC if anchor else YAW_DOC if yaw else COLLECTION_DOC if collection else SMOKE_DOC if smoke else CONTINUATION_DOC if continuation else DOC)/'STORAGE.json')
    if value['fixed_margin_bytes']!=1<<30 or value['required_bytes']!=value['estimated_payload_bytes']+(1<<30):
        raise ValueError('translated_storage_contract')
    free=shutil.disk_usage(output).free
    return dict(estimate=value,free_bytes=free,passed=free>=value['required_bytes'])
