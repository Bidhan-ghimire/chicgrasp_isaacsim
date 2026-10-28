"""Offline regression tests grouped by scene."""
from __future__ import annotations



# Scene setup

"""Editable scene plans and identities; ordinary Python, no simulator."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pipeline import scene as scenes
from pipeline import settings

ROOT = Path(__file__).resolve().parents[1]


class SceneSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = (ROOT/'scene_setup.py').read_text(encoding='utf-8')
        (self.root/'scene_setup.py').write_text(self.source, encoding='utf-8')
        for path in scenes.BRIDGE_FILES:
            destination = self.root/path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text('# fixture dependency\n', encoding='utf-8')
        self.patch_root = patch.object(scenes, 'ROOT', self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)
        self.cfg = {'scene_sha256':scenes.BASE_SCENE_SHA256, 'scene_setup':'scene_setup.py', 'scene_options':{}}
        self.position = {'id':'P1','authored_xyz_m':[.65,0,.8155], 'authored_quaternion_wxyz':[1,0,0,0]}

    def custom(self):
        (self.root/'scene_setup.py').write_text(self.source.replace('USE_ORIGINAL_TABLE_TASK = True', 'USE_ORIGINAL_TABLE_TASK = False'), encoding='utf-8')

    def test_table_default_is_noop_and_keeps_original_evaluation(self):
        module, identity = scenes.load_scene_setup(self.cfg)
        cfg = SimpleNamespace(unchanged=[1,2,3])
        before = copy.deepcopy(vars(cfg))
        module.configure_scene(cfg,self.position,{})
        self.assertEqual(vars(cfg),before)
        self.assertEqual(module.build_scene(object(),self.position,{})['created_prims'],[])
        for status in ('PASS','FAIL'):
            result = module.evaluate_episode(None,{'status':status,'rise':.1},{})
            self.assertEqual(result['status'],status)
            self.assertEqual(result['rise'],.1)
            self.assertEqual(result['training_eligible'],status=='PASS')
        self.assertEqual(identity['definition']['kind'],'table')

    def test_table_rejects_changed_height_or_orientation(self):
        scenes.validate_scene_position(self.cfg,self.position)
        for field, value in [('authored_xyz_m',[.65,0,1.15]), ('authored_quaternion_wxyz',[0,1,0,0])]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                scenes.validate_scene_position(self.cfg,dict(self.position,**{field:value}))

    def test_scene_name_is_identity_label_not_task_selection(self):
        for original in (True, False):
            identities = []
            for name in ('table', 'sleeping', 'eating'):
                with self.subTest(original=original, name=name):
                    source = self.source.replace('SCENE_NAME = "table"', 'SCENE_NAME = '+repr(name))
                    if not original:
                        source = source.replace('USE_ORIGINAL_TABLE_TASK = True', 'USE_ORIGINAL_TABLE_TASK = False')
                    (self.root/'scene_setup.py').write_text(source, encoding='utf-8')
                    module, identity = scenes.load_scene_setup(self.cfg)
                    identities.append(identity)
                    self.assertEqual(identity['definition'], {'id':name, 'kind':'table' if original else 'custom'})
                    cfg = SimpleNamespace(physics={'unchanged':True})
                    before = copy.deepcopy(vars(cfg))
                    module.configure_scene(cfg,self.position,{})
                    self.assertEqual(vars(cfg),before)
                    # An object without any stage API proves the default authors nothing.
                    self.assertEqual(module.build_scene(object(),self.position,{})['created_prims'],[])
                    capture = SimpleNamespace(completed=True,lifecycle_events=[],recording_errors=[])
                    result = module.evaluate_episode(capture,{'status':'PASS'}, {})
                    self.assertEqual(result['status'], 'PASS' if original else 'UNSCORED')
                    elevated = dict(self.position, authored_xyz_m=[.65,0,1.15])
                    if original:
                        with self.assertRaises(ValueError): module.validate_position(elevated,{})
                    else:
                        module.validate_position(elevated,{})
            self.assertEqual(len({value['source']['sha256'] for value in identities}),3)

    def test_scene_name_and_original_task_switch_are_explicit(self):
        for old, value in [('SCENE_NAME = "table"', 'SCENE_NAME = " "'),
                           ('SCENE_NAME = "table"', 'SCENE_NAME = 1'),
                           ('USE_ORIGINAL_TABLE_TASK = True', 'USE_ORIGINAL_TABLE_TASK = "False"'),
                           ('USE_ORIGINAL_TABLE_TASK = True', 'USE_ORIGINAL_TABLE_TASK = 0')]:
            with self.subTest(value=value):
                (self.root/'scene_setup.py').write_text(self.source.replace(old,value),encoding='utf-8')
                with self.assertRaises(ValueError): scenes.load_scene_setup(self.cfg)

    def test_custom_allows_height_and_unit_rotation_without_creating_physics(self):
        self.custom()
        scenes.validate_scene_position(self.cfg,dict(self.position,authored_xyz_m=[.65,0,1.15],authored_quaternion_wxyz=[0,1,0,0]))
        module, identity = scenes.load_scene_setup(self.cfg)
        self.assertEqual(identity['definition']['kind'],'custom')
        self.assertEqual(module.build_scene(object(), self.position, {}), {'created_prims':[], 'scene':'table'})

    def test_invalid_quaternions_positions_and_booleans_are_rejected(self):
        self.custom()
        for xyz, quat in [([.65,0,True],[1,0,0,0]),([float('nan'),0,1],[1,0,0,0]),([.65,0,1],[2,0,0,0]),([.65,0,1],[0,0,0,0])]:
            with self.subTest(xyz=xyz,quat=quat), self.assertRaises(ValueError):
                scenes.validate_scene_position(self.cfg,dict(self.position,authored_xyz_m=xyz,authored_quaternion_wxyz=quat))

    def test_custom_completion_is_never_reported_as_task_success(self):
        self.custom()
        module,_ = scenes.load_scene_setup(self.cfg)
        for complete, lifecycle, errors in [(True,[],[]),(False,[],[]),(True,['reset'],[]),(True,[],['nonfinite'])]:
            capture = SimpleNamespace(completed=complete,lifecycle_events=lifecycle,recording_errors=errors)
            result = module.evaluate_episode(capture,{'status':'PASS'},{})
            self.assertEqual(result['status'],'UNSCORED')
            self.assertFalse(result['temporary_gripper_attachment'])
            self.assertEqual(result['training_eligible'],complete and not lifecycle and not errors)

    def test_source_change_is_rejected_before_changed_code_executes(self):
        identity = scenes.scene_setup_identity(self.cfg)
        (self.root/'scene_setup.py').write_text("raise AssertionError('modified code executed')\n",encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'source changed'):
            scenes.authenticate_scene_setup(identity)

    def test_option_changes_have_different_identity(self):
        self.custom()
        first = scenes.scene_setup_identity(self.cfg)
        second = scenes.scene_setup_identity(dict(self.cfg,scene_options={'custom_note':'test-only option'}))
        self.assertNotEqual(first,second)
        self.assertEqual(first['source'],second['source'])
        self.assertEqual(scenes.authenticate_scene_setup(second)[1],second)

    def test_declared_asset_and_bridge_changes_invalidate_identity(self):
        asset = self.root/'extra.usda'
        asset.write_text('original',encoding='utf-8')
        (self.root/'scene_setup.py').write_text(self.source.replace('ASSET_FILES = []','ASSET_FILES = ["extra.usda"]'),encoding='utf-8')
        identity = scenes.scene_setup_identity(self.cfg)
        asset.write_text('changed',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'dependency changed'):
            scenes.authenticate_scene_setup(identity)
        asset.write_text('original',encoding='utf-8')
        (self.root/scenes.BRIDGE_FILES[0]).write_text('# changed',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'dependency changed'):
            scenes.authenticate_scene_setup(identity)

    def test_asset_paths_cannot_escape_repository(self):
        for value in ('../outside.usd',str(self.root/'absolute.usd')):
            with self.subTest(value=value):
                (self.root/'scene_setup.py').write_text(self.source.replace('ASSET_FILES = []','ASSET_FILES = '+repr([value])),encoding='utf-8')
                with self.assertRaises(ValueError): scenes.scene_setup_identity(self.cfg)

    def test_options_must_be_finite_objects(self):
        self.custom()
        for options in ([], {'custom_value':float('nan')}, {'custom_value':float('inf')}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                scenes.scene_setup_identity(dict(self.cfg,scene_options=options))

    def test_pose_only_height_changes_are_distinct_and_split_distance_is_3d(self):
        self.custom()
        rows=[dict(self.position,split='train'),dict(self.position,id='P2',split='test',authored_xyz_m=[.65,0,1.15])]
        (self.root/'positions.json').write_text(json.dumps({'schema':'research-positions/v1','frame':'WORLD','positions':rows}),encoding='utf-8')
        with patch.object(settings,'ROOT',self.root):
            result = settings.load_positions(dict(self.cfg,positions=['positions.json']))
            self.assertEqual(len(result),2)
            rows[1]['authored_xyz_m']=[.65,0,.82]
            (self.root/'positions.json').write_text(json.dumps({'schema':'research-positions/v1','frame':'WORLD','positions':rows}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'cross-split'):
                settings.load_positions(dict(self.cfg,positions=['positions.json']))

    def test_equivalent_quaternion_signs_are_duplicate_poses(self):
        self.custom()
        rows=[dict(self.position,split='train'),dict(self.position,id='P2',split='train',authored_quaternion_wxyz=[-1,0,0,0])]
        (self.root/'positions.json').write_text(json.dumps({'schema':'research-positions/v1','frame':'WORLD','positions':rows}),encoding='utf-8')
        with patch.object(settings,'ROOT',self.root), self.assertRaisesRegex(ValueError,'Duplicate'):
            settings.load_positions(dict(self.cfg,positions=['positions.json']))

    def test_scene_review_can_identify_missing_script_without_executing_it(self):
        cfg = dict(self.cfg,scene_profile='attempt005',environment_seed=42,script='not-written.py',positions=[])
        for relative in ('camera_presets/HAND_DEMO_ONLY.json','camera_presets/HD_TWINS_DEMO_ONLY.json','camera_presets/apply_cameras.py'):
            path = self.root/relative
            path.parent.mkdir(exist_ok=True)
            path.write_text('{}',encoding='utf-8')
        with patch.object(settings,'ROOT',self.root):
            identity = settings.scene_identity(cfg)
            self.assertIsNone(identity['inputs']['not-written.py'])
            (self.root/'not-written.py').write_text("raise AssertionError('do not run demo while previewing')",encoding='utf-8')
            self.assertNotEqual(identity,settings.scene_identity(cfg))

    def test_module_import_never_loads_simulator_or_models(self):
        code = "import sys,scene_setup; from pipeline import scene as scenes; assert not ({'torch','omni','isaaclab','pxr'} & set(sys.modules))"
        result = subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)


# Scene provenance

"""Saved scene identity checks with temporary metadata and mocked process boundaries."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from pipeline import settings
from pipeline import training
from pipeline import training as data
from pipeline.runtime import workflow


def scene():
    return {'schema':'research-scene-setup/v1', 'base_scene_sha256':settings.SCENE_SHA256,
            'source':{'path':'scene_setup.py','bytes':123,'sha256':'c'*64},
            'options':{}, 'definition':{'id':'fixture','kind':'table'}, 'assets':[], 'bridge_files':[]}


def row(name='a', split='train', identity=None):
    identity = copy.deepcopy(identity or scene())
    return {'id':name,'path':name+'/episode','split':split,'position_id':name,'complete':True,
            'script_sha256':'a'*64,'scene_sha256':identity['base_scene_sha256'],
            'scene_setup_identity':identity,
            'episode_binding':{'path':name+'/episode_binding.json','bytes':1,'sha256':'d'*64}}


def save_dataset(root, rows):
    path = root/'dataset.json'
    path.write_text(json.dumps({'schema':'research-dataset/v1','episodes':rows}), encoding='utf-8')
    return path


def save_episode(root, identity=None):
    record = row(identity=identity)
    episode = root/record['path']
    episode.mkdir(parents=True)
    (episode/'manifest.json').write_text('{"fixture":true}', encoding='utf-8')
    binding = root/record['episode_binding']['path']
    binding.write_text(json.dumps({'schema':'research-episode-binding/v1',
        'scene_setup_identity':record['scene_setup_identity'],
        'episode_manifest':data.identity(episode/'manifest.json')}), encoding='utf-8')
    record['episode_binding'].update(data.identity(binding))
    return record, episode, binding


class SceneProvenanceTests(unittest.TestCase):
    def test_saved_identity_is_validated_without_source_execution(self):
        value = scene()
        with patch.object(Path, 'read_text', side_effect=AssertionError('No current source read')):
            self.assertEqual(data.validate_scene_identity(value), value)
        for changed in (None, {}, dict(value, schema='legacy'), dict(value, options={'bad':float('nan')})):
            with self.subTest(changed=changed), self.assertRaises(ValueError): data.validate_scene_identity(changed)

    def test_train_validation_must_match_every_scene_component(self):
        for key, replacement in (
            ('source', {'path':'scene_setup.py','bytes':123,'sha256':'e'*64}),
            ('options', {'height_m':1.2}), ('definition', {'id':'hanging','kind':'custom'}),
            ('assets', [{'path':'assets/custom.usda','bytes':2,'sha256':'a'*64}]),
            ('bridge_files', [{'path':'pipeline/scene.py','bytes':2,'sha256':'f'*64}])):
            rows = [row(), row('b','validation')]
            rows[1]['scene_setup_identity'][key] = replacement
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'mixed'):
                data.consistent_scene_identity(rows)

    def test_missing_binding_is_not_filled_from_current_scene(self):
        saved = row()
        saved.pop('scene_setup_identity')
        with self.assertRaisesRegex(ValueError,'Missing or invalid'): data.consistent_scene_identity([saved])
        with tempfile.TemporaryDirectory() as folder:
            path = save_dataset(Path(folder), [saved])
            with patch.object(data,'frozen_module',side_effect=AssertionError('must stop before episode loader')):
                with self.assertRaises(ValueError): data.Dataset(path)

    def test_test_and_incomplete_rows_do_not_contribute_scene_consensus(self):
        test = row('test','test'); test.pop('scene_setup_identity')
        incomplete = row('incomplete'); incomplete['complete'] = False; incomplete.pop('scene_setup_identity')
        self.assertEqual(data.consistent_scene_identity([row(), test, incomplete]), scene())

    def test_base_hash_must_agree_with_binding(self):
        saved = row(); saved['scene_sha256'] = '9'*64
        with self.assertRaisesRegex(ValueError,'base scene hash'): data.consistent_scene_identity([saved])

    def test_sibling_receipt_and_manifest_are_both_authenticated(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            saved, episode, binding = save_episode(root)
            checked = data.verify_episode_binding(episode,saved['episode_binding'],scene(),base_dir=root)
            self.assertEqual(checked['episode_manifest'],data.identity(episode/'manifest.json'))
            self.assertNotIn(binding, list(episode.iterdir()))
            (episode/'manifest.json').write_text('{"different":true}')
            with self.assertRaisesRegex(ValueError,'manifest mismatch'):
                data.verify_episode_binding(episode,saved['episode_binding'],scene(),base_dir=root)

    def test_swapped_receipt_fails_even_with_refreshed_byte_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); saved,episode,binding=save_episode(root)
            value=json.loads(binding.read_text()); value['scene_setup_identity']['options']={'height_m':2}
            binding.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'byte identity'):
                data.verify_episode_binding(episode,saved['episode_binding'],scene(),base_dir=root)
            saved['episode_binding'].update(data.identity(binding))
            with self.assertRaisesRegex(ValueError,'scene mismatch'):
                data.verify_episode_binding(episode,saved['episode_binding'],scene(),base_dir=root)

    def test_binding_cannot_be_placed_inside_episode_or_outside_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); saved,episode,_=save_episode(root)
            for path in ('a/episode/episode_binding.json','../episode_binding.json'):
                changed=dict(saved['episode_binding'],path=path)
                with self.subTest(path=path), self.assertRaisesRegex(ValueError,'contained sibling'):
                    data.verify_episode_binding(episode,changed,scene(),base_dir=root)

    def test_scene_plan_and_native_request_do_not_compile_a_demonstration(self):
        cfg=settings.load_experiment()
        with patch.object(workflow,'compiled_script',side_effect=AssertionError('No demonstration for scene preview')):
            plan=workflow.make_plan(cfg,'scene')
            self.assertEqual(plan['action_count'],0)
            self.assertIsNone(plan['script'])
            with tempfile.TemporaryDirectory() as folder:
                def fake_runner(command,**kwargs):
                    request=settings.read(command[command.index('--request')+1])
                    self.assertNotIn('script',request)
                    self.assertNotIn('script_source_sha256',request)
                    attempt=Path(command[command.index('--attempt')+1]); attempt.mkdir()
                    settings.write(attempt/'research_result.json',{'completion':'COMPLETED',
                        'scene_setup_identity':request['scene_setup_identity']})
                    return {'success':True}
                host={'isaac_sim':'C:/unused','isaac_lab':'C:/unused','policy_support':'C:/unused','config_path':'C:/unused.json'}
                output=Path(folder)/'scene'
                result=workflow.run_native(cfg,host,'scene',plan['positions'][0],output,runner=fake_runner)
                self.assertEqual(result['completion'],'COMPLETED')
                self.assertFalse((output/'commands.npy').exists())

    def test_technical_success_with_wrong_scene_is_not_admitted(self):
        cfg=settings.load_experiment()
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'collection'
            def fake_run(cfg,host,mode,position,attempt,**kwargs):
                Path(attempt).mkdir()
                return {'completion':'COMPLETED','native':{'completion':'COMPLETED','training_eligible':True,
                    'scene_setup_identity':scene()}, 'result_path':str(Path(attempt)/'native/research_result.json')}
            with patch.object(workflow,'load_host',return_value={}),patch.object(workflow,'preflight',return_value={'status':'PASS'}), \
                 patch.object(workflow,'require_approval',return_value={}),patch.object(workflow,'run_native',side_effect=fake_run):
                result=workflow.execute_simulation(cfg,'record',output,limit=1)
            self.assertEqual(result['completion'],'FAILED')
            self.assertEqual(result['training_episodes'],0)

    def test_training_rejects_wrong_approved_scene_before_worker_launch(self):
        cfg=settings.load_experiment()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); dataset=save_dataset(root,[row()]); output=root/'training'
            with patch.object(workflow,'load_host',return_value={'training_python':sys.executable}), \
                 patch.object(workflow,'require_approval',return_value={}),patch('pipeline.runtime.process.run_owned') as launch:
                with self.assertRaisesRegex(ValueError,'approved experiment scene'):
                    workflow.execute_training(cfg,output,dataset=str(dataset))
                launch.assert_not_called()
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError,'approved experiment scene'):
                training.training_plan(dataset,expected_scene_setup_identity=workflow.scene_setup_identity(cfg))

    def test_collection_keeps_failed_provenance_report_and_admits_nothing(self):
        cfg=settings.load_experiment()
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'collection'
            def fake_run(cfg,host,mode,position,attempt,**kwargs):
                episode=Path(attempt)/'native/episode'; episode.mkdir(parents=True)
                return {'completion':'COMPLETED','native':{'completion':'COMPLETED','training_eligible':True,
                    'scene_setup_identity':workflow.scene_setup_identity(cfg), 'episode':{'path':str(episode)}},
                    'result_path':str(episode.parent/'research_result.json')}
            with patch.object(workflow,'load_host',return_value={}),patch.object(workflow,'preflight',return_value={'status':'PASS'}), \
                 patch.object(workflow,'require_approval',return_value={}),patch.object(workflow,'run_native',side_effect=fake_run):
                result=workflow.execute_simulation(cfg,'record',output,limit=1)
            self.assertEqual(result['completion'],'FAILED')
            self.assertEqual(result['training_episodes'],0)
            record=settings.read(output/'R000059.json')
            self.assertIn('Episode provenance:',record['parent_errors'][0])
            self.assertTrue((output/'summary.json').is_file())

    def test_live_gate_rejects_other_editable_scene_before_checkpoint_read(self):
        cfg=settings.load_experiment()
        artifact={'schema':'research-model-artifact/v1','training_window_start':50,
                  'scene_sha256s':[settings.SCENE_SHA256], 'scene_setup_identity':scene(),
                  'identities':{'episodes':[row()]}}
        with patch.object(workflow,'read',return_value=artifact):
            with self.assertRaisesRegex(ValueError,'editable scene identity'):
                workflow.checkpoint_record(cfg)

    def test_training_worker_forwards_expected_scene(self):
        from pipeline import training as training_entry
        request={'dataset':'dataset.json','output':'model','scene_setup_identity':scene(),
                 'training':{'model_type':'diffusion','updates':1,'seed':42,'device':'cpu'}}
        with patch.object(training_entry,'read',return_value=request),patch.object(sys,'argv',['worker','request.json']), \
             patch.object(training,'run_training') as train:
            training_entry.worker_main()
            self.assertEqual(train.call_args.kwargs['expected_scene_setup_identity'],scene())

    def test_training_receipt_requires_matching_exported_scene(self):
        cfg=settings.load_experiment()
        approved=workflow.scene_setup_identity(cfg)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); dataset=save_dataset(root,[row(identity=approved)])
            def fake_worker(command,**kwargs):
                request=settings.read(command[-1]); output=Path(request['output']); output.mkdir()
                self.assertEqual(request['scene_setup_identity'],approved)
                settings.write(output/'artifact.json',{'scene_setup_identity':scene(),'identities':{'episodes':[row()]}})
                return {'success':True}
            host={'training_python':sys.executable,'isaac_sim':'C:/unused'}
            with patch.object(workflow,'load_host',return_value=host),patch.object(workflow,'require_approval',return_value={}):
                result=workflow.execute_training(cfg,root/'training',dataset=str(dataset),runner=fake_worker)
            self.assertEqual(result['completion'],'FAILED')
            self.assertIn('scene differs',result['errors'][0])


# Scene model identity

"""Saved model scene bindings: CPU metadata fixtures only, no actual model imports."""


import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from pipeline import models
from pipeline import models as runtime
from pipeline.training import CONTRACT
from pipeline.training import identity


def scene_fixture():
    return {"schema": "research-scene-setup/v1", "base_scene_sha256": "b" * 64,
            "source": {"path": "scene_setup.py", "bytes": 123, "sha256": "a" * 64},
            "options": {"support_height_m": 0.7}, "definition": {"id": "synthetic-table", "kind": "table"},
            "assets": [], "bridge_files": [{"path": 'pipeline/scene.py', "bytes": 456, "sha256": "c" * 64}]}


def episode_fixture(name="train_a", *, split="train", scene=None):
    scene = copy.deepcopy(scene if scene is not None else scene_fixture())
    return {"id": name, "split": split, "complete": True, "scene_sha256": scene["base_scene_sha256"],
            "scene_setup_identity": scene}


def artifact_fixture():
    return {"schema": "research-model-artifact/v1", "model_type": "diffusion", "policy_contract": CONTRACT,
            "model_config": {"training_window_start": 50}, "training_window_start": 50,
            "live_compatible": True, "checkpoint_selection": "FINAL_EMA", "completed_updates": 1,
            "script_options": {}, "scene_setup_identity": scene_fixture(),
            "identities": {"episodes": [episode_fixture(), episode_fixture("val_a", split="validation")]}}


def saved_fixture(artifact):
    return {"schema": "research-policy-checkpoint/v1", "policy_contract": CONTRACT,
            "metadata": {"plan": {"scene_setup_identity": copy.deepcopy(artifact["scene_setup_identity"])},
                         "identities": copy.deepcopy(artifact["identities"])}}


class SceneModelIdentityTests(unittest.TestCase):
    def test_equal_source_consensus_is_accepted_without_mutating_metadata(self):
        artifact = artifact_fixture()
        before = copy.deepcopy(artifact)
        scene = models.validate_artifact_scene_identity(artifact)
        self.assertEqual(scene, scene_fixture())
        self.assertEqual(artifact, before)
        self.assertIsNot(scene, artifact["scene_setup_identity"])

    def test_missing_top_level_scene_is_not_filled_from_sources(self):
        artifact = artifact_fixture()
        del artifact["scene_setup_identity"]
        with self.assertRaises(ValueError): models.validate_artifact_scene_identity(artifact)
        self.assertNotIn("scene_setup_identity", artifact)

    def test_missing_source_scene_is_not_filled_from_artifact(self):
        artifact = artifact_fixture()
        del artifact["identities"]["episodes"][0]["scene_setup_identity"]
        with self.assertRaises(ValueError): models.validate_artifact_scene_identity(artifact)

    def test_mixed_admitted_source_scenes_are_rejected(self):
        artifact = artifact_fixture()
        artifact["identities"]["episodes"][1]["scene_setup_identity"]["options"]["support_height_m"] = 0.8
        with self.assertRaises(ValueError): models.validate_artifact_scene_identity(artifact)

    def test_artifact_differs_from_unanimous_sources_is_rejected(self):
        artifact = artifact_fixture()
        artifact["scene_setup_identity"]["source"]["sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "differs from admitted source"):
            models.validate_artifact_scene_identity(artifact)

    def test_source_base_scene_binding_must_agree(self):
        artifact = artifact_fixture()
        artifact["identities"]["episodes"][0]["scene_sha256"] = "e" * 64
        with self.assertRaises(ValueError): models.validate_artifact_scene_identity(artifact)

    def test_no_admitted_sources_cannot_establish_identity(self):
        for records in ([], [episode_fixture(split="test")]):
            artifact = artifact_fixture()
            artifact["identities"]["episodes"] = records
            with self.assertRaises(ValueError): models.validate_artifact_scene_identity(artifact)

    def test_artifact_file_inspection_uses_saved_identity_and_never_deserializes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "model.pt"
            normalizer = root / "normalization.json"
            checkpoint.write_bytes(b"Not a model: metadata-only inspection fixture")
            normalizer.write_text("{}", encoding="utf-8")
            artifact = artifact_fixture()
            artifact.update(checkpoint_path=checkpoint.name, checkpoint=identity(checkpoint),
                            normalizer_path=normalizer.name, normalizer=identity(normalizer))
            (root / "artifact.json").write_text(json.dumps(artifact), encoding="utf-8")
            result = models.inspect_checkpoint(checkpoint)
            self.assertEqual(result["scene_setup_identity"], scene_fixture())
            artifact["scene_setup_identity"]["options"]["support_height_m"] = 1.0
            (root / "artifact.json").write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(ValueError): models.inspect_checkpoint(checkpoint)
            with patch.object(models, "_adapter_class", side_effect=AssertionError("Adapter import must not occur")):
                with self.assertRaises(ValueError): models.load_predictor(checkpoint, device="cpu")

    def test_checkpoint_saved_plan_and_sources_equal_artifact(self):
        artifact = artifact_fixture()
        saved = saved_fixture(artifact)
        self.assertEqual(runtime.validate_checkpoint_scene_identity(saved, artifact), scene_fixture())

    def test_checkpoint_saved_plan_missing_identity_is_rejected(self):
        artifact = artifact_fixture()
        saved = saved_fixture(artifact)
        del saved["metadata"]["plan"]["scene_setup_identity"]
        with self.assertRaises(ValueError): runtime.validate_checkpoint_scene_identity(saved, artifact)

    def test_checkpoint_saved_plan_and_its_sources_agree_but_artifact_differs(self):
        artifact = artifact_fixture()
        saved = saved_fixture(artifact)
        changed = copy.deepcopy(scene_fixture())
        changed["options"]["support_height_m"] = 0.9
        saved["metadata"]["plan"]["scene_setup_identity"] = changed
        saved["metadata"]["identities"]["episodes"] = [episode_fixture(scene=changed)]
        with self.assertRaisesRegex(ValueError, "saved-plan scene identity differs"):
            runtime.validate_checkpoint_scene_identity(saved, artifact)

    def test_checkpoint_plan_matches_but_saved_source_scene_does_not(self):
        artifact = artifact_fixture()
        saved = saved_fixture(artifact)
        saved["metadata"]["identities"]["episodes"][0]["scene_setup_identity"]["options"] = {}
        with self.assertRaises(ValueError): runtime.validate_checkpoint_scene_identity(saved, artifact)

    def test_adapter_load_rejects_missing_binding_before_model_construction(self):
        artifact = artifact_fixture()
        saved = saved_fixture(artifact)
        del saved["metadata"]["plan"]["scene_setup_identity"]
        loads = []
        def fake_load(*args, **kwargs):
            loads.append(kwargs)
            return saved
        fake_torch = types.SimpleNamespace(load=fake_load)
        with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(runtime.Adapter, "__init__", side_effect=AssertionError("Model must not construct")):
            with self.assertRaises(ValueError): runtime.Adapter.load("synthetic.pt", metadata=artifact, device="cpu")
        self.assertEqual(loads, [{"map_location": "cpu", "weights_only": True}])

    def test_no_heavy_modules_imported(self):
        forbidden = {"torch", "torchvision", "diffusers", "isaacsim", "isaaclab", "omni", "carb", "pxr"}
        self.assertFalse(forbidden & {name.split(".")[0] for name in sys.modules})


# Scene native

"""Offline scene-hook and capture tests. Fake APIs only; never starts Kit."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
from types import ModuleType
import unittest
from unittest.mock import patch

from pipeline.runtime import native as backend
from pipeline.recording import CustomCapture
from pipeline.scene import install_scene_builder
from pipeline.scene import validate_custom_build


class StageBuilderTests(unittest.TestCase):
    def test_all_three_modes_build_before_reset_without_mutating_reset_pose(self):
        for mode in ("scene", "scripted", "live"):
            events = []
            cfg = NS(scene=NS(replicate_physics=True, chicken=NS(init_state=NS(pos=(.7,.1,1.2),rot=(0.,1.,0.,0.)))), events=NS())
            session = NS(mode=mode)
            module = NS(configure_scene=lambda *args: events.append("configure"),
                build_scene=lambda *args: events.append("build") or {"created": True})
            install_scene_builder(cfg, session, module, {"options": {}, "definition": {"kind":"custom"}},
                                  {}, event_factory=lambda **kw:NS(**kw))
            events.append("construct_scene")
            cfg.events.research_scene_prestartup.func(NS(sim=NS(get_initial_stage=lambda: object())), None)
            events.append("initial_reset")
            self.assertEqual(events, ["configure", "construct_scene", "build", "initial_reset"])
            self.assertFalse(cfg.scene.replicate_physics)
            self.assertEqual(cfg.scene.chicken.init_state.pos, (.7,.1,1.2))
            self.assertEqual(cfg.scene.chicken.init_state.rot, (0.,1.,0.,0.))
            self.assertEqual(validate_custom_build(session)["calls"], 1)
            with self.assertRaisesRegex(RuntimeError, "exactly once"):
                cfg.events.research_scene_prestartup.func(NS(), None)

    def test_table_registers_no_extra_event_and_preserves_replication(self):
        cfg = NS(scene=NS(replicate_physics=True), events=NS())
        module = NS(configure_scene=lambda *args: None,
                    build_scene=lambda *args: self.fail("default table build should not author"))
        session = NS()
        install_scene_builder(cfg, session, module, {"options": {}, "definition":{"kind":"table"}}, {})
        self.assertTrue(cfg.scene.replicate_physics)
        self.assertEqual(vars(cfg.events), {})

    def test_selected_module_configuration_all_modes_and_robot_reset_receipt(self):
        for mode in ("scene", "scripted", "live"):
            calls=[]
            cfg=NS(actions={"dimension":5}, observations={"keys":"policy-four"},
                scene=NS(replicate_physics=True, ground=NS(spawn=NS(usd_path=backend.host.ORIGINAL_GROUND_URL)),
                    chicken=NS(init_state=NS(pos=(0,0,0),rot=(1,0,0,0))), robot=NS(init_state={"joint":0})),events=NS())
            session=NS(mode=mode)
            def configure(cfg,position,options):
                calls.append((mode,"configure"))
                cfg.scene.robot.init_state={"joint":1}
                cfg.scene.chicken.init_state.pos=(.7,.2,1.3)
            module=NS(configure_scene=configure,
                build_scene=lambda *args:calls.append((mode,"build")) or {})
            identity={"definition":{"kind":"custom","id":"fixture"},"options":{}}
            managers=ModuleType("isaaclab.managers")
            managers.EventTermCfg=lambda **kw:NS(**kw)
            with patch.dict(sys.modules,{"isaaclab.managers":managers}):
                backend.configure_research(cfg,session,{"authored_xyz_m":[.7,.2,1.2],"authored_quaternion_wxyz":[1,0,0,0]},
                    {"path":"same-bytes.usd"},scene_module=module,scene_identity=identity)
            cfg.events.research_scene_prestartup.func(NS(sim=NS(get_initial_stage=lambda:object())),None)
            self.assertEqual(calls,[(mode,"configure"),(mode,"build")])
            self.assertTrue(session.research_start_configuration["robot_reset_configuration_changed"])
            self.assertEqual(session.research_start_configuration["after"]["xyz_m"],[.7,.2,1.3])
            self.assertEqual(session.research_start_configuration["requested_pose"]["authored_xyz_m"],[.7,.2,1.2])

    def test_unreached_build_is_failure(self):
        with self.assertRaisesRegex(RuntimeError, "not built"):
            validate_custom_build(NS(research_scene_build={"status":"PENDING"}))


def capture_fixture(folder, *, bad_pose_at=None, reset_at=None, privileged=False):
    import numpy as np
    sys.path.insert(0, str(backend.FROZEN/"src"))
    from chicgrasp_policy import interface
    source = (backend.FROZEN/"scripts/policy/attachment_assisted_lift_record.py").read_text()
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_valid_pose")
    space = {"np":np, "CaptureFailure": RuntimeError}
    exec(compile(ast.Module(body=[function],type_ignores=[]), "<original pose validator>", "exec"), space)
    class Runtime:
        def __init__(self):
            self.owner, self.origin_clock, self.episode_id = None, (0,0.), "custom-test"
            self.output_dir, self.seed, self.counter, self.jaws = Path(folder), 42, 0, [1,1]
            self.calls, self.current = [], np.array([.65,0.,.2,1.,1.], dtype=np.float32)
        def clock(self): return self.counter, self.counter/120
        def observation(self):
            result = dict(camera_left_rgb=np.zeros((240,320,3),np.uint8),
                camera_right_rgb=np.zeros((240,320,3),np.uint8), arm_joint_pos=np.zeros(6,np.float32),
                jaw_state=np.array(self.jaws,np.uint8))
            if privileged: result["chicken_pose"] = [0,0,1]
            return result
        def measure(self):
            value = [.65,.1,.9,1.,0.,0.,0.]
            if self.counter == bad_pose_at: value[0] = float("nan")
            return dict(valid=True, issues=[], clock_before=list(self.clock()), clock_after=list(self.clock()),
                poses={name:list(value) for name in ("robot_root_world_wxyz", "gripper_world_wxyz",
                    "chicken_actor_world_wxyz", "tcp_world_wxyz", "tcp_base_wxyz", "chicken_in_gripper_wxyz")})
        def read_jaws(self): return {"clock_before":list(self.clock()),"clock_after":list(self.clock())}
        def set_hooks(self, boundary, lifecycle, pre_reset): self.boundary,self.lifecycle,self.pre_reset=boundary,lifecycle,pre_reset
        def camera_frames(self): return {"left":self.counter//2,"right":self.counter//2}
        def camera_metadata(self): return {"policy_only":True}
        def pre_candidate_observation(self, before): return {"pose":before}
        def step(self, action):
            self.calls.append(action.copy()); self.jaws=action[3:].tolist()
            for _ in range(2):
                self.counter += 1
                if self.counter == reset_at: self.pre_reset()
                self.boundary()
            return {"resetting":False,"truncated":False,"terminated":False,"invalid":False}
    rt = Runtime()
    core = NS(_valid_pose=space["_valid_pose"], CaptureFailure=RuntimeError,
        write_json=lambda path,data: path.write_text(json.dumps(data)),
        RuntimeSession=NS(commanded_jaw_metadata=lambda runtime,bits:{"bits":list(runtime.jaws)}))
    records, bundles, commands = [], [], []
    writer = NS(write_substep=records.append, write_pre_candidate=lambda *args: None,
        write_bundle=lambda index,obs,sensor: bundles.append((index,sensor["native_clock"])),
        command_event=lambda name,index,**kw: commands.append((name,index,kw)))
    class Controller:
        total_actions=200
        next_index=0
        def select(self,index,observation,sensor):
            if index != self.next_index: raise AssertionError("order")
            self.next_index += 1
            return np.array([.65,0.,.2,*( [1.,1.] if index<120 else [0.,0.])],np.float32)
    capture = CustomCapture(core,rt,writer,Controller(),dependencies=(np,interface))
    return capture,rt,records,bundles,commands


class CustomCaptureTests(unittest.TestCase):
    def test_200_actions_exact_schedule_no_owner_and_unscored(self):
        with tempfile.TemporaryDirectory() as folder:
            capture,rt,records,bundles,commands=capture_fixture(folder)
            result=capture.run()
            self.assertEqual(result["status"],"UNSCORED")
            self.assertIsNone(result["success"])
            self.assertEqual((len(rt.calls),rt.counter,len(capture.executed_actions)),(1200,2400,200))
            self.assertEqual(len(records),2400)
            self.assertEqual(bundles[0],(0,[0,0.]))
            self.assertEqual(bundles[-1],(199,[2388,19.9]))
            self.assertEqual(sum(x[0]=="ACCEPTANCE_OBSERVED" for x in commands),200)
            self.assertIsNone(rt.owner)
            self.assertTrue(all(row["attachment"]["gripper_attachment"] is False for row in records))
            self.assertTrue(all(x.attachment_state is False for x in capture.diagnostics))
            self.assertTrue(all(x.outcome_label is None for x in capture.diagnostics[:-1]))
            self.assertEqual(capture.diagnostics[-1].outcome_label,"CUSTOM_SCENE_TASK_NOT_SCORED")

    def test_nonfinite_pose_stops_without_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            capture,rt,records,*_=capture_fixture(folder,bad_pose_at=7)
            with self.assertRaisesRegex(RuntimeError,"invalid_required_pose"):
                capture.run()
            self.assertEqual(rt.counter,7)
            self.assertFalse(capture.completed)
            self.assertTrue(records[-1]["runtime"]["invalid"])

    def test_reset_stops_before_reset_or_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            capture,rt,*_=capture_fixture(folder,reset_at=5)
            with self.assertRaisesRegex(RuntimeError,"unexpected_reset"):
                capture.run()
            self.assertFalse(capture.completed)
            self.assertEqual(len(capture.lifecycle_events),1)

    def test_privileged_input_rejected_before_action(self):
        with tempfile.TemporaryDirectory() as folder:
            capture,rt,*_=capture_fixture(folder,privileged=True)
            with self.assertRaises(ValueError): capture.run()
            self.assertEqual(rt.calls,[])

    def test_unexpected_owner_rejected(self):
        with self.assertRaisesRegex(ValueError,"attachment owner"):
            CustomCapture(None,NS(owner=object()),None,NS(total_actions=200))

    def test_custom_evaluator_cannot_inherit_original_label(self):
        for label in ("PIPELINE_VERTICAL_SLICE_PASS_WITH_TEMPORARY_ATTACHMENT","PHYSICAL_GRASP_PASS","G1_PASS"):
            with self.assertRaisesRegex(ValueError,"inherit"):
                backend.validate_scene_evaluation({"status":"PASS","training_eligible":True,"label":label},custom_scene=True)


class NativeCustomOrchestrationTests(unittest.TestCase):
    def execute(self, mode, *, evaluator_eligible=True, failure=None):
        from tests.test_runtime import NativeOrchestrationTests
        fixture = NativeOrchestrationTests()
        fixture.setUp()
        fixture.custom = True
        fixture.objective_status = "UNSCORED"
        fixture.capture_exception = failure
        core, native, deps, receipt = fixture.install
        class Runtime(native.RuntimeSession):
            def initialize(self):
                super().initialize()
                self.research_scene_build = dict(status="BUILT", calls=1,
                    replicate_physics_after=False, phase="prestartup_before_initial_sim_reset")
        fixture.install = core, NS(RuntimeSession=Runtime), deps, receipt
        module = NS(validate_position=lambda *args:None,
            evaluate_episode=lambda capture,metrics,options:dict(status="UNSCORED",
                label="CUSTOM_SCENE_TASK_NOT_SCORED", training_eligible=evaluator_eligible))
        def authenticate(identity):
            identity = deepcopy(identity)
            identity["definition"] = {"kind":"custom","id":"fake-custom-test"}
            return module,identity
        def capture_factory(core,runtime,writer,controller):
            self.assertIsNone(runtime.owner)
            fixture.calls.append("custom_capture")
            return fixture.modules["chicgrasp_policy"].geometric_capture.make_capture(
                core,runtime,writer,action_source=controller)
        # The actual custom scheduler has separate complete/finite/reset tests.
        with patch('pipeline.scene.authenticate_scene_setup', side_effect=authenticate), \
             patch('pipeline.recording.CustomCapture', side_effect=capture_factory):
            if mode == "live":
                real_load = backend.load_json
                def load(path):
                    value=real_load(path)
                    value["scene_setup_identity"]["definition"]={"kind":"custom","id":"fake-custom-test"}
                    return value
                with patch.object(backend,"load_json",side_effect=load):
                    result=fixture.execute(mode=mode)
            else:
                result=fixture.execute(mode=mode)
        return result,fixture.calls

    def test_custom_script_uses_no_owner_or_original_visual_verdict(self):
        result,calls=self.execute("scripted")
        self.assertEqual(result["completion"],"COMPLETED")
        self.assertEqual(result["custom_task_status"],"UNSCORED")
        self.assertTrue(result["training_eligible"])
        self.assertNotIn("owner_bind",calls)
        self.assertNotIn("save_visuals",calls)
        self.assertIn("custom_capture",calls)
        self.assertFalse(result["original_hc_evidence"])

    def test_custom_explicit_eligibility_denial_preserved(self):
        result,_=self.execute("scripted",evaluator_eligible=False)
        self.assertEqual(result["completion"],"COMPLETED")
        self.assertFalse(result["training_eligible"])

    def test_custom_live_never_training_eligible_and_no_v3c_attachment(self):
        result,calls=self.execute("live")
        self.assertEqual(result["completion"],"COMPLETED",result["failures"])
        self.assertFalse(result["training_eligible"])
        self.assertNotIn("owner_bind",calls)

    def test_custom_capture_failure_cannot_publish_episode(self):
        result,calls=self.execute("scripted",failure=RuntimeError("nonfinite fake pose"))
        self.assertEqual(result["completion"],"FAILED")
        self.assertFalse(result["training_eligible"])
        self.assertIsNone(result["episode"])
        self.assertIn("close_scene",calls)


class EpisodeBindingTests(unittest.TestCase):
    def test_sidecar_binds_actual_manifest_and_scene_identity(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            capture,runtime,*_=capture_fixture(folder)
            # Small output fixture only. Full 200-action schedule is tested above.
            capture.completed=True
            capture.executed_actions=[np.array([.65,0.,.2,1.,1.],np.float32)]*2
            capture.observations=[capture.interface.validate_policy_observation(runtime.observation())]*2
            capture.diagnostics=[capture.interface.DiagnosticFrame(),
                capture.interface.DiagnosticFrame(outcome_label="CUSTOM_SCENE_TASK_NOT_SCORED")]
            capture.rows=[{}]*24
            runtime.research_scene_setup_identity={"schema":"fixture","definition":{"kind":"custom"}}
            receipt=backend.finalize_episode(capture,runtime,Path(folder))
            binding=Path(receipt["binding"]["path"])
            value=json.loads(binding.read_text())
            manifest=Path(receipt["path"])/"manifest.json"
            self.assertEqual(value["schema"],"research-episode-binding/v1")
            self.assertEqual(value["scene_setup_identity"],runtime.research_scene_setup_identity)
            self.assertEqual(value["episode_manifest"],{"bytes":manifest.stat().st_size,"sha256":backend.sha256(manifest)})
            self.assertEqual(receipt["binding"]["sha256"],backend.sha256(binding))
            self.assertNotIn(binding.name,[p.name for p in Path(receipt["path"]).iterdir()])


# Scene inspection

"""Offline inspection navigation checks; no SDK app, physics or rendering."""
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
from types import ModuleType
import unittest
from unittest.mock import Mock
from unittest.mock import patch
from pipeline.runtime import native as backend


class InspectionNavigationTests(unittest.TestCase):
    def runtime(self, *, change_clock=False, on_update=None):
        state = {'open': True, 'updates': 0, 'clock': (8, .0666667)}
        def update():
            state['updates'] += 1
            if on_update: on_update()
            if change_clock: state['clock'] = (9, .075)
            if state['updates'] == 3: state['open'] = False
        return NS(env=NS(sim=NS(pause=Mock())), clock=lambda: state['clock'],
                  app=NS(is_running=lambda: state['open'], update=update)), state

    def test_navigation_and_focal_length_changes_do_not_end_inspection(self):
        camera = NS(transform='saved', focal_length=15)
        def navigate():
            camera.transform = 'temporary pan/rotate/zoom'
            camera.focal_length = 20
        runtime, state = self.runtime(on_update=navigate)
        guard = Mock(side_effect=RuntimeError('Approved HAND local transform changed'))
        with tempfile.TemporaryDirectory() as directory, patch.object(backend.time, 'monotonic', return_value=0), patch.object(backend.time, 'sleep'):
            result = backend.paused_scene_review(runtime, Path(directory), 30, NS(before_render=guard))
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(state['updates'], 3)
        guard.assert_not_called()
        runtime.env.sim.pause.assert_called_once()
        self.assertEqual(result['reason'], 'WINDOW_CLOSED')
        self.assertEqual(result['candidate_actions'], 0)
        self.assertFalse(result['camera_edits_saved'])
        self.assertEqual(result['camera_navigation'], 'temporary_inspection_only')

    def test_physics_advance_is_still_rejected(self):
        runtime, state = self.runtime(change_clock=True)
        with tempfile.TemporaryDirectory() as directory, patch.object(backend.time, 'monotonic', return_value=0), patch.object(backend.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'scene_review_physics_advanced'):
                backend.paused_scene_review(runtime, Path(directory), 30)
        self.assertEqual(state['updates'], 1)

    def test_close_marker_requires_no_update(self):
        runtime, state = self.runtime()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'CLOSE_SCENE').touch()
            result = backend.paused_scene_review(runtime, folder, 30)
        self.assertEqual(result['reason'], 'CLOSE_SCENE')
        self.assertEqual(state['updates'], 0)

    def test_review_time_limit_remains(self):
        runtime, state = self.runtime()
        with tempfile.TemporaryDirectory() as directory, patch.object(backend.time, 'monotonic', side_effect=[0, 31]):
            result = backend.paused_scene_review(runtime, Path(directory), 30)
        self.assertEqual(result['reason'], 'REVIEW_TIME_LIMIT')
        self.assertEqual(state['updates'], 0)

    def cameras_fixture(self, guard):
        runtime = NS(stage=object(), env=NS(sim=NS(pause=Mock())),
                     clock=lambda: (8, .0666667), camera_frames=lambda: {'left': 2, 'right': 2},
                     effective_configuration={'physics': 'frozen'})
        sensor = NS(pause=Mock(), destroy=Mock())
        def creator(stage, *, enabled, camera_factory):
            for name in ('HAND', 'LEFT_HD', 'RIGHT_HD'): camera_factory(name=name)
            return NS(before_render=guard, paths={'HAND': '/hand'}, configs={}, policy_before={})
        return runtime, sensor, creator

    def test_default_view_is_free_perspective_after_initial_validation(self):
        guard = Mock()
        runtime, sensor, creator = self.cameras_fixture(guard)
        viewport = NS(camera_path='/old')
        module = ModuleType('omni.kit.viewport.utility')
        module.get_active_viewport = lambda: viewport
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', {'omni.kit.viewport.utility': module}):
            _, _, receipt = backend.create_scene_cameras(runtime, Path(directory), creator=creator, camera_factory=lambda **kw: sensor)
        guard.assert_called_once()
        self.assertEqual(viewport.camera_path, '/OmniverseKit_Persp')
        self.assertFalse(receipt['camera_edits_saved'])
        self.assertEqual(receipt['preset_validation'], 'INITIAL_CONFIGURATION_ONLY')
        self.assertEqual(receipt['viewport_errors'], [])

    def test_initial_invalid_camera_still_rejected_and_sensors_cleaned(self):
        guard = Mock(side_effect=RuntimeError('Approved HAND local transform changed'))
        runtime, sensor, creator = self.cameras_fixture(guard)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'Approved HAND local transform changed'):
                backend.create_scene_cameras(runtime, Path(directory), creator=creator, camera_factory=lambda **kw: sensor)
        self.assertEqual(sensor.destroy.call_count, 3)
        guard.assert_called_once()

    def test_unavailable_viewport_does_not_abort_loaded_scene(self):
        runtime, sensor, creator = self.cameras_fixture(Mock())
        module = ModuleType('omni.kit.viewport.utility')
        module.get_active_viewport = lambda: None
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', {'omni.kit.viewport.utility': module}):
            _, _, receipt = backend.create_scene_cameras(runtime, Path(directory), creator=creator, camera_factory=lambda **kw: sensor)
        self.assertEqual(len(receipt['viewport_errors']), 1)
        self.assertIn('no_active_viewport', receipt['viewport_errors'][0])


class FinishInspectionTests(unittest.TestCase):
    def fake_ui(self):
        from contextlib import nullcontext
        self.click = None
        self.window = NS(frame=nullcontext(), destroy=Mock())
        ui = ModuleType('omni.ui')
        ui.Window = Mock(return_value=self.window)
        ui.VStack = lambda **kw: nullcontext()
        ui.Label = Mock()
        def button(label, **kwargs):
            self.assertEqual(label, 'Finish inspection')
            self.click = kwargs['clicked_fn']
        ui.Button = button
        omni = ModuleType('omni'); omni.__path__ = []; omni.ui = ui
        return {'omni':omni, 'omni.ui':ui}

    def test_button_click_finishes_paused_loop_without_closing_app_early(self):
        modules = self.fake_ui()
        events = []
        runtime = NS(env=NS(sim=NS(pause=lambda:events.append('pause'))),
            clock=lambda:(8,.0666667), app=NS(is_running=lambda:True,
                update=lambda: (events.append('update'), self.click())))
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', modules), patch.object(backend.time, 'sleep'):
            result = backend.paused_scene_review(runtime, Path(directory), 30)
        self.assertEqual(events, ['pause', 'update'])
        self.assertEqual(result['reason'], 'FINISH_BUTTON')
        self.assertEqual(result['candidate_actions'], 0)
        self.assertFalse(result['approved'])
        self.assertEqual(result['controls_errors'], [])
        self.window.destroy.assert_called_once()

    def test_ui_failure_is_recorded_without_aborting_review(self):
        modules = self.fake_ui()
        modules['omni.ui'].Window.side_effect = RuntimeError('UI unavailable')
        runtime, state = InspectionNavigationTests().runtime()
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', modules), patch.object(backend.time, 'sleep'):
            result = backend.paused_scene_review(runtime, Path(directory), 30)
        self.assertEqual(result['reason'], 'WINDOW_CLOSED')
        self.assertEqual(state['updates'], 3)
        self.assertIn('UI unavailable', result['controls_errors'][0])

    def test_ui_cleanup_failure_does_not_prevent_result(self):
        controls = dict(finish_requested=True, window=NS(destroy=Mock(side_effect=RuntimeError('already closed'))), errors=[])
        runtime, state = InspectionNavigationTests().runtime()
        with tempfile.TemporaryDirectory() as directory:
            result = backend.paused_scene_review(runtime, Path(directory), 30, controls_factory=lambda:controls)
        self.assertEqual(result['reason'], 'FINISH_BUTTON')
        self.assertIn('already closed', result['controls_errors'][0])

    def test_finish_result_is_durable_before_application_close(self):
        from tests.test_runtime import NativeOrchestrationTests
        fixture = NativeOrchestrationTests(); fixture.setUp()
        core = fixture.install[0]
        writer = core.write_json
        def save(path, value):
            if Path(path).name == 'research_result.json':
                self.assertEqual(value['completion'], 'COMPLETED')
                self.assertEqual(value['scene_review']['reason'], 'FINISH_BUTTON')
                fixture.calls.append('durable_scene_result')
            writer(path, value)
        core.write_json = save
        with patch.object(backend, 'create_scene_cameras', return_value=(NS(paths={}), [], {})), patch.object(backend, 'paused_scene_review', return_value={'reason':'FINISH_BUTTON', 'candidate_actions':0}):
            result = fixture.execute(mode='scene')
        self.assertEqual(result['counts']['completed_actions'], 0)
        self.assertLess(fixture.calls.index('durable_scene_result'), fixture.calls.index('close_app'))
        self.assertNotIn('capture_run', fixture.calls)


# Scene outputs

"""Offline scene-output allocation and approval routing; no simulator launches."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline import settings
from pipeline.runtime import workflow


class SceneOutputsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.requested=self.root/'scene001'
        self.cfg=dict(settings.load_experiment(),scene_approval=str(self.root/'approval.json'))
        self.stack=ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(workflow,'load_host',return_value={}))
        self.stack.enter_context(patch.object(workflow,'preflight',return_value={'status':'PASS'}))
        self.child_calls=[]

    def run_scene(self, requested=None, *, completion='COMPLETED', raise_child=False):
        requested=Path(requested or self.requested).resolve()
        def child(cfg,host,mode,position,output,**kwargs):
            # The pointer must identify this attempt before any child work starts.
            pointer=settings.read(requested.with_name(requested.name+'.latest.json'))
            self.assertEqual(Path(pointer['output_directory']),Path(output).parent)
            self.assertEqual(pointer['requested_output'],str(requested))
            self.child_calls.append(str(output))
            if raise_child: raise RuntimeError('synthetic child interruption')
            result_path=Path(output)/'native/research_result.json'
            result_path.parent.mkdir(parents=True)
            native={'completion':completion,'scene_setup_identity':workflow.scene_setup_identity(cfg)}
            settings.write(result_path,native)
            return {'completion':completion,'native':native,'result_path':str(result_path)}
        with patch.object(workflow,'run_native',side_effect=child):
            return workflow.execute_simulation(self.cfg,'scene',requested)

    def test_repeated_identical_calls_allocate_new_folders_and_keep_old_bytes(self):
        snapshots=[]
        for index in range(1,4):
            result=self.run_scene()
            actual=self.root/f'scene{index:03d}'
            self.assertEqual(result['requested_output'],str(self.requested))
            self.assertEqual(result['output_directory'],str(actual))
            sentinel=actual/'owner-notes.bin'; sentinel.write_bytes(bytes([index,0,255]))
            snapshots.append({p:p.read_bytes() for p in actual.rglob('*') if p.is_file()})
        for snapshot in snapshots:
            for path,raw in snapshot.items(): self.assertEqual(path.read_bytes(),raw)
        self.assertEqual(workflow._latest_scene_review(self.requested),self.root/'scene003')
        self.assertEqual(len(self.child_calls),3)

    def test_existing_numbered_files_and_directories_are_never_reused(self):
        self.requested.mkdir(); (self.requested/'old.log').write_text('keep original')
        (self.root/'scene002').write_text('unrelated file occupying this name')
        (self.root/'scene003').mkdir()
        result=self.run_scene()
        self.assertEqual(result['output_directory'],str(self.root/'scene004'))
        self.assertEqual((self.requested/'old.log').read_text(),'keep original')
        self.assertEqual((self.root/'scene002').read_text(),'unrelated file occupying this name')

    def test_custom_name_series_does_not_consume_unrelated_names(self):
        (self.root/'scene999').mkdir(); (self.root/'inspection-other002').mkdir()
        requested=self.root/'inspection'
        first=self.run_scene(requested); second=self.run_scene(requested)
        self.assertEqual(first['output_directory'],str(requested))
        self.assertEqual(second['output_directory'],str(self.root/'inspection-002'))
        self.assertTrue((self.root/'scene999').is_dir())
        self.assertTrue((self.root/'inspection-other002').is_dir())

    def test_concurrent_mkdir_collision_advances_without_touching_winner(self):
        self.requested.mkdir()
        competing=self.root/'scene002'
        mkdir=Path.mkdir
        collided=[]
        def racing_mkdir(path,*args,**kwargs):
            if path==competing and not collided:
                collided.append(True); mkdir(path)
                (path/'other-process.log').write_text('preserve')
                raise FileExistsError(str(path))
            return mkdir(path,*args,**kwargs)
        with patch.object(Path,'mkdir',racing_mkdir): result=self.run_scene()
        self.assertEqual(result['output_directory'],str(self.root/'scene003'))
        self.assertEqual((competing/'other-process.log').read_text(),'preserve')

    def test_parallel_allocators_leave_latest_pointing_to_last_allocated_folder(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            paths=list(pool.map(lambda _:workflow._allocate_scene_output(self.requested),range(8)))
        self.assertEqual(len(set(paths)),8)
        self.assertEqual({p.name for p in paths},{f'scene{i:03d}' for i in range(1,9)})
        self.assertEqual(workflow._latest_scene_review(self.requested),self.root/'scene008')

    def test_original_approval_path_follows_latest_and_actual_folder_can_be_explicit(self):
        self.run_scene(); result=self.run_scene()
        actual=Path(result['output_directory'])
        receipt=workflow.approve_review(self.cfg,self.requested)
        approval=settings.read(receipt['approval'])
        self.assertEqual(receipt['review_directory'],str(actual))
        self.assertEqual(approval['review_summary_sha256'],settings.digest(actual/'summary.json'))
        direct=workflow.approve_review(self.cfg,actual,output=str(self.root/'explicit-approval.json'))
        self.assertEqual(direct['review_directory'],str(actual))

    def test_repeated_approval_is_idempotent_then_archives_previous_receipt(self):
        self.run_scene()
        first=workflow.approve_review(self.cfg,self.requested)
        original=Path(first['approval']).read_bytes()
        duplicate=workflow.approve_review(self.cfg,self.requested)
        self.assertTrue(duplicate['unchanged'])
        self.assertEqual(Path(first['approval']).read_bytes(),original)
        self.assertEqual(list(self.requested.glob('previous-approval*.json')),[])
        actual=Path(self.run_scene()['output_directory'])
        # A previously occupied archive name must remain byte-for-byte intact.
        occupied=actual/'previous-approval001.json'; occupied.write_bytes(b'preserved unrelated archive')
        updated=workflow.approve_review(self.cfg,self.requested)
        self.assertFalse(updated['unchanged'])
        self.assertEqual(Path(updated['previous_approval']).name,'previous-approval002.json')
        self.assertEqual(Path(updated['previous_approval']).read_bytes(),original)
        self.assertEqual(occupied.read_bytes(),b'preserved unrelated archive')
        self.assertNotEqual(Path(updated['approval']).read_bytes(),original)
        self.assertEqual(settings.read(updated['approval'])['review_directory'],str(actual))
        self.assertTrue(workflow.require_approval(self.cfg)['OWNER_APPROVED'])

    def test_failed_latest_does_not_fall_back_or_replace_existing_approval(self):
        self.run_scene(); approved=workflow.approve_review(self.cfg,self.requested)
        original=Path(approved['approval']).read_bytes()
        failed=self.run_scene(completion='FAILED')
        self.assertEqual(failed['completion'],'FAILED')
        with self.assertRaisesRegex(ValueError,'completed scene review'):
            workflow.approve_review(self.cfg,self.requested)
        self.assertEqual(Path(approved['approval']).read_bytes(),original)

    def test_interrupted_latest_with_no_summary_refuses_stale_success(self):
        self.run_scene()
        with self.assertRaisesRegex(RuntimeError,'synthetic child interruption'):
            self.run_scene(raise_child=True)
        self.assertEqual(workflow._latest_scene_review(self.requested),self.root/'scene002')
        with self.assertRaises(ValueError) as caught:
            workflow.approve_review(self.cfg,self.requested)
        message=str(caught.exception)
        self.assertIn(str(self.root/'scene002/summary.json'),message)
        self.assertIn('Finish inspection',message)
        self.assertIn('wait for COMPLETED',message)
        self.assertIn('already closed',message)
        self.assertIn('rerun the same load_scene.py --execute',message)
        self.assertFalse(Path(self.cfg['scene_approval']).exists())

    def test_missing_latest_summary_does_not_mutate_existing_approval_or_create_summary(self):
        self.run_scene(); workflow.approve_review(self.cfg,self.requested)
        with self.assertRaisesRegex(RuntimeError,'synthetic child interruption'):
            self.run_scene(raise_child=True)
        before={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with self.assertRaisesRegex(ValueError,'No saved completion summary'):
            workflow.approve_review(self.cfg,self.requested)
        after={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        self.assertFalse((self.root/'scene002/summary.json').exists())
        self.assertEqual(settings.read(self.cfg['scene_approval'])['review_directory'],str(self.requested))

    def test_pending_first_inspection_has_same_guidance_without_approval(self):
        actual=workflow._allocate_scene_output(self.requested)
        with self.assertRaisesRegex(ValueError,'may still be running') as caught:
            workflow.approve_review(self.cfg,self.requested)
        self.assertIn('Finish inspection',str(caught.exception))
        self.assertIn(str(actual/'summary.json'),str(caught.exception))
        self.assertFalse((actual/'summary.json').exists())
        self.assertFalse(Path(self.cfg['scene_approval']).exists())

    def test_pointer_schema_requested_path_sibling_and_series_are_validated(self):
        self.run_scene()
        pointer=self.requested.with_name(self.requested.name+'.latest.json')
        valid=settings.read(pointer)
        outside=self.root/'other-parent'; outside.mkdir(); (outside/'scene002').mkdir()
        (self.root/'unrelated').mkdir()
        for update in ({'schema':'wrong'}, {'requested_output':str(self.root/'different001')},
                       {'output_directory':str(outside/'scene002')},
                       {'output_directory':str(self.root/'unrelated')},
                       {'output_directory':'scene001'}, {'output_directory':str(self.root/'scene999')},
                       {'output_directory':str(self.root/'scene000')}):
            pointer.write_text(json.dumps(dict(valid,**update)),encoding='utf-8')
            with self.subTest(update=update),self.assertRaises(ValueError):
                workflow.approve_review(self.cfg,self.requested)
        pointer.write_text(json.dumps(valid),encoding='utf-8')
        self.assertEqual(workflow.approve_review(self.cfg,self.requested)['review_directory'],str(self.requested))

    def test_bad_pointer_is_not_replaced_by_another_execute(self):
        self.run_scene()
        pointer=self.requested.with_name(self.requested.name+'.latest.json')
        pointer.write_text('{"unrelated":true}',encoding='utf-8')
        before=pointer.read_bytes()
        with self.assertRaisesRegex(ValueError,'Invalid latest'): self.run_scene()
        self.assertEqual(pointer.read_bytes(),before)
        self.assertFalse((self.root/'scene002').exists())

    def test_identity_changes_still_reject_approval(self):
        self.run_scene(); changed=copy.deepcopy(self.cfg)
        changed['script_options']={'lift_extra_m':.02}
        with self.assertRaisesRegex(ValueError,'Configuration changed'):
            workflow.approve_review(changed,self.requested)
        self.assertFalse(Path(self.cfg['scene_approval']).exists())

    def test_unrelated_active_approval_file_is_preserved(self):
        self.run_scene(); approval=Path(self.cfg['scene_approval'])
        approval.write_text('{"purpose":"unrelated"}',encoding='utf-8'); before=approval.read_bytes()
        with self.assertRaisesRegex(ValueError,'unrelated file'):
            workflow.approve_review(self.cfg,self.requested)
        self.assertEqual(approval.read_bytes(),before)

    def test_record_and_live_keep_existing_directory_refusal(self):
        for mode in ('record','live'):
            existing=self.root/mode; existing.mkdir(); (existing/'keep.bin').write_bytes(b'keep')
            with self.subTest(mode=mode),patch.object(workflow,'require_approval',return_value={}), \
                 patch.object(workflow,'checkpoint_record',return_value={}), \
                 patch.object(workflow,'run_native',side_effect=AssertionError('Must not dispatch')), \
                 self.assertRaises(FileExistsError):
                workflow.execute_simulation(self.cfg,mode,existing,limit=1)
            self.assertEqual((existing/'keep.bin').read_bytes(),b'keep')
            self.assertFalse(existing.with_name(existing.name+'.latest.json').exists())
            self.assertFalse(existing.with_name(existing.name+'-002').exists())


if __name__ == "__main__":
    unittest.main()
