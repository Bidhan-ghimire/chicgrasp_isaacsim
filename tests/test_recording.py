"""Offline regression tests grouped by recording."""
from __future__ import annotations



# Recording selection

"""Recording selection with saved catalogue metadata and fake process boundaries."""
import contextlib
import copy
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import Mock
from unittest.mock import patch

from pipeline import runtime as bootstrap
from pipeline import settings
from pipeline.runtime import workflow


class RecordingSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = settings.load_experiment()
        cls.rows = settings.load_positions(cls.cfg)
        cls.training = [row for row in cls.rows if row['split'] in ('train','validation')]
        cls.ids = [row['id'] for row in cls.training]

    def test_default_and_explicit_all_preserve_exact_original_order(self):
        self.assertEqual(len(self.training),50)
        for limit in (None,50):
            for seed in (None,42,17):
                with self.subTest(limit=limit,seed=seed), patch.object(random.Random,'sample',side_effect=AssertionError('All-position path must not draw')):
                    rows=workflow.select_positions(self.cfg,'record',limit=limit,selection_seed=seed)
                    self.assertEqual([row['id'] for row in rows],self.ids)

    def test_requested_counts_are_unique_reproducible_and_non_test(self):
        for count in (1,2,5,10,20,40):
            with self.subTest(count=count):
                first=workflow.select_positions(self.cfg,'record',limit=count)
                second=workflow.select_positions(self.cfg,'record',limit=count,selection_seed=42)
                self.assertEqual(first,second)
                self.assertEqual(len({row['id'] for row in first}),count)
                self.assertTrue(all(row['split']=='train' for row in first))

    def test_golden_selection_and_seed_change(self):
        chosen=lambda seed:[p['id'] for p in workflow.select_positions(self.cfg,'record',limit=5,selection_seed=seed)]
        self.assertEqual(chosen(42),['R000059','R000010','R000003','R000066','R000024'])
        self.assertNotEqual(set(chosen(42)),set(chosen(43)))
        self.assertEqual([p['id'] for p in workflow.select_positions(self.cfg,'record',limit=1)],['R000059'])

    def test_selection_does_not_mutate_global_rng_or_catalogue(self):
        original=copy.deepcopy(self.rows); before=random.getstate()
        with patch.object(workflow,'load_positions',return_value=self.rows):
            workflow.select_positions(self.cfg,'record',limit=20,selection_seed=7)
        self.assertEqual(random.getstate(),before)
        self.assertEqual(self.rows,original)

    def test_validation_recording_support_remains(self):
        rows=copy.deepcopy(self.rows)
        rows[0]['split']='validation'
        with patch.object(workflow,'load_positions',return_value=rows):
            selected=workflow.select_positions(self.cfg,'record',limit=50)
            self.assertEqual(selected[0]['split'],'validation')
            self.assertEqual([p['id'] for p in selected],self.ids)
            self.assertEqual(workflow.select_positions(self.cfg,'record',rows[0]['id'])[0]['split'],'validation')

    def test_invalid_record_limits_are_rejected(self):
        for value in (0,-1,51,True,False,1.5,'2'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError,'limit'):
                workflow.select_positions(self.cfg,'record',limit=value)

    def test_invalid_seeds_and_seed_on_other_modes_are_rejected(self):
        for value in (-1,True,2.0,'2'):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'selection-seed'):
                workflow.select_positions(self.cfg,'record',limit=1,selection_seed=value)
        for mode in ('live','scene','script','train'):
            with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,'only to recorder'):
                workflow.make_plan(self.cfg,mode,selection_seed=42)

    def test_explicit_position_is_exact_and_never_sampled(self):
        with patch.object(random.Random,'sample',side_effect=AssertionError('Explicit position must not draw')):
            selected=workflow.select_positions(self.cfg,'record','R000002',limit=1,selection_seed=999)
        self.assertEqual([p['id'] for p in selected],['R000002'])
        with self.assertRaisesRegex(ValueError,'exceeds 1'):
            workflow.select_positions(self.cfg,'record','R000002',limit=2)
        with self.assertRaisesRegex(ValueError,'test positions'):
            workflow.select_positions(self.cfg,'record','HC07',limit=1)
        with self.assertRaisesRegex(ValueError,'Unknown position'):
            workflow.select_positions(self.cfg,'record','unknown')

    def test_live_and_scene_prefix_behavior_is_unchanged(self):
        with patch.object(random.Random,'sample',side_effect=AssertionError('Never sample held-out rows')):
            self.assertEqual([p['id'] for p in workflow.select_positions(self.cfg,'live',limit=2)],['HC01','HC02'])
            self.assertEqual(len(workflow.select_positions(self.cfg,'live',limit=100)),10)
            self.assertEqual([p['id'] for p in workflow.select_positions(self.cfg,'scene',limit=2)],['R000002'])

    def test_plan_has_complete_selection_metadata(self):
        plan=workflow.make_plan(self.cfg,'record',limit=2,selection_seed=42)
        selected=plan['position_selection']
        self.assertEqual(selected['selected_ids'],['R000059','R000010'])
        self.assertEqual(selected['algorithm'],'python-random-sample/v1')
        self.assertTrue(selected['seed_applied'])
        self.assertEqual((selected['selection_seed'],selected['available_count'],selected['requested_count'],selected['selected_count']),(42,50,2,2))
        self.assertEqual(plan['planned_kit_starts'],2)
        self.assertEqual(selected['eligible_ids'],self.ids)
        self.assertEqual(len(selected['eligible_pool_sha256']),64)
        self.assertEqual(set(selected['source_files']),set(self.cfg['positions']))

    def test_plan_execution_dataset_summary_share_selection_without_refill(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'collection'; called=[]
            def fake_run(cfg,host,mode,position,path,**kwargs):
                called.append(position['id'])
                native=Path(path)/'native'; native.mkdir(parents=True)
                value={'completion':'COMPLETED','training_eligible':False,
                       'scene_setup_identity':workflow.scene_setup_identity(cfg)}
                settings.write(native/'research_result.json',value)
                return {'completion':'COMPLETED','native':value,'result_path':str(native/'research_result.json')}
            with patch.object(workflow,'load_host',return_value={}),patch.object(workflow,'preflight',return_value={'status':'PASS'}), \
                 patch.object(workflow,'require_approval',return_value={}),patch.object(workflow,'run_native',side_effect=fake_run):
                result=workflow.execute_simulation(self.cfg,'record',output,limit=2,selection_seed=42)
            plan=settings.read(output/'plan.json'); dataset=settings.read(output/'dataset.json'); summary=settings.read(output/'summary.json')
            self.assertEqual(called,['R000059','R000010'])
            self.assertEqual(result['training_episodes'],0)
            self.assertEqual(result['runs_completed'],2)
            self.assertEqual(dataset['episodes'],[])
            for value in (dataset,summary,result):
                self.assertEqual(value['position_selection'],plan['position_selection'])

    def test_oversize_fails_before_output_or_native_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'collection'
            with patch.object(workflow,'run_native') as native:
                with self.assertRaisesRegex(ValueError,'available recording positions'):
                    workflow.execute_simulation(self.cfg,'record',output,limit=51)
                native.assert_not_called()
            self.assertFalse(output.exists())

    def test_cli_passes_selection_seed_to_plan_without_execution(self):
        stdout=io.StringIO()
        with patch.object(sys,'argv',['recorder.py','--limit','2','--selection-seed','42']), \
             patch.object(workflow,'execute_simulation',side_effect=AssertionError('Plan must not launch')),contextlib.redirect_stdout(stdout):
            workflow.main('record')
        result=json.loads(stdout.getvalue())
        self.assertEqual(result['position_selection']['selected_ids'],['R000059','R000010'])
        self.assertFalse(result['executed'])

    def test_cli_passes_selection_seed_to_execution(self):
        stdout=io.StringIO()
        with patch.object(sys,'argv',['recorder.py','--limit','5','--selection-seed','9','--execute','--output','unused']), \
             patch.object(workflow,'execute_simulation',return_value={'completion':'COMPLETED'}) as launch,contextlib.redirect_stdout(stdout):
            workflow.main('record')
        self.assertEqual(launch.call_args.kwargs['limit'],5)
        self.assertEqual(launch.call_args.kwargs['selection_seed'],9)

    def test_bootstrap_preserves_flags_without_launching_sdk(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); python=root/'sim/kit/python/python.exe'; python.parent.mkdir(parents=True); python.write_bytes(b'fixture')
            host=root/'host.json'; host.write_text('{}'); config=root/'config.json'
            config.write_text(json.dumps({'host':str(host)}))
            settings_value={'isaac_sim':str(root/'sim'),'isaac_lab':str(root/'lab'),'policy_support':str(root/'support')}
            argv=['--config',str(config),'--limit','5','--selection-seed','17']
            runner=Mock(return_value=Mock(returncode=0))
            with patch.object(bootstrap,'load_host',return_value=settings_value),patch('pipeline.runtime.host.environment',return_value={}):
                bootstrap.handoff('record',argv,runner=runner,executable=str(root/'other.exe'),env={})
            command=runner.call_args.args[0]
            self.assertEqual(command[-len(argv):],argv)
            self.assertNotIn('--execute',command)


# Research workflow

"""Offline integration: pure scripts and mocked native/process boundaries only."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from pipeline import settings
from pipeline.runtime import workflow

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.cfg = settings.load_experiment()
        self.positions = settings.load_positions(self.cfg)

    def test_default_script_matches_frozen_arithmetic(self):
        settings.add_frozen_source()
        from chicgrasp_policy.attachment_capture import frozen_candidate_actions
        original = frozen_candidate_actions()
        for p in self.positions:
            expected = original.astype(np.float64)
            expected[:, :3] += np.asarray(p['authored_xyz_m']) - [.65, 0, .8155]
            np.testing.assert_array_equal(workflow.compiled_script(self.cfg, p), expected.astype(np.float32))

    def test_alternate_demo_changes_lift_not_setup(self):
        p = self.positions[0]
        changed = copy.deepcopy(self.cfg)
        changed['script_options'] = {'lift_extra_m': .02}
        baseline = workflow.compiled_script(self.cfg, p)
        alternate = workflow.compiled_script(changed, p)
        np.testing.assert_array_equal(baseline[:140], alternate[:140])
        np.testing.assert_array_equal(baseline[:, 3:], alternate[:, 3:])
        self.assertGreater(float(alternate[-1, 2] - baseline[-1, 2]), .0199)
        self.assertNotEqual(settings.scene_identity(changed), settings.scene_identity(self.cfg))

    def test_manifest_separation_and_explicit_selection(self):
        self.assertEqual(len(workflow.select_positions(self.cfg, 'record')), 50)
        self.assertEqual(len(workflow.select_positions(self.cfg, 'live')), 10)
        with self.assertRaisesRegex(ValueError, 'test positions'):
            workflow.select_positions(self.cfg, 'record', 'HC07')
        self.assertEqual(workflow.make_plan(self.cfg, 'live', position_id='HC07')['planned_kit_starts'], 3)
        self.assertEqual(workflow.make_plan(self.cfg, 'live', position_id='HC07', evaluation_seed=15345)['planned_kit_starts'], 1)
        with self.assertRaisesRegex(ValueError, 'configured evaluation seed'):
            workflow.make_plan(self.cfg, 'live', position_id='HC07', evaluation_seed=1)

    def test_planning_does_not_import_native_or_models(self):
        code = "import sys\nfrom unittest.mock import patch\nfrom pipeline.runtime import workflow\nfrom pipeline import settings\nwith patch('pipeline.training.run_training', side_effect=AssertionError('Planning called training')) as train, patch('pipeline.models.create_adapter', side_effect=AssertionError('Planning constructed a model')) as model:\n    workflow.make_plan(settings.load_experiment(), 'record', limit=1)\n    assert not {'pipeline.runtime.native', 'torch', 'isaacsim'} & set(sys.modules)\n    train.assert_not_called()\n    model.assert_not_called()\n"
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_approval_requires_review_and_invalidates_on_change(self):
        with tempfile.TemporaryDirectory() as td:
            review = Path(td) / 'review'
            review.mkdir()
            approval = Path(td) / 'approval.json'
            cfg = dict(self.cfg, scene_approval=str(approval))
            settings.write(review / 'summary.json', {'mode':'scene','completion':'COMPLETED','scene_identity':settings.scene_identity(cfg)})
            workflow.approve_review(cfg, review)
            self.assertTrue(workflow.require_approval(cfg)['OWNER_APPROVED'])
            cfg['script_options'] = {'lift_extra_m': .02}
            with self.assertRaisesRegex(ValueError, 'stale'): workflow.require_approval(cfg)

    def test_recorder_uses_selected_script_and_failed_process_is_not_success(self):
        with tempfile.TemporaryDirectory() as td:
            p = self.positions[0]
            cfg = copy.deepcopy(self.cfg)
            cfg['script_options'] = {'lift_extra_m': .02}
            def fake_runner(command, **kwargs):
                attempt = Path(command[command.index('--attempt') + 1])
                request = settings.read(command[command.index('--request') + 1])
                actual = np.load(request['script']['path'], allow_pickle=False)
                np.testing.assert_array_equal(actual, workflow.compiled_script(cfg, p))
                attempt.mkdir()
                settings.write(attempt / 'research_result.json', {'completion':'COMPLETED','training_eligible':True,
                               'scene_setup_identity':request['scene_setup_identity']})
                return {'success':False}
            host = {'isaac_sim':'C:/unused', 'isaac_lab':'C:/unusedlab', 'policy_support':'C:/unusedsupport', 'config_path':'C:/unused.json'}
            result = workflow.run_native(cfg, host, 'scripted', p, Path(td)/'run', runner=fake_runner)
            self.assertEqual(result['completion'], 'FAILED')

    def test_collection_does_not_admit_refused_episode(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(workflow, 'load_host', return_value={}), patch.object(workflow, 'preflight', return_value={'status':'PASS'}), patch.object(workflow, 'require_approval', return_value={}):
                def refused(cfg, host, mode, position, output, **kwargs):
                    Path(output).mkdir()
                    native = Path(output)/'native'
                    native.mkdir()
                    result = {'completion':'COMPLETED','training_eligible':False,'scene_setup_identity':workflow.scene_setup_identity(cfg)}
                    settings.write(native/'research_result.json', result)
                    return {'completion':'COMPLETED','native':result,'result_path':str(native/'research_result.json')}
                with patch.object(workflow, 'run_native', side_effect=refused):
                    result = workflow.execute_simulation(self.cfg, 'record', Path(td)/'collection', limit=1)
                self.assertEqual(result['training_episodes'],0)
                self.assertEqual(result['completion'],'COMPLETED')
                self.assertEqual(settings.read(Path(td)/'collection/dataset.json')['episodes'],[])

    def test_schema_rejects_wrong_jaw_or_duration(self):
        with tempfile.TemporaryDirectory() as td:
            script = Path(td)/'invalid.py'
            script.write_text('import numpy as np\ndef build_demo(position, options): return np.zeros((199,5),dtype=np.float32)\n')
            cfg = dict(self.cfg, script=str(script))
            with self.assertRaisesRegex(ValueError, '200'): workflow.compiled_script(cfg, self.positions[0])

    def test_new_script_collection_manifest_is_accepted_by_training_planner(self):
        from pipeline.training import training_plan
        with tempfile.TemporaryDirectory() as td:
            cfg = copy.deepcopy(self.cfg)
            cfg['script_options'] = {'lift_extra_m': .02}
            host = {'isaac_sim':'C:/unused','isaac_lab':'C:/unused','policy_support':'C:/unused','config_path':'C:/unused.json'}
            def fake_runner(command, **kwargs):
                request = settings.read(command[command.index('--request') + 1])
                actual = np.load(request['script']['path'], allow_pickle=False)
                np.testing.assert_array_equal(actual, workflow.compiled_script(cfg, request['position']))
                attempt = Path(command[command.index('--attempt') + 1])
                episode = attempt/'episode'
                episode.mkdir(parents=True)
                settings.write(episode/'manifest.json', {'fixture':'metadata-only, no runtime capture'})
                from pipeline.training import identity
                binding = attempt/'episode_binding.json'
                settings.write(binding, {'schema':'research-episode-binding/v1','scene_setup_identity':request['scene_setup_identity'],
                                         'episode_manifest':identity(episode/'manifest.json')})
                settings.write(attempt/'research_result.json', {'completion':'COMPLETED','training_eligible':True,
                    'scene_setup_identity':request['scene_setup_identity'],
                    'episode':{'path':str(episode),'binding':{'path':str(binding),**identity(binding)}}})
                return {'success':True}
            with patch.object(workflow,'load_host',return_value=host), patch.object(workflow,'preflight',return_value={'status':'PASS'}), patch.object(workflow,'require_approval',return_value={}):
                result = workflow.execute_simulation(cfg,'record',Path(td)/'collection',limit=1,runner=fake_runner)
            self.assertEqual(result['training_episodes'],1)
            plan = training_plan(Path(td)/'collection/dataset.json',model_type='bc_cnn',updates=10000)
            self.assertEqual(plan['train_ids'],['R000059'])
            self.assertEqual(plan['script_options'],{'lift_extra_m':.02})
            self.assertEqual(plan['scene_setup_identity'],workflow.scene_setup_identity(cfg))
            self.assertFalse(plan['test_data_loaded'])


if __name__ == "__main__":
    unittest.main()
