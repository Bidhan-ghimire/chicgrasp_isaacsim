"""Offline regression tests grouped by original evaluation."""
from __future__ import annotations



# Launcher

"""Offline contract/integration tests. Child test processes are ordinary Python."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from unittest.mock import Mock

from pipeline.runtime import host as common
from pipeline import original_evaluation as launcher
from pipeline import original_evaluation as native_entry

def config(root):
    return dict(schema='hc-reproduction-host/v1',isaac_sim=str(root/'SDK with spaces'),
        isaac_lab=str(root/'Lab with spaces'),policy_support=str(root/'support'),
        checkpoint=str(root/'models/BEST.pt'),config_path=str(root/'host.json'))

class LauncherTests(unittest.TestCase):
    def test_every_frozen_scientific_binding_survives_relocation(self):
        old=common.load_json(common.ROOT/'pipeline/runtime/data/HC_BINDINGS.json')['allocations']
        c=config(Path('D:/different machine/root with spaces'))
        for point in common.ALLOCATION_ORDER:
            new=common.make_binding(c,point,checkpoint_verified=True)
            self.assertEqual(common.scientific_projection(old[point]),common.scientific_projection(new))
            self.assertEqual(new['checkpoint']['path'],c['checkpoint'])
            self.assertEqual(new['policy_support_path'],c['policy_support'])
            changes={k for k in new if new[k]!=old[point][k]}
            self.assertLessEqual(changes,{'checkpoint','checkpoint_identity','normalizer','scene_profile','policy_support_path','checkpoint_verified'})
        plan=launcher.make_plan(c)
        self.assertEqual(len(plan['allocations']),30)
        self.assertEqual([r['evaluation_seed'] for r in plan['allocations']],[15345]*10+[16345]*10+[17345]*10)
        self.assertFalse(plan['optional_demo_cameras_enabled'])

    def test_unknown_allocation_and_changed_binding_fail(self):
        with self.assertRaises(ValueError):common.make_binding(config(Path('.')),'HC99_R0')
        with patch.object(common,'sha256',return_value='0'*64):
            with self.assertRaisesRegex(ValueError,'bindings changed'):
                common.make_binding(config(Path('.')),'HC01_R0')

    def test_relocated_cli_plan_never_imports_runtime(self):
        with tempfile.TemporaryDirectory(prefix='HC repo with spaces ') as tmp:
            root=Path(tmp);checkout=root/'relocated checkout';checkout.mkdir()
            shutil.copy2(common.ROOT/'reproduce.py',checkout/'reproduce.py')
            shutil.copytree(common.ROOT/'pipeline',checkout/'pipeline',ignore=shutil.ignore_patterns('__pycache__'))
            c=config(root);common.write_json(root/'host.json',c)
            target=root/'allocation plan.json'
            call=[sys.executable,'-B','-s',str(checkout/'reproduce.py'),'plan','--config',str(root/'host.json'),'--output',str(target)]
            result=subprocess.run(call,cwd=root,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            plan=common.load_json(target)
            self.assertEqual(len(plan['allocations']),30)
            self.assertFalse(plan['simulator_started']);self.assertFalse(plan['model_loaded'])
            # The copy has no frozen_runtime, SDK or checkpoint: plan cannot have loaded them.
            self.assertFalse((checkout/'frozen_runtime').exists())
            second=subprocess.run(call,cwd=root,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(second.returncode,0)

    def test_relative_host_paths_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();p=root/'host.json'
            common.write_json(p,dict(schema='hc-reproduction-host/v1',isaac_sim='sim',isaac_lab='lab',policy_support='deps',checkpoint='BEST.pt'))
            self.assertEqual(common.load_config(p)['checkpoint'],str(root/'BEST.pt'))
            with self.assertRaises(FileExistsError):common.write_json(p,{})

    def test_native_entry_preserves_exact_scientific_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();attempt=root/'HC07_R0/native001';attempt.mkdir(parents=True)
            c=config(root);common.write_json(root/'host.json',c)
            inv=dict(r4_live_point='HC07_R0',output_directory=str(attempt/'capture'),timeout_ms=1200000,
                run_type='HARNESS',execution_approved=False,evidence_authority=False,stage3_learned=True,
                stage3_witness=True,envelope_v2=True,envelope_protocol='v3c')
            common.write_json(attempt/'invocation.json',inv)
            recorder=SimpleNamespace(main=Mock(return_value=0))
            with (patch.object(native_entry,'preflight',return_value={'status':'PASS'}) as pre,
                  patch.object(native_entry,'install',return_value=(recorder,{}))):
                self.assertEqual(native_entry.native_main(['--config',str(root/'host.json'),'--point','HC07_R0','--attempt',str(attempt),'--mode','execute']),0)
            pre.assert_called_once_with(common.load_config(root/'host.json'),include_checkpoint=True)
            self.assertEqual(recorder.main.call_args.args[0],['--execute','--output-dir',str(attempt/'capture/experiment'),
                '--journal',str(attempt/'capture/structured/events.jsonl'),'--seed','42','--geometric-mode',
                '--run-type','HARNESS','--stage3-learned','--stage3-witness','--r4-live-point','HC07_R0','--envelope-v3c'])

    def test_batch_stops_after_first_technical_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=config(root);output=root/'new run'
            with patch.object(launcher,'ROOT',root),patch.object(launcher,'preflight',return_value={'status':'PASS'}), \
                 patch.object(launcher.shutil,'disk_usage',return_value=SimpleNamespace(free=10*1024**3)), \
                 patch.object(launcher,'trial',return_value=dict(point='HC01_R0',status='TECHNICAL_STOP',actual_kit_starts=None)) as run:
                self.assertEqual(launcher.run_batch(c,output,runner=Mock()),1)
            self.assertEqual(run.call_count,1)
            saved=common.load_json(output/'RUN_RESULT.json')
            self.assertEqual(saved['completed_trials'],0);self.assertTrue(saved['start_accounting_unknown'])
            self.assertFalse((root/'.local/ACTIVE_RUN.lock').exists())

    def test_run_requires_explicit_execute(self):
        with self.assertRaises(SystemExit) as error:
            launcher.main(['run','--output','never-created'])
        self.assertEqual(error.exception.code,2)

    def test_failed_capture_cannot_qualify(self):
        value=dict(exit_code=0,owned_tree_gone_verified=True,capture_complete=True,actual_kit_starts=1)
        self.assertTrue(launcher.capture_clean(value))
        for key in ('timed_out','forced_termination','stop_requested','cleanup_budget_exceeded'):
            self.assertFalse(launcher.capture_clean(dict(value,**{key:True})))
        self.assertFalse(launcher.capture_clean(dict(value,actual_kit_starts=None)))


if __name__ == "__main__":
    unittest.main()
