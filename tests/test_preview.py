"""Offline regression tests grouped by preview."""
from __future__ import annotations



# Preview ui

import sys
import unittest
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import patch

from pipeline import scene as p


class PreviewControlsTests(unittest.TestCase):
    def test_buttons_only_set_requests_and_finish_starts_disabled(self):
        buttons, events = {}, []
        def button(text, **kw):
            buttons[text] = NS(**kw)
            return buttons[text]
        ui = NS(Window=lambda *a, **k: NS(frame=nullcontext(), destroy=lambda: events.append('destroy')),
                VStack=lambda **kw: nullcontext(), Label=lambda text, **kw: NS(text=text), Button=button)
        with patch.dict(sys.modules, {'omni': NS(ui=ui), 'omni.ui': ui}):
            c = p.create_controls()
        self.assertFalse(buttons['Finish preview'].enabled)
        for text, name in [('Start demonstration','start'), ('Stop preview','stop'), ('Finish preview','finish')]:
            self.assertFalse(c[name]); buttons[text].clicked_fn(); self.assertTrue(c[name])
        self.assertEqual(events, [])


class PreviewSessionTests(unittest.TestCase):
    def setUp(self):
        self.wall, self.physics, self.steps, self.events = 0., 0, [], []
        self.playing, self.running, self.callbacks = True, True, []
        self.controls = dict(start=False, stop=False, finish=False, window=NS(destroy=lambda:self.events.append('destroy')),
                            status=NS(text=''), start_button=NS(enabled=True), stop_button=NS(enabled=True),
                            finish_button=NS(enabled=False))
        def update():
            self.events.append('render')
            if self.callbacks: self.callbacks.pop(0)()
        def step(action):
            self.steps.append(action)
            self.physics += 2
            self.wall += .001
            return {'terminated':False}
        def forbidden_transition(): self.fail("preview must not pause/play the timeline")
        def bare_update(): self.fail("use sim.render to synchronize Fabric without physics")
        self.runtime = NS(env=NS(sim=NS(pause=forbidden_transition,play=forbidden_transition,
                                      render=update,is_playing=lambda:self.playing)),
                          app=NS(is_running=lambda:self.running, update=bare_update), clock=lambda:(self.physics,self.physics/120),
                          step=step, media_deadline_monotonic=1200)
        self.original_step=step
        self.cameras=NS(paths={'LEFT_HD':'/left'},before_render=lambda:self.events.append('camera_check'))
        self.session=p.PreviewSession(self.runtime,self.cameras,seconds=1,controls_factory=lambda:self.controls,
                                      monotonic=lambda:self.wall,sleep=self.sleep)
    def sleep(self, duration): self.wall += duration
    def start(self):
        self.callbacks.append(lambda:self.controls.update(start=True))
        self.session.wait_for_start()
        self.session.wrap_step()

    def test_wait_start_does_not_step_physics(self):
        self.start()
        self.assertEqual(self.physics,0)
        self.assertEqual(self.steps,[])
        self.assertTrue(self.playing)
        self.assertFalse(self.controls['start_button'].enabled)

    def test_full_run_exact_ticks_paced_and_finish_without_physics(self):
        self.start()
        for i in range(1200):
            before=self.wall
            self.assertEqual(self.runtime.step(i),{'terminated':False})
            self.assertGreaterEqual(self.wall-before,1/60-1e-10)
        self.assertEqual(self.steps,list(range(1200)))
        self.assertEqual(self.physics,2400)
        self.callbacks.append(lambda:self.controls.update(finish=True))
        self.session.finished()
        self.assertTrue(self.playing)
        self.assertEqual(self.physics,2400)
        self.assertEqual(self.session.record['completed_actions'],200)
        self.assertEqual(self.session.record['close_reason'],'FINISH_BUTTON')
        self.session.close()
        self.assertIs(self.runtime.step,self.original_step)
        self.assertIn('destroy',self.events)

    def test_slow_tick_does_not_cause_next_tick_to_catch_up(self):
        self.start()
        self.wall += 8
        before=self.wall
        self.runtime.step(1)
        self.assertGreaterEqual(self.wall-before,1/60-1e-10)

    def test_idle_wait_renders_and_keeps_initialized_timeline(self):
        self.callbacks.extend([lambda:None, lambda:None, lambda:self.controls.update(start=True)])
        self.session.wait_for_start()
        self.assertEqual(self.events.count('render'),3)
        self.assertTrue(self.playing)
        self.assertEqual(self.physics,0)
        self.assertEqual(self.steps,[])

    def test_manual_timeline_pause_during_wait_never_autoplays(self):
        self.playing=False
        with self.assertRaisesRegex(p.PreviewStopped,'timeline_paused'):
            self.session.wait_for_start()
        self.assertEqual(self.physics,0)
        self.assertEqual(self.events,[])

    def test_stop_before_start_executes_nothing(self):
        self.controls['stop']=True
        with self.assertRaises(p.PreviewStopped):self.session.wait_for_start()
        self.assertEqual(self.physics,0)
        self.session.close()
        self.assertEqual(self.session.record['status'],'INCOMPLETE')

    def test_stop_midrun_has_no_extra_step(self):
        self.start();self.runtime.step(0);self.controls['stop']=True
        with self.assertRaises(p.PreviewStopped):self.runtime.step(1)
        self.assertEqual(self.steps,[0])

    def test_timeline_pause_aborts_before_entering_blocking_step(self):
        self.start();self.playing=False
        with self.assertRaisesRegex(p.PreviewStopped,'timeline_paused'):self.runtime.step(0)
        self.assertEqual(self.physics,0)

    def test_incomplete_cannot_finish(self):
        self.start();self.runtime.step(0)
        with self.assertRaisesRegex(RuntimeError,'1200'):self.session.finished()
        self.assertNotEqual(self.session.record['status'],'COMPLETED')

    def test_window_closed_during_run_aborts_without_step(self):
        self.start();self.running=False
        with self.assertRaises(p.PreviewStopped):self.runtime.step(0)
        self.assertEqual(self.steps,[])

    def test_idle_wait_detects_unrequested_physics(self):
        def unsafe():self.physics+=1
        self.callbacks.append(unsafe)
        with self.assertRaisesRegex(RuntimeError,'idle_wait'):self.session.wait_for_start()

    def test_wait_timeout_never_autoplays(self):
        with self.assertRaisesRegex(p.PreviewStopped,'no_actions_executed'):self.session.wait_for_start()
        self.assertEqual(self.physics,0)

    def test_existing_camera_guard_failure_stops_before_command(self):
        self.start()
        def fail():raise ValueError('policy camera changed')
        self.cameras.before_render=fail
        with self.assertRaisesRegex(ValueError,'policy camera changed'):self.runtime.step(0)
        self.assertEqual(self.steps,[])

    def test_configuration_allows_only_gui_readbacks(self):
        before=dict(physics_dt_s=1/120,has_gui=False,has_gui_raw=None,offscreen_render=True)
        after=dict(before,has_gui=True,has_gui_raw=True,offscreen_render=False)
        self.assertEqual(len(p.validate_preview_configuration(after,before)),3)
        after['physics_dt_s']=1/60
        with self.assertRaisesRegex(ValueError,'physics'):p.validate_preview_configuration(after,before)


# Script preview native

"""Offline preview integration through the native boundary; no Kit imports."""
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from pipeline.runtime import native as backend
from pipeline import scene as preview_ui
from tests import test_runtime as fixtures


class PreviewRequestTests(unittest.TestCase):
    def test_preview_requires_script_but_no_model(self):
        value = fixtures.request("preview")
        value["evaluation_seed"] = None
        validated = backend.validate_request(value, Path.cwd())
        self.assertEqual(validated["mode"], "preview")
        self.assertEqual(validated["script"]["sha256"], "a" * 64)
        self.assertNotIn("checkpoint", validated)
        self.assertFalse(validated["video"]["enabled"])

    def test_preview_missing_script_fails_before_launch(self):
        value = fixtures.request("preview")
        value.pop("script")
        with self.assertRaisesRegex(ValueError, "script_path_required"):
            backend.validate_request(value, Path.cwd())

    def test_preview_rejects_video_encoding_request(self):
        value = fixtures.request("preview")
        value["video"]["enabled"] = True
        with self.assertRaises(ValueError):
            backend.validate_request(value, Path.cwd())


class PreviewNativeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.NativeOrchestrationTests()
        self.fixture.setUp()
        self.calls = self.fixture.calls
        self.launches, self.captures, self.previews = [], [], []
        self.wait_error = None
        self.changed_physics = False
        test = self
        original_runtime = self.fixture.install[1].RuntimeSession

        class Runtime(original_runtime):
            def __init__(self, output, **kwargs):
                super().__init__(output, **kwargs)
                test.launches.append(dict(kwargs))
                if kwargs.get("stage0_mode") == "TUNE_GUI":
                    self.effective_configuration.update(
                        has_gui=True, has_gui_raw=True, offscreen_render=False)
                if test.changed_physics:
                    self.effective_configuration["physics"] = "CHANGED"

        self.fixture.install[1].RuntimeSession = Runtime
        package = self.fixture.modules["chicgrasp_policy"]
        original_make = package.geometric_capture.make_capture

        def make_capture(core, runtime, writer, action_source):
            capture = original_make(core, runtime, writer, action_source)
            test.captures.append(capture)
            return capture

        package.geometric_capture.make_capture = make_capture
        real_write = self.fixture.install[0].write_json

        def save(path, value):
            real_write(path, value)
            if Path(path).name == "research_result.json":
                test.calls.append("durable_result")

        self.fixture.install[0].write_json = save

        class Preview:
            def __init__(self, runtime, cameras, *, seconds):
                self.runtime = runtime
                self.record = dict(status="READY", saved_as_training_episode=False)
                test.previews.append(self)
                test.calls.append("preview_create")

            def wait_for_start(self):
                test.calls.append("preview_wait")
                if test.wait_error:
                    raise test.wait_error
                self.record["status"] = "RUNNING"

            def wrap_step(self):
                test.calls.append("preview_wrap")

            def finished(self):
                test.calls.append("preview_finished")
                self.record["status"] = "COMPLETED"

            def close(self):
                test.calls.append("preview_close")

        self.preview_class = Preview

    def run_preview(self, **kwargs):
        camera = NS(paths={"LEFT_HD": "/left", "RIGHT_HD": "/right", "HAND": "/hand"})
        with patch.object(preview_ui, "PreviewSession", self.preview_class), \
                patch.object(backend, "create_scene_cameras", return_value=(camera, [], {})) as create:
            result = self.fixture.execute(mode="preview", **kwargs)
            if create.called:
                self.assertEqual(create.call_args.kwargs, {"pause": False})
            return result

    def test_full_preview_uses_gui_exact_script_and_saves_before_close(self):
        result = self.run_preview()
        self.assertEqual(result["completion"], "COMPLETED")
        self.assertEqual(self.launches[0]["stage0_mode"], "TUNE_GUI")
        runtime = self.previews[0].runtime
        self.assertTrue(runtime.stage1_demo)
        self.assertFalse(runtime.stage3_learned)
        self.assertEqual(result["counts"], dict(
            completed_actions=200, selected_actions=200,
            observed_substeps=2400, completed_control_ticks=1200))
        self.assertEqual(self.captures[0].executed_actions, [[0., 0., .5, 1., 1.]] * 200)
        self.assertIsNone(result["episode"])
        self.assertFalse(result["training_eligible"])
        self.assertIsNone(result["media"])
        for unexpected in ("episode", "inspect_new_checkpoint", "load_new_predictor", "save_visuals"):
            self.assertNotIn(unexpected, self.calls)
        self.assertLess(self.calls.index("preview_wait"), self.calls.index("capture_run"))
        self.assertLess(self.calls.index("preview_wrap"), self.calls.index("capture_run"))
        self.assertLess(self.calls.index("capture_run"), self.calls.index("preview_finished"))
        self.assertLess(self.calls.index("preview_close"), self.calls.index("close_app"))
        self.assertLess(self.calls.index("durable_result"), self.calls.index("close_app"))

    def test_physical_miss_is_distinct_from_completed_preview(self):
        self.fixture.objective_status = "FAIL"
        result = self.run_preview()
        self.assertEqual(result["completion"], "COMPLETED")
        self.assertEqual(result["objective_metrics"]["status"], "FAIL")
        self.assertFalse(result["training_eligible"])
        self.assertIsNone(result["episode"])

    def test_stop_before_start_never_runs_script_and_still_closes(self):
        self.wait_error = RuntimeError("owner_stopped_preview_before_start")
        result = self.run_preview()
        self.assertEqual(result["completion"], "FAILED")
        self.assertEqual(result["counts"]["completed_actions"], 0)
        self.assertFalse(result["training_eligible"])
        self.assertNotIn("capture_run", self.calls)
        self.assertNotIn("episode", self.calls)
        self.assertIn("preview_close", self.calls)
        self.assertIn("close_app", self.calls)
        self.assertTrue(any("owner_stopped_preview" in x["message"] for x in result["failures"]))

    def test_interrupted_capture_cannot_become_episode_or_completed_preview(self):
        self.fixture.capture_exception = RuntimeError("owner_stopped_preview_during_demo")
        result = self.run_preview()
        self.assertEqual(result["completion"], "FAILED")
        self.assertFalse(result["training_eligible"])
        self.assertIsNone(result["episode"])
        self.assertNotIn("preview_finished", self.calls)
        self.assertIn("preview_close", self.calls)
        self.assertLess(self.calls.index("durable_result"), self.calls.index("close_app"))

    def test_gui_allowance_does_not_accept_changed_physics(self):
        self.changed_physics = True
        result = self.run_preview()
        self.assertEqual(result["completion"], "FAILED")
        self.assertNotIn("capture_run", self.calls)
        self.assertFalse(result["training_eligible"])
        self.assertIsNone(result["episode"])
        self.assertIn("close_app", self.calls)

    def test_failed_preflight_never_launches_or_constructs_preview(self):
        result = self.run_preview(bad_preflight=True)
        self.assertEqual(result["completion"], "FAILED")
        self.assertEqual(self.launches, [])
        self.assertEqual(self.previews, [])
        self.assertNotIn("capture_run", self.calls)

    def test_recording_mode_keeps_headless_episode_path(self):
        result = self.fixture.execute(mode="scripted")
        self.assertEqual(result["completion"], "COMPLETED")
        self.assertNotIn("stage0_mode", self.launches[0])
        self.assertTrue(result["training_eligible"])
        self.assertIsNotNone(result["episode"])
        self.assertNotIn("preview_create", self.calls)


# Script preview workflow

"""Offline preview dispatch checks. No simulator, model or GPU is started."""
from contextlib import ExitStack
from contextlib import redirect_stdout
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pipeline import settings
from pipeline.runtime import workflow


class ScriptPreviewWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.cfg = dict(settings.load_experiment(), scene_approval=str(self.root/'scene-approval001.json'))
        self.host = {'isaac_sim':'C:/unused', 'isaac_lab':'C:/unusedlab',
                     'policy_support':'C:/unusedsupport', 'config_path':'C:/unused.json'}
        settings.write(Path(self.cfg['scene_approval']), {'OWNER_APPROVED':True,
                       'scene_identity':settings.scene_identity(self.cfg)})
        self.calls = []
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(workflow,'load_host',return_value=self.host))
        self.stack.enter_context(patch.object(workflow,'preflight',return_value={'status':'PASS'}))

    def child(self, command, **kwargs):
        request = settings.read(command[command.index('--request')+1])
        attempt = Path(command[command.index('--attempt')+1]); attempt.mkdir(parents=True)
        self.calls.append((command, kwargs, request))
        self.assertEqual(request['mode'], 'preview')
        self.assertFalse(request['video']['enabled'])
        self.assertIsNone(request['checkpoint'])
        actual = np.load(request['script']['path'], allow_pickle=False)
        np.testing.assert_array_equal(actual, workflow.compiled_script(self.cfg, request['position']))
        settings.write(attempt/'research_result.json', {'completion':'COMPLETED',
            'scene_setup_identity':request['scene_setup_identity'], 'training_eligible':False,
            'episode':None, 'media':None})
        return {'success':True}

    def cli(self, args, mode='script'):
        stream = io.StringIO()
        with patch.object(workflow,'load_experiment',return_value=self.cfg), \
             patch.object(sys,'argv',['scripted_run.py',*args]), redirect_stdout(stream):
            workflow.main(mode)
        return json.loads(stream.getvalue())

    def test_default_command_is_inert_plan_and_preview_plan_is_one_start(self):
        with patch.object(workflow,'execute_simulation',side_effect=AssertionError('must not execute')):
            plan = self.cli(['--position','R000002'])
        self.assertEqual(plan['mode'],'script')
        self.assertEqual(plan['planned_kit_starts'],0)
        self.assertFalse(plan['executed'])
        preview = workflow.make_plan(self.cfg,'preview')
        self.assertEqual(preview['mode'],'preview')
        self.assertEqual(preview['planned_kit_starts'],1)
        self.assertEqual(len(preview['positions']),1)
        self.assertEqual(preview['video'],{'enabled':False})
        self.assertEqual(preview['action_count'],200)

    def test_preview_flag_dispatches_with_default_output_and_execute_alias(self):
        with patch.object(workflow,'execute_simulation',return_value={'completion':'COMPLETED'}) as execute:
            self.cli(['--preview','--position','R000002'])
            call = execute.call_args
            self.assertEqual(call.args[1],'preview')
            self.assertEqual(Path(call.args[2]), self.root/'preview001')
            self.assertEqual(call.kwargs['position_id'],'R000002')
            self.cli(['--execute','--output',str(self.root/'explicit001')])
            self.assertEqual(execute.call_args.args[1],'preview')
            self.assertEqual(Path(execute.call_args.args[2]),self.root/'explicit001')

    def test_preview_rejects_flags_for_other_modes_without_dispatch(self):
        cases = [('scene',['--preview']), ('live',['--preview']), ('record',['--preview']),
                 ('script',['--preview','--limit','2']), ('script',['--preview','--seed','15345']),
                 ('script',['--preview','--artifact','model.json']), ('script',['--preview','--dataset','data.json']),
                 ('script',['--preview','--approve','scene001']), ('script',['--preview','--updates','100']),
                 ('script',['--preview','--resume','LAST.pt']), ('script',['--preview','--selection-seed','1'])]
        for mode,args in cases:
            with self.subTest(mode=mode,args=args), redirect_stderr(io.StringIO()), \
                 patch.object(workflow,'execute_simulation') as execute, self.assertRaises(SystemExit) as error:
                self.cli(args, mode)
            self.assertEqual(error.exception.code,2)
            execute.assert_not_called()

    def test_one_native_preview_exact_commands_containment_and_no_dataset(self):
        with patch.object(workflow,'checkpoint_record',side_effect=AssertionError('no model')):
            result = workflow.execute_simulation(self.cfg,'preview',self.root/'preview001',
                                                 position_id='R000002',runner=self.child)
        self.assertEqual(result['completion'],'COMPLETED')
        self.assertEqual(result['training_episodes'],0)
        self.assertEqual(result['runs_completed'],1)
        self.assertEqual(len(self.calls),1)
        _,options,_ = self.calls[0]
        self.assertEqual(options['timeout_seconds'],self.cfg['limits']['per_simulator_seconds'])
        self.assertEqual(options['cleanup_seconds'],self.cfg['limits']['cleanup_seconds'])
        self.assertFalse(list(self.root.rglob('dataset.json')))
        self.assertFalse(list(self.root.rglob('artifact.json')))

    def test_repeat_preview_preserves_scene_pointer_approval_and_previous_output(self):
        pointer = self.root/'scene001.latest.json'; pointer.write_bytes(b'untouched scene pointer')
        approval_before = Path(self.cfg['scene_approval']).read_bytes()
        first = workflow.execute_simulation(self.cfg,'preview',self.root/'preview001',runner=self.child)
        before = {p:p.read_bytes() for p in Path(first['output_directory']).rglob('*') if p.is_file()}
        second = workflow.execute_simulation(self.cfg,'preview',self.root/'preview001',runner=self.child)
        self.assertEqual(Path(second['output_directory']),self.root/'preview002')
        self.assertEqual(pointer.read_bytes(),b'untouched scene pointer')
        self.assertEqual(Path(self.cfg['scene_approval']).read_bytes(),approval_before)
        for p,value in before.items(): self.assertEqual(p.read_bytes(),value)
        self.assertFalse((self.root/'preview001.latest.json').exists())

    def test_missing_or_stale_approval_prevents_output_creation_or_child(self):
        for approved in (False,True):
            Path(self.cfg['scene_approval']).write_text(json.dumps({'OWNER_APPROVED':approved,'scene_identity':{}}))
            with self.assertRaisesRegex(ValueError,'missing or stale'):
                workflow.execute_simulation(self.cfg,'preview',self.root/'preview001',runner=self.child)
            self.assertFalse(self.calls)
            self.assertFalse((self.root/'preview001').exists())

    def test_selection_is_explicit_and_preview_cannot_be_batch(self):
        self.assertEqual(workflow.make_plan(self.cfg,'preview',position_id='R000002')['positions'][0]['id'],'R000002')
        with self.assertRaisesRegex(ValueError,'Unknown position'):
            workflow.make_plan(self.cfg,'preview',position_id='does-not-exist')
        with self.assertRaisesRegex(ValueError,'one position'):
            workflow.make_plan(self.cfg,'preview',limit=2)

    def test_incomplete_native_result_is_not_success_or_training_data(self):
        def stopped(command, **kwargs):
            self.child(command,**kwargs)
            attempt = Path(command[command.index('--attempt')+1])
            result = settings.read(attempt/'research_result.json')
            result.update(completion='STOPPED', stop_reason='STOP_PREVIEW')
            (attempt/'research_result.json').write_text(json.dumps(result))
            return {'success':True}
        result = workflow.execute_simulation(self.cfg,'preview',self.root/'preview001',runner=stopped)
        self.assertEqual(result['completion'],'FAILED')
        self.assertEqual(result['training_episodes'],0)
        self.assertEqual(settings.read(result['results'][0])['stop_reason'],'STOP_PREVIEW')
        self.assertFalse(list(self.root.rglob('dataset.json')))


if __name__ == "__main__":
    unittest.main()
