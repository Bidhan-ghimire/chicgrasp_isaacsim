"""Offline regression tests grouped by runtime."""
from __future__ import annotations



# Owned process

"""Real Windows stub-process checks; never imports Kit, Torch, or CUDA."""

import ctypes
import ast
from ctypes import wintypes
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from pipeline.runtime import process as owned_process


def _pid_is_gone(pid):
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenProcess.restype = wintypes.HANDLE
    api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    api.WaitForSingleObject.restype = wintypes.DWORD
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    handle = api.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE only
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:
            return True
        raise ctypes.WinError(error)
    try:
        return api.WaitForSingleObject(handle, 0) == 0
    finally:
        api.CloseHandle(handle)


@unittest.skipUnless(os.name == "nt", "Windows job objects required")
class OwnedProcessTests(unittest.TestCase):
    receipts = []
    original_checker_results = []

    @classmethod
    def setUpClass(cls):
        parent = os.environ.get("CHICGRASP_OWNED_PROCESS_TEST_ROOT")
        if not parent:
            raise RuntimeError("Set CHICGRASP_OWNED_PROCESS_TEST_ROOT to an authorized existing output folder")
        cls.root = Path(tempfile.mkdtemp(prefix="owned process tests ", dir=parent))

    def directory(self, name):
        return self.root/name

    def run_stub(self, name, code, *, timeout_seconds=8, stop_file=None):
        directory = self.directory(name)
        receipt = owned_process.run_owned(
            [sys.executable, "-B", "-c", code, str(directory)],
            cwd=self.root, env=os.environ.copy(), directory=directory,
            timeout_seconds=timeout_seconds, cleanup_seconds=5, stop_file=stop_file)
        self.receipts.append({"case": name, "receipt": str(directory/"receipt.json"), "result": receipt})
        saved = json.loads((directory/"receipt.json").read_text())
        self.assertEqual(saved, receipt)
        for stream in ("stdout", "stderr"):
            data = Path(receipt[stream]["path"]).read_bytes()
            self.assertEqual(receipt[stream]["bytes"], len(data))
            self.assertEqual(receipt[stream]["sha256"], hashlib.sha256(data).hexdigest())
        self.assertTrue(receipt["tree_empty_verified"])
        self.assertTrue(receipt["root_exited_verified"])
        self.assertTrue(receipt["owned_tree_gone_verified"])
        self.assertLessEqual(receipt["cleanup_elapsed_ms"], 5000)
        if receipt["root_pid"] is not None:
            self.assertTrue(_pid_is_gone(receipt["root_pid"]))
        return receipt

    def test_normal_success_and_native_job_membership(self):
        code = '''import ctypes,json,os,sys
from ctypes import wintypes
from pathlib import Path
s=json.loads((Path(sys.argv[1])/'started.json').read_text())
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.OpenJobObjectW.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.LPCWSTR]; k.OpenJobObjectW.restype=wintypes.HANDLE
k.GetCurrentProcess.argtypes=[]; k.GetCurrentProcess.restype=wintypes.HANDLE
k.IsProcessInJob.argtypes=[wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)]; k.IsProcessInJob.restype=wintypes.BOOL
k.CloseHandle.argtypes=[wintypes.HANDLE]; k.CloseHandle.restype=wintypes.BOOL
h=k.OpenJobObjectW(4,False,s['job_name']); assert h
try:
 b=wintypes.BOOL(False); assert k.IsProcessInJob(k.GetCurrentProcess(),h,ctypes.byref(b)); assert b.value
 assert s['assigned_before_resume'] and not s['resumed']
 print(json.dumps({'native_job_membership':b.value,'actual_pid':os.getpid(),'receipt_written_before_resume':True}))
 sys.stderr.buffer.write(b'diagnostic\\n')
finally:k.CloseHandle(h)
'''
        r=self.run_stub("normal",code)
        self.assertTrue(r["success"])
        self.assertEqual(r["exit_code"],0)
        self.assertTrue(r["assigned_before_resume"])
        self.assertTrue(r["resumed"])
        self.assertFalse(r["forced_termination"])
        self.assertFalse(r["timed_out"])
        self.assertTrue(json.loads(Path(r["stdout_path"]).read_text())["native_job_membership"])

    def test_nonzero_exit_is_failure_without_force(self):
        r=self.run_stub("exit7","import sys;sys.stdout.buffer.write(b'failure stdout\\n');sys.stderr.buffer.write(b'failure stderr\\n');sys.exit(7)")
        self.assertEqual(r["exit_code"],7)
        self.assertFalse(r["success"])
        self.assertFalse(r["forced_termination"])
        self.assertFalse(r["timed_out"])

    def test_timeout_contains_root_and_descendant(self):
        code="import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)']);Path(sys.argv[1],'descendant.pid').write_text(str(p.pid));print(p.pid,flush=True);time.sleep(30)"
        r=self.run_stub("timeout_descendant",code,timeout_seconds=2)
        child=int((self.directory("timeout_descendant")/"descendant.pid").read_text())
        self.assertTrue(_pid_is_gone(child))
        self.assertTrue(r["timed_out"])
        self.assertTrue(r["forced_termination"])
        self.assertFalse(r["success"])
        self.assertLess(r["elapsed_seconds"],7.5)

    def test_lingering_descendant_reaches_timeout_then_forced_cleanup(self):
        code="import subprocess,sys;from pathlib import Path;p=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)']);Path(sys.argv[1],'descendant.pid').write_text(str(p.pid));print(p.pid,flush=True)"
        r=self.run_stub("root_exit_descendant",code,timeout_seconds=.75)
        child=int((self.directory("root_exit_descendant")/"descendant.pid").read_text())
        self.assertTrue(_pid_is_gone(child))
        self.assertEqual(r["exit_code"],0)
        self.assertTrue(r["forced_termination"])
        self.assertFalse(r["success"])
        self.assertTrue(r["timed_out"])

    def test_descendant_naturally_finishes_after_parent_exit(self):
        code="import subprocess,sys;from pathlib import Path;p=subprocess.Popen([sys.executable,'-B','-c',\"import time;time.sleep(.25);print('descendant completed',flush=True)\"]);Path(sys.argv[1],'descendant.pid').write_text(str(p.pid));print('root completed',flush=True)"
        r=self.run_stub("natural_descendant",code)
        child=int((self.directory("natural_descendant")/"descendant.pid").read_text())
        self.assertTrue(_pid_is_gone(child))
        self.assertTrue(r["success"])
        self.assertTrue(r["autonomous_tree_exit"])
        self.assertFalse(r["forced_termination"])
        self.assertFalse(r["timed_out"])
        self.assertFalse(r["parent_termination_requested"])
        self.assertGreaterEqual(r["job_empty_observed_ms"]-r["root_exit_observed_ms"],180)
        self.assertIn("descendant completed",Path(r["stdout_path"]).read_text())

    def test_exact_original_checker_qualifies_normal_stub_receipt(self):
        r=self.run_stub("original_checker","print('complete',flush=True)")
        # This is an actual read of the final saved receipt, after run_owned has
        # returned. The launcher supplies its journal-read timestamp similarly.
        process=json.loads((self.directory("original_checker")/"receipt.json").read_text())
        process["final_journal_read_completed_ms"]=math.ceil((time.monotonic()-r["started_monotonic"])*1000)
        process["duration_seconds"]=time.monotonic()-r["started_monotonic"]
        source_path=Path(__file__).resolve().parents[1]/"frozen_runtime/src/chicgrasp_policy/stage3_check.py"
        source=source_path.read_text(encoding="utf-8")
        tree=ast.parse(source)
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="check_process")
        # Execute only the original pure checker function, with its sole math
        # dependency; importing the scientific checker module is unnecessary.
        namespace={"math":math}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(source_path),"exec"),namespace)
        issues=namespace["check_process"](process,absolute_ms=8000)
        proof={"source_path":str(source_path),"source_sha256":hashlib.sha256(source_path.read_bytes()).hexdigest(),
               "function":"check_process","method":"Unmodified original AST function on actual native Windows stub receipt", "issues":issues,"process":process}
        proof_path=self.directory("original_checker")/"checker_qualification.json"
        proof_path.write_text(json.dumps(proof,indent=2)+"\n",encoding="utf-8")
        self.original_checker_results.append({"proof_path":str(proof_path),"issues":issues,"source_sha256":proof["source_sha256"]})
        self.assertEqual(issues,[])
        for field in ("root_exit_observed_ms","job_empty_observed_ms","drain_completed_observed_ms","stream_files_finalized_ms","job_disposed_ms"):
            self.assertIs(type(process[field]),int)
            self.assertGreaterEqual(process[field],0)
            self.assertLessEqual(process[field],process["final_journal_read_completed_ms"])

    def test_stop_file_terminates_owned_child(self):
        directory=self.directory("stop_file")
        stop=directory/"stop.flag"
        def request_stop():
            end=time.monotonic()+4
            while not (directory/"ready").exists() and time.monotonic()<end:
                time.sleep(.01)
            if (directory/"ready").exists():
                stop.write_text("owner stop")
        worker=threading.Thread(target=request_stop,daemon=True)
        worker.start()
        r=self.run_stub("stop_file","import sys,time;from pathlib import Path;Path(sys.argv[1],'ready').write_text('ready');time.sleep(30)",stop_file=stop)
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.assertTrue(r["stop_requested"])
        self.assertFalse(r["timed_out"])
        self.assertTrue(r["forced_termination"])
        self.assertFalse(r["success"])

    def test_assignment_failure_never_executes_child(self):
        with mock.patch.object(owned_process.OwnedJob,"assign",side_effect=OSError("injected assignment failure")):
            r=self.run_stub("assignment_failure","import sys;from pathlib import Path;Path(sys.argv[1],'executed').write_text('unsafe')")
        self.assertFalse((self.directory("assignment_failure")/"executed").exists())
        self.assertFalse(r["assigned_before_resume"])
        self.assertFalse(r["resumed"])
        self.assertTrue(r["forced_termination"])
        self.assertFalse(r["success"])
        self.assertEqual(r["errors"][0]["message"],"injected assignment failure")

    def test_resume_failure_never_executes_child(self):
        with mock.patch.object(owned_process,"_resume_process",side_effect=OSError("injected resume failure")):
            r=self.run_stub("resume_failure","import sys;from pathlib import Path;Path(sys.argv[1],'executed').write_text('unsafe')")
        self.assertFalse((self.directory("resume_failure")/"executed").exists())
        self.assertTrue(r["assigned_before_resume"])
        self.assertFalse(r["resumed"])
        self.assertTrue(r["forced_termination"])
        self.assertFalse(r["success"])

    def test_preexisting_stop_never_launches(self):
        stop=self.root/"preexisting.stop"
        stop.write_text("stop before dispatch")
        with mock.patch.object(owned_process.subprocess,"Popen") as popen:
            r=self.run_stub("preexisting_stop","raise RuntimeError('must not run')",stop_file=stop)
        popen.assert_not_called()
        self.assertTrue(r["stop_requested"])
        self.assertFalse(r["resumed"])
        self.assertFalse(r["forced_termination"])
        self.assertIsNone(r["root_pid"])

    def test_log_read_error_preserves_failed_receipt_after_cleanup(self):
        directory=self.directory("log_read_failure")
        with mock.patch.object(owned_process,"_identity",side_effect=OSError("injected log read error")):
            r=owned_process.run_owned([sys.executable,"-B","-c","pass"],cwd=self.root,env=os.environ.copy(),directory=directory)
        self.receipts.append({"case":"log_read_failure","receipt":str(directory/"receipt.json"),"result":r})
        self.assertEqual(json.loads((directory/"receipt.json").read_text()),r)
        self.assertTrue(r["owned_tree_gone_verified"])
        self.assertEqual(r["exit_code"],0)
        self.assertIsNone(r["stdout_sha256"])
        self.assertIsNone(r["stderr_sha256"])
        self.assertFalse(r["success"])
        self.assertFalse(r["forced_termination"])
        self.assertTrue(_pid_is_gone(r["root_pid"]))

    def test_existing_directory_is_untouched_and_no_launch(self):
        directory=self.directory("already_exists")
        directory.mkdir()
        sentinel=directory/"keep.txt"
        sentinel.write_bytes(b"preserve me\x00\xff")
        with mock.patch.object(owned_process.subprocess,"Popen") as popen:
            with self.assertRaises(FileExistsError):
                owned_process.run_owned([sys.executable,"-B","-c","raise RuntimeError('must not run')"],cwd=self.root,env=os.environ.copy(),directory=directory)
        popen.assert_not_called()
        self.assertEqual(sentinel.read_bytes(),b"preserve me\x00\xff")
        self.assertEqual([p.name for p in directory.iterdir()],["keep.txt"])

    def test_invalid_limits_fail_before_launch(self):
        for name,value in [("cleanup_seconds",5.01),("cleanup_seconds",0),("timeout_seconds",1201),("timeout_seconds",float('nan'))]:
            with self.subTest(name=name,value=value),mock.patch.object(owned_process.subprocess,"Popen") as popen:
                with self.assertRaises(ValueError):
                    owned_process.run_owned([sys.executable,"-B","-c","pass"],cwd=self.root,env=os.environ.copy(),directory=self.directory("invalid"),**{name:value})
                popen.assert_not_called()
        self.assertFalse(self.directory("invalid").exists())


# Training process

"""Unbounded training supervision with Python stubs/fake clocks, never training."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from pipeline.runtime import process as owned_process



@unittest.skipUnless(os.name == "nt", "Windows job objects required")
class TrainingProcessTests(unittest.TestCase):
    receipts = []

    @classmethod
    def setUpClass(cls):
        parent = os.environ.get("CHICGRASP_OWNED_PROCESS_TEST_ROOT")
        if not parent:
            raise RuntimeError("Set CHICGRASP_OWNED_PROCESS_TEST_ROOT to an authorized output folder")
        cls.root = Path(tempfile.mkdtemp(prefix="training process tests ", dir=parent))

    def run_stub(self, name, code, *, stop_file=None):
        directory = self.root/name
        receipt = owned_process.run_training_owned(
            [sys.executable, "-B", "-c", code, str(directory)], cwd=self.root,
            env=os.environ.copy(), directory=directory, timeout_seconds=None,
            cleanup_seconds=5, stop_file=stop_file)
        self.receipts.append({"case":name,"receipt":str(directory/"receipt.json"),"result":receipt})
        self.assertEqual(json.loads((directory/"receipt.json").read_text()),receipt)
        self.assertEqual(receipt["runtime_limit_policy"],"none")
        for field in ("timeout_seconds","timeout_ms","absolute_deadline_ms","effective_deadline_ms"):
            self.assertIsNone(receipt[field])
        self.assertFalse(receipt["timed_out"])
        self.assertEqual(receipt["termination_and_drain_cap_ms"],5000)
        self.assertTrue(receipt["owned_tree_gone_verified"])
        self.assertLessEqual(receipt["cleanup_elapsed_ms"],5000)
        if receipt["root_pid"] is not None:
            self.assertTrue(_pid_is_gone(receipt["root_pid"]))
        for name in ("stdout","stderr"):
            data=Path(receipt[name]["path"]).read_bytes()
            self.assertEqual(receipt[name]["bytes"],len(data))
            self.assertEqual(receipt[name]["sha256"],hashlib.sha256(data).hexdigest())
        return receipt

    def test_normal_success_owned_before_resume_and_null_deadlines(self):
        code = '''import ctypes,json,sys
from ctypes import wintypes
from pathlib import Path
s=json.loads((Path(sys.argv[1])/'started.json').read_text())
assert s['assigned_before_resume'] and not s['resumed']
assert s['timeout_seconds'] is None and s['runtime_limit_policy']=='none'
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.GetCurrentProcess.restype=wintypes.HANDLE
k.IsProcessInJob.argtypes=[wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)]
k.OpenJobObjectW.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.LPCWSTR]
k.OpenJobObjectW.restype=wintypes.HANDLE
k.CloseHandle.argtypes=[wintypes.HANDLE]
h=k.OpenJobObjectW(4,False,s['job_name']);assert h
try:
 b=wintypes.BOOL(False);assert k.IsProcessInJob(k.GetCurrentProcess(),h,ctypes.byref(b));assert b.value
 print('verified native training job membership',flush=True)
finally:k.CloseHandle(h)
'''
        receipt=self.run_stub("normal",code)
        self.assertTrue(receipt["success"])
        self.assertTrue(receipt["autonomous_tree_exit"])
        self.assertFalse(receipt["forced_termination"])
        self.assertEqual(receipt["status"],"COMPLETE")

    def test_nonzero_exit_is_failure_without_timeout(self):
        receipt=self.run_stub("nonzero","import sys;print('failed normally');sys.exit(7)")
        self.assertEqual(receipt["exit_code"],7)
        self.assertEqual(receipt["status"],"FAILED")
        self.assertFalse(receipt["forced_termination"])

    def test_two_day_mock_elapsed_time_does_not_create_deadline(self):
        clock=SimpleNamespace(now=100.)
        child=SimpleNamespace(pid=424242,poll=lambda:0)
        class Job:
            name="mock-owned-training-job"
            def assign(self, process):
                self.assigned=process
            def active(self): return 0
            def terminate(self): raise AssertionError("No runtime deadline should terminate")
            def close(self): pass
        job=Job()
        def resume(process):
            self.assertIs(job.assigned,process)
            clock.now += 2*24*60*60
        directory=self.root/"two_days_mock"
        with mock.patch.object(owned_process,"OwnedJob",return_value=job), \
             mock.patch.object(owned_process.subprocess,"Popen",return_value=child) as popen, \
             mock.patch.object(owned_process,"_resume_process",side_effect=resume), \
             mock.patch.object(owned_process.time,"monotonic",side_effect=lambda:clock.now):
            receipt=owned_process.run_training_owned(["fake-only"],cwd=self.root,
                env={},directory=directory)
        self.receipts.append({"case":"two_days_mock","receipt":str(directory/"receipt.json"),"result":receipt})
        self.assertEqual(receipt["elapsed_seconds"],172800)
        self.assertIsNone(receipt["effective_deadline_ms"])
        self.assertTrue(receipt["success"])
        self.assertFalse(receipt["timed_out"])
        self.assertFalse(receipt["forced_termination"])
        self.assertTrue(popen.call_args.kwargs["creationflags"] & 0x4)

    def test_explicit_stop_contains_root_and_descendant(self):
        directory=self.root/"stop_descendant"
        stop=directory/"STOP"
        def request_stop():
            end=time.monotonic()+5
            while not (directory/"descendant.pid").exists() and time.monotonic()<end:
                time.sleep(.01)
            if (directory/"descendant.pid").exists(): stop.write_text("owner stop")
        thread=threading.Thread(target=request_stop,daemon=True)
        thread.start()
        code="import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)']);Path(sys.argv[1],'descendant.pid').write_text(str(p.pid));time.sleep(30)"
        receipt=self.run_stub("stop_descendant",code,stop_file=stop)
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(receipt["stop_file"],str(stop.resolve()))
        self.assertEqual(receipt["status"],"STOPPED")
        self.assertTrue(receipt["forced_termination"])
        self.assertTrue(_pid_is_gone(int((directory/"descendant.pid").read_text())))

    def test_exited_or_crashed_root_does_not_wait_forever_for_descendant(self):
        for exit_code in (0,7):
            with self.subTest(exit_code=exit_code):
                case="root_exit_descendant_"+str(exit_code)
                code="import subprocess,sys;from pathlib import Path;p=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)']);Path(sys.argv[1],'descendant.pid').write_text(str(p.pid));sys.exit("+str(exit_code)+")"
                receipt=self.run_stub(case,code)
                self.assertEqual(receipt["exit_code"],exit_code)
                self.assertFalse(receipt["success"])
                self.assertTrue(receipt["forced_termination"])
                self.assertTrue(receipt["training_root_exit_with_active_job"])
                self.assertEqual(receipt["status"],"FAILED")
                self.assertLess(receipt["elapsed_seconds"],6)
                child=int((self.root/case/"descendant.pid").read_text())
                self.assertTrue(_pid_is_gone(child))

    def test_assignment_and_resume_failure_remain_fail_closed(self):
        for target,name in ((owned_process.OwnedJob,"assign"),(owned_process,"_resume_process")):
            case="failed_"+name
            with self.subTest(name=name), mock.patch.object(target,name,side_effect=OSError("injected failure")):
                receipt=self.run_stub(case,"import sys;from pathlib import Path;Path(sys.argv[1],'executed').write_text('unsafe')")
            self.assertFalse((self.root/case/"executed").exists())
            self.assertFalse(receipt["resumed"])
            self.assertTrue(receipt["forced_termination"])
            self.assertEqual(receipt["status"],"FAILED")

    def test_job_accounting_lag_does_not_force_normal_root_exit(self):
        active_counts=iter((1,1,0))
        class Job:
            name="mock-accounting-lag-job"
            def assign(self, process): pass
            def active(self): return next(active_counts,0)
            def terminate(self): raise AssertionError("Accounting lag must not force termination")
            def close(self): pass
        directory=self.root/"accounting_lag_mock"
        child=SimpleNamespace(pid=424243,poll=lambda:0)
        with mock.patch.object(owned_process,"OwnedJob",return_value=Job()), \
             mock.patch.object(owned_process.subprocess,"Popen",return_value=child), \
             mock.patch.object(owned_process,"_resume_process"):
            receipt=owned_process.run_training_owned(["fake-only"],cwd=self.root,env={},directory=directory)
        self.receipts.append({"case":"accounting_lag_mock","receipt":str(directory/"receipt.json"),"result":receipt})
        self.assertTrue(receipt["training_root_exit_with_active_job"])
        self.assertTrue(receipt["success"])
        self.assertTrue(receipt["autonomous_tree_exit"])
        self.assertFalse(receipt["forced_termination"])
        self.assertLessEqual(receipt["cleanup_elapsed_ms"],5000)

    def test_keyboard_interrupt_cleans_up_and_saves_receipt_before_reraising(self):
        directory=self.root/"keyboard_interrupt"
        with mock.patch.object(owned_process,"_resume_process",side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            owned_process.run_training_owned([sys.executable,"-B","-c","raise RuntimeError('must not execute')"],
                cwd=self.root,env=os.environ.copy(),directory=directory)
        receipt=json.loads((directory/"receipt.json").read_text())
        self.receipts.append({"case":"keyboard_interrupt","receipt":str(directory/"receipt.json"),"result":receipt})
        self.assertTrue(receipt["owned_tree_gone_verified"])
        self.assertTrue(receipt["forced_termination"])
        self.assertFalse(receipt["resumed"])
        self.assertEqual(receipt["errors"][0]["type"],"KeyboardInterrupt")
        self.assertTrue(_pid_is_gone(receipt["root_pid"]))
        self.assertLessEqual(receipt["cleanup_elapsed_ms"],5000)

    def test_preexisting_stop_prevents_launch(self):
        stop=self.root/"early.STOP";stop.write_text("stop")
        with mock.patch.object(owned_process.subprocess,"Popen") as popen:
            receipt=self.run_stub("early_stop","raise RuntimeError('must not run')",stop_file=stop)
        popen.assert_not_called()
        self.assertEqual(receipt["status"],"STOPPED")
        self.assertIsNone(receipt["root_pid"])

    def test_simulator_limits_and_training_cleanup_fail_before_launch(self):
        cases=[(owned_process.run_owned,"timeout_seconds",value) for value in (None,1201,float("inf"))]
        cases += [(owned_process.run_training_owned,"timeout_seconds",value) for value in (1200,21600,0,True,float("inf"),float("nan"))]
        cases += [(owned_process.run_training_owned,"cleanup_seconds",value) for value in (5.01,0,True,None)]
        for runner,key,value in cases:
            with self.subTest(runner=runner.__name__,key=key,value=value), mock.patch.object(owned_process.subprocess,"Popen") as popen:
                with self.assertRaises(ValueError):
                    runner([sys.executable,"-B","-c","pass"],cwd=self.root,env={},
                           directory=self.root/"invalid",**{key:value})
                popen.assert_not_called()
        self.assertFalse((self.root/"invalid").exists())

    def test_existing_directory_is_never_overwritten(self):
        directory=self.root/"existing";directory.mkdir()
        sentinel=directory/"receipt.json";sentinel.write_bytes(b"preserve previous run")
        with mock.patch.object(owned_process.subprocess,"Popen") as popen, self.assertRaises(FileExistsError):
            owned_process.run_training_owned([sys.executable,"-B","-c","pass"],cwd=self.root,
                env={},directory=directory)
        popen.assert_not_called()
        self.assertEqual(sentinel.read_bytes(),b"preserve previous run")


# Runtime adapter

"""Standard-library path/provenance checks; never import the frozen runtime."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pipeline.runtime import host as adapter
from pipeline.runtime.host import FROZEN


class AdapterTests(unittest.TestCase):
    def test_ground_changes_only_location_and_calls_original_once(self):
        cfg = SimpleNamespace(scene=SimpleNamespace(ground=SimpleNamespace(spawn=SimpleNamespace(
            usd_path=adapter.ORIGINAL_GROUND_URL, size=(4.0, 4.0), color=(0.0, 0.0, 0.0))),
            chicken=SimpleNamespace(pos=(0.1, 0.2, 0.8))), physics_dt=1/120)
        before = deepcopy(cfg)
        session = object()
        called = []
        def original(actual_cfg, actual_session):
            called.append((actual_cfg, actual_session))
            return "original-result"
        result = adapter.configure_with_bundled_ground(cfg, session, original, dict(path="D:/repo/ground.usd"))
        self.assertEqual(result, "original-result")
        self.assertEqual(called, [(cfg, session)])
        before.scene.ground.spawn.usd_path = "D:/repo/ground.usd"
        self.assertEqual(cfg, before)

    def test_ground_unexpected_url_does_not_call_original_or_mutate(self):
        cfg = SimpleNamespace(scene=SimpleNamespace(ground=SimpleNamespace(spawn=SimpleNamespace(usd_path="other.usd"))))
        before = deepcopy(cfg)
        called = []
        with self.assertRaisesRegex(ValueError, "unexpected_original_ground"):
            adapter.configure_with_bundled_ground(cfg, None, lambda *args: called.append(args), dict(path="new.usd"))
        self.assertEqual(called, [])
        self.assertEqual(cfg, before)

    def test_literal_allowlist_rejects_missing_or_duplicate(self):
        with self.assertRaisesRegex(ValueError, "allowlist"):
            adapter.rewrite_literal_paths("def f(): return 'other'", {"old": "new"})
        with self.assertRaisesRegex(ValueError, "allowlist"):
            adapter.rewrite_literal_paths("def f(): return 'old', 'old'", {"old": "new"})

    def test_native_functions_change_only_enumerated_literals(self):
        cases = [
            ("attachment_capture_runtime.py", "launch_app", {
                "C:/isaacsim": "D:/sdk/sim", "C:/isaacsim/kit": "D:/sdk/sim/kit",
                "C:/isaacsim/kit/kernel/plugins": "D:/sdk/sim/kit/kernel/plugins"}),
            ("attachment_owner.py", "_installed_identity", {
                "C:/isaacsim/VERSION": "D:/sdk/sim/VERSION",
                "C:/robotics/IsaacLab/VERSION": "D:/sdk/lab/VERSION"}),
        ]
        for filename, name, paths in cases:
            source = (FROZEN / "src/chicgrasp_policy" / filename).read_text()
            module = ast.parse(source)
            function = next(node for node in ast.walk(module)
                            if isinstance(node, ast.FunctionDef) and node.name == name)
            text = ast.get_source_segment(source, function)
            result = adapter.rewrite_literal_paths(text, paths)
            # Reverse precisely the allowed literals; every other AST node must
            # be identical, including native calls, timing and error handling.
            reverse = {new: old for old, new in paths.items()}
            for node in ast.walk(result):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in reverse:
                    node.value = reverse[node.value]
            self.assertEqual(ast.dump(result, include_attributes=False),
                             ast.dump(ast.parse(text), include_attributes=False))

    def test_renderer_alias_preserves_all_scientific_fields(self):
        resource = dict(original_path="C:/old/render.kit", path="D:/new/render.kit", bytes=100, sha256="f" * 64)
        record = dict(schema="shared-camera-renderer/v1",
                      experience=dict(path=resource["original_path"], bytes=100, sha256="f" * 64),
                      settings={"brightness": 0.4, "seed": 42})
        before = deepcopy(record)
        changed = adapter.relocate_renderer_record(record, resource)
        self.assertEqual(record, before)
        self.assertEqual(changed["settings"], before["settings"])
        self.assertEqual(changed["experience"]["path"], resource["path"])
        for key, bad in (("path", "E:/unexpected.kit"), ("bytes", 99), ("sha256", "e" * 64)):
            rejected = deepcopy(record)
            rejected["experience"][key] = bad
            with self.assertRaises(ValueError):
                adapter.relocate_renderer_record(rejected, resource)

    def test_shader_alias_is_one_exact_dependency_and_copy_only(self):
        resource = dict(original_path="C:/old/OmniPBR.mdl", path="D:/new/OmniPBR.mdl")
        row = dict(property="/shader.info:mdl:sourceAsset", authored=resource["path"],
                   resolved=resource["path"], source_layer="anonymous-session", anchored=resource["path"])
        record = dict(dependencies=[row], materials={"roughness": 0.8})
        before = deepcopy(record)
        changed = adapter.relocate_shader_report(record, resource, row["property"])
        self.assertEqual(record, before)
        self.assertEqual(changed["materials"], before["materials"])
        self.assertEqual(changed["dependencies"][0]["authored"], resource["original_path"])
        self.assertEqual(changed["dependencies"][0]["source_layer"], "anonymous-session")
        for invalid in ([dict(row, resolved="E:/other.mdl")], [row, row], []):
            with self.assertRaises(ValueError):
                adapter.relocate_shader_report(dict(dependencies=invalid), resource, row["property"])

    def test_execution_identity_is_truthful_and_output_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            invocation = dict(executed_commit=None, runtime_snapshot_sha256="a" * 64,
                              adapter_snapshot_sha256="b" * 64)
            result = adapter.execution_identity(invocation, root, root / "capture/experiment")
            self.assertIsNone(result["executed_commit"])
            self.assertEqual(result["adapter_snapshot_sha256"], "b" * 64)
            self.assertEqual(result["errors"], [])
            with self.assertRaises(ValueError):
                adapter.execution_identity(invocation, root, root / "wrong")

    def test_sdk_hashes_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records = []
            for name in ("experience", "ground_shader", "physx_binding", "sim_version", "lab_version"):
                raw = (name + "\n").encode()
                (root / name).write_bytes(raw)
                records.append(dict(name=name, config_key="sdk", relative_path=name,
                    original_path="C:/old/" + name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
            manifest = root / 'pipeline/runtime/data/SDK_RESOURCE_IDENTITIES.json'
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps(dict(schema="hc-sdk-resources/v1", resources=records)))
            with patch.object(adapter, "ROOT", root):
                checked = adapter.verify_sdk_resources(dict(sdk=str(root)))
                self.assertEqual(len(checked), 5)
                (root / "ground_shader").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "sdk_resource_changed"):
                    adapter.verify_sdk_resources(dict(sdk=str(root)))


# Tests native

"""Offline contract tests: fake native objects, no Kit/model/encoder starts."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from pipeline.runtime import native as backend
from pipeline.scene import scene_setup_identity


def request(mode="scripted"):
    return dict(schema=backend.REQUEST_SCHEMA, mode=mode, scene_profile="attempt005",
        environment_seed=42, position=dict(id="NEW01", authored_xyz_m=[.67, .1, .8155]),
        scene_setup_identity=scene_setup_identity({}),
        script=dict(path="actions.npy", sha256="a"*64), evaluation_seed=15345,
        video=dict(enabled=False))


class HostIdentityTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "Windows native binding identity")
    def test_resource_path_matches_loaded_module_without_relaxing_identity(self):
        # SDK manifests use URI-style separators, but the frozen runtime reads
        # a Windows-native binding path. All other identity fields must survive.
        owner=NS(PINNED_RUNTIME=dict(binding_path="old", binding_sha256="fixed-hash",
            binding_bytes=123, native_python="3.11.13"), _installed_identity=lambda x:x)
        native=NS(SHARED_EXPERIENCE="old", APP_CONFIGURATION={},
            _NativeDependencies=type("Dependencies", (), {
                "launch_app":lambda self:None, "construct_environment":lambda self:None}))
        package=ModuleType("chicgrasp_policy")
        package.render_profile=NS(validate_renderer=lambda *a:None)
        package.grey_studio_check=NS(SHADER="shader", validate_report=lambda *a:None)
        package.dome_recovery=NS(apply_overrides=lambda *a:None)
        package.attachment_capture_runtime=native
        package.attachment_owner=owner
        resources={"experience":{"path":"D:/SDK/render.kit"},
            "sim_version":{"path":"D:/SDK/VERSION"},
            "lab_version":{"path":"D:/Lab/VERSION"},
            "physx_binding":{"path":"D:/SDK/bindings/_physx.pyd"},
            "ground_shader":{"path":"D:/SDK/shader.mdl"}}
        loader=NS(exec_module=lambda module:None)
        spec=NS(name="research_original_recorder", loader=loader)
        with patch.dict(sys.modules, {"chicgrasp_policy":package}), \
                patch.object(backend.host,"verify_sdk_resources",return_value=resources), \
                patch.object(backend.host,"verify_bundled_ground",return_value={}), \
                patch.object(backend.host,"relocate_function",side_effect=lambda fn,paths:fn), \
                patch.object(backend,"_hook_configuration",side_effect=lambda fn:fn), \
                patch.object(backend.importlib.util,"spec_from_file_location",return_value=spec), \
                patch.object(backend.importlib.util,"module_from_spec",return_value=ModuleType(spec.name)):
            backend.install_host({"isaac_sim":"D:/SDK"}, {})
        self.assertEqual(owner.PINNED_RUNTIME, dict(binding_path=r"D:\SDK\bindings\_physx.pyd",
            binding_sha256="fixed-hash", binding_bytes=123, native_python="3.11.13"))


class RequestTests(unittest.TestCase):
    def test_scene_does_not_require_checkpoint_or_script(self):
        value = request("scene")
        value.pop("script")
        out = backend.validate_request(value, Path.cwd())
        self.assertNotIn("checkpoint", out)
        self.assertEqual(out["position"]["authored_quaternion_wxyz"], [1., 0., 0., 0.])

    def test_unsupported_scene_seed_and_rotation_rejected(self):
        for field, value in (("scene_profile", "hanging"), ("environment_seed", 43)):
            candidate = request()
            candidate[field] = value
            with self.assertRaises(ValueError):
                backend.validate_request(candidate, Path.cwd())
        candidate = request()
        candidate["position"]["authored_quaternion_wxyz"] = [0, 0, 0, 1]
        with self.assertRaises(ValueError):
            backend.validate_request(candidate, Path.cwd())

    def test_live_requires_new_checkpoint_identity(self):
        with self.assertRaisesRegex(ValueError, "checkpoint_path_required"):
            backend.validate_request(request("live"), Path.cwd())

    def test_parent_null_sampling_seed_is_allowed_only_when_unused(self):
        value = request()
        value["evaluation_seed"] = None
        self.assertEqual(backend.validate_request(value, Path.cwd())["evaluation_seed"], 15345)
        value.update(mode="live", checkpoint=dict(path="new-model.pt", sha256="b"*64))
        with self.assertRaisesRegex(ValueError, "evaluation_seed"):
            backend.validate_request(value, Path.cwd())

    def test_artifact_hash_and_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"input.npy"
            path.write_bytes(b"unchanged")
            record = dict(path=str(path), bytes=9, sha256=hashlib.sha256(b"unchanged").hexdigest())
            self.assertEqual(backend.authenticate_artifact(record)["bytes"], 9)
            path.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "sha256_changed"):
                backend.authenticate_artifact(record)


class HostHookTests(unittest.TestCase):
    def test_frozen_configuration_source_has_exact_injection_site(self):
        path = backend.FROZEN / "src/chicgrasp_policy/attachment_capture_runtime.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        native = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_NativeDependencies")
        function = next(node for node in native.body if isinstance(node, ast.FunctionDef) and node.name == "construct_environment")
        hooked = backend.configuration_hook_tree(ast.get_source_segment(source, function))
        compile(hooked, str(path), "exec")  # Source only, never imports or executes native code.

    def test_only_requested_pose_and_ground_are_changed(self):
        cfg = NS(seed=42, sim=NS(device="cuda:0", physics="frozen"),
            scene=NS(chicken=NS(init_state=NS(pos=(1, 2, 3), rot=(1, 0, 0, 0))),
                     robot=NS(init_state="unchanged"),
                     ground=NS(spawn=NS(usd_path=backend.host.ORIGINAL_GROUND_URL))))
        session = NS()
        backend.configure_research(cfg, session,
            dict(authored_xyz_m=[.6, .2, .1], authored_quaternion_wxyz=[1, 0, 0, 0]),
            dict(path="same-bytes.usd"))
        self.assertEqual(cfg.scene.chicken.init_state.pos, (.6, .2, .1))
        self.assertEqual(cfg.scene.ground.spawn.usd_path, "same-bytes.usd")
        self.assertEqual(cfg.scene.robot.init_state, "unchanged")
        self.assertEqual(cfg.sim.physics, "frozen")
        self.assertFalse(session.research_start_configuration["robot_reset_configuration_changed"])

    def test_configuration_hook_calls_once_after_seed(self):
        source = "def build(self, cfg, session):\n    cfg.seed = session.seed\n    return cfg.seed\n"
        tree = backend.configuration_hook_tree(source)
        namespace = {}
        exec(compile(tree, "<fake registered cfg>", "exec"), namespace)
        calls = []
        self_obj = NS(configure_research=lambda cfg, session: calls.append(cfg.seed))
        self.assertEqual(namespace["build"](self_obj, NS(), NS(seed=42)), 42)
        self.assertEqual(calls, [42])
        with self.assertRaisesRegex(ValueError, "hook_count"):
            backend.configuration_hook_tree("def build(self):\n    return None\n")

    def test_unexpected_ground_rejected_before_placement(self):
        cfg = NS(scene=NS(ground=NS(spawn=NS(usd_path="other.usd"))))
        with self.assertRaisesRegex(ValueError, "unexpected_original_ground"):
            backend.configure_research(cfg, NS(), {}, {})


class ScriptTests(unittest.TestCase):
    def test_sequential_exact_script_dispatch_and_no_padding(self):
        events, selected = [], []
        writer = NS(action=selected.append,
                    command_event=lambda *a, **k: events.append((a, k)))
        actions = [[float(i), 0., .5, 1., 1.] for i in range(200)]
        controller = backend.ScriptController(actions, writer, {"sha256": "a"*64})
        for index in range(200):
            actual = controller.select(index, None, {"native_clock": [index*12, index*.1]})
            self.assertEqual(actual, actions[index])
            self.assertIsNot(actual, actions[index])
        self.assertEqual(len(events), 200)
        self.assertEqual(controller.summary(200)["learned_actions"], 0)
        with self.assertRaisesRegex(ValueError, "sequence"):
            controller.select(200, None, {"native_clock": []})

    def test_skipped_or_repeated_index_rejected(self):
        controller = backend.ScriptController([[0]*5]*200, NS(), {})
        with self.assertRaisesRegex(ValueError, "sequence"):
            controller.select(1, None, {})

    def test_incomplete_capture_never_becomes_episode(self):
        capture = NS(completed=False, executed_actions=[1], observations=[1],
                     diagnostics=[1], rows=[1]*12, lifecycle_events=[])
        with self.assertRaisesRegex(ValueError, "incomplete_or_reset"):
            backend.episode_arrays(capture, NS(), np=None, interface=None, episode_io=None)


class SceneCameraTests(unittest.TestCase):
    def test_scene_creates_presentation_cameras_without_encoder_or_physics_step(self):
        events, sensors = [], []
        runtime = NS(stage=object(), env=NS(sim=NS(pause=lambda: events.append("pause_physics"))),
            clock=lambda: (12, .1), camera_frames=lambda: {"left": 2, "right": 2},
            effective_configuration={"physics": "unchanged"})
        def factory(**kwargs):
            sensor = NS(pause=lambda: events.append("pause_sensor"), destroy=lambda: events.append("destroy_sensor"))
            sensors.append(sensor)
            return sensor
        def creator(stage, *, enabled, camera_factory):
            self.assertIs(stage, runtime.stage)
            self.assertTrue(enabled)
            for name in ("HAND", "LEFT_HD", "RIGHT_HD"):
                camera_factory(name=name)
            return NS(before_render=lambda: events.append("read_only_camera_check"),
                paths={"HAND": "/hand", "LEFT_HD": "/left_hd", "RIGHT_HD": "/right_hd"},
                configs={"hand": "existing preset"}, policy_before={"left": "unchanged"})
        with tempfile.TemporaryDirectory() as directory:
            cameras, created, receipt = backend.create_scene_cameras(runtime, Path(directory),
                creator=creator, camera_factory=factory)
            self.assertEqual(len(created), 3)
            self.assertFalse(receipt["encoding_started"])
            self.assertTrue(receipt["policy_cameras_unchanged"])
            self.assertTrue((Path(directory)/"scene_cameras.json").is_file())
            self.assertEqual(backend.close_scene_cameras(created), [])
        self.assertEqual(events.count("destroy_sensor"), 3)
        self.assertEqual(events[0], "pause_physics")

    def test_partial_scene_camera_creation_cleans_up(self):
        events = []
        runtime = NS(stage=object(), env=NS(sim=NS(pause=lambda: None)),
            clock=lambda: (0, 0.), camera_frames=lambda: {}, effective_configuration={})
        def creator(stage, *, enabled, camera_factory):
            camera_factory()
            raise RuntimeError("second camera failed")
        sensor = NS(pause=lambda: events.append("pause"), destroy=lambda: events.append("destroy"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "second camera failed"):
                backend.create_scene_cameras(runtime, Path(directory), creator=creator,
                    camera_factory=lambda **kwargs: sensor)
        self.assertEqual(events, ["pause", "destroy"])


class NativeOrchestrationTests(unittest.TestCase):
    """Exercise control flow through fake native objects, never the native modules."""
    def setUp(self):
        self.calls = []
        self.objective_status = "PASS"
        self.custom = False
        self.capture_exception = None
        calls = self.calls
        test = self

        class Runtime:
            def __init__(self, output, **kwargs):
                self.output_dir, self.seed = output, kwargs["seed"]
                self.episode_id = kwargs["episode_id"]
                self.owner, self.origin_clock, self.scene_identity = None, (0, 0.), None
                self.env, self.stage = NS(), NS()
                self.initialization = {"candidate_started": False}
                self.effective_configuration = {"physics": "frozen"}

            def initialize(self):
                calls.append("initialize")
                self.research_start_configuration = {"after": "requested"}

            def close_scene(self, context):
                calls.append("close_scene")
                return {"errors": [], "owner": None, "managed_environment_closed": True}

            def close_app(self):
                calls.append("close_app")

        class Writer:
            def __init__(self, rgb, output):
                self.rgb = rgb
            def action(self, record):
                pass
            def command_event(self, *args, **kwargs):
                pass
            def finish(self, complete, errors):
                calls.append(("writer_finish", complete))

        class Capture:
            def __init__(self, runtime, controller):
                self.runtime, self.controller = runtime, controller
                self.rows, self.sensor_samples, self.lifecycle_events = [], [], []
                self.executed_actions, self.control_ticks = [], 0
                self.context, self.completed = None, False
            def run(self):
                calls.append("capture_run")
                if test.capture_exception:
                    raise test.capture_exception
                for i in range(200):
                    self.executed_actions.append(self.controller.select(i, None, {"native_clock": [i*12, i*.1]}))
                self.rows, self.control_ticks, self.completed = [None]*2400, 1200, True
                return {"status": test.objective_status, "label": "ATTACHMENT_ASSISTED"}

        class FakeLiveController:
            def __init__(self, predictor, writer, **kwargs):
                test.assertEqual(len(kwargs["setup_actions"]), 50)
                test.assertEqual(kwargs["envelope_v2"], not test.custom)
                self.next_index, self.total_actions = 0, 200
                self.predictor = predictor
            def select(self, index, observation, sensor):
                self.next_index += 1
                return [0., 0., .5, 1., 1.]

        package = ModuleType("chicgrasp_policy")
        package.__path__ = []
        profile = dict(physical_configuration={"physics": "frozen"}, profile_id="attempt005", asset_root_identities=[])
        package.stage1_demo = NS(approved_scaled_profile=lambda: profile, SCALED_PROFILE_SHA256="b"*64,
            save_visuals=lambda *a, **kw: calls.append("save_visuals"))
        package.geometric_capture = NS(make_capture=lambda core, runtime, writer, action_source:
                                     Capture(runtime, action_source))
        package.attachment_owner = NS(bind_runtime=lambda *a, **kw: NS(
            journal=NS(select_geometric_admission=lambda *a, **kw: calls.append("owner_bind")),
            export=lambda: {"operations": []}))
        rgb_module = ModuleType("chicgrasp_policy.attachment_observation_evidence")
        rgb_module.IncrementalRGBWriter = lambda *a, **kw: NS()
        policy_module = ModuleType("chicgrasp_policy.stage3_policy")
        policy_module.PolicyWriter, policy_module.LiveController = Writer, FakeLiveController
        models = ModuleType("pipeline.models")
        def inspect_checkpoint(path):
            calls.append("inspect_new_checkpoint")
            return dict(training_window_start=50, live_compatible=True)
        def load_predictor(path, **kwargs):
            calls.append("load_new_predictor")
            return lambda history, chunk_index: None
        models.inspect_checkpoint, models.load_predictor = inspect_checkpoint, load_predictor
        self.modules = {"chicgrasp_policy": package,
            "chicgrasp_policy.attachment_observation_evidence": rgb_module,
            "chicgrasp_policy.stage3_policy": policy_module, "pipeline.models": models}
        core = NS(TASK_ID="original-task", write_json=backend.write_json, owner_cleanup_issues=lambda x: [])
        self.install = (core, NS(RuntimeSession=Runtime), NS(), {"schema": "fake-test-only"})

    def execute(self, *, bad_preflight=False, mode="scripted"):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, self.modules), \
                patch.object(backend, "install_host", return_value=self.install), \
                patch.object(backend, "sha256", return_value="b"*64), \
                patch.object(backend, "preflight", return_value={"status": "FAIL" if bad_preflight else "PASS", "errors": []}), \
                patch.object(backend, "authenticate_artifact", return_value={"path": "new-model.pt", "bytes": 4, "sha256": "c"*64}), \
                patch.object(backend, "load_script", return_value=([[0., 0., .5, 1., 1.]]*200, {"sha256": "a"*64})), \
                patch.object(backend, "finalize_episode", side_effect=lambda *a: self.calls.append("episode") or {"path": "episode"}):
            value = request(mode)
            if mode == "live":
                artifact = Path(directory)/"artifact.json"
                artifact.write_text(json.dumps({"scene_setup_identity": value["scene_setup_identity"]}))
                value["checkpoint"] = {"path": str(Path(directory)/"new-model.pt"), "sha256": "c"*64,
                    "artifact_path": str(artifact), "artifact_sha256": "c"*64}
            result = backend.run({}, value, Path(directory)/"new-attempt")
            saved = json.loads((Path(directory)/"new-attempt/research_result.json").read_text())
            self.assertEqual(saved, result)
            self.assertTrue((Path(directory)/"new-attempt/application_close_returned.json").is_file() or bad_preflight)
            return result

    def test_script_complete_pass_is_dataset_eligible(self):
        result = self.execute()
        self.assertEqual(result["completion"], "COMPLETED")
        self.assertTrue(result["training_eligible"])
        self.assertEqual(result["counts"]["completed_actions"], 200)
        self.assertLess(self.calls.index("episode"), self.calls.index("close_scene"))
        self.assertLess(self.calls.index("close_scene"), self.calls.index("close_app"))

    def test_physical_miss_keeps_complete_episode_but_not_training_eligible(self):
        self.objective_status = "FAIL"
        result = self.execute()
        self.assertEqual(result["completion"], "COMPLETED")
        self.assertIsNotNone(result["episode"])
        self.assertFalse(result["training_eligible"])

    def test_capture_failure_retains_raw_and_closes_without_episode(self):
        self.capture_exception = RuntimeError("measured admission refused")
        result = self.execute()
        self.assertEqual(result["completion"], "FAILED")
        self.assertIsNone(result["episode"])
        self.assertFalse(result["training_eligible"])
        self.assertIn("close_app", self.calls)

    def test_failed_media_retains_physical_metrics_but_blocks_training(self):
        with patch.object(backend, "attach_media", return_value=NS(finish=lambda:
                {"status": "FAIL", "errors": ["encoder_drain_failed"]})):
            result = self.execute()
        self.assertEqual(result["completion"], "FAILED")
        self.assertEqual(result["objective_metrics"]["status"], "PASS")
        self.assertIsNotNone(result["episode"])
        self.assertFalse(result["training_eligible"])

    def test_failed_preflight_never_initializes_native(self):
        result = self.execute(bad_preflight=True)
        self.assertEqual(result["completion"], "FAILED")
        self.assertNotIn("initialize", self.calls)

    def test_live_uses_new_predictor_after_preflight_and_is_not_a_training_demo(self):
        result = self.execute(mode="live")
        self.assertEqual(result["completion"], "COMPLETED")
        self.assertFalse(result["training_eligible"])
        self.assertEqual(result["checkpoint_identity"]["sha256"], "c"*64)
        self.assertLess(self.calls.index("inspect_new_checkpoint"), self.calls.index("initialize"))
        self.assertLess(self.calls.index("initialize"), self.calls.index("load_new_predictor"))


if __name__ == "__main__":
    unittest.main()
