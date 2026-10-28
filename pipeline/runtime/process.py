"""Windows child supervision; standard library, no simulator imports.

run_owned retains the simulator's finite limit. run_training_owned has no
wall-clock deadline; both use the same owned-job and bounded-cleanup machinery.

The OwnedJob ctypes layouts and kill-on-close semantics are adapted from
frozen_runtime/scripts/policy/exposure_control_process.py (original HC source
commit 3d98c85228821d955479730292d2b026fd1d63ed). This adapter additionally assigns
the root while CREATE_SUSPENDED and only then calls NtResumeProcess. It never
kills processes by name or enumerates unrelated process trees.
"""
from __future__ import annotations


import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid


class OwnedJob:
    """One private, non-inheritable, kill-on-close Windows job handle."""

    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("Owned-process supervision requires Windows")

        class BasicLimit(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IOCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class ExtendedLimit(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimit), ("IoInfo", IOCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        class Accounting(ctypes.Structure):
            _fields_ = [(name, ctypes.c_longlong) for name in (
                "TotalUserTime", "TotalKernelTime", "ThisPeriodTotalUserTime",
                "ThisPeriodTotalKernelTime")] + [(name, wintypes.DWORD) for name in (
                    "TotalPageFaultCount", "TotalProcesses", "ActiveProcesses",
                    "TotalTerminatedProcesses")]

        self.accounting = Accounting
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([wintypes.LPVOID, wintypes.LPCWSTR], wintypes.HANDLE),
            "SetInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD], wintypes.BOOL),
            "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            "QueryInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD, wintypes.LPVOID], wintypes.BOOL),
            "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (arguments, returns) in signatures.items():
            method = getattr(self.api, name)
            method.argtypes, method.restype = arguments, returns
        self.name = "Local\\ChicGraspExposure-" + str(uuid.uuid4())
        self.handle = None
        ctypes.set_last_error(0)
        self.handle = self.api.CreateJobObjectW(None, self.name)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if ctypes.get_last_error() == 183:
            self.close()
            raise RuntimeError("Fresh named job unexpectedly already exists")
        limits = ExtendedLimit()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE; no breakaway.
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, process):
        if not self.api.AssignProcessToJobObject(self.handle, wintypes.HANDLE(int(process._handle))):
            raise ctypes.WinError(ctypes.get_last_error())

    def active(self):
        record = self.accounting()
        if not self.api.QueryInformationJobObject(self.handle, 1, ctypes.byref(record), ctypes.sizeof(record), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(record.ActiveProcesses)

    def terminate(self):
        if not self.api.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            if not self.api.CloseHandle(self.handle):
                raise ctypes.WinError(ctypes.get_last_error())
            self.handle = None


def _resume_process(process):
    """Popen closes its primary thread handle; resume through its process handle."""
    api = ctypes.WinDLL("ntdll", use_last_error=True)
    method = api.NtResumeProcess
    method.argtypes = [wintypes.HANDLE]
    method.restype = ctypes.c_long  # NTSTATUS is a signed 32-bit value on Windows.
    status = method(wintypes.HANDLE(int(process._handle)))
    if status < 0:
        raise OSError(f"NtResumeProcess failed with NTSTATUS 0x{status & 0xffffffff:08x}")


def _write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _identity(path, may_hash):
    before = path.stat()
    result = {"path": str(path), "bytes": before.st_size, "sha256": None}
    if not may_hash:
        result["error"] = "Tree disappearance unverified; file may still have a writer"
        return result
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        result["error"] = "Log changed while hashing"
    else:
        result["sha256"] = sha
    return result


def run_owned(command, *, cwd, env, directory, timeout_seconds=1200,
              cleanup_seconds=5, stop_file=None):
    """Supervise a bounded process; timeout must be positive and at most 1200s."""
    return _run_owned(command, cwd=cwd, env=env, directory=directory,
                      timeout_seconds=timeout_seconds, cleanup_seconds=cleanup_seconds,
                      stop_file=stop_file, training=False)


def run_training_owned(command, *, cwd, env, directory, timeout_seconds=None,
                       cleanup_seconds=5, stop_file=None):
    """Supervise training without a runtime deadline; explicit stops still apply.

    Only None is accepted for timeout_seconds. Cleanup retains its independent
    five-second maximum. This supervisor does not save or resume checkpoints.
    """
    return _run_owned(command, cwd=cwd, env=env, directory=directory,
                      timeout_seconds=timeout_seconds, cleanup_seconds=cleanup_seconds,
                      stop_file=stop_file, training=True)


def _run_owned(command, *, cwd, env, directory, timeout_seconds,
               cleanup_seconds, stop_file, training):
    """Launch exactly one suspended root, contain it, and save a truthful receipt.

    A fresh directory and an existing parent directory are required. Input
    validation/overwrite failures raise before launch. Launch/containment errors
    are returned in receipt.json after fail-closed cleanup. KeyboardInterrupt and
    SystemExit are re-raised only after cleanup and receipt persistence. Log
    hashing occurs after the bounded process-containment phase.
    """
    if os.name != "nt":
        raise RuntimeError("Owned-process supervision requires Windows")
    if isinstance(command, (str, bytes)) or not command:
        raise ValueError("command must be a nonempty argument sequence; shell strings are prohibited")
    command = [os.fspath(arg) for arg in command]
    if not command:
        raise ValueError("command argument sequence is empty")
    if any(not isinstance(arg, str) or "\0" in arg for arg in command):
        raise ValueError("command arguments must be strings without NUL")
    if training and timeout_seconds is not None:
        raise ValueError("Training has no wall-clock limit; timeout_seconds must be None")
    limits = [("cleanup_seconds", cleanup_seconds, 5)]
    if not training:
        limits.append(("timeout_seconds", timeout_seconds, 1200))
    for name, value, maximum in limits:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError(f"{name} must be finite, positive and at most {maximum}")
    cwd, directory = Path(cwd).resolve(), Path(directory).resolve()
    if not cwd.is_dir():
        raise NotADirectoryError(str(cwd))
    if not directory.parent.is_dir():
        raise FileNotFoundError(f"Create the output parent first: {directory.parent}")
    stop_file = Path(stop_file).resolve() if stop_file is not None else None
    directory.mkdir(exist_ok=False)
    stdout_path, stderr_path = directory/"stdout.bin", directory/"stderr.bin"
    start = time.monotonic()
    deadline = None if timeout_seconds is None else start + float(timeout_seconds)
    timeout_ms = None if timeout_seconds is None else int(timeout_seconds*1000)
    record = {
        "schema": "portable-owned-process/v1", "command": command, "cwd": str(cwd),
        "directory": str(directory), "started_unix": time.time(), "started_monotonic": start,
        "timeout_seconds": timeout_seconds, "cleanup_seconds": cleanup_seconds,
        "runtime_limit_policy": "none" if training else "bounded",
        "stop_file": str(stop_file) if stop_file is not None else None,
        "training_root_exit_with_active_job": False,
        "root_pid": None, "pid": None, "job_name": None, "job_assigned": False,
        "assigned_before_resume": False, "resumed": False, "exit_code": None,
        "timed_out": False, "stop_requested": False, "forced_termination": False,
        "tree_empty_verified": False, "owned_tree_gone_verified": False,
        "root_exited_verified": False, "cleanup_elapsed_ms": 0.0,
        "cleanup_budget_exceeded": False, "success": False, "errors": [],
        "root_exit_verified": False, "job_assigned_before_resume": False,
        "root_resumed": False, "job_disposed_after_observation": False,
        "job_members_before_disposal": None, "autonomous_tree_exit": False,
        "kill_on_job_close_used": False, "kill_on_job_close_possible": False,
        "startup_containment_used": False, "parent_termination_requested": False,
        "capture_complete": False, "containment_budget_overrun": False,
        "timeout_ms": timeout_ms, "absolute_deadline_ms": timeout_ms,
        "effective_deadline_ms": timeout_ms, "parent_launch_boundary_ms": 0,
        "termination_and_drain_cap_ms": int(cleanup_seconds*1000),
        "root_exit_observed_ms": None, "job_empty_observed_ms": None,
        "drain_completed_observed_ms": None, "stream_files_finalized_ms": None,
        "job_disposed_ms": None,
        "capture_method": "Direct inherited binary file handles; no pipe forwarding. Observe all owned writers gone, fsync and close before receipt.",
    }
    record["containment_errors"] = record["errors"]
    process = job = None
    propagate = None
    out = err = None

    def observed_ms():
        # Ceiling does not report an observation earlier than its actual instant.
        return math.ceil((time.monotonic()-start)*1000)

    def observe_tree():
        root_gone = process is None or process.poll() is not None
        if process is not None and root_gone:
            record["root_exit_verified"] = True
            if record["root_exit_observed_ms"] is None:
                record["root_exit_observed_ms"] = observed_ms()
        active = job.active() if job is not None else 0
        if job is not None and active == 0 and record["job_empty_observed_ms"] is None:
            record["job_empty_observed_ms"] = observed_ms()
        if active == 0 and root_gone:
            record["tree_empty_verified"] = True
            record["root_exited_verified"] = True
            record["owned_tree_gone_verified"] = True
            record["tree_verification_method"] = "QueryInformationJobObject.ActiveProcesses=0 and root process signalled" if job is not None else "No job or child process was created"
            # There is no forwarding pipe to drain: every inherited writer has
            # exited. File durability and closure are observed separately below.
            if process is not None and record["drain_completed_observed_ms"] is None:
                record["drain_completed_observed_ms"] = observed_ms()
        return active, root_gone

    def request_termination():
        record["forced_termination"] = True
        record["parent_termination_requested"] = True
        if process is not None and not record["resumed"]:
            record["startup_containment_used"] = True
        if job is not None:
            job.terminate()
        if process is not None and not record["job_assigned"] and process.poll() is None:
            process.kill()

    try:
        out = stdout_path.open("xb", buffering=0)
        err = stderr_path.open("xb", buffering=0)
        job = OwnedJob()
        record["job_name"] = job.name
        if stop_file is not None and stop_file.exists():
            record["stop_requested"] = True
        elif deadline is not None and time.monotonic() >= deadline:
            record["timed_out"] = True
        else:
            process = subprocess.Popen(
                command, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, shell=False, close_fds=True,
                creationflags=subprocess.CREATE_NO_WINDOW | 0x00000004)
            record["root_pid"] = record["pid"] = process.pid
            job.assign(process)
            record["job_assigned"] = record["assigned_before_resume"] = True
            record["job_assigned_before_resume"] = True
            _write_json(directory/"started.json", record)
            # Recheck before executing any child code if startup used the budget.
            if stop_file is not None and stop_file.exists():
                record["stop_requested"] = True
            elif deadline is not None and time.monotonic() >= deadline:
                record["timed_out"] = True
            else:
                _resume_process(process)
                record["resumed"] = record["root_resumed"] = True
                while True:
                    now = time.monotonic()
                    if deadline is not None and now >= deadline:
                        record["timed_out"] = True
                        break
                    if stop_file is not None and stop_file.exists():
                        record["stop_requested"] = True
                        break
                    active, root_gone = observe_tree()
                    if root_gone and active == 0:
                        record["autonomous_tree_exit"] = True
                        break
                    if training and root_gone:
                        # An exited/crashed training worker cannot supervise its
                        # remaining writers. Do not wait forever without a root.
                        record["training_root_exit_with_active_job"] = True
                        break
                    time.sleep(0.02 if deadline is None else min(0.02, deadline-now))
    except BaseException as error:
        record["errors"].append({"phase": "launch_or_wait", "type": type(error).__name__, "message": str(error)})
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            propagate = error
    finally:
        cleanup_start = time.monotonic()
        cleanup_deadline = cleanup_start + float(cleanup_seconds)

        try:
            active, root_gone = observe_tree()
            if training and record["training_root_exit_with_active_job"]:
                # Windows may signal root exit slightly before updating job
                # accounting. A short observation window is part of the SAME
                # cleanup budget, never an extension or an unlimited wait.
                observation_deadline = min(cleanup_deadline, cleanup_start + 0.1)
                while root_gone and active and time.monotonic() < observation_deadline:
                    time.sleep(min(0.005, max(0., observation_deadline-time.monotonic())))
                    active, root_gone = observe_tree()
                if root_gone and active == 0 and not record["errors"]:
                    record["autonomous_tree_exit"] = True
            if not root_gone or active:
                request_termination()
            while True:
                active, root_gone = observe_tree()
                if active == 0 and root_gone:
                    break
                left = cleanup_deadline-time.monotonic()
                if left <= 0:
                    raise TimeoutError("Shared cleanup deadline expired before root exit and zero active job processes")
                time.sleep(min(0.005, left))
        except BaseException as error:
            record["errors"].append({"phase": "cleanup", "type": type(error).__name__, "message": str(error)})
            # Even a query/terminate failure must close the kill-on-close handle.
            # An unassigned suspended child still needs its exact handle killed.
            if process is not None and not record["job_assigned"] and process.poll() is None:
                record["forced_termination"] = True
                record["parent_termination_requested"] = True
                record["startup_containment_used"] = not record["resumed"]
                try:
                    process.kill()
                except BaseException as kill_error:
                    record["errors"].append({"phase": "unassigned_root_kill", "type": type(kill_error).__name__, "message": str(kill_error)})
        finally:
            if job is not None:
                try:
                    record["job_members_before_disposal"] = job.active()
                except BaseException as query_error:
                    record["errors"].append({"phase": "before_disposal_query", "type": type(query_error).__name__, "message": str(query_error)})
                try:
                    before_disposal = record["job_members_before_disposal"]
                    record["kill_on_job_close_possible"] = before_disposal is None or before_disposal > 0
                    if record["kill_on_job_close_possible"]:
                        record["forced_termination"] = True
                        record["parent_termination_requested"] = True
                        record["autonomous_tree_exit"] = False
                        record["startup_containment_used"] = process is not None and not record["resumed"]
                    job.close()
                    record["job_disposed_ms"] = observed_ms()
                    record["job_disposed_after_observation"] = before_disposal is not None
                    record["kill_on_job_close_used"] = None if before_disposal is None else before_disposal > 0
                except BaseException as close_error:
                    record["errors"].append({"phase": "job_close", "type": type(close_error).__name__, "message": str(close_error)})
            record["exit_code"] = process.poll() if process is not None else None
            streams_finalized = out is not None and err is not None
            for name, stream in (("stdout",out),("stderr",err)):
                if stream is not None:
                    try:
                        stream.flush()
                        os.fsync(stream.fileno())
                    except OSError as error:
                        streams_finalized = False
                        record["errors"].append({"phase":name+"_flush","type":type(error).__name__,"message":str(error)})
                    finally:
                        try:
                            stream.close()
                        except OSError as error:
                            streams_finalized = False
                            record["errors"].append({"phase":name+"_close","type":type(error).__name__,"message":str(error)})
            if streams_finalized:
                record["stream_files_finalized_ms"] = observed_ms()
            record["capture_complete"] = streams_finalized and record["owned_tree_gone_verified"]
            record["cleanup_elapsed_ms"] = (time.monotonic()-cleanup_start)*1000
            record["cleanup_budget_exceeded"] = record["cleanup_elapsed_ms"] > cleanup_seconds*1000
            record["containment_budget_overrun"] = record["cleanup_budget_exceeded"]

    for name, path in (("stdout", stdout_path), ("stderr", stderr_path)):
        try:
            if path.exists():
                identity = _identity(path, record["owned_tree_gone_verified"])
            else:
                identity = {"path": str(path), "bytes": None, "sha256": None, "error": "Log creation failed"}
        except OSError as error:
            record["errors"].append({"phase": name+"_identity", "type": type(error).__name__, "message": str(error)})
            identity = {"path": str(path), "bytes": None, "sha256": None, "error": str(error)}
        record[name] = identity
        record[name+"_path"] = identity["path"]
        record[name+"_bytes"] = identity["bytes"]
        record[name+"_sha256"] = identity["sha256"]
    record["elapsed_seconds"] = time.monotonic()-start
    record["duration_seconds"] = record["elapsed_seconds"]
    record["finished_unix"] = time.time()
    if timeout_seconds is not None and record["elapsed_seconds"] > timeout_seconds:
        record["timed_out"] = True
    record["success"] = bool(
        process is not None and record["assigned_before_resume"] and record["resumed"]
        and record["exit_code"] == 0 and record["owned_tree_gone_verified"]
        and not any(record[x] for x in ("errors", "timed_out", "stop_requested", "forced_termination", "cleanup_budget_exceeded"))
        and all(record[name]["sha256"] is not None for name in ("stdout", "stderr")))
    record["status"] = "COMPLETE" if record["success"] else "STOPPED" if record["stop_requested"] else "TIMED_OUT" if record["timed_out"] else "FAILED"
    _write_json(directory/"receipt.json", record)
    if propagate is not None:
        raise propagate
    return record
