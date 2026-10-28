"""Explicit CPU construction dimension query, qualified against installed source.

No native imports or manager replacement. Only the configured jaw callable is
wrapped; its original strict implementation handles every other call.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import dis
from functools import wraps
import hashlib
from pathlib import Path
import sys
import time
from types import CodeType

from .jaw_state_diagnostics import real_guard_observation


INSTALLED_MANAGER_SHA256 = "120a77f2a7fee19a21478946b3e991a2a635ad4333c6a3432e2296f53fda202f"
_DIMENSION_STATEMENT = "obs_dims = tuple(term_cfg.func(self._env, **term_cfg.params).shape)"


def _find_code(code, qualname):
    if code.co_qualname == qualname:
        return code
    for child in code.co_consts:
        if isinstance(child, CodeType):
            found = _find_code(child, qualname)
            if found is not None:
                return found
    return None


def qualify_dimension_query(observation_manager_type):
    """Read/compile source without executing it and verify the loaded callable."""
    prepare = observation_manager_type._prepare_terms
    code = prepare.__code__
    path = Path(code.co_filename)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != INSTALLED_MANAGER_SHA256:
        raise ValueError("unsupported_observation_manager_source")
    source = payload.decode("utf-8")
    tree = ast.parse(source, filename=code.co_filename)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "ObservationManager")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_prepare_terms")
    queries = [node for node in ast.walk(method) if isinstance(node, ast.Assign)
               and ast.unparse(node) == _DIMENSION_STATEMENT]
    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)
             and ast.unparse(node.func) == "term_cfg.func"]
    if len(queries) != 1 or len(calls) != 1:
        raise ValueError("unsupported_observation_dimension_query")
    compiled = compile(source, code.co_filename, "exec", dont_inherit=True)
    expected = _find_code(compiled, "ObservationManager._prepare_terms")
    if expected != code:
        raise ValueError("loaded_observation_manager_code_differs_from_source")
    offsets = [instruction.offset for instruction in dis.get_instructions(code)
               if instruction.opname == "CALL_FUNCTION_EX"
               and instruction.positions.lineno == calls[0].lineno]
    if len(offsets) != 1:
        raise ValueError("unsupported_observation_dimension_call_instruction")
    return code, offsets[0], {
        "schema": "observation-dimension-boundary/v1", "source": str(path.resolve()),
        "sha256": digest, "callable": code.co_qualname, "line": calls[0].lineno,
        "call_instruction_offset": offsets[0], "statement": _DIMENSION_STATEMENT,
        "consumes": "shape only; no returned sizing data retained",
    }


@contextmanager
def cpu_observation_metadata(cfg, observation_manager_type, *, dimension_func, phase_provider, sink):
    """Bind one CPU cfg during construction; its copied manager stays strict later.

    Sink receives one strict-JSON-compatible metadata or real-guard record per
    call. The caller owns durable writing; this adapter retains no frame list.
    No stack-name inference: admission requires the exact verified code object
    and call instruction, the bound environment cfg and explicit active scope.
    """
    if str(cfg.sim.device) != "cpu" or cfg.scene.num_envs != 1:
        raise ValueError("metadata_scope_requires_explicit_single_cpu_configuration")
    if not all(callable(value) for value in (dimension_func, phase_provider, sink)):
        raise TypeError("metadata_scope_requires_dimension_phase_and_sink_callables")
    term = cfg.observations.policy.jaw_state
    original = term.func
    if not callable(original) or getattr(original, "_chicgrasp_metadata_adapter", False):
        raise ValueError("metadata_scope_requires_unwrapped_jaw_term")
    code, offset, qualification = qualify_dimension_query(observation_manager_type)
    state = {"qualification": qualification, "dimension_queries": 0, "metadata_scope_active": True}
    active = True

    @wraps(original)
    def scoped_jaw_state(env, *args, **params):
        frame = sys._getframe(1)
        try:
            dimension_query = (active and str(cfg.sim.device) == "cpu"
                and getattr(env, "cfg", None) is cfg and str(env.device) == "cpu"
                and frame.f_code is code and frame.f_lasti == offset)
        finally:
            del frame
        if dimension_query:
            sizing = dimension_func(env, *args, **params)
            state["dimension_queries"] += 1
            sink({"schema": "jaw-state-dimension-discovery/v1",
                  "phase": "DIMENSION_DISCOVERY", "observation_phase": "CONSTRUCTION",
                  "monotonic_ns": time.monotonic_ns(), "policy_frame_emitted": False,
                  "physical_jaw_positions_read": False, "paired_state_validity_applied": False,
                  "shape": list(sizing.shape), "dtype": str(sizing.dtype),
                  "device": str(sizing.device), "boundary": dict(qualification)})
            return sizing
        phase = phase_provider()
        with real_guard_observation(sink, phase) as observation:
            result = original(env, *args, **params)
        if observation["errors"]:
            raise RuntimeError("real_jaw_guard_recording_failed:" + ";".join(observation["errors"]))
        return result

    scoped_jaw_state._chicgrasp_metadata_adapter = True
    term.func = scoped_jaw_state
    try:
        yield state
    finally:
        active = False
        state["metadata_scope_active"] = False
        term.func = original
