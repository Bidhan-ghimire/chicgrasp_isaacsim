"""Read-only, partial constructor-boundary observations; never create native objects.

This module imports no simulator packages. It accepts the actual environment
already supplied to the observation term and keeps only ordinary copied values.
It deliberately does not invoke the older PhysicsContext device/GPU properties:
their installed SimulationManager getter has a CUDA-setting repair branch.
"""
from __future__ import annotations

import math
import time


def _type(value):
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _plain(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    if type(value) in (list, tuple):
        return [_plain(item) for item in value]
    if _type(value) == "torch.device":
        return str(value)
    raise TypeError(f"unsupported observation value: {_type(value)}")


def observe_existing_backend(env, *, phase="CONSTRUCTOR_GUARD", asset_name="robot"):
    """Copy available fields, including errors; partial data never qualifies CPU.

    Every getter below is backed by the inspected installed source in the Cycle46
    source note. Attribute access uses already constructed objects only. No state
    tensor, SimulationContext singleton, creation method or stage cache is read.
    """
    fields, access_errors = {}, []

    def field(name, source, getter, *, kind="observed", derivation=None):
        row = {"source": source, "raw_value": None, "raw_type": None,
               "phase": phase, "kind": kind, "status": "UNAVAILABLE", "error": None}
        if derivation is not None:
            row["derivation"] = derivation
        try:
            value = getter()
            row.update(raw_type=_type(value), raw_value=_plain(value))
            row["status"] = "OBSERVED" if value is not None else "UNAVAILABLE"
            if value is None:
                row["reason"] = "observed_none"
        except AttributeError as exc:
            row.update(reason="existing_member_unavailable", error=f"{_type(exc)}:{exc}")
        except Exception as exc:
            row.update(status="ERROR", error=f"{_type(exc)}:{exc}")
        fields[name] = row
        return row

    # These ordinary attribute handles exist only during this call, never in output.
    def existing(source, getter):
        try:
            return getter()
        except Exception as exc:
            access_errors.append({"source": source, "phase": phase,
                                  "error": f"{_type(exc)}:{exc}"})
            return None

    sim = existing("env.sim", lambda: env.sim)
    physics = existing("env.sim._physics_context", lambda: sim._physics_context)
    settings = existing("env.sim.carb_settings", lambda: sim.carb_settings)
    asset = existing("env.scene[asset_name]", lambda: env.scene[asset_name])
    view = existing("asset._physics_sim_view", lambda: asset._physics_sim_view)
    prim = existing("physics._physics_scene.GetPrim()", lambda: physics._physics_scene.GetPrim())

    field("configured_simulation_device", "env.cfg.sim.device", lambda: env.cfg.sim.device, kind="configured")
    field("environment_device", "env.device -> env.sim.device -> cached _physics_device", lambda: env.device)
    field("simulation_device", "env.sim.device -> cached _physics_device", lambda: sim.device)
    field("physics_context_type", "env.sim._physics_context", lambda: _type(physics) if physics is not None else None)
    field("physics_context_path", "env.sim._physics_context.prim_path", lambda: physics.prim_path)
    field("configured_physics_scene_path", "env.cfg.sim.physics_prim_path", lambda: env.cfg.sim.physics_prim_path,
          kind="configured")
    field("physics_scene_path", "existing PhysicsContext._physics_scene.GetPrim().GetPath()", lambda: str(prim.GetPath()))
    field("physics_scene_valid", "existing PhysicsContext._physics_scene.GetPrim().IsValid()", lambda: prim.IsValid())
    field("gpu_dynamics", "PhysicsContext.is_gpu_dynamics_enabled -> existing scene API attribute Get",
          lambda: physics.is_gpu_dynamics_enabled())
    field("broadphase", "PhysicsContext.get_broadphase_type -> existing scene API attribute Get",
          lambda: physics.get_broadphase_type())
    for name, path in (
        ("suppress_readback", "/physics/suppressReadback"),
        ("physics_cuda_device", "/physics/cudaDevice"),
        ("renderer_active_gpu", "/renderer/activeGpu"),
        ("cameras_enabled", "/isaaclab/cameras_enabled"),
        ("rtx_sensors", "/isaaclab/render/rtx_sensors"),
        ("offscreen_render", "/isaaclab/render/offscreen"),
        ("disable_contact_processing", "/physics/disableContactProcessing"),
        ("fabric_enabled", "/physics/fabricEnabled"),
    ):
        field(name, f"env.sim.carb_settings.get({path!r})", lambda path=path: settings.get(path))
    field("native_view_valid", "env.scene[asset_name]._physics_sim_view.is_valid", lambda: view.is_valid)
    field("native_view_device", "env.scene[asset_name]._physics_sim_view.device", lambda: view.device)
    field("native_view_device_ordinal", "env.scene[asset_name]._physics_sim_view.device_ordinal", lambda: view.device_ordinal)
    field("physics_cached_device", "env.sim._physics_context._device (construction-time cached)", lambda: physics._device)

    # The installed property getter is not unconditionally read-only. Record this
    # limitation, not a guessed readback or a call that could repair a setting.
    for name, source in (("physics_device", "PhysicsContext.device"),
                         ("gpu_sim", "PhysicsContext.use_gpu_sim"),
                         ("gpu_pipeline", "PhysicsContext.use_gpu_pipeline")):
        fields[name] = {"source": source, "raw_value": None, "raw_type": None,
                        "phase": phase, "kind": "observed", "status": "UNAVAILABLE", "error": None,
                        "reason": "not_queried: SimulationManager.get_physics_sim_device may set cudaDevice"}

    # Source-derived explanation from captured independent operands, not another
    # native query or proof of full native scene/device association.
    copied = {name: row["raw_value"] for name, row in fields.items() if row["status"] == "OBSERVED"}
    derived_device = None
    if copied.get("suppress_readback") is False:
        derived_device = "cpu"
    elif (copied.get("suppress_readback") is True and type(copied.get("broadphase")) is str
          and type(copied.get("gpu_dynamics")) is bool):
        if copied["broadphase"] == "GPU" and copied["gpu_dynamics"]:
            ordinal = copied.get("physics_cuda_device")
            if type(ordinal) is int and ordinal >= 0:
                derived_device = f"cuda:{ordinal}"
        else:
            derived_device = "cpu"
    derivation = "source branch replay from copied raw settings/scene values; no manager getter; not independent proof"
    field("derived_physics_device", "SimulationManager.get_physics_sim_device source branch",
          lambda: derived_device, kind="derived", derivation=derivation)
    for name in ("derived_gpu_sim", "derived_gpu_pipeline"):
        field(name, "PhysicsContext GPU property source branch",
              lambda: ("cuda" in derived_device) if derived_device is not None else None,
              kind="derived", derivation=derivation)
    clock = {
        "step": field("native_clock_step", "env.sim.current_time_step_index -> existing _number_of_steps",
                      lambda: sim.current_time_step_index),
        "time_s": field("native_clock_time_s", "env.sim.current_time -> existing _current_time",
                        lambda: sim.current_time),
    }
    field("environment_step_counter", "env._sim_step_counter (not a native clock)", lambda: env._sim_step_counter)
    return {"schema": "cpu-startup-backend-observation/v1", "phase": phase,
            "monotonic_ns": time.monotonic_ns(), "fields": fields, "native_clock": clock,
            "access_errors": access_errors,
            "verification": "CPU_CONFIGURATION_UNAVAILABLE",
            "qualification_limit": "partial boundary observation; full managed scene contract not evaluated"}
