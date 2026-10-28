"""Passive, early scene light inventory. Native imports are deferred."""
from contextlib import contextmanager
import json
import os
from pathlib import Path

from .render_profile import SETTING_PATHS
from .stage0_render_diagnostic import SETTINGS, _value

ATTRIBUTES = ("visibility", "inputs:intensity", "intensity", "inputs:color",
              "inputs:exposure", "inputs:texture:file", "texture:file",
              "visibleInPrimaryRay")


def snapshot(scene, *, bindings=None, phase="CALLER_READBACK_PHASE_UNSPECIFIED"):
    """Read the scene without processing it; the caller declares the timing.

    TraverseAll includes inactive prims. A separate supported filtered-child
    walk includes instance proxies; the union is deduplicated by exact path.
    Unloaded payload contents cannot be inventoried without loading them.
    """
    if bindings is None:
        from pxr import Usd, UsdGeom, UsdLux
        bindings = Usd, UsdGeom, UsdLux
    Usd, UsdGeom, UsdLux = bindings
    stage, sim = scene.stage, scene.sim
    errors = []

    def read(source, getter):
        try:
            return dict(source=source, status="OBSERVED", value=_value(getter()))
        except Exception as exc:
            error = type(exc).__name__ + ":" + str(exc)
            errors.append(dict(source=source, error=error))
            return dict(source=source, status="UNAVAILABLE", error=error)

    prims = {str(p.GetPath()): p for p in stage.TraverseAll()}
    stack = [stage.GetPseudoRoot()]
    predicate = Usd.TraverseInstanceProxies()
    while stack:
        prim = stack.pop()
        for child in prim.GetFilteredChildren(predicate):
            prims[str(child.GetPath())] = child
            stack.append(child)
    lights = []
    for path, prim in sorted(prims.items()):
        if not (prim.HasAPI(UsdLux.LightAPI) or prim.IsA(UsdLux.BoundableLightBase)
                or prim.IsA(UsdLux.NonboundableLightBase)):
            continue
        attributes = {}
        for name in ATTRIBUTES:
            attr = prim.GetAttribute(name)
            valid = attr.IsValid()
            attributes[name] = dict(present=valid)
            if valid:
                attributes[name].update(
                    value=read(path + "." + name, attr.Get),
                    authored=read(path + "." + name + ".authored", attr.HasAuthoredValueOpinion))
        lights.append(dict(path=path, type=str(prim.GetTypeName()),
            active=bool(prim.IsActive()), loaded=bool(prim.IsLoaded()),
            instance_proxy=bool(prim.IsInstanceProxy()),
            computed_visibility=read(path + ".ComputeVisibility", lambda prim=prim:
                str(UsdGeom.Imageable(prim).ComputeVisibility(Usd.TimeCode.Default()))),
            attributes=attributes))
    result = dict(schema="stage0-early-light-inventory/v1", read_only=True,
        phase=phase,
        earlier_application_startup_updates="ALREADY_OCCURRED_NOT_COUNTED",
        coverage=dict(traversed_prims=len(prims),
            traversal="TraverseAll union recursive GetFilteredChildren(TraverseInstanceProxies())",
            unloaded_paths=[p for p, prim in sorted(prims.items()) if not prim.IsLoaded()],
            limit="Unloaded payload contents and unpopulated prims are not materialized"),
        lights=lights, rendering_mode=read("SimulationContext.render_mode", lambda: sim.render_mode),
        renderer_settings={key: read(key, lambda key=key: sim.get_setting(key))
            for key in dict.fromkeys((*SETTING_PATHS, *SETTINGS))}, errors=errors,
        status="PARTIAL" if errors else "COMPLETE")
    return json.loads(json.dumps(result, allow_nan=False))


def write_inventory(scene, output_dir, mode):
    if mode not in ("CHECK_HEADLESS", "TUNE_GUI"):
        raise ValueError("light_inventory_requires_named_stage0_mode")
    data = snapshot(scene, phase="AFTER_INTERACTIVE_SCENE_BEFORE_CONTEXT_ATTACH_AND_INITIAL_RESET_RENDER")
    data["mode"] = mode
    path = Path(output_dir) / ("lights_gui.json" if mode == "TUNE_GUI" else "lights_headless.json")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if data["status"] != "COMPLETE":
        raise RuntimeError("early_light_inventory_incomplete:" + str(path))
    # Same existing scene-construction hook: verify the owner rig before the
    # normal reset's first render or any policy/checkpoint image acquisition.
    from .grey_studio_check import write_check
    write_check(scene.stage, output_dir)
    from .dome_recovery import validate_lights
    validate_lights(data)
    return path


@contextmanager
def observe_construction(module, observer):
    """Preserve the existing constructor result, call order and exceptions."""
    original = module.InteractiveScene
    calls = 0

    def observed(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls != 1:
            raise RuntimeError("light_inventory_unexpected_second_scene")
        scene = original(*args, **kwargs)
        observer(scene)
        return scene

    module.InteractiveScene = observed
    try:
        yield
        if calls != 1:
            raise RuntimeError("light_inventory_scene_hook_not_reached")
    finally:
        module.InteractiveScene = original
