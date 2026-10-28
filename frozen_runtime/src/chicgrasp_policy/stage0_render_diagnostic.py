"""Read-only Stage0 render-state diagnostics, never a rendering correction.

Native imports occur only inside snapshot(). Unknown values/errors are preserved;
these observations do not certify native renderer consumption of a USD light.
"""
import json
import math

SCHEMA = "stage0-render-state-diagnostic/v1"
CAMERAS = ("camera_left", "camera_right")
SETTINGS = ("/app/window/enabled", "/app/window/hideUi", "/app/useFabricSceneDelegate",
            "/app/renderer/skipWhileInvisible", "/app/renderer/skipWhileMinimized",
            "/rtx/hydra/readTransformsFromFabricInRenderDelegate",
            "/rtx/scenedb/domeLight/sdiio/enabled", "/rtx/directLighting/domeLight/approxIbl/enabled",
            "/rtx/post/backgroundZeroAlpha/enabled", "/rtx/post/backgroundZeroAlpha/backgroundComposite",
            "/rtx/post/backgroundZeroAlpha/blackBackgroundInComposite",
            "/persistent/app/viewport/displayOptions")


def _value(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (list, tuple):
        return [_value(v) for v in value]
    if type(value) is dict:
        return {str(k): _value(v) for k, v in value.items()}
    # Native Sdf/Gf/Vt values stay explicitly represented, not invented arrays.
    return {"python_type": type(value).__module__ + "." + type(value).__qualname__,
            "representation": repr(value)}


def snapshot(session, *, bindings=None):
    """Read existing objects only. Injected bindings are offline test substitutes."""
    if bindings is None:
        from pxr import Usd, UsdGeom, UsdLux
        from omni.kit.viewport.utility import get_active_viewport
        from omni.ui import Workspace
        bindings = (Usd, UsdGeom, UsdLux, get_active_viewport, Workspace.get_window)
    Usd, UsdGeom, UsdLux, get_viewport, get_window = bindings
    errors = []

    def read(source, getter):
        try:
            return {"source": source, "status": "OBSERVED", "value": _value(getter())}
        except Exception as exc:
            error = type(exc).__name__ + ":" + str(exc)
            errors.append({"source": source, "error": error})
            return {"source": source, "status": "UNAVAILABLE", "error": error}

    def attribute(attr):
        source = str(attr.GetPath())
        return {"path": source,
                "value": read(source + ".Get", attr.Get),
                "authored": read(source + ".HasAuthoredValueOpinion", attr.HasAuthoredValueOpinion),
                "property_stack": read(source + ".GetPropertyStack", lambda: [
                    {"layer": str(s.layer.identifier), "path": str(s.path), "default": _value(s.default)}
                    for s in attr.GetPropertyStack()])}

    def prim_record(prim, *, authored_only=False):
        if not prim.IsValid():
            return {"valid": False}
        path = str(prim.GetPath())
        record = {"valid": True, "path": path, "type": str(prim.GetTypeName()),
                  "active": bool(prim.IsActive()), "loaded": bool(prim.IsLoaded()),
                  "applied_schemas": list(prim.GetAppliedSchemas())}
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            record["computed_visibility"] = read(path + ".ComputeVisibility", lambda:
                str(imageable.ComputeVisibility(Usd.TimeCode.Default())))
            record["computed_purpose"] = read(path + ".ComputePurpose", lambda: str(imageable.ComputePurpose()))
        record["attributes"] = [attribute(a) for a in prim.GetAttributes()
                                if not authored_only or a.HasAuthoredValueOpinion()]
        record["relationships"] = [{"name": str(r.GetName()),
            "targets": read(str(r.GetPath()) + ".GetTargets", lambda r=r: [str(p) for p in r.GetTargets()])}
            for r in prim.GetRelationships()]
        return record

    def viewport_record():
        viewport = get_viewport()
        if viewport is None:
            return {"present": False}
        return {"present": True, **{name: read("active_viewport." + name,
                lambda name=name: getattr(viewport, name)) for name in
                ("updates_enabled", "freeze_frame", "frame_info", "camera_path",
                 "render_product_path", "resolution", "usd_context_name", "hydra_engine")}}

    def window_record():
        window = get_window("Viewport")
        if window is None:
            return {"present": False}
        return {"present": True, **{name: read("Viewport_window." + name,
                lambda name=name: getattr(window, name)) for name in ("visible", "docked", "selected_in_dock")}}

    stage = session.stage
    result = {"schema": SCHEMA, "read_only": True,
              "viewport": read("get_active_viewport", viewport_record),
              "window": read("Workspace.get_window(Viewport)", window_record),
              "simulation": {name: read("SimulationContext." + name, lambda name=name:
                    getattr(session.env.sim, name)) for name in
                    ("render_mode", "_has_gui", "_offscreen_render", "_render_viewport")},
              "settings": {key: read("carb.settings" + key, lambda key=key:
                    session.env.sim.get_setting(key)) for key in SETTINGS},
              "prims": {}, "camera_render_products": {}, "render_scope": None}
    for path in ("/World", "/World/DomeLight", "/OmniKit_Viewport_LightRig"):
        result["prims"][path] = read(path, lambda path=path: prim_record(stage.GetPrimAtPath(path)))
    for name in CAMERAS:
        result["camera_render_products"][name] = read(name + ".render_product_paths", lambda name=name:
            [prim_record(stage.GetPrimAtPath(path)) for path in session.env.scene[name].render_product_paths])
    result["resolved_dome_selection"] = {
        name: {"status": "UNKNOWN", "resolved_prim_path": None,
               "reason": "All exposed product attributes and relationship targets are recorded above; "
                         "a domeLight source token does not identify a resolved light prim."}
        for name in CAMERAS}

    def render_scope():
        root = stage.GetPrimAtPath("/Render")
        if not root.IsValid():
            return []
        # Inspect existing rendering prims, not the robot/chicken/asset tree.
        return [prim_record(prim, authored_only=True) for prim in Usd.PrimRange(root)]
    result["render_scope"] = read("Usd.PrimRange(/Render); authored opinions only", render_scope)

    def light_links():
        api = UsdLux.LightAPI(stage.GetPrimAtPath("/World/DomeLight"))
        return {name: {"include_root": read(name + ".GetIncludeRootAttr.Get", lambda c=collection:
                        c.GetIncludeRootAttr().Get()),
                       "includes": read(name + ".GetIncludesRel.GetTargets", lambda c=collection:
                        [str(p) for p in c.GetIncludesRel().GetTargets()]),
                       "excludes": read(name + ".GetExcludesRel.GetTargets", lambda c=collection:
                        [str(p) for p in c.GetExcludesRel().GetTargets()])}
                for name, collection in (("light_link", api.GetLightLinkCollectionAPI()),
                                         ("shadow_link", api.GetShadowLinkCollectionAPI()))}
    result["light_links"] = read("UsdLux.LightAPI(DomeLight) collections", light_links)
    result["errors"] = errors
    result["status"] = "PARTIAL" if errors else "COMPLETE"
    return json.loads(json.dumps(result, allow_nan=False))
