"""Read-only, import-isolated check of the owner's exact Grey Studio export.

The native snapshot resolves asset values in their originating layers and saves
actual composed values. Passing verifies composition, not rendered appearance.
"""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .stage0_visuals import config_snapshot

SCHEMA = "stage0-grey-studio-check/v1"
EARLY_PHASE = "BEFORE_INITIAL_RESET_AND_POLICY_RGB"
ASSET = Path(__file__).resolve().parents[2] / "assets/lighting/grey_studio/light_rig_environment.usda"
ASSET_SHA256 = "42e90211ef403c4d6e61fd07afaf9481ebe4734bfabcd27556a1b6b7e814723c"
ROOT = "/Environment"
RIG = ROOT + "/Grey_Studio"
PLANE = RIG + "/Plane"
MATERIAL = RIG + "/Looks/Ground_Material"
SHADER = MATERIAL + "/Shader"
MDL = "C:/isaacsim/kit/mdl/core/Base/OmniPBR.mdl"
RIG_MATRIX = [[0., 1., 0., 0.], [0., 0., 1., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]]
IDENTITY = np.eye(4).tolist()
EXPECTED = {
    ROOT: {"xformOp:translate": [0., 0., 0.], "xformOp:orient": [1., 0., 0., 0.],
           "xformOp:scale": [1., 1., 1.],
           "xformOpOrder": ["xformOp:translate", "xformOp:orient", "xformOp:scale"]},
    RIG: {"xformOp:transform": RIG_MATRIX, "xformOpOrder": ["xformOp:transform"]},
    PLANE: {"visibility": "invisible", "xformOp:translate": [0., 0., 0.],
            "xformOp:rotateXYZ": [0., 0., 0.], "xformOp:scale": [286.92724609375] * 3,
            "xformOpOrder": ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]},
    RIG + "/DistantLight": {"inputs:intensity": 3000., "inputs:angle": 34.3,
        "inputs:shaping:cone:angle": 180., "xformOp:rotateXYZ": [255.70177427072963, 0., 0.],
        "xformOp:translate": [0., 0., 0.], "xformOp:scale": [1., 1., 1.],
        "xformOpOrder": ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]},
    RIG + "/DomeLight": {"inputs:intensity": 1003.3, "inputs:exposure": .4,
        "inputs:shaping:cone:angle": 180., "inputs:texture:format": "latlong",
        "visibleInPrimaryRay": True, "xformOp:rotateXYZ": [270., -128.27698468300977, 0.],
        "xformOp:translate": [0., 0., 0.], "xformOp:scale": [1., 1., 1.],
        "xformOpOrder": ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]},
}
TYPES = {ROOT: "Xform", RIG: "Xform", PLANE: "Mesh",
         RIG + "/DistantLight": "DistantLight", RIG + "/DomeLight": "DomeLight"}


def _value(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise ValueError("nonfinite_native_value")
        return float(value)
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        return [float(value.GetReal()), *map(float, value.GetImaginary())]
    return [_value(v) for v in value]


def _path(value):
    return str(value).replace("\\", "/").casefold()


def _same_export_location(value):
    """Accept an existing filesystem alias only for the unchanged frozen export."""
    if _path(value) == _path(ASSET):
        return True
    try:
        path = Path(value)
        return (path.is_absolute() and path.resolve(strict=True) == ASSET.resolve(strict=True)
                and path.samefile(ASSET)
                and hashlib.sha256(ASSET.read_bytes()).hexdigest() == ASSET_SHA256)
    except (TypeError, ValueError, OSError):
        return False


def _close(actual, expected, *, matrix=False):
    if isinstance(expected, (str, bool)):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, list) and expected and isinstance(expected[0], str):
        return actual == expected
    try:
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        return (a.shape == b.shape and bool(np.isfinite(a).all())
                and bool(np.allclose(a, b, rtol=0 if matrix else 1.2e-7,
                                     atol=2e-7 if matrix else 1e-9)))
    except (ValueError, TypeError):
        return False


def validate_report(report, calibration):
    """Recheck saved facts, never trust a PASS label in place of observations."""
    def require(ok, message):
        if not ok:
            raise ValueError("grey_studio:" + message)
    require(report.get("schema") == SCHEMA and report.get("read_only") is True, "schema")
    require(not report.get("errors") and report.get("status") == "COMPLETE", "read_errors")
    require(report.get("asset", {}).get("sha256") == ASSET_SHA256, "export_identity")
    require(_same_export_location(report["asset"]["path"]), "export_location")
    require(report.get("rig_paths") == [RIG] and report.get("nested_environment") is False,
            "duplicate_or_nested_rig")
    references = report.get("references", [])
    require(len(references) == 1, "root_reference_count")
    require(_same_export_location(references[0].get("resolved"))
            and references[0].get("prim_path") in (ROOT, ""), "root_reference_target")
    require(report.get("root_export_spec_count") == 1, "composed_export_count")
    for path, expected in EXPECTED.items():
        item = report.get("prims", {}).get(path, {})
        require(item.get("valid") is True and item.get("active") is True
                and item.get("loaded") is True and item.get("type") == TYPES[path], "prim:" + path)
        require(item.get("reset_xform_stack") is False, "reset_transform:" + path)
        require(not any("physics" in str(s).lower() or "physx" in str(s).lower()
                        for s in item.get("applied_schemas", [])), "physics_schema:" + path)
        require(item.get("effective_visibility") == ("invisible" if path == PLANE else "inherited"),
                "visibility:" + path)
        for key, value in expected.items():
            require(_close(item.get("attributes", {}).get(key), value), "attribute:" + path + "." + key)
    require(_close(report["prims"][ROOT].get("world_matrix"), IDENTITY, matrix=True), "root_world_transform")
    require(_close(report["prims"][RIG].get("world_matrix"), RIG_MATRIX, matrix=True), "rig_world_transform")
    require(report["prims"][PLANE].get("material_binding") == [MATERIAL], "plane_material")
    require(report["prims"][PLANE].get("material_strength") == "weakerThanDescendants", "material_strength")
    require(report["prims"][RIG + "/DomeLight"].get("texture_file") in (None, ""), "unexpected_dome_texture")
    require(report.get("rig_physics_schemas") == [], "rig_physics_schema")
    dependencies = report.get("dependencies", [])
    mdl = [d for d in dependencies if d.get("property") == SHADER + ".info:mdl:sourceAsset"]
    require(len(mdl) == 1 and _path(mdl[0].get("authored")) == _path(MDL)
            and _path(mdl[0].get("resolved")) == _path(MDL), "rig_mdl_resolution")
    for dependency in dependencies:
        require(bool(dependency.get("source_layer")) and bool(dependency.get("anchored"))
                and bool(dependency.get("resolved")), "unresolved_dependency:" + str(dependency.get("property")))
    expected_cameras = config_snapshot(calibration)["cameras"]
    require(set(report.get("cameras", {})) == set(expected_cameras), "camera_keys")
    for name, expected in expected_cameras.items():
        item = report["cameras"][name]
        require(item.get("path") == expected["prim_path"] and item.get("type") == "Camera"
                and item.get("active") is True and item.get("loaded") is True
                and item.get("effective_visibility") == "inherited", "camera_identity:" + name)
        for key in ("parent_world_matrix", "world_matrix", "local_matrix"):
            require(_close(item.get(key), expected[key], matrix=True), "camera_transform:" + name + ":" + key)
    return True


def snapshot(stage, calibration, *, bindings=None):
    """Observe only. Bindings=(Usd, UsdGeom, Sdf, Ar) are substituted offline."""
    if bindings is None:
        from pxr import Usd, UsdGeom, Sdf, Ar
        bindings = (Usd, UsdGeom, Sdf, Ar)
    Usd, UsdGeom, Sdf, Ar = bindings
    report = dict(schema=SCHEMA, read_only=True, asset={}, prims={}, cameras={},
                  dependencies=[], references=[], rig_paths=[], nested_environment=False,
                  dependency_coverage="Stage.TraverseAll authored asset attributes at default time; no instance proxies",
                  root_export_spec_count=0, rig_physics_schemas=[], errors=[], status="COMPLETE")

    def collect(label, function):
        try:
            return function()
        except Exception as exc:
            report["errors"].append(dict(source=label, error=type(exc).__name__ + ":" + str(exc)))
            return None

    collect("export_identity", lambda: report["asset"].update(
        path=str(ASSET), sha256=hashlib.sha256(ASSET.read_bytes()).hexdigest()))
    cache = collect("XformCache", lambda: UsdGeom.XformCache(Usd.TimeCode.Default()))

    def prim_record(prim):
        if not prim.IsValid():
            return {"valid": False}
        result = dict(valid=True, path=str(prim.GetPath()), type=str(prim.GetTypeName()),
            active=bool(prim.IsActive()), loaded=bool(prim.IsLoaded()),
            applied_schemas=list(prim.GetAppliedSchemas()))
        imageable = UsdGeom.Imageable(prim)
        result["effective_visibility"] = str(imageable.ComputeVisibility(Usd.TimeCode.Default())) if imageable else None
        result["ancestor_visibility"] = []
        ancestor = prim
        while ancestor.IsValid() and str(ancestor.GetPath()) != "/":
            visibility = ancestor.GetAttribute("visibility")
            result["ancestor_visibility"].append(dict(path=str(ancestor.GetPath()),
                visibility=_value(visibility.Get()) if visibility.IsValid() else None,
                active=bool(ancestor.IsActive())))
            ancestor = ancestor.GetParent()
        xform = UsdGeom.Xformable(prim)
        if xform:
            result.update(reset_xform_stack=bool(xform.GetResetXformStack()),
                local_matrix=_value(xform.GetLocalTransformation()),
                world_matrix=_value(cache.GetLocalToWorldTransform(prim)),
                parent_world_matrix=_value(cache.GetLocalToWorldTransform(prim.GetParent())))
        return result

    for path, expected in EXPECTED.items():
        def read_prim(path=path, expected=expected):
            prim = stage.GetPrimAtPath(path)
            item = prim_record(prim)
            if item["valid"]:
                item["attributes"] = {key: _value(prim.GetAttribute(key).Get()) for key in expected}
                if path == PLANE:
                    relation = prim.GetRelationship("material:binding")
                    item["material_binding"] = [str(p) for p in relation.GetTargets()]
                    item["material_strength"] = str(relation.GetMetadata("bindMaterialAs"))
                if path.endswith("/DomeLight"):
                    texture = prim.GetAttribute("inputs:texture:file")
                    value = texture.Get() if texture.IsValid() else None
                    item["texture_file"] = None if value is None else str(value.path)
            report["prims"][path] = item
        collect(path, read_prim)
    for name, camera in calibration["cameras"].items():
        collect(name, lambda name=name, camera=camera: report["cameras"].update(
            {name: prim_record(stage.GetPrimAtPath(camera["camera_prim_path"]))}))

    def resolve_assets():
        resolver = Ar.GetResolver()
        with Ar.ResolverContextBinder(stage.GetPathResolverContext()):
            root = stage.GetPrimAtPath(ROOT)
            for spec in root.GetPrimStack():
                if _path(spec.layer.realPath or spec.layer.identifier) == _path(ASSET) and str(spec.path) == ROOT:
                    report["root_export_spec_count"] += 1
                # GetAppliedItems includes prepend/append edits used by AddReference.
                # Added-or-explicit alone must not be assumed to include them.
                for ref in spec.referenceList.GetAppliedItems():
                    anchored = Sdf.ComputeAssetPathRelativeToLayer(spec.layer, ref.assetPath)
                    report["references"].append(dict(source_layer=str(spec.layer.identifier),
                        authored=str(ref.assetPath), prim_path=str(ref.primPath), anchored=str(anchored),
                        resolved=str(resolver.Resolve(anchored))))
            for prim in stage.TraverseAll():
                path = str(prim.GetPath())
                if path.endswith("/Grey_Studio"):
                    report["rig_paths"].append(path)
                if path == ROOT + ROOT or path.startswith(ROOT + ROOT + "/"):
                    report["nested_environment"] = True
                if path == ROOT or path.startswith(ROOT + "/"):
                    for schema in prim.GetAppliedSchemas():
                        if "physics" in str(schema).lower() or "physx" in str(schema).lower():
                            report["rig_physics_schemas"].append(dict(path=path, schema=str(schema)))
                for attr in prim.GetAttributes():
                    if str(attr.GetTypeName()) not in ("asset", "asset[]") or not attr.HasAuthoredValueOpinion():
                        continue
                    value = attr.Get()
                    values = list(value) if str(attr.GetTypeName()) == "asset[]" else [value]
                    source = next((s for s in attr.GetPropertyStack() if s.default is not None), None)
                    for value in values:
                        if value is None or not value.path:
                            continue
                        if source is None:
                            raise ValueError("asset_source_layer_unavailable:" + str(attr.GetPath()))
                        anchored = Sdf.ComputeAssetPathRelativeToLayer(source.layer, value.path)
                        report["dependencies"].append(dict(property=str(attr.GetPath()),
                            authored=str(value.path), source_layer=str(source.layer.identifier),
                            source_property=str(source.path), anchored=str(anchored),
                            resolved=str(resolver.Resolve(anchored)),
                            usd_resolved_path=str(value.resolvedPath)))
    collect("resolver_and_composition", resolve_assets)
    report["status"] = "PARTIAL" if report["errors"] else "COMPLETE"
    # Validation occurs after the caller persists this strict JSON report.
    return json.loads(json.dumps(report, allow_nan=False))


def write_check(stage, output_dir, calibration=None):
    """Construction-observer boundary: persist even a failed check before raising.

    Called after InteractiveScene construction and before its initial reset;
    this function never initializes sensors or acquires images itself.
    """
    if calibration is None:
        calibration = json.loads((Path(__file__).resolve().parents[2] /
                                  "config/policy_camera_calibration.json").read_text(encoding="utf-8"))
    report = snapshot(stage, calibration)
    report["phase"] = EARLY_PHASE
    destination = Path(output_dir) / "grey_studio_check.json"
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    validate_report(report, calibration)
    return report
