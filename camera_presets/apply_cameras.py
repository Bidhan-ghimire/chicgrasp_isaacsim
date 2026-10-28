"""Optional presentation cameras for an already loaded Isaac Sim scene.

Importing this module uses only the Python standard library. Creation is explicit;
it never starts Kit, steps physics, renders, changes lighting, or loads a model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PRESET_DIR = Path(__file__).resolve().parent


def load_presets(directory=None):
    directory = Path(directory) if directory is not None else PRESET_DIR
    return tuple(json.loads((directory / name).read_text(encoding="utf-8"))
                 for name in ("HAND_DEMO_ONLY.json", "HD_TWINS_DEMO_ONLY.json"))


def checked_matrix(value):
    if len(value) != 4 or any(len(row) != 4 for row in value):
        raise ValueError("Expected a 4x4 matrix")
    matrix = [[float(x) for x in row] for row in value]
    if not all(math.isfinite(x) for row in matrix for x in row):
        raise ValueError("Nonfinite camera matrix")
    if any(abs(matrix[i][3] - (1 if i == 3 else 0)) > 1e-12 for i in range(4)):
        raise ValueError("Expected a Gf row-vector affine matrix")
    return matrix


def compose_world_matrix(local, parent_world):
    """Diagnostic only: Gf row-vector world = local @ parent_world.

    The runtime uses native parent inheritance, not this function as a follower.
    Keep the complete owner-approved local matrix, including its scale.
    """
    a, b = checked_matrix(local), checked_matrix(parent_world)
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)]


def _matrix_error(a, b):
    return max(abs(float(a[i][j]) - float(b[i][j]))
               for i in range(4) for j in range(4))


def _optics_snapshot(prim):
    return {a.GetName(): str(a.Get()) for a in prim.GetAttributes()
            if not a.GetName().startswith("xformOp")}


def copy_camera_attributes(source, target):
    """Read the source; author only on the new destination camera."""
    if source is target or str(source.GetPath()) == str(target.GetPath()):
        raise ValueError("A policy camera cannot be its own presentation twin")
    if any("physics" in str(s).lower() or "physx" in str(s).lower()
           for s in source.GetAppliedSchemas()):
        raise ValueError("Refuse physical schemas on a camera source")
    schemas = source.GetMetadata("apiSchemas")
    if schemas is not None:
        target.SetMetadata("apiSchemas", schemas)
    for attr in source.GetAttributes():
        if attr.GetName().startswith("xformOp"):
            continue
        value = attr.Get()
        if value is not None:
            target.CreateAttribute(attr.GetName(), attr.GetTypeName(),
                                   attr.IsCustom(), attr.GetVariability()).Set(value)


def _snapshot(prim, UsdGeom):
    return {
        "path": str(prim.GetPath()),
        "attributes": {a.GetName(): str(a.Get()) for a in prim.GetAttributes()},
        "schemas": list(prim.GetAppliedSchemas()),
        "parent": str(prim.GetParent().GetPath()),
        "world": str(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)),
        "reset": UsdGeom.Xformable(prim).GetResetXformStack(),
    }


def _set_matrix(prim, matrix, Gf, UsdGeom):
    matrix = checked_matrix(matrix)
    transform = UsdGeom.Xformable(prim)
    transform.ClearXformOpOrder()
    transform.AddTransformOp().Set(Gf.Matrix4d(*[x for row in matrix for x in row]))
    transform.SetResetXformStack(False)


def _apply_hand(prim, config, Gf, UsdGeom):
    # Sensor initialization can author defaults. Restore approval afterwards.
    _set_matrix(prim, config["local_matrix_row_major"], Gf, UsdGeom)
    for key, value in config["raw_optics"].items():
        prim.GetAttribute(key).Set(value)
    prim.GetAttribute("clippingRange").Set(Gf.Vec2f(*config["clipping"]))
    for key, value in config["other_optics"].items():
        prim.GetAttribute(key).Set(value)


class DemoCameras:
    """Render handles and read-only checks; no simulation-clock ownership."""

    def __init__(self, stage, cameras, paths, configs, policy_before, UsdGeom):
        self.stage, self.cameras, self.paths = stage, cameras, paths
        self.configs, self.policy_before = configs, policy_before
        self._UsdGeom = UsdGeom

    def before_render(self):
        """Call before an already scheduled render; this never moves anything."""
        U = self._UsdGeom
        for path, original in self.policy_before.items():
            if _snapshot(self.stage.GetPrimAtPath(path), U) != original:
                raise RuntimeError("Policy camera changed: " + path)
        for name, config in self.configs.items():
            prim = self.stage.GetPrimAtPath(self.paths[name])
            if any("physics" in str(s).lower() or "physx" in str(s).lower()
                   for s in prim.GetAppliedSchemas()):
                raise RuntimeError("Presentation camera acquired physical schema")
            if tuple(self.cameras[name].get_resolution()) != tuple(config["resolution"]):
                raise RuntimeError("Presentation render-product resolution changed: " + name)
            if name == "HAND":
                xform = U.Xformable(prim)
                if (str(prim.GetParent().GetPath()) != config["parent_link"] or
                        xform.GetResetXformStack() or
                        _matrix_error(xform.GetLocalTransformation(),
                                      config["local_matrix_row_major"]) > 1e-12):
                    raise RuntimeError("Approved HAND local transform changed")
                expected = {**config["raw_optics"], **config["other_optics"]}
                if any(prim.GetAttribute(key).Get() != value for key, value in expected.items()):
                    raise RuntimeError("Approved HAND optics changed")
                if list(prim.GetAttribute("clippingRange").Get()) != config["clipping"]:
                    raise RuntimeError("Approved HAND clipping changed")
            else:
                source = self.stage.GetPrimAtPath(config["copy_from"])
                if (_optics_snapshot(source) != _optics_snapshot(prim) or
                        _matrix_error(U.Xformable(source).ComputeLocalToWorldTransform(0),
                                      U.Xformable(prim).ComputeLocalToWorldTransform(0)) > 1e-10):
                    raise RuntimeError("HD twin differs from its policy source: " + name)

    def live_hand_world_pose(self):
        """Read the sensor's live USD-axis world pose, without moving the camera."""
        return self.cameras["HAND"].get_world_pose(camera_axes="usd")

    def pause(self):
        for camera in self.cameras.values():
            camera.pause()


def create_demo_cameras(stage=None, *, enabled=False, camera_factory=None,
                        preset_directory=None):
    """Opt in after the frozen scene has loaded, reset, and initialized.

    Returns None while disabled. With enabled=True, returns DemoCameras holding
    HAND, LEFT_HD and RIGHT_HD sensor objects. It does not import a preview stage
    or capped chicken. It never enables USD write-back or disables Fabric.
    """
    if not enabled:
        return None
    hand, hd = load_presets(preset_directory)
    for preset in (hand, hd):
        if preset.get("DEMO_ONLY") is not True or preset.get("policy_observation_member") is not False:
            raise ValueError("Expected presentation-only camera presets")
    from pxr import Gf, UsdGeom
    if stage is None:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise ValueError("Load and initialize the frozen scene first")
    if UsdGeom.GetStageMetersPerUnit(stage) != hand["transform_application"]["stage_meters_per_unit"]:
        raise ValueError("HAND preset requires a one-metre stage unit")
    configs = {"HAND": hand["camera"], **hd["views"]}
    paths = {"HAND": hand["prim_path"],
             **{name: "/World/DemoOnlyFilming/" + name for name in hd["views"]}}
    if any(stage.GetPrimAtPath(path).IsValid() for path in paths.values()):
        raise ValueError("Presentation prim already exists; refuse to overwrite it")
    if not stage.GetPrimAtPath(configs["HAND"]["parent_link"]).IsValid():
        raise ValueError("Expected gripper_base_link is missing")
    policy_before = {}
    for config in hd["views"].values():
        source = stage.GetPrimAtPath(config["copy_from"])
        if not source.IsValid() or not source.IsA(UsdGeom.Camera):
            raise ValueError("Expected policy camera is missing")
        policy_before[config["copy_from"]] = _snapshot(source, UsdGeom)
    if camera_factory is None:
        from isaacsim.sensors.camera import Camera
        camera_factory = Camera
    cameras = {}
    for name, config in configs.items():
        camera = UsdGeom.Camera.Define(stage, paths[name])
        prim = camera.GetPrim()
        if name == "HAND":
            _apply_hand(prim, config, Gf, UsdGeom)
        else:
            source = stage.GetPrimAtPath(config["copy_from"])
            copy_camera_attributes(source, prim)
        sensor = camera_factory(prim_path=paths[name], name="DEMO_ONLY_" + name,
                                frequency=-1, resolution=tuple(config["resolution"]),
                                annotator_device="cpu")
        sensor.initialize()
        cameras[name] = sensor
        if name == "HAND":
            _apply_hand(prim, config, Gf, UsdGeom)
        else:
            copy_camera_attributes(source, prim)
            world = UsdGeom.Xformable(source).ComputeLocalToWorldTransform(0)
            parent_world = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(0)
            _set_matrix(prim, world * parent_world.GetInverse(), Gf, UsdGeom)
    result = DemoCameras(stage, cameras, paths, configs, policy_before, UsdGeom)
    result.before_render()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect optional camera presets; never start Kit.")
    parser.add_argument("--describe", action="store_true", help="Print exact saved settings offline")
    options = parser.parse_args()
    if options.describe:
        hand, twins = load_presets()
        print(json.dumps({"enabled_by_default": False, "HAND": hand["camera"],
                          "HD_TWINS": twins["views"]}, indent=2))
    else:
        parser.print_help()
