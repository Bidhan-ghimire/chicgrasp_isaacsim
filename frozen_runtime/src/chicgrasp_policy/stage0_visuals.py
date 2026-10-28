"""Pure Stage0 camera/light configuration and USD row-matrix round trips.

No native imports, file writes, or simulator operations. Camera poses are raw
USD/OpenGL WXYZ; all translations are metres. Matrices use USD row vectors.
"""
from copy import deepcopy
import math

import numpy as np


CAMERA_NAMES = ("PolicyCameraLeft", "PolicyCameraRight")
CAMERA_PARENT = "/World/envs/env_0"
CONFIG_SCHEMA = "gate6-stage0-visuals/v1"
PENDING_STATUS = "STAGE0_PENDING_HEADLESS"
APPROVED_STATUS = "STAGE0_OWNER_APPROVED"
IDENTITY = np.eye(4).tolist()
OPTIC_KEYS = frozenset(("projection", "focal_length", "horizontal_aperture",
    "vertical_aperture", "horizontal_aperture_offset", "vertical_aperture_offset",
    "focus_distance", "f_stop", "clipping_range", "width", "height", "data_types"))
LIGHT_KEYS = frozenset(("color", "intensity", "exposure", "normalize",
    "enable_color_temperature", "color_temperature", "texture_file", "texture_format",
    "visible_in_primary_ray"))
LIGHT_DEFAULTS = dict(color=[0.9, 0.9, 0.9], intensity=1200.0, exposure=0.0,
    normalize=False, enable_color_temperature=False, color_temperature=6500.0,
    texture_file=None, texture_format="automatic", visible_in_primary_ray=True)


def _number(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
        raise ValueError("finite_number_required:" + label)
    return float(value)


def _vector(value, size, label):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != size:
        raise ValueError("vector_shape:" + label)
    return np.array([_number(x, label) for x in value], dtype=np.float64)


def rigid_matrix(value, label="matrix"):
    """Copy a rigid, proper, unit-scale USD row-vector transform or reject it."""
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("matrix_shape:" + label) from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("matrix_shape_or_nonfinite:" + label)
    if not np.allclose(matrix[:, 3], [0, 0, 0, 1], atol=1e-10, rtol=0):
        raise ValueError("matrix_not_affine:" + label)
    rotation = matrix[:3, :3]
    if (not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8, rtol=0)
            or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-8, rel_tol=0)):
        raise ValueError("matrix_scale_shear_or_reflection:" + label)
    return matrix.copy()


def pose_matrix(position, quaternion_wxyz):
    """Construct a rigid USD row-vector matrix from a parent-relative pose."""
    position = _vector(position, 3, "position")
    quaternion = _vector(quaternion_wxyz, 4, "quaternion_wxyz")
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, abs_tol=1e-8, rel_tol=0):
        raise ValueError("unit_quaternion_required")
    w, x, y, z = quaternion / norm
    column_rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    result = np.eye(4)
    result[:3, :3], result[3, :3] = column_rotation.T, position
    return result.tolist()


def matrix_pose(matrix):
    """Return translation and WXYZ without rounding or discarding scale/shear."""
    matrix = rigid_matrix(matrix)
    r = matrix[:3, :3].T
    trace = float(np.trace(r))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = np.array([s/4, (r[2, 1]-r[1, 2])/s, (r[0, 2]-r[2, 0])/s, (r[1, 0]-r[0, 1])/s])
    else:
        index = int(np.argmax(np.diag(r)))
        j, k = (index+1) % 3, (index+2) % 3
        s = math.sqrt(1.0 + r[index, index] - r[j, j] - r[k, k]) * 2
        q = np.zeros(4)
        q[0] = (r[k, j]-r[j, k])/s
        q[index+1] = s/4
        q[j+1] = (r[j, index]+r[index, j])/s
        q[k+1] = (r[k, index]+r[index, k])/s
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q *= -1
    return matrix[3, :3].tolist(), q.tolist()


def _transforms(record, label):
    parent = rigid_matrix(record["parent_world_matrix"], label+":parent")
    local = rigid_matrix(record["local_matrix"], label+":local")
    world = rigid_matrix(record["world_matrix"], label+":world")
    if not np.allclose(local @ parent, world, atol=1e-8, rtol=0):
        raise ValueError("local_parent_world_mismatch:" + label)
    return parent, local, world


def _light_properties(properties):
    if not isinstance(properties, dict) or set(properties) != LIGHT_KEYS:
        raise ValueError("light_property_keys")
    color = _vector(properties["color"], 3, "light_color")
    if np.any(color < 0):
        raise ValueError("negative_light_color")
    for name in ("intensity", "exposure", "color_temperature"):
        _number(properties[name], "light_"+name)
    if properties["intensity"] < 0 or not 1000 <= properties["color_temperature"] <= 10000:
        raise ValueError("light_property_range")
    for name in ("normalize", "enable_color_temperature", "visible_in_primary_ray"):
        if type(properties[name]) is not bool:
            raise ValueError("light_boolean:" + name)
    if properties["texture_file"] is not None and not isinstance(properties["texture_file"], str):
        raise ValueError("light_texture_file")
    if properties["texture_format"] not in ("automatic", "latlong", "mirroredBall", "angular", "cubeMapVerticalCross"):
        raise ValueError("light_texture_format")


def _optics(shared):
    """Full effective values expected before native USD float representation."""
    return dict(projection="perspective", focal_length=shared["focal_length"],
        horizontal_aperture=shared["horizontal_aperture"],
        vertical_aperture=shared.get("vertical_aperture", shared["horizontal_aperture"]*shared["height"]/shared["width"]),
        horizontal_aperture_offset=shared.get("horizontal_aperture_offset", 0.0),
        vertical_aperture_offset=shared.get("vertical_aperture_offset", 0.0),
        focus_distance=shared["focus_distance"], f_stop=shared.get("f_stop", 0.0),
        clipping_range=list(shared["clipping_range"]), width=shared["width"],
        height=shared["height"], data_types=list(shared["data_types"]))


def _validate_optics(actual, expected, *, exact=False):
    if not isinstance(actual, dict) or set(actual) != OPTIC_KEYS:
        raise ValueError("effective_optic_keys")
    for key in OPTIC_KEYS:
        value, baseline = actual[key], expected[key]
        if key in ("projection", "width", "height", "data_types"):
            if value != baseline or type(value) is not type(baseline):
                raise ValueError("immutable_optics:" + key)
        else:
            a = _vector(value, 2, key) if key == "clipping_range" else np.array(_number(value, key))
            b = np.asarray(baseline, dtype=np.float64)
            # Initial native fields can be USD float32 representations of JSON doubles.
            matches = np.array_equal(a, b) if exact else np.allclose(a, b, rtol=1.2e-7, atol=1e-9)
            if not matches:
                raise ValueError("immutable_optics:" + key)


def _validate_initial_light(actual, configured):
    _light_properties(actual)
    for key in LIGHT_KEYS:
        a, b = actual[key], configured[key]
        if key in ("color", "intensity", "exposure", "color_temperature"):
            if not np.allclose(a, b, rtol=1.2e-7, atol=1e-9):
                raise ValueError("initial_light_configuration_mismatch:"+key)
        elif a != b or type(a) is not type(b):
            raise ValueError("initial_light_configuration_mismatch:"+key)


def validate_config(config):
    """Return a validated independent configuration; pending is never approval."""
    result = deepcopy(config)
    if result.get("calibration_status") not in ("USER_APPROVED", PENDING_STATUS, APPROVED_STATUS):
        raise ValueError("unknown_visual_approval_status")
    if set(result.get("cameras", {})) != set(CAMERA_NAMES):
        raise ValueError("camera_identity_keys")
    shared = result["shared_camera_settings"]
    if (shared.get("width") != 320 or type(shared.get("width")) is not int
            or shared.get("height") != 240 or type(shared.get("height")) is not int
            or shared.get("data_types") != ["rgb"] or shared.get("rgb_only") is not True
            or shared.get("projection_type", "pinhole") != "pinhole"):
        raise ValueError("fixed_camera_format")
    for field in ("focal_length", "horizontal_aperture", "focus_distance"):
        if _number(shared[field], field) <= 0:
            raise ValueError("positive_optic_required:"+field)
    clip = _vector(shared["clipping_range"], 2, "clipping_range")
    if not 0 < clip[0] < clip[1]:
        raise ValueError("clipping_range_order")
    effective = _optics(shared)
    _validate_optics(effective, effective, exact=True)
    if effective["vertical_aperture"] <= 0 or effective["f_stop"] < 0:
        raise ValueError("optic_range")
    for name in CAMERA_NAMES:
        record = result["cameras"][name]
        if (record.get("camera_prim_name") != name or record.get("camera_prim_path") != CAMERA_PARENT+"/"+name
                or record.get("camera_parent_prim") != CAMERA_PARENT
                or record.get("direct_child_of_environment_root") is not True
                or record.get("camera_not_under_robot_or_chicken") is not True):
            raise ValueError("camera_identity_or_parent:"+name)
        if record.get("quaternion_convention", "wxyz") != "wxyz" or record.get("camera_convention", "opengl") != "opengl":
            raise ValueError("camera_quaternion_convention")
        pose_matrix(record["position_environment_m"], record["quaternion_environment_wxyz"])
        rigid_matrix(record.get("parent_world_matrix", IDENTITY), name+":parent")
    light = result["scene_light"]
    if (light.get("type") != "DomeLight" or light.get("prim_path") != "/World/DomeLight"
            or light.get("parent_path") != "/World" or light.get("quaternion_convention") != "wxyz"):
        raise ValueError("light_identity_or_parent")
    pose_matrix(light["position_parent_m"], light["quaternion_parent_wxyz"])
    rigid_matrix(light["parent_world_matrix"], "light_parent")
    _light_properties(light["properties"])
    return result


def config_snapshot(config):
    """Synthesize the snapshot schema for offline roundtrip tests, never evidence."""
    config = validate_config(config)
    cameras = {}
    for name, camera in config["cameras"].items():
        parent = rigid_matrix(camera.get("parent_world_matrix", IDENTITY))
        local = pose_matrix(camera["position_environment_m"], camera["quaternion_environment_wxyz"])
        cameras[name] = dict(prim_path=camera["camera_prim_path"], parent_path=camera["camera_parent_prim"],
            parent_world_matrix=parent.tolist(), local_matrix=local,
            world_matrix=(np.asarray(local) @ parent).tolist(), effective_optics=_optics(config["shared_camera_settings"]))
    light = config["scene_light"]
    parent = rigid_matrix(light["parent_world_matrix"])
    local = pose_matrix(light["position_parent_m"], light["quaternion_parent_wxyz"])
    return dict(cameras=cameras, light=dict(type=light["type"], prim_path=light["prim_path"],
        parent_path=light["parent_path"], parent_world_matrix=parent.tolist(), local_matrix=local,
        world_matrix=(np.asarray(local) @ parent).tolist(), properties=deepcopy(light["properties"])))


def validate_export(snapshot, baseline_config, *, initial_snapshot=None):
    """Convert native transforms to paste-ready config, preserving frozen optics.

    Pass the initial native snapshot to compare optics exactly throughout tuning.
    Without it, only USD float representation tolerance against config is allowed.
    No round-trip changes gain approval; the returned status is always pending.
    """
    result = validate_config(baseline_config)
    if set(snapshot.get("cameras", {})) != set(CAMERA_NAMES):
        raise ValueError("camera_identity_keys")
    expected_optics = _optics(result["shared_camera_settings"])
    configured_snapshot = config_snapshot(result) if initial_snapshot is not None else None
    for name in CAMERA_NAMES:
        current = snapshot["cameras"][name]
        if current.get("prim_path") != CAMERA_PARENT+"/"+name or current.get("parent_path") != CAMERA_PARENT:
            raise ValueError("camera_identity_or_parent:"+name)
        parent, local, world = _transforms(current, name)
        _validate_optics(current["effective_optics"], expected_optics)
        if initial_snapshot is not None:
            initial = initial_snapshot["cameras"][name]
            if initial.get("prim_path") != current["prim_path"] or initial.get("parent_path") != current["parent_path"]:
                raise ValueError("initial_camera_identity_or_parent:"+name)
            initial_parent, initial_local, _ = _transforms(initial, name+":initial")
            configured = configured_snapshot["cameras"][name]
            if (not np.allclose(initial_local, configured["local_matrix"], atol=2e-7, rtol=0)
                    or not np.allclose(initial_parent, configured["parent_world_matrix"], atol=1e-8, rtol=0)):
                raise ValueError("initial_camera_configuration_mismatch:"+name)
            _validate_optics(initial["effective_optics"], expected_optics)
            _validate_optics(current["effective_optics"], initial["effective_optics"], exact=True)
            if not np.array_equal(parent, rigid_matrix(initial["parent_world_matrix"])):
                raise ValueError("camera_parent_transform_changed:"+name)
        position, quaternion = matrix_pose(local)
        record = result["cameras"][name]
        historical_fields = ("source_json_path", "robot_command_issued", "jaw_command_issued",
            "requested_tcp_translation_m", "requested_tcp_movement_norm_m", "actual_tcp_movement",
            "chicken_movement", "quaternion_norm_before_normalization", "quaternion_norm_after_normalization")
        historical = record.setdefault("historical_calibration_provenance", {})
        for field in historical_fields:
            if field in record:
                historical[field] = record.pop(field)
        result["cameras"][name].update(position_environment_m=position,
            quaternion_environment_wxyz=quaternion, quaternion_convention="wxyz", camera_convention="opengl",
            parent_world_matrix=parent.tolist(), exported_world_matrix=world.tolist(),
            pose_source="Stage0 native USD snapshot", snapshot_reference="stage0_export.cameras."+name)
    light = snapshot["light"]
    if light.get("type") != "DomeLight" or light.get("prim_path") != "/World/DomeLight" or light.get("parent_path") != "/World":
        raise ValueError("light_identity_or_parent")
    parent, local, world = _transforms(light, "light")
    if initial_snapshot is not None:
        initial_light = initial_snapshot["light"]
        if any(initial_light.get(key) != light[key] for key in ("type", "prim_path", "parent_path")):
            raise ValueError("initial_light_identity_or_parent")
        initial_parent, initial_local, _ = _transforms(initial_light, "initial_light")
        configured = configured_snapshot["light"]
        if (not np.allclose(initial_local, configured["local_matrix"], atol=2e-7, rtol=0)
                or not np.allclose(initial_parent, configured["parent_world_matrix"], atol=1e-8, rtol=0)):
            raise ValueError("initial_light_configuration_mismatch:transform")
        if not np.array_equal(parent, initial_parent):
            raise ValueError("light_parent_transform_changed")
        _validate_initial_light(initial_light["properties"], result["scene_light"]["properties"])
    _light_properties(light["properties"])
    position, quaternion = matrix_pose(local)
    result["scene_light"].update(position_parent_m=position, quaternion_parent_wxyz=quaternion,
        parent_world_matrix=parent.tolist(), exported_world_matrix=world.tolist(), properties=deepcopy(light["properties"]))
    result.setdefault("historical_calibration_status", result["calibration_status"])
    result["calibration_status"] = PENDING_STATUS
    result["visual_configuration_schema"] = CONFIG_SCHEMA
    result["stage0_export"] = deepcopy(snapshot)
    return validate_config(result)
