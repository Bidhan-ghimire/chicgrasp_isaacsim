"""Observation-only primitive control and read-only state adapters.

Ordinary import is native-isolated. Only the explicit scene-spawn callback writes
USD, exclusively below CONTROL_ROOT; no helper advances simulation or queries a
contact report. The classifier consumes the collector's already copied snapshot.
"""
from copy import deepcopy
import math
import numpy as np

CONTROL_ROOT = "/World/ChicGraspObservationControl"
CONTROL_BODY = CONTROL_ROOT + "/Body"
CONTROL_SUPPORT = CONTROL_ROOT + "/Support"
BODY_COLLIDER = CONTROL_BODY + "/geometry/mesh"
SUPPORT_COLLIDER = CONTROL_SUPPORT + "/geometry/mesh"
PHYSICS_SCENE = "/physicsScene"
JAW_NAMES = tuple("PrismaticJoint" + str(i) for i in range(1, 5))
JAW_LINKS = tuple("jaw_" + str(i) + "_link" for i in range(1, 5))


def observation_control_spec():
    """Frozen diagnostic settings, not grasp/force acceptance criteria."""
    return {
        "schema": 1, "root": CONTROL_ROOT, "body_path": CONTROL_BODY,
        "support_path": CONTROL_SUPPORT, "body_collider_path": BODY_COLLIDER,
        "support_collider_path": SUPPORT_COLLIDER, "physics_scene": PHYSICS_SCENE,
        "shape": "Cube", "body_dimensions_m": [0.05, 0.05, 0.05],
        "body_initial_position_world_m": [10.0, 0.0, 0.15],
        "support_dimensions_m": [0.4, 0.4, 0.05],
        "support_position_world_m": [10.0, 0.0, 0.025],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "mass_kg": 0.1,
        "initial_surface_separation_m": 0.075, "disable_gravity": False,
        "kinematic_enabled": False, "sleep_threshold": 0.0,
        "contact_report_threshold": 0.0, "report_pairs": [],
        "contact_offset_m": 0.001, "rest_offset_m": 0.0,
        "linear_damping": 0.0, "angular_damping": 0.0,
        "static_friction": 0.5, "dynamic_friction": 0.5, "restitution": 0.0,
        "collision_isolation": "new control colliders filtered against existing non-control rigid bodies/colliders",
        "camera_exclusion": "x=10 m; existing fixed and wrist cameras have 5 m far clip; no camera changes",
        "physical_expectation": "gravity can close the initial gap; actual motion must be observed",
        "inference_only": {
            "minimum_samples": 3, "minimum_window_s": 0.05,
            "surface_gap_abs_max_m": 0.002, "linear_speed_max_m_s": 0.01,
            "angular_speed_max_rad_s": 0.1, "rotation_matrix_identity_max_abs": 0.01,
            "meaning": "sustained near-support low motion under downward gravity, consistent with support; not contact proof or grasp gate",
        },
    }


def add_observation_control(cfg):
    """Append one ordinary scene configuration hook before gym.make only."""
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    if cfg.scene.num_envs != 1 or cfg.sim.device != "cuda:0":
        raise ValueError("observation_control_requires_existing_single_gpu_scene")
    if getattr(cfg.scene, "observation_control", None) is not None:
        raise ValueError("observation_control_configuration_already_present")
    cfg.scene.observation_control = AssetBaseCfg(
        prim_path=CONTROL_ROOT, spawn=SpawnerCfg(func=spawn_control_pair))
    return observation_control_spec()


def _production_filter_targets(stage, UsdPhysics):
    # A rigid-body target covers its colliders, including instanced shapes.
    return sorted({str(prim.GetPath()) for prim in stage.Traverse()
                   if not str(prim.GetPath()).startswith(CONTROL_ROOT + "/")
                   and (prim.HasAPI(UsdPhysics.RigidBodyAPI)
                        or prim.HasAPI(UsdPhysics.CollisionAPI))})


def spawn_control_pair(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Normal initialization callback. No post-start repair or adaptive values."""
    from pxr import UsdPhysics
    from isaaclab import sim as sim_utils
    from isaaclab.sim.utils import get_current_stage
    from isaaclab.sim.schemas import activate_contact_sensors
    if prim_path != CONTROL_ROOT or translation not in (None, (0, 0, 0)) or orientation not in (None, (1, 0, 0, 0)):
        raise ValueError("exact_control_path_and_identity_parent_required")
    stage = get_current_stage()
    if stage.GetPrimAtPath(CONTROL_ROOT).IsValid():
        raise ValueError("control_namespace_already_exists")
    if not stage.GetPrimAtPath(PHYSICS_SCENE).IsA(UsdPhysics.Scene):
        raise ValueError("production_physics_scene_missing")
    targets = _production_filter_targets(stage, UsdPhysics)
    if not targets or not any(path.startswith("/World/envs/env_0/Robot/") for path in targets):
        raise ValueError("production_assets_must_precede_control_spawn")
    spec = observation_control_spec()
    sim_utils.create_prim(CONTROL_ROOT, "Xform")
    common = dict(collision_props=sim_utils.CollisionPropertiesCfg(
        collision_enabled=True, contact_offset=spec["contact_offset_m"], rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5,
            dynamic_friction=0.5, restitution=0.0))
    support_cfg = sim_utils.CuboidCfg(size=tuple(spec["support_dimensions_m"]), **common)
    body_cfg = sim_utils.CuboidCfg(size=tuple(spec["body_dimensions_m"]),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True,
            kinematic_enabled=False, disable_gravity=False, sleep_threshold=0.0,
            linear_damping=0.0, angular_damping=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1), **common)
    sim_utils.spawn_cuboid(CONTROL_SUPPORT, support_cfg,
        translation=tuple(spec["support_position_world_m"]), orientation=(1, 0, 0, 0))
    sim_utils.spawn_cuboid(CONTROL_BODY, body_cfg,
        translation=tuple(spec["body_initial_position_world_m"]), orientation=(1, 0, 0, 0))
    UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(CONTROL_BODY)).CreateSimulationOwnerRel().SetTargets([PHYSICS_SCENE])
    for path in (BODY_COLLIDER, SUPPORT_COLLIDER):
        prim = stage.GetPrimAtPath(path)
        UsdPhysics.CollisionAPI(prim).CreateSimulationOwnerRel().SetTargets([PHYSICS_SCENE])
        UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(targets)
    activate_contact_sensors(CONTROL_BODY, threshold=0.0, stage=stage)
    return stage.GetPrimAtPath(CONTROL_ROOT)


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=np.float64, copy=True)


def _finite_array(value, shape):
    arr = _array(value)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise ValueError("nonfinite_or_unexpected_buffer_shape")
    return arr


def _rotation(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(q).all() or norm <= 0:
        raise ValueError("invalid_quaternion")
    w, x, y, z = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _clock_valid(clock):
    return (isinstance(clock, (tuple, list)) and len(clock) == 2
            and type(clock[0]) is int and clock[0] >= 0
            and type(clock[1]) in (int, float) and math.isfinite(clock[1]))


def capture_collision_context(stage, paths):
    """Direct description only: no cooking, mesh extraction or transform cache."""
    result = []
    names = ("size", "radius", "height", "axis", "xformOp:scale",
             "physics:collisionEnabled", "physics:approximation",
             "physxCollision:contactOffset", "physxCollision:restOffset")
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        row = {"path": str(path), "valid": bool(prim.IsValid()), "attributes": {},
               "cooked_surface": "UNAVAILABLE", "source": "USD collision description, not cooked geometry"}
        if row["valid"]:
            row["type"] = str(prim.GetTypeName())
            for name in names:
                attr = prim.GetAttribute(name)
                if attr and attr.IsValid():
                    value = attr.Get()
                    if value is not None:
                        row["attributes"][name] = (value if isinstance(value, (str, bool, int, float))
                                                   else list(value))
        result.append(row)
    return deepcopy(result)


def validate_saved_control_configuration(configuration):
    """Recheck actual captured fields, never trust a cached ``valid`` verdict.

    This uses only saved values. The float32 alternatives are the exact USD
    float authoring representations, not an acceptance tolerance or repair.
    """
    issues = []
    spec = observation_control_spec()
    if not isinstance(configuration, dict):
        return ["control_configuration_missing"]
    if configuration.get("spec") != spec:
        issues.append("control_spec_changed")
    expected = {"body_exists": True, "support_exists": True, "support_is_static": True,
        "body_rigid_enabled": True, "body_kinematic": False, "disable_gravity": False,
        "sleep_threshold": 0.0, "linear_damping": 0.0, "angular_damping": 0.0,
        "mass_kg": spec["mass_kg"], "report_api": True, "report_threshold": 0.0,
        "report_pairs": [], "body_simulation_owner": [PHYSICS_SCENE]}
    for name, wanted in expected.items():
        actual = configuration.get(name)
        if type(wanted) is bool:
            equal = actual is wanted
        elif type(wanted) is float:
            equal = (type(actual) in (int, float) and math.isfinite(actual)
                     and actual in (wanted, float(np.float32(wanted))))
        else:
            equal = actual == wanted
        if not equal:
            issues.append("saved_configuration_mismatch:" + name)
    try:
        pose = _finite_array(configuration["support_static_usd_world_wxyz"], (7,))
        expected_pose = spec["support_position_world_m"] + spec["quaternion_wxyz"]
        if not np.allclose(pose, expected_pose, atol=1e-7, rtol=0):
            issues.append("saved_support_transform_mismatch")
        gravity = _finite_array(configuration["scene_gravity_world_m_s2"], (3,))
        configured_gravity = _finite_array(configuration["gravity_world_m_s2"], (3,))
        if (gravity[2] >= 0 or np.any(gravity[:2] != 0)
                or not np.allclose(gravity, configured_gravity, atol=1e-5, rtol=0)):
            issues.append("saved_gravity_mismatch")
    except Exception as exc:
        issues.append("saved_transform_or_gravity_invalid:" + type(exc).__name__)
    targets = configuration.get("isolation_target_paths")
    if (not isinstance(targets, list) or not targets
            or any(not isinstance(path, str) or not path.startswith("/")
                   or path.startswith(CONTROL_ROOT) for path in targets)
            or len(set(targets)) != len(targets)
            or not any(path.startswith("/World/envs/env_0/Robot/") for path in targets)):
        issues.append("saved_isolation_targets_invalid")
    filters = configuration.get("filter_targets")
    owners = configuration.get("collider_simulation_owners")
    if not isinstance(filters, dict):
        issues.append("saved_filter_relationships_invalid")
        filters = {}
    if not isinstance(owners, dict):
        issues.append("saved_scene_owner_relationships_invalid")
        owners = {}
    for path in (BODY_COLLIDER, SUPPORT_COLLIDER):
        if filters.get(path) != targets:
            issues.append("saved_isolation_mismatch:" + path)
        if owners.get(path) != [PHYSICS_SCENE]:
            issues.append("saved_collider_scene_owner_mismatch:" + path)
    colliders = configuration.get("colliders")
    if not isinstance(colliders, list) or len(colliders) != 2:
        issues.append("saved_collider_descriptions_missing")
    else:
        for record, path, dimensions in zip(colliders, (BODY_COLLIDER, SUPPORT_COLLIDER),
                (spec["body_dimensions_m"], spec["support_dimensions_m"])):
            try:
                attrs = record["attributes"]
                size = attrs["size"]
                scale = _finite_array(attrs.get("xformOp:scale", [1, 1, 1]), (3,))
                if (record.get("valid") is not True or record.get("path") != path
                        or record.get("type") != "Cube" or type(size) not in (int, float)
                        or not math.isfinite(size)
                        or not np.allclose(size * scale, dimensions, atol=1e-7, rtol=0)
                        or attrs.get("physics:collisionEnabled") is not True
                        or attrs.get("physxCollision:contactOffset") not in (0.001, float(np.float32(0.001)))
                        or attrs.get("physxCollision:restOffset") != 0.0):
                    issues.append("saved_collider_description_mismatch:" + path)
            except Exception as exc:
                issues.append("saved_collider_description_invalid:" + path + ":" + type(exc).__name__)
    return issues


class ControlReader:
    """Existing-stage readback; a new body view, never a new physics scene."""
    def __init__(self, env, stage, scene_identity, *, view=None, schemas=None):
        self.env, self.stage, self.scene_identity = env, stage, scene_identity
        if schemas is None:
            from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
            schemas = (Usd, UsdGeom, UsdPhysics, PhysxSchema)
        self.schemas = schemas
        self._isolation_targets = None
        self.view = (view if view is not None else
                     env.scene["chicken"]._physics_sim_view.create_rigid_body_view(CONTROL_BODY))
        self.configuration = self._configuration()

    def _configuration(self):
        Usd, UsdGeom, UsdPhysics, PhysxSchema = self.schemas
        spec = observation_control_spec()
        body = self.stage.GetPrimAtPath(CONTROL_BODY)
        support = self.stage.GetPrimAtPath(CONTROL_SUPPORT)
        rb, physx = UsdPhysics.RigidBodyAPI(body), PhysxSchema.PhysxRigidBodyAPI(body)
        report = PhysxSchema.PhysxContactReportAPI(body)
        def targets(prim, name):
            return [str(x) for x in prim.GetRelationship(name).GetTargets()]
        matrix = UsdGeom.Xformable(support).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        quat = matrix.ExtractRotationQuat()
        support_pose = list(matrix.ExtractTranslation()) + [float(quat.GetReal())] + list(quat.GetImaginary())
        if self._isolation_targets is None:
            self._isolation_targets = _production_filter_targets(self.stage, UsdPhysics)
        expected_targets = self._isolation_targets
        colliders = capture_collision_context(self.stage, (BODY_COLLIDER, SUPPORT_COLLIDER))
        config = {"spec": spec, "issues": [], "body_exists": bool(body.IsValid()),
            "support_exists": bool(support.IsValid()), "support_is_static": not support.HasAPI(UsdPhysics.RigidBodyAPI),
            "body_rigid_enabled": rb.GetRigidBodyEnabledAttr().Get(),
            "body_kinematic": rb.GetKinematicEnabledAttr().Get(),
            "disable_gravity": physx.GetDisableGravityAttr().Get(),
            "sleep_threshold": physx.GetSleepThresholdAttr().Get(),
            "linear_damping": physx.GetLinearDampingAttr().Get(),
            "angular_damping": physx.GetAngularDampingAttr().Get(),
            "mass_kg": UsdPhysics.MassAPI(body).GetMassAttr().Get(),
            "report_api": bool(report), "report_threshold": report.GetThresholdAttr().Get(),
            "report_pairs": targets(body, "physxContactReport:reportPairs"),
            "body_simulation_owner": targets(body, "physics:simulationOwner"),
            "support_static_usd_world_wxyz": support_pose,
            "support_pose_source": "read-only USD world transform of static collider parent; not a dynamic native pose",
            "gravity_world_m_s2": list(self.env.cfg.sim.gravity),
            "gravity_source": "env.cfg.sim.gravity; native scene gravity readback below",
            "colliders": colliders, "filter_targets": {}, "isolation_target_paths": list(expected_targets),
            "collider_simulation_owners": {},
            "native_drop_accounting": "NATIVE_DROP_ACCOUNTING_UNAVAILABLE"}
        scene = UsdPhysics.Scene(self.stage.GetPrimAtPath(PHYSICS_SCENE))
        direction = np.asarray(scene.GetGravityDirectionAttr().Get(), dtype=float)
        magnitude = float(scene.GetGravityMagnitudeAttr().Get())
        config["scene_gravity_world_m_s2"] = (direction * magnitude).tolist()
        expected = {"body_exists": True, "support_exists": True, "support_is_static": True,
            "body_rigid_enabled": True, "body_kinematic": False, "disable_gravity": False,
            "linear_damping": 0.0, "angular_damping": 0.0,
            "sleep_threshold": 0.0, "mass_kg": 0.1, "report_api": True,
            "report_threshold": 0.0, "report_pairs": [], "body_simulation_owner": [PHYSICS_SCENE]}
        for key, value in expected.items():
            observed = config[key]
            equal = observed == value
            if type(value) is float and value != 0.0:
                equal = type(observed) in (float, int) and float(observed) == float(np.float32(value))
            if not equal:
                config["issues"].append("configuration_mismatch:" + key)
        if not np.allclose(support_pose, spec["support_position_world_m"] + spec["quaternion_wxyz"], atol=1e-7, rtol=0):
            config["issues"].append("support_transform_mismatch")
        for record, dimensions in zip(colliders, (spec["body_dimensions_m"], spec["support_dimensions_m"])):
            attr = record["attributes"]
            size = attr.get("size")
            scale = np.asarray(attr.get("xformOp:scale", [1, 1, 1]))
            if (not record["valid"] or record.get("type") != "Cube" or size is None
                    or not np.allclose(scale * size, dimensions, atol=1e-7, rtol=0)
                    or attr.get("physics:collisionEnabled") is not True
                    or attr.get("physxCollision:contactOffset") not in (0.001, float(np.float32(0.001)))
                    or attr.get("physxCollision:restOffset") != 0.0):
                config["issues"].append("collider_description_mismatch:" + record["path"])
            filters = targets(self.stage.GetPrimAtPath(record["path"]), "physics:filteredPairs")
            config["filter_targets"][record["path"]] = filters
            if sorted(filters) != expected_targets:
                config["issues"].append("control_isolation_mismatch:" + record["path"])
            scene_owners = targets(self.stage.GetPrimAtPath(record["path"]), "physics:simulationOwner")
            config["collider_simulation_owners"][record["path"]] = scene_owners
            if scene_owners != [PHYSICS_SCENE]:
                config["issues"].append("control_scene_owner_mismatch:" + record["path"])
        gravity = config["scene_gravity_world_m_s2"]
        if (not np.isfinite(gravity).all() or gravity[2] >= 0 or gravity[:2] != [0.0, 0.0]
                or not np.allclose(gravity, config["gravity_world_m_s2"], atol=1e-5, rtol=0)):
            config["issues"].append("gravity_not_verified_downward")
        config["valid"] = not config["issues"]
        return config

    def read(self):
        clock = lambda: [self.env.sim.current_time_step_index, self.env.sim.current_time]
        result = {"clock_before": clock(), "clock_after": None, "issues": [],
            "source": "existing physics view get_transforms/get_velocities; no refresh",
            "pose_frame": "world", "pose_order": "xyz_wxyz", "velocity_frame": "world",
            "linear_velocity_reference": "body center of mass", "velocity_units": ["m/s"]*3 + ["rad/s"]*3}
        try:
            self.scene_identity.validate("observation_control_read")
            if list(self.view.prim_paths) != [CONTROL_BODY]:
                raise ValueError("control_body_view_identity_changed")
            pose = _finite_array(self.view.get_transforms(), (1, 7))[0]
            vel = _finite_array(self.view.get_velocities(), (1, 6))[0]
            _rotation(pose[[6, 3, 4, 5]])
            result["body_world_wxyz"] = pose[[0, 1, 2, 6, 3, 4, 5]].tolist()
            result["body_velocity_world"] = vel.tolist()
            current = self._configuration()
            result["configuration_valid"] = current["valid"]
            result["configuration_issues"] = current["issues"]
            result["support_static_usd_world_wxyz"] = current["support_static_usd_world_wxyz"]
            result["scene_gravity_world_m_s2"] = current["scene_gravity_world_m_s2"]
            result["body_kinematic"] = current["body_kinematic"]
            result["disable_gravity"] = current["disable_gravity"]
            if not current["valid"]:
                result["issues"].append("control_configuration_invalid")
        except Exception as exc:
            result["issues"].append(f"control_read_error:{type(exc).__name__}:{exc}")
        result["clock_after"] = clock()
        if not _clock_valid(result["clock_before"]) or result["clock_before"] != result["clock_after"]:
            result["issues"].append("control_clock_invalid_or_changed")
        result["valid"] = not result["issues"]
        return deepcopy(result)


def bind_observation_control(env, stage, scene_identity):
    return ControlReader(env, stage, scene_identity)


def read_articulation_observations(env, clock=None):
    """Direct current native getters; optional actuator buffers are labelled old commands/estimates."""
    now = lambda: [env.sim.current_time_step_index, env.sim.current_time]
    before = now()
    result = {"clock_before": before, "clock_after": None, "issues": [], "channels": {},
        "sample_interval_s": float(env.cfg.sim.dt), "joint_mapping": [], "link_mapping": [],
        "frame": "joint coordinates (prismatic positive axis); link poses world xyz_wxyz"}
    try:
        robot = env.scene["robot"]
        view = robot.root_physx_view
        names = list(robot.joint_names)
        link_paths = list(view.link_paths[0])
        ids = [names.index(name) for name in JAW_NAMES]
        matches = [[i for i, path in enumerate(link_paths) if path.endswith("/" + name)] for name in JAW_LINKS]
        if any(len(match) != 1 for match in matches):
            raise ValueError("ambiguous_jaw_link_mapping")
        links = [match[0] for match in matches]
        if (len(set(names)) != len(names) or len(set(ids)) != 4 or len(set(links)) != 4
                or len(view.link_paths) != 1):
            raise ValueError("ambiguous_articulation_mapping")
        result["joint_mapping"] = [{"name": name, "dof_index": idx, "type": "prismatic"} for name, idx in zip(JAW_NAMES, ids)]
        result["link_mapping"] = [{"name": name, "link_index": idx, "path": link_paths[idx]} for name, idx in zip(JAW_LINKS, links)]
        q = _finite_array(view.get_dof_positions(), (1, len(names)))[0, ids]
        qd = _finite_array(view.get_dof_velocities(), (1, len(names)))[0, ids]
        poses = _finite_array(view.get_link_transforms(), (1, len(link_paths), 7))[0, links]
        for pose in poses:
            _rotation(pose[[6, 3, 4, 5]])
        result["joint_positions_m"] = q.tolist()
        result["joint_velocities_m_s"] = qd.tolist()
        result["jaw_link_world_wxyz"] = poses[:, [0, 1, 2, 6, 3, 4, 5]].tolist()
        result["required_source"] = "root_physx_view.get_dof_positions/get_dof_velocities/get_link_transforms; no cache refresh"
        def optional(name, source, getter, shape, indices, units, frame, status, timing):
            channel = {"source": source, "units": units, "frame": frame, "quantity_status": status,
                       "timing": timing, "native_sample_clock": before, "status": "UNAVAILABLE", "values": None}
            try:
                value = _finite_array(getter(), shape)[0, indices]
                channel.update(status="AVAILABLE", values=value.tolist())
            except Exception as exc:
                channel["reason"] = f"{type(exc).__name__}:{exc}"
            result["channels"][name] = channel
        for field, units, status in (
            ("default_joint_pos", "m", "configured default position, NOT an actual measured reset state"),
            ("default_joint_vel", "m/s", "configured default velocity, NOT a measured state"),
            ("joint_pos_target", "m", "commanded joint position"),
            ("joint_vel_target", "m/s", "commanded joint velocity"),
            ("joint_effort_target", "N", "commanded feed-forward generalized effort"),
            ("computed_torque", "N", "implicit actuator approximate PD effort before clipping, not measured drive/contact force"),
            ("applied_torque", "N", "implicit actuator approximate PD effort after clipping, not measured drive/contact force")):
            optional(field, "robot.data." + field, lambda f=field: getattr(robot.data, f),
                (1, len(names)), ids, units, "jaw prismatic joint positive axes", status,
                "last controller write/compute; may precede this substep; native clock labels observation, not recomputation")
        optional("native_actuation_force", "root_physx_view.get_dof_actuation_forces",
            lambda: view.get_dof_actuation_forces(), (1, len(names)), ids, "N", "jaw prismatic joint positive axes",
            "native actuation input; not total implicit-drive/contact force", "direct getter at observation boundary")
        optional("native_position_target", "root_physx_view.get_dof_position_targets",
            lambda: view.get_dof_position_targets(), (1, len(names)), ids, "m", "jaw prismatic joint positive axes",
            "native current commanded position, NOT measured position", "direct getter at observation boundary")
        optional("measured_generalized_effort", "root_physx_view.get_dof_projected_joint_forces",
            lambda: view.get_dof_projected_joint_forces(), (1, len(names)), ids, "N", "jaw prismatic motion axes",
            "incoming joint force projected on active motion direction", "direct getter at observation boundary")
        optional("incoming_joint_wrench", "root_physx_view.get_link_incoming_joint_force",
            lambda: view.get_link_incoming_joint_force(), (1, len(link_paths), 6), links,
            ["N"]*3 + ["N m"]*3, "joints' child frames; indexed by link_mapping, NOT dof_index",
            "measured total active/passive incoming joint force and torque", "direct getter at observation boundary")
    except Exception as exc:
        result["issues"].append(f"articulation_measurement_error:{type(exc).__name__}:{exc}")
    result["clock_after"] = now()
    if (not _clock_valid(before) or result["clock_after"] != before
            or (clock is not None and list(clock) != before)):
        result["issues"].append("articulation_clock_invalid_or_changed")
    result["valid"] = not result["issues"]
    return deepcopy(result)


def classify_control_observations(samples, *, configuration, raw_observations, persistence_complete):
    """Diagnostic result only. Never called to choose candidate actions or wait.

    Three near-support observations spanning 50 ms distinguish sustained support-
    consistent motion from a single touching/overlapping description. In free
    fall gravity alone changes vertical speed by ~0.49 m/s over that window;
    requiring <=0.01 m/s at every sample plus the 2 mm offset envelope is
    conservative evidence of support, not proof of a collision or report.
    """
    spec = observation_control_spec()
    rules = spec["inference_only"]
    result = {"result": "CONTROL_OBSERVATION_INCOMPLETE", "issues": [],
        "window": None, "support_consistent_window": None, "reported_records": 0,
        "inference_rules": rules, "native_drop_accounting": "NATIVE_DROP_ACCOUNTING_UNAVAILABLE",
        "coverage_limits": ["positive control applies only to this primitive pair/reporting configuration",
            "does not prove jaw/cooked-mesh completeness, zero native drops, or no physical jaw contact",
            "support-consistent motion is independent inference, not physical contact proof"]}
    if not persistence_complete:
        result["issues"].append("control_persistence_incomplete")
    if (not isinstance(configuration, dict) or not configuration.get("valid")
            or configuration.get("spec") != spec):
        result["issues"].append("control_configuration_missing_or_invalid")
    result["issues"].extend(validate_saved_control_configuration(configuration))
    if not samples or not raw_observations or len(raw_observations) != len(samples):
        result["issues"].append("control_sample_or_raw_coverage_incomplete")
    callback_mode = any(raw.get("collector_transport") == "callback-contact/v1" for raw in raw_observations)
    delivery_issues = []
    last_clock, stable = None, []
    for index, (sample, raw) in enumerate(zip(samples, raw_observations)):
        try:
            clock = sample["clock_before"]
            callback = raw.get("collector_transport") == "callback-contact/v1"
            # A completed local callback envelope is an observation even when
            # no callback occurred. It is not a native empty-batch heartbeat.
            acquisition_status = "CALLBACK_BOUNDARY_FINALIZED" if callback else "RAW_REPORT_OBSERVED"
            if (not sample.get("valid") or not sample.get("configuration_valid") or not _clock_valid(clock)
                    or sample["clock_after"] != clock or raw.get("clock_before") != clock
                    or raw.get("clock_after") != clock or raw.get("container_status") != "present"
                    or raw.get("acquisition_status") not in ((acquisition_status, "UNKNOWN") if callback else (acquisition_status,))
                    or (not callback and (raw.get("acquisition_issues") or raw.get("context", {}).get("issues")
                        or raw.get("observation_issues") or raw.get("local_unclassified_headers", 0) != 0))
                    or (last_clock is not None and (clock[0] <= last_clock[0] or clock[1] <= last_clock[1]))):
                raise ValueError("invalid_missing_stale_or_unaligned_control_observation")
            last_clock = clock
            headers = raw["headers"]
            if not isinstance(headers, list):
                raise ValueError("corrupt_raw_header")
            degraded = (raw.get("acquisition_status") == "UNKNOWN" or raw.get("acquisition_issues")
                or raw.get("observation_issues") or raw.get("local_unclassified_headers", 0) != 0
                or any(not h.get("valid") or h.get("issues") for h in headers))
            if degraded:
                if not callback:
                    raise ValueError("corrupt_raw_header")
                delivery_issues.append("sample_" + str(index) + ":callback_delivery_unavailable")
            else:
                result["reported_records"] += sum(h.get("pair_classification") == "CONTROL_PAIR"
                    and h.get("type") in (0, 2) and h.get("num_contact_data", 0) > 0 for h in headers)
            pose = _finite_array(sample["body_world_wxyz"], (7,))
            velocity = _finite_array(sample["body_velocity_world"], (6,))
            support = _finite_array(sample["support_static_usd_world_wxyz"], (7,))
            gravity = _finite_array(sample["scene_gravity_world_m_s2"], (3,))
            rotation = _rotation(pose[3:])
            relative = pose[:3] - support[:3]
            gap = relative[2] - (spec["body_dimensions_m"][2] + spec["support_dimensions_m"][2]) / 2
            suitable = (sample["body_kinematic"] is False and sample["disable_gravity"] is False
                and gravity[2] < 0 and np.all(gravity[:2] == 0)
                and np.max(np.abs(rotation - np.eye(3))) <= rules["rotation_matrix_identity_max_abs"]
                and np.max(np.abs(_rotation(support[3:]) - np.eye(3))) <= 1e-7
                and np.all(np.abs(relative[:2]) + 0.025 <= np.asarray(spec["support_dimensions_m"][:2]) / 2)
                and abs(gap) <= rules["surface_gap_abs_max_m"]
                and np.linalg.norm(velocity[:3]) <= rules["linear_speed_max_m_s"]
                and np.linalg.norm(velocity[3:]) <= rules["angular_speed_max_rad_s"])
            stable = stable + [(index, clock)] if suitable else []
            if (len(stable) >= rules["minimum_samples"]
                    and stable[-1][1][1] - stable[0][1][1] >= rules["minimum_window_s"]
                    and result["support_consistent_window"] is None):
                result["support_consistent_window"] = {"start_index": stable[0][0], "end_index": index,
                    "clock_start": stable[0][1], "clock_end": clock, "samples": len(stable)}
        except Exception as exc:
            stable = []
            result["issues"].append(f"sample_{index}:{type(exc).__name__}:{exc}")
    if samples:
        result["window"] = {"first_clock": samples[0].get("clock_before"),
                            "last_clock": samples[-1].get("clock_after"), "sample_count": len(samples)}
    if callback_mode:
        result["contact_delivery_issues"] = delivery_issues
        result["negative_contact_qualification"] = "NOT_ESTABLISHED"
    if not result["issues"] and not delivery_issues:
        result["result"] = ("CONTROL_CONTACT_REPORTED" if result["reported_records"] else
            "CONTROL_CONTACT_EXPECTED_BUT_NOT_REPORTED" if result["support_consistent_window"] else
            "CONTROL_PHYSICAL_CONDITION_UNCONFIRMED")
    return deepcopy(result)
