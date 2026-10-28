"""Cycle 11 raw jaw contacts. No policy, contact classifier, owner or runner.

Installed source: Isaac Lab 2.3.2; Isaac Sim 5.1.0-rc.19;
omni.physx 107.3.26 _physx.pyi:492-710,1365-1407. The direct report
associates data with shapes through header offsets, NOT tensor ordering.
Native imports are restricted to explicit coverage/setup/binding calls.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

TASK_ID = "Isaac-ChicGrasp-GraspLift-DynamicChicken-IK-Abs-v0"
ENV_ROOT = "/World/envs/env_0"
PHYSICS_DT_S = 1.0 / 120.0
SAMPLE_CAPACITY = 2400
# Local copy/storage limits, never native scene capacities or contact thresholds.
MAX_HEADERS = 4096
MAX_CONTACTS = 16384
HEADER_FIELDS = ("stage_id", "actor0", "actor1", "collider0", "collider1", "type",
                 "num_contact_data", "contact_data_offset", "num_friction_anchors_data",
                 "friction_anchors_offset", "proto_index0", "proto_index1")
CONTACT_EVENT_NAMES = {0: "CONTACT_FOUND", 1: "CONTACT_LOST", 2: "CONTACT_PERSIST"}
GPU_CAPACITIES = (
    ("gpu_mem_rigid_contact_count", "physxScene:gpuMaxRigidContactCount"),
    ("gpu_mem_rigid_patch_count", "physxScene:gpuMaxRigidPatchCount"),
    ("gpu_mem_found_lost_pairs", "physxScene:gpuFoundLostPairsCapacity"),
    ("gpu_mem_found_lost_aggregate_pairs", "physxScene:gpuFoundLostAggregatePairsCapacity"),
    ("gpu_mem_total_aggregate_pairs", "physxScene:gpuTotalAggregatePairsCapacity"),
    ("gpu_mem_collision_stack_size", "physxScene:gpuCollisionStackSize"),
)
ROOT = Path(__file__).resolve().parents[2]
CHICKEN_ASSET = "assets/grasp_contract/chicken_rigid_wide_torso_spread_legs_protected_shank_grasp_v2/asset"
ROBOT_ASSET = "assets/grasp_contract/ur10e_chicgrasp_serrated_tip_collision_package"
SOURCE_PINS = {
    "config/robot_station_frames.json": "9192da25d0117405e812a85104ab14a6cce94372bdf0591b0805a34e87c4c181",
    CHICKEN_ASSET + "/grasp_contract.json": "2d3a7527395e916e558c64af9f776c3d337b82e931e191ec87d65dd793d6b822",
    ROBOT_ASSET + "/ur10e_chicgrasp_serrated_tip_collision.usda": "4a2b0f7502f6ff19841857f6466f6fa7b245417f84cacf91efe6c53c9e29eb33",
}
RAW_SEMANTICS = {
    "source": "omni.physx 107.3.26 _physx.pyi:492-710,1365-1389",
    "impulse": "native ContactData.impulse: Contact impulse (Float3), unchanged",
    "normal": "native ContactData.normal: Contact normal, unchanged",
    "position": "native ContactData.position: Contact position, unchanged",
    "separation": "native ContactData.separation: Contact separation value, unchanged",
    "units": {"stage_length": "m", "stage_mass": "kg", "time": "s",
              "raw_impulse_units_established": False,
              "raw_vector_reference_established": False,
              "note": "Direct binding does not specify vector units/frame; no cross-API inference"},
    "native_integration_interval_s": PHYSICS_DT_S,
    "impulse_integration_interval_established": False,
    "acted_on_body": None,
    "normal_vs_tangential_content": None,
    "normal_direction_sign": None,
    "association": "same native header contact_data_offset/num_contact_data and source index",
    "report_tensor_association": "unsupported; no tensor channel read or joined",
    "derived_normal_force_N": None,
    "normal_force_unavailable_reason": "component, sign, units and acted-on-body interpretation unestablished",
}


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _guard(task_id, num_envs):
    if task_id != TASK_ID or type(num_envs) is not int or num_envs != 1:
        raise ValueError("exact_dynamic_task_and_one_environment_required")


def _json(value):
    """Loss-signalling JSON, never silently turn NaN/missing evidence into zero."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError("diagnostic_json_requires_string_keys")
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"unavailable_native_type": type(value).__name__}


def load_contract_map():
    """Pinned identity/geometry only; does not import legacy code or thresholds."""
    contents = {}
    for relative, expected in SOURCE_PINS.items():
        raw = (ROOT / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("contract_source_hash_mismatch:" + relative)
        contents[relative] = raw
    gc = json.loads(contents[CHICKEN_ASSET + "/grasp_contract.json"])
    rf = json.loads(contents["config/robot_station_frames.json"])
    chicken = ENV_ROOT + "/Chicken/" + gc["rigid_body_frame_relative_path"]
    if chicken != ENV_ROOT + "/Chicken/raw_chicken_visual/raw_chicken_visual_mesh":
        raise ValueError("unexpected_chicken_actor")
    stations = gc["stations"]
    if set(stations) != {"negX_shank", "posX_shank"}:
        raise ValueError("unexpected_stations")
    result = {"source_sha256": dict(SOURCE_PINS), "chicken_actor": chicken,
              "jaws": {}, "contract_geometry": {}, "units": {"length": "m", "mass": "kg", "up_axis": "Z"}}
    all_chicken = None
    finger_variants = ("vp0V0", "1_us0V0", "2_us0V0", "3_us0V0")
    cylinder_variants = ("dg0w0", "1_vi0y0", "5_vi0y0", "2_vi0y0")
    for station_id, ids in (("negX_shank", [1, 2]), ("posX_shank", [3, 4])):
        station = stations[station_id]
        if station["jaw_ids"] != ids:
            raise ValueError("station_jaw_map_mismatch")
        allowed, forbidden = station["allowed_colliders"], station["forbidden_colliders"]
        paths = allowed + forbidden
        if (len(allowed) != 1 or len(forbidden) != 123 or len(set(paths)) != 124
                or any(p.startswith("/") or ".." in p.split("/") for p in paths)):
            raise ValueError("chicken_collider_inventory_mismatch")
        full = sorted(chicken + "/" + p for p in paths)
        if all_chicken is not None and all_chicken != full:
            raise ValueError("station_collider_unions_disagree")
        all_chicken = full
        # Geometry only: historical cooked-force traces are not live evidence.
        result["contract_geometry"][station_id] = {
            key: deepcopy(station[key]) for key in ("allowed_region", "axes_rigid_body",
                                                   "center_rigid_body_m", "quaternion_rigid_body_wxyz")}
        result["contract_geometry"][station_id]["patches"] = [
            {key: deepcopy(patch[key]) for key in ("jaw_id", "center_rigid_body_m",
              "expected_normal_local", "expected_normal_rigid_body", "obb_half_extents_m",
              "prim_path", "quaternion_rigid_body_wxyz")} for patch in station["patches"]]
        if sorted(p["jaw_id"] for p in station["patches"]) != ids:
            raise ValueError("contract_patch_jaw_map_mismatch")
        for j in ids:
            patch = rf["jaw_patch_frames_tcp"][str(j)]
            if patch["jaw_id"] != j or patch["station_id"] != station_id:
                raise ValueError("robot_station_map_mismatch")
            gc_patch = next(p for p in station["patches"] if p["jaw_id"] == j)
            if (patch["target_collider"] != allowed[0].split("/")[-1]
                    or patch["contributing_serrated_collision_pieces"] != gc_patch["cooked_geometry_evidence"]["contributing_pieces"]):
                raise ValueError("robot_chicken_contributor_maps_disagree")
            actor = ENV_ROOT + f"/Robot/GripperRig/jaw_{j}_link"
            first = "tn__fingers_for_chicken_right_08092024_" + finger_variants[j-1]
            pieces_root = actor + "/visual_00_" + first + "/tn__fingers_for_chicken_right_08092024_vp0V0/refined_serrated_tip_collision/"
            pieces = [pieces_root + f"finger_collision_piece_{i:02d}" for i in range(1, 28)]
            cylinder = actor + "/visual_01_tn__Linear_cylinder_HFK32CL_modified_" + cylinder_variants[j-1] + "/tn__Linear_cylinder_HFK32CL_modified_dg0w0/Mesh"
            contributors = [pieces_root + p for p in patch["contributing_serrated_collision_pieces"]]
            if len(set(contributors)) != len(contributors) or not set(contributors).issubset(pieces):
                raise ValueError("unexpected_contributors")
            result["jaws"][actor] = {
                "jaw_id": j, "station_id": station_id,
                "enabled_colliders": sorted(pieces + [cylinder]),
                "contributor_colliders": contributors,
                "allowed_chicken_colliders": [chicken + "/" + p for p in allowed],
                "forbidden_chicken_colliders": [chicken + "/" + p for p in forbidden],
                "patch_frame": deepcopy(patch),
            }
    result["chicken_colliders"] = all_chicken
    return result


def inspect_runtime_coverage(stage, contracts):
    """Validate the composed actors AND all enabled child collider ownership."""
    from pxr import Usd, UsdGeom, UsdPhysics
    envs = stage.GetPrimAtPath("/World/envs")
    if not envs.IsValid() or [str(p.GetPath()) for p in envs.GetChildren()] != [ENV_ROOT]:
        raise ValueError("exactly_one_composed_environment_required")
    if (UsdGeom.GetStageMetersPerUnit(stage) != 1.0
            or UsdPhysics.GetStageKilogramsPerUnit(stage) != 1.0
            or str(UsdGeom.GetStageUpAxis(stage)) != "Z"):
        raise ValueError("expected_metre_kilogram_Z_up_stage")
    expected = {actor: jaw["enabled_colliders"] for actor, jaw in contracts["jaws"].items()}
    expected[contracts["chicken_actor"]] = contracts["chicken_colliders"]
    for actor, paths in expected.items():
        prim = stage.GetPrimAtPath(actor)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("missing_rigid_actor:" + actor)
        api = UsdPhysics.RigidBodyAPI(prim)
        if api.GetRigidBodyEnabledAttr().Get() is not True or api.GetKinematicEnabledAttr().Get() is not False:
            raise ValueError("unexpected_rigid_body_state:" + actor)
        found = []
        for child in Usd.PrimRange(prim):
            if child != prim and child.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError("unexpected_nested_rigid_body:" + str(child.GetPath()))
            if child.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get():
                found.append(str(child.GetPath()))
        if sorted(found) != sorted(paths):
            raise ValueError("enabled_collider_inventory_mismatch:" + actor)
    return deepcopy(contracts)


class ReportingSetupError(RuntimeError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = deepcopy(evidence)


def prepare_reporting(stage, *, task_id, num_envs=1, contracts=None):
    """Explicit pre-physics setup. No helper, sleep/filter/scene edits or step.

    Only missing jaw PhysxContactReportAPI and necessary threshold=0 are written.
    All restrictions preflight before any mutation. Errors retain partial writes.
    """
    _guard(task_id, num_envs)
    from pxr import PhysxSchema
    import carb.settings
    contracts = load_contract_map() if contracts is None else contracts
    # Caller-supplied metadata cannot replace pinned identities/geometry.
    if contracts != load_contract_map():
        raise ValueError("unapproved_contract_map")
    coverage = inspect_runtime_coverage(stage, contracts)
    evidence = {"task_id": task_id, "num_envs": num_envs, "coverage": coverage,
                "properties": [], "completed": False}
    planned = []
    try:
        if carb.settings.get_settings().get("/physics/disableContactProcessing") is not False:
            raise ValueError("global_contact_processing_not_confirmed_enabled")
        for actor in coverage["jaws"]:
            prim = stage.GetPrimAtPath(actor)
            report = PhysxSchema.PhysxContactReportAPI(prim)
            if report.GetReportPairsRel().GetTargets():
                raise ValueError("existing_reportPairs_restriction:" + actor)
            # An authored attribute can already exist even before API enablement.
            before = report.GetThresholdAttr().Get()
            planned.append((actor, prim, bool(report), before))
        for actor, prim, present, before in planned:
            api_record = {"actor": actor, "property": "apiSchemas:PhysxContactReportAPI",
                          "existing_value": present, "desired_value": True, "authored": False,
                          "reason": "native contact reporting on this existing jaw"}
            evidence["properties"].append(api_record)
            report = PhysxSchema.PhysxContactReportAPI(prim)
            if not present:
                report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                api_record["authored"] = bool(report)
                if not report:
                    raise RuntimeError("report_API_apply_failed:" + actor)
            cutoff = {"actor": actor, "property": "physxContactReport:threshold",
                      "existing_value": before, "desired_value": 0.0, "authored": False,
                      "reason": "retain weak reports; NOT a station-force threshold"}
            evidence["properties"].append(cutoff)
            if before != 0.0 or isinstance(before, bool):
                if not report.CreateThresholdAttr().Set(0.0):
                    raise RuntimeError("report_threshold_write_failed:" + actor)
                cutoff["authored"] = True
            if report.GetThresholdAttr().Get() != 0.0 or isinstance(report.GetThresholdAttr().Get(), bool):
                raise RuntimeError("report_threshold_readback_failed:" + actor)
        evidence["completed"] = True
        return deepcopy(evidence)
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}:{exc}"
        raise ReportingSetupError(evidence["error"], evidence) from exc


class RuntimeProbe:
    """Read-only installed stage/report/error/sleep/capacity boundaries."""
    def health(self):
        from pxr import PhysxSchema
        issues = list(self.errors)
        info = {"issues": issues, "jaws": {}, "native_drop_count": None,
                "native_drop_counter_available": False, "gpu_capacities": {},
                "completeness_proven": False,
                "completeness_note": "health checks cannot prove absence of unreported contacts"}
        try:
            managed = getattr(self, "scene_identity", None)
            if managed is not None:
                validation = managed.validate("jaw_health")
                info["stage_identity_domain"] = "constructed_managed_USD_stage_cache"
                info["managed_stage_cache_id"] = managed.stage_id
                info["managed_identity_validation"] = {"mode": validation["mode"],
                    "verdict": validation["verdict"], "phase": validation["phase"],
                    "monotonic_ns": validation["monotonic_ns"],
                    "stage_cache_id": validation["anchor"]["stage_cache_id"]}
            else:
                info["attached_stage_id"] = self.physics.get_attached_stage()
                if info["attached_stage_id"] != self.stage_id:
                    issues.append("native_stage_mismatch")
            info["disable_contact_processing"] = self.settings.get("/physics/disableContactProcessing")
            if info["disable_contact_processing"] is not False:
                issues.append("contact_processing_not_confirmed_enabled")
            if inspect_runtime_coverage(self.stage, self.coverage) != self.coverage:
                issues.append("composed_coverage_changed")
            for actor in (*self.coverage["jaws"], self.coverage["chicken_actor"]):
                sleeping = self.physics.is_sleeping(self.stage_id, self.encode(actor))
                details = {"sleeping": sleeping}
                if actor in self.coverage["jaws"]:
                    report = PhysxSchema.PhysxContactReportAPI(self.stage.GetPrimAtPath(actor))
                    details.update(report_api_present=bool(report), cutoff=report.GetThresholdAttr().Get(),
                                   report_pairs=[str(p) for p in report.GetReportPairsRel().GetTargets()])
                    if not report or details["cutoff"] != 0.0 or isinstance(details["cutoff"], bool) or details["report_pairs"]:
                        issues.append("reporting_restricted_or_unavailable:" + actor)
                info["jaws"][actor] = details
                if sleeping is not False:
                    issues.append("sleeping_or_sleep_state_unknown:" + actor)
            info["scene_statistics_inputs"] = {"stage_cache_id": self.stage_id,
                "encoded_scene_path": self.scene_id,
                "scene_path": str(self.scene.GetPath()),
                "limit": "caller-supplied inputs, not independent active-stage identity"}
            if not self.statistics.get_physx_scene_statistics(self.stage_id, self.scene_id, self.stats):
                issues.append("native_capacity_statistics_unavailable")
            else:
                for field, attr in GPU_CAPACITIES:
                    needed, limit = getattr(self.stats, field), self.scene.GetAttribute(attr).Get()
                    info["gpu_capacities"][field] = {"needed": needed, "limit": limit}
                    if not _integer(needed) or not _integer(limit) or needed < 0 or limit <= 0 or needed >= limit:
                        issues.append("native_capacity_unknown_or_saturated:" + field)
        except Exception as exc:
            issues.append(f"health_unavailable:{type(exc).__name__}:{exc}")
        return deepcopy(info)

    def identity_issues(self, actor, collider):
        from pxr import UsdPhysics
        prim = self.stage.GetPrimAtPath(collider)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            return ["unresolved_collider_identity:" + collider]
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True:
            return ["reported_collider_not_enabled:" + collider]
        owner = prim
        while owner.IsValid() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            owner = owner.GetParent()
        if not owner.IsValid() or str(owner.GetPath()) != actor:
            return ["reported_collider_actor_ownership_mismatch:" + collider]
        return []


def _clock(clock):
    value = (clock.current_time_step_index, clock.current_time)
    if not _integer(value[0]) or value[0] < 0 or not _finite(value[1]):
        raise ValueError("missing_or_nonfinite_native_clock")
    return value


def _vector_container(value):
    return (value is not None and not isinstance(value, (str, bytes, bytearray, dict))
            and hasattr(value, "__len__") and hasattr(value, "__getitem__"))


def _copy_contact(data, index):
    item = {"source_data_index": index, "issues": []}
    for name in ("position", "normal", "impulse"):
        try:
            values = list(getattr(data, name))
            item[name + "_raw"] = deepcopy(values)
            if len(values) != 3 or not all(_finite(v) for v in values):
                item["issues"].append("malformed_or_nonfinite:" + name)
        except Exception as exc:
            item["issues"].append(f"unavailable:{name}:{type(exc).__name__}:{exc}")
    for name in ("separation", "face_index0", "face_index1", "material0", "material1"):
        try:
            item[name + "_raw"] = deepcopy(getattr(data, name))
            good = _finite(item[name + "_raw"]) if name == "separation" else _integer(item[name + "_raw"])
            if not good:
                item["issues"].append("malformed_or_nonfinite:" + name)
        except Exception as exc:
            item["issues"].append(f"unavailable:{name}:{type(exc).__name__}:{exc}")
    return item


class NativeJawReader:
    """Explicit polling or copied callback transport; never a fallback vote."""
    def __init__(self, physics, clock, decode, probe, coverage, *, pose_reader=None,
                 max_headers=MAX_HEADERS, max_contacts=MAX_CONTACTS, callback_collector=None):
        if any(not _integer(n) or n <= 0 for n in (max_headers, max_contacts)):
            raise ValueError("positive_local_copy_capacities_required")
        self.physics, self.clock, self.decode, self.probe = physics, clock, decode, probe
        self.coverage, self.pose_reader = deepcopy(coverage), pose_reader
        self.max_headers, self.max_contacts = max_headers, max_contacts
        self.closed = False
        self.observation_control = None
        self.callback_collector = callback_collector
        self.collector_transport = "callback-contact/v1" if callback_collector is not None else "polling-contact/v1"
        self._callback_pair_state = None

    def configure_observation(self, control_body_path, control_support_path, *,
                              body_collider_path=None, support_collider_path=None):
        """Enable a saved sidecar of the existing acquisition, never another read.

        Exact fixture identities are diagnostic routing inputs, not additions to
        the production jaw map or evidence of native delivery/physical contact.
        """
        if not all(isinstance(p, str) for p in (control_body_path, control_support_path)):
            raise ValueError("exact_observation_control_paths_required")
        paths = {"body_actor": control_body_path, "support_path": control_support_path,
                 "body_collider": body_collider_path or control_body_path + "/geometry/mesh",
                 "support_collider": support_collider_path or control_support_path + "/geometry/mesh"}
        if any(not isinstance(p, str) or not p.startswith("/World/ChicGraspObservationControl/")
               for p in paths.values()):
            raise ValueError("exact_observation_control_paths_required")
        if paths["body_collider"] == paths["support_collider"] or paths["body_actor"] == paths["support_path"]:
            raise ValueError("distinct_observation_control_paths_required")
        if self.observation_control is not None and self.observation_control != paths:
            raise ValueError("observation_control_identity_cannot_change")
        self.observation_control = paths

    def _observation_header(self, raw, data):
        """Classify copied fields only; never contributes to production issues."""
        item = deepcopy(raw)
        problems = [item["error"]] if "error" in item else []
        paths = [item.get(f"actor{i}_path") for i in (0, 1)]
        colliders = [item.get(f"collider{i}_path") for i in (0, 1)]
        control = self.observation_control
        classification = "UNRESOLVED"
        if (set(colliders) == {control["body_collider"], control["support_collider"]}
                and any(paths[i] == control["body_actor"] and colliders[i] == control["body_collider"]
                        for i in (0, 1))):
            classification = "CONTROL_PAIR"
        elif all(isinstance(p, str) and p.startswith("/") for p in paths + colliders):
            if any(p in self.coverage["jaws"] for p in paths):
                classification = ("PRODUCTION_CHICKEN_JAW" if self.coverage["chicken_actor"] in paths
                                  else "PRODUCTION_JAW_OTHER")
            elif self.coverage["chicken_actor"] in paths:
                classification = "CHICKEN_OTHER"
            else:
                classification = "OTHER"
        start, count = item.get("contact_data_offset"), item.get("num_contact_data")
        if _integer(start) and start >= 0 and _integer(count) and count >= 0:
            end = start + count
            item["copied_data_indices"] = list(range(start, min(end, len(data))))
            if end > len(data):
                problems.append("contact_data_not_fully_copied")
            for di in item["copied_data_indices"]:
                problems.extend(f"contact[{di}]:{error}" for error in data[di]["issues"])
        else:
            item["copied_data_indices"] = []
        if item.get("proto_index0") != 0xFFFFFFFF or item.get("proto_index1") != 0xFFFFFFFF:
            problems.append("instanced_contact_identity_unestablished")
        item.update(pair_classification=classification,
                    event_name=(CONTACT_EVENT_NAMES.get(item.get("type"), "UNRECOGNIZED_CONTACT_EVENT")
                                if _integer(item.get("type")) else "UNRECOGNIZED_CONTACT_EVENT"),
                    issues=list(dict.fromkeys(problems)), valid=not problems,
                    production_route=("EXISTING_HEADER_ERROR_RETAINED" if "error" in item else
                                      "MONITORED_JAW" if item.get("monitored_jaws") else "UNMONITORED_PAIR_FILTERED"),
                    identity_basis="exact copied decoded paths; static actor domain not independently certified")
        return item

    def clock_identity(self):
        return _clock(self.clock)

    def _recorded_decode(self, encoded, observations):
        """Record only the result of the existing decode; never call it twice."""
        try:
            value = str(self.decode(encoded))
        except Exception as exc:
            observations.append({"encoded": deepcopy(encoded), "outcome": "RAISE",
                                 "exception_type": type(exc).__name__, "message": str(exc)})
            raise
        observations.append({"encoded": deepcopy(encoded), "outcome": "RETURN", "value": value})
        return value

    def _recorded_identity(self, actor, collider, observations):
        identity = {"actor": actor, "collider": collider}
        try:
            value = self.probe.identity_issues(actor, collider)
        except Exception as exc:
            observations.append(dict(identity, outcome="RAISE", exception_type=type(exc).__name__, message=str(exc)))
            raise
        observations.append(dict(identity, outcome="RETURN", value=deepcopy(value)))
        return value

    def decode_initialization_evidence(self, lifecycle):
        """Decode a completed initialization snapshot once, never seed admission.

        Called after ordinary binding/control configuration. No new acquisition,
        scene or pose read occurs here; native path decoding only interprets
        already-copied scalar identities. In-flight receipts may finish later;
        they remain unarchived raw diagnostics outside this recorded ID set.
        """
        if self.callback_collector is None or self.observation_control is None:
            raise ValueError("callback_observation_binding_required")
        records = lifecycle["initialization_callbacks"]
        result = {"schema": "gate5-callback-initialization-routing/v1",
                  "candidate_current": False, "stage_id": self.probe.stage_id,
                  "source_sequences": [r["sequence"] for r in records], "callbacks": [],
                  "decode_observations": [],
                  "limit": "completed initialization snapshot only; no candidate clock or production negative qualification"}
        for record in records:
            item = {"sequence": record["sequence"], "candidate_current": False, "headers": []}
            for original in record["raw_headers"]:
                raw = deepcopy(original)
                try:
                    for key in ("actor0", "actor1", "collider0", "collider1"):
                        raw[key + "_path"] = self._recorded_decode(raw[key], result["decode_observations"])
                    if raw.get("stage_id") != self.probe.stage_id:
                        raise ValueError("initialization_stage_not_managed_stage")
                except Exception as exc:
                    raw["error"] = type(exc).__name__ + ":" + str(exc)[:512]
                routed = self._observation_header(raw, record["data"])
                routed["production_route"] = "INITIALIZATION_DIAGNOSTIC_ONLY"
                item["headers"].append(routed)
            result["callbacks"].append(item)
        return deepcopy(_json(result))

    def read(self):
        result = {"clock_before": None, "clock_after": None, "stage_id": self.probe.stage_id,
                  "headers": [], "contacts": [], "unassociated_contacts": [], "issues": [], "health": {},
                  "container_status": "unavailable", "no_contacts_reported": None,
                  "chicken_pose": None, "raw_semantics": deepcopy(RAW_SEMANTICS)}
        issues = result["issues"]
        if self.callback_collector is not None:
            result["collector_transport"] = self.collector_transport
            result["callback_decoded_headers"] = []
            result["replay_observations"] = {"physics_dt": None, "decode": [], "identity": []}
        snapshot = None
        if self.observation_control is not None:
            snapshot = {"schema": "gate5-observation-contact-snapshot/v1", "diagnostic_only": True,
                "native_api": "IPhysxSimulation.get_contact_report",
                "acquisition_phase": "existing post-physics collector boundary; copy before decode/pose/health",
                "native_api_calls": 0, "container_status": "unavailable", "raw_headers": [],
                "data": [], "headers": [], "observation_issues": [],
                "control_identity": deepcopy(self.observation_control),
                "native_drop_accounting": {"status": "NATIVE_DROP_ACCOUNTING_UNAVAILABLE", "count": None,
                    "units": None, "scope": None, "reset_behavior": None,
                    "source": "installed omni.physx 107.3.26 _physx.pyi IPhysxSimulation contact interface",
                    "limit": "contact lifecycle events and capacity statistics are not transport drop counters"},
                "coverage_limit": "existing reporting restrictions unchanged; nonreporting other bodies may be absent; positive control does not certify production coverage",
                "friction_anchor_payload": "UNAVAILABLE_IN_EXISTING_TWO_VECTOR_ACQUISITION; header metadata retained"}
            if self.callback_collector is not None:
                snapshot.update(collector_transport=self.collector_transport,
                    native_api="IPhysxSimulation.subscribe_contact_report_events",
                    acquisition_phase="completed local callback copies from the existing sim.step interval; post-fetch/pre-render drain")
        try:
            if self.closed:
                raise ValueError("diagnostic_binding_closed")
            result["clock_before"] = self.clock_identity()
            physics_dt = self.clock.get_physics_dt()
            if self.callback_collector is not None:
                result["replay_observations"]["physics_dt"] = physics_dt
            if physics_dt != PHYSICS_DT_S:
                raise ValueError("physics_interval_changed")
            result["health"]["before"] = self.probe.health()
            issues.extend(result["health"]["before"]["issues"])
            if self.callback_collector is not None:
                from .callback_contact_collector import flatten_boundary
                boundary = self.callback_collector.take_boundary(result["clock_before"], self.probe.stage_id)
                result["callback_boundary"] = deepcopy(boundary)
                batch = flatten_boundary(boundary)
                copied_headers, copied_data = batch["headers"], batch["data"]
                for name in ("source_header_count", "source_data_count", "local_omitted_headers", "local_omitted_data"):
                    result[name] = batch[name]
                issues.extend(batch["issues"])
                if snapshot is not None:
                    snapshot["callback_boundary"] = deepcopy(boundary)
            else:
                if snapshot is not None:
                    snapshot["native_api_calls"] += 1
                native = self.physics.get_contact_report()
                if not isinstance(native, (tuple, list)) or len(native) != 2:
                    raise ValueError("missing_or_malformed_report_container")
                headers, data = native
                if not _vector_container(headers) or not _vector_container(data):
                    raise ValueError("missing_or_malformed_report_vector")
                result["source_header_count"], result["source_data_count"] = len(headers), len(data)
                result["local_omitted_headers"] = max(0, len(headers)-self.max_headers)
                result["local_omitted_data"] = max(0, len(data)-self.max_contacts)
                if len(headers) >= self.max_headers or len(data) >= self.max_contacts:
                    issues.append("local_copy_capacity_reached_or_exceeded")
                if not len(headers) and len(data):
                    issues.append("orphan_contact_data_without_headers")
                copied_data = [_copy_contact(data[di], di) for di in range(min(len(data), self.max_contacts))]
                copied_headers = []
                for hi in range(min(len(headers), self.max_headers)):
                    raw = {"source_header_index": hi}
                    for key in HEADER_FIELDS:
                        try:
                            value = getattr(headers[hi], key)
                            raw[key] = deepcopy(int(value) if key == "type" and not isinstance(value, (bool, float, str))
                                                else value)
                        except Exception as exc:
                            raw.setdefault("error", f"{type(exc).__name__}:{exc}")
                    copied_headers.append(raw)
            result["container_status"] = "present"
            if snapshot is not None:
                snapshot.update(container_status="present", raw_headers=deepcopy(copied_headers),
                                data=deepcopy(copied_data), source_header_count=result["source_header_count"],
                                source_data_count=result["source_data_count"], local_omitted_headers=result["local_omitted_headers"],
                                local_omitted_data=result["local_omitted_data"])
            occupied = set()
            for hi, raw in enumerate(copied_headers):
                try:
                    if "error" in raw:
                        result["headers"].append(deepcopy(raw))
                        issues.append(f"header[{hi}]:{raw['error']}")
                        continue
                    decode = (lambda value: self._recorded_decode(value, result["replay_observations"]["decode"])) \
                        if self.callback_collector is not None else lambda value: str(self.decode(value))
                    raw["actor0_path"], raw["actor1_path"] = (decode(raw[k]) for k in ("actor0", "actor1"))
                    raw["collider0_path"], raw["collider1_path"] = (decode(raw[k]) for k in ("collider0", "collider1"))
                    if any(not _integer(raw[k]) for k in ("stage_id", "actor0", "actor1", "collider0", "collider1",
                                                         "type", "num_contact_data", "contact_data_offset",
                                                         "num_friction_anchors_data", "friction_anchors_offset",
                                                         "proto_index0", "proto_index1")):
                        raise ValueError("noninteger_native_header")
                    if any(not raw[k].startswith("/") for k in ("actor0_path", "actor1_path", "collider0_path", "collider1_path")):
                        raise ValueError("unresolved_native_path")
                    count, start, kind = raw["num_contact_data"], raw["contact_data_offset"], raw["type"]
                    if raw["stage_id"] != self.probe.stage_id:
                        raise ValueError("header_stage_mismatch")
                    if kind not in (0, 1, 2) or count < 0 or start < 0 or (count and start + count > result["source_data_count"]):
                        raise ValueError("invalid_header_type_count_or_offset")
                    if (kind == 1 and count != 0) or (kind != 1 and count == 0):
                        raise ValueError("header_type_count_inconsistent")
                    if raw["num_friction_anchors_data"] < 0 or raw["friction_anchors_offset"] < 0:
                        raise ValueError("invalid_friction_anchor_metadata")
                    slots = set(range(start, min(start+count, self.max_contacts)))
                    if occupied.intersection(slots):
                        raise ValueError("overlapping_contact_ranges")
                    occupied.update(slots)
                    raw["monitored_jaws"] = []
                    for side in (0, 1):
                        actor, collider = raw[f"actor{side}_path"], raw[f"collider{side}_path"]
                        if actor in self.coverage["jaws"]:
                            jaw = self.coverage["jaws"][actor]
                            other = raw[f"actor{1-side}_path"]
                            other_shape = raw[f"collider{1-side}_path"]
                            raw["monitored_jaws"].append({
                                "reporting_actor_slot": side, "reporting_actor": actor,
                                "opposing_actor": other, "jaw_id": jaw["jaw_id"], "station_id": jaw["station_id"],
                                "jaw_collider_is_contributor": collider in jaw["contributor_colliders"],
                                "opposing_actor_is_chicken": other == self.coverage["chicken_actor"],
                                "chicken_child_in_allowed_identity_list": other == self.coverage["chicken_actor"] and other_shape in jaw["allowed_chicken_colliders"],
                                "chicken_child_in_forbidden_identity_list": other == self.coverage["chicken_actor"] and other_shape in jaw["forbidden_chicken_colliders"],
                            })
                    if not raw["monitored_jaws"]:
                        continue
                    result["headers"].append(deepcopy(raw))
                    if raw["proto_index0"] != 0xFFFFFFFF or raw["proto_index1"] != 0xFFFFFFFF:
                        issues.append("instanced_contact_identity_unestablished")
                    for side in (0, 1):
                        try:
                            actor, collider = raw[f"actor{side}_path"], raw[f"collider{side}_path"]
                            observed = (self._recorded_identity(actor, collider, result["replay_observations"]["identity"])
                                        if self.callback_collector is not None else self.probe.identity_issues(actor, collider))
                            issues.extend(observed)
                        except Exception as exc:
                            issues.append(f"native_identity_unavailable:{type(exc).__name__}:{exc}")
                    for di in sorted(slots):
                        item = deepcopy(copied_data[di])
                        item["source_header_index"] = hi
                        result["contacts"].append(item)
                        issues.extend(f"contact[{di}]:{i}" for i in item["issues"])
                except Exception as exc:
                    raw["error"] = f"{type(exc).__name__}:{exc}"
                    result["headers"].append(deepcopy(raw))
                    issues.append(f"header[{hi}]:{raw['error']}")
                finally:
                    if self.callback_collector is not None:
                        result["callback_decoded_headers"].append(deepcopy(raw))
                    if snapshot is not None:
                        try:
                            snapshot["headers"].append(self._observation_header(raw, copied_data))
                        except Exception as exc:
                            # Diagnostic-only routing must not change the old
                            # production failure/success path or hide its error.
                            snapshot["observation_issues"].append(f"header[{hi}]:{type(exc).__name__}:{exc}")
            result["unassociated_contacts"] = [item for di, item in enumerate(copied_data) if di not in occupied]
            if (result["source_data_count"] <= self.max_contacts and result["source_header_count"] <= self.max_headers
                    and len(occupied) != result["source_data_count"]):
                issues.append("unidentified_or_unreferenced_contact_data")
            if self.pose_reader is not None:
                pose = deepcopy(self.pose_reader.read())
                result["chicken_pose"] = pose
                if (pose.get("valid") is not True or pose.get("clock_before") != result["clock_before"]
                        or pose.get("clock_after") != result["clock_before"]
                        or pose.get("identities", {}).get("chicken_actor") != self.coverage["chicken_actor"]):
                    issues.append("missing_or_stale_same_step_chicken_pose")
        except Exception as exc:
            issues.append(f"acquisition_unavailable:{type(exc).__name__}:{exc}")
        # Include health and clock changes across the COMPLETE copy/pose operation.
        try:
            result["health"]["after"] = self.probe.health()
            issues.extend(result["health"]["after"]["issues"])
            result["clock_after"] = self.clock_identity()
            if result["clock_after"] != result["clock_before"]:
                issues.append("native_clock_changed_during_acquisition")
        except Exception as exc:
            issues.append(f"final_health_or_clock_unavailable:{type(exc).__name__}:{exc}")
        result["issues"] = list(dict.fromkeys(issues))
        if self.callback_collector is not None and "callback_boundary" in result:
            from .callback_contact_collector import reduce_boundary
            awake = {actor: details.get("sleeping") is False for actor, details in
                     result.get("health", {}).get("after", {}).get("jaws", {}).items()}
            reduced = reduce_boundary(result["callback_boundary"], result["callback_decoded_headers"],
                self.coverage, awake=awake, previous_state=self._callback_pair_state)
            self._callback_pair_state = deepcopy(reduced["state"])
            result["callback_reduction"] = reduced
            if snapshot is not None:
                snapshot["callback_reduction"] = deepcopy(reduced)
        result["acquisition_status"] = ("UNKNOWN" if result["issues"] else
            "CALLBACK_BOUNDARY_FINALIZED" if self.callback_collector is not None else "RAW_REPORT_OBSERVED")
        if not result["issues"] and result["container_status"] == "present":
            result["no_contacts_reported"] = not result["contacts"]
        if snapshot is not None:
            snapshot.update(clock_before=result["clock_before"], clock_after=result["clock_after"],
                            stage_id=result["stage_id"], acquisition_status=result["acquisition_status"],
                            acquisition_issues=deepcopy(result["issues"]),
                            copied_header_count=len(snapshot["raw_headers"]), copied_data_count=len(snapshot["data"]),
                            retained_header_count=len(snapshot["headers"]),
                            decoded_header_error_count=sum(bool(h["issues"]) for h in snapshot["headers"]),
                            production_header_count=len(result["headers"]), production_contact_count=len(result["contacts"]),
                            persistence_status="IN_MEMORY_COPIED_NOT_YET_PERSISTED")
            snapshot["local_unclassified_headers"] = len(snapshot["raw_headers"]) - len(snapshot["headers"])
            if self.callback_collector is not None:
                snapshot["decoded_header_count"] = sum(all(isinstance(h.get(key), str)
                    and h[key].startswith("/") for key in
                    ("actor0_path", "actor1_path", "collider0_path", "collider1_path"))
                    for h in snapshot["headers"])
            result["observation_snapshot"] = _json(snapshot)
        return deepcopy(result)

    def close(self):
        """Release only this binding's error observer; never close the simulator."""
        subscription = getattr(self.probe, "error_subscription", None)
        if subscription is not None:
            subscription.unsubscribe()
            self.probe.error_subscription = None
        self.closed = True


_NOT_EVALUATED = object()


def _stage_operand(value, source, *, conversion=None):
    """Diagnostic representation only; never fed back into the comparison."""
    item = {"source": source, "conversion": conversion}
    if value is _NOT_EVALUATED:
        return dict(item, availability="not_evaluated")
    item.update(availability="captured", python_type=type(value).__module__ + "." + type(value).__qualname__,
                repr=repr(value))
    if type(value) is int:
        item.update(value_text=str(value), encoding="integer_decimal")
    elif type(value) is bool:
        item.update(value_text="True" if value else "False", encoding="boolean_literal")
    elif value is None:
        item.update(value_text="None", encoding="none_literal")
    else:
        item.update(value_text=item["repr"], encoding="python_repr",
                    representation_limit="wrapper repr is not an independent native identity certificate")
    return item


def _record_stage_guard(diagnostics, cache_id, cached, attached, nonpositive, mismatch,
                        verdict, comparison_error=None):
    if diagnostics is None:
        return
    try:
        record = {"guard": "native_stage_identity_mismatch",
            "callable": "chicgrasp_policy.jaw_contact_diagnostic.bind_runtime",
            "location": "src/chicgrasp_policy/jaw_contact_diagnostic.py:bind_runtime stage comparison",
            "sequence": len(diagnostics["records"]), "monotonic_ns": time.monotonic_ns(),
            "context": deepcopy(diagnostics.get("context", {})),
            "operands": {
                "cache_id_wrapper": _stage_operand(cache_id, "UsdUtils.StageCache.Get().GetId(stage)"),
                "cached_stage_id": _stage_operand(cached,
                    "UsdUtils.StageCache.Get().GetId(stage).ToLongInt()", conversion="existing ToLongInt()"),
                "attached_stage_id": _stage_operand(attached,
                    "get_physx_simulation_interface().get_attached_stage()")},
            "identity_domains": {"cached_stage_id": "USD StageCache identifier for caller-supplied stage",
                "attached_stage_id": "stage attached through IPhysxSimulation; equivalence not assumed"},
            "nonpositive_cache": _stage_operand(nonpositive, "p.stage_id <= 0"),
            "mismatch": _stage_operand(mismatch, "p.physics.get_attached_stage() != p.stage_id"),
            "verdict": verdict}
        if comparison_error is not None:
            record["comparison_error"] = type(comparison_error).__name__ + ":" + str(comparison_error)
        # Validate/copy before exposing a record to another component. No float
        # normalization, default-zero value or repeat native getter is permitted.
        record = json.loads(json.dumps(record, allow_nan=False))
        diagnostics["records"].append(deepcopy(record))
        diagnostics["errors"].extend("execution_provenance:" + str(error)
                                     for error in record["context"].get("errors", []))
        if diagnostics.get("persist") is not None:
            diagnostics["persist"](deepcopy(record))
    except Exception as exc:
        diagnostics["errors"].append(type(exc).__name__ + ":" + str(exc))


def bind_runtime(env, stage, *, task_id, setup, diagnostics=None, scene_identity=None, callback_collector=None):
    """Explicit future binding; no setup, callback, scene creation or step."""
    _guard(task_id, env.num_envs)
    from pxr import PhysicsSchemaTools, UsdPhysics, UsdUtils
    import carb.settings
    from omni.physx import get_physx_interface, get_physx_simulation_interface, get_physx_statistics_interface
    from omni.physx.bindings._physx import PhysicsSceneStats
    from .lift_event_pose_evidence import NativePoseReader
    contracts = load_contract_map()
    if setup.get("completed") is not True or setup.get("coverage") != contracts or setup.get("task_id") != task_id:
        raise ValueError("matching_completed_reporting_preparation_required")
    p = RuntimeProbe()
    p.stage, p.coverage = stage, inspect_runtime_coverage(stage, contracts)
    cache_id = UsdUtils.StageCache.Get().GetId(stage)
    p.stage_id = cache_id.ToLongInt()
    p.physics = get_physx_simulation_interface()
    p.scene_identity = scene_identity
    if scene_identity is not None:
        if scene_identity.env is not env or scene_identity.stage_id != p.stage_id:
            raise ValueError("managed_identity_requires_its_capture_environment_and_stage")
        scene_identity.validate("jaw_bind", emit=True)
    else:
        attached = nonpositive = mismatch = _NOT_EVALUATED
        try:
            # Explicit-attachment callers retain original ordering and rejection.
            nonpositive = p.stage_id <= 0
            if nonpositive:
                rejected = True
            else:
                attached = p.physics.get_attached_stage()
                mismatch = attached != p.stage_id
                rejected = bool(mismatch)
        except Exception as exc:
            _record_stage_guard(diagnostics, cache_id, p.stage_id, attached, nonpositive, mismatch,
                                "COMPARISON_ERROR", exc)
            raise
        _record_stage_guard(diagnostics, cache_id, p.stage_id, attached, nonpositive, mismatch,
                            "REJECT" if rejected else "PASS")
        if rejected:
            raise ValueError("native_stage_identity_mismatch")
    scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if len(scenes) != 1:
        raise ValueError("one_existing_physics_scene_required")
    p.scene, = scenes
    p.encode, decode = PhysicsSchemaTools.sdfPathToInt, PhysicsSchemaTools.intToSdfPath
    p.scene_id = p.encode(str(p.scene.GetPath()))
    p.settings = carb.settings.get_settings()
    p.statistics, p.stats = get_physx_statistics_interface(), PhysicsSceneStats()
    p.errors = []
    p.error_overflow = False
    def on_error(event):
        if len(p.errors) < 64:
            p.errors.append(f"native_error:{event.type}:{event.payload}")
        elif not p.error_overflow:
            p.errors.append("native_error_log_local_capacity_reached")
            p.error_overflow = True
    p.error_subscription = get_physx_interface().get_error_event_stream().create_subscription_to_push(on_error)
    try:
        initial = p.health()
        if initial["issues"]:
            raise ValueError("native_reporting_health_unavailable:" + ";".join(initial["issues"]))
        reader = NativeJawReader(p.physics, env.sim, decode, p, p.coverage,
                                 pose_reader=NativePoseReader(env, task_id=task_id,
                                                            scene_identity=scene_identity),
                                 callback_collector=callback_collector)
        reader.setup_evidence = deepcopy(setup)
        return reader
    except Exception:
        p.error_subscription.unsubscribe()
        raise


class JawContactDiagnostic:
    """Bounded one-episode evidence, never eligibility.contacts or validity."""
    def __init__(self, reader, *, episode_id, origin_step=None, origin_time=None, sample_capacity=SAMPLE_CAPACITY,
                 retain_samples=True):
        if not isinstance(episode_id, str) or not episode_id.strip() or not _integer(sample_capacity) or sample_capacity <= 0:
            raise ValueError("episode_identity_and_positive_sample_capacity_required")
        origin = reader.clock_identity()
        if origin_step is None and origin_time is None:
            origin_step, origin_time = origin
        if not _integer(origin_step) or not _finite(origin_time) or (origin_step, origin_time) != origin:
            raise ValueError("episode_origin_must_match_current_native_clock")
        self.reader, self.episode_id = reader, episode_id
        self.origin = (origin_step, origin_time)
        self.sample_capacity, self._samples, self._issues = sample_capacity, [], []
        if type(retain_samples) is not bool:
            raise TypeError("retain_samples_must_be_boolean")
        self.retain_samples, self._sample_count = retain_samples, 0
        self._last_clock, self._seen = self.origin, set()

    def use_external_storage(self):
        """Select reference-only history before any sampled boundary exists."""
        if self._sample_count or self._samples:
            raise RuntimeError("external_storage_must_precede_first_contact_sample")
        self.retain_samples = False

    def post_physics_step(self, control_tick, substep, *, episode_id=None):
        if self._sample_count >= self.sample_capacity:
            if "sample_storage_full" not in self._issues:
                self._issues.append("sample_storage_full")
            raise ValueError("sample_storage_full_no_overwrite")
        sample = {"episode_id": self.episode_id if episode_id is None else episode_id,
                  "control_tick": control_tick, "substep": substep, "issues": []}
        issues = sample["issues"]
        if sample["episode_id"] != self.episode_id:
            issues.append("wrong_episode")
        context_ok = _integer(control_tick) and 0 <= control_tick <= 1199 and _integer(substep) and substep in (0, 1)
        if not context_ok:
            issues.append("invalid_control_tick_or_substep")
        reading = self.reader.read()
        sample["reading"] = reading
        issues.extend(reading["issues"])
        clock = reading["clock_before"]
        if context_ok:
            relative = 2*control_tick + substep + 1
            if (control_tick, substep) in self._seen:
                issues.append("duplicate_substep_read")
            self._seen.add((control_tick, substep))
            if clock is None or clock[0] != self.origin[0] + relative:
                issues.append("stale_or_wrong_native_step")
            if relative != self._sample_count + 1:
                issues.append("missing_or_out_of_order_substeps")
        if clock is not None:
            if clock[0] <= self._last_clock[0] or clock[1] <= self._last_clock[1]:
                issues.append("duplicate_stale_or_regressed_native_clock")
            if clock[0] > self._last_clock[0]:
                self._last_clock = clock
        sample["issues"] = list(dict.fromkeys(issues))
        if sample["issues"]:
            reading["no_contacts_reported"] = None
        sample["acquisition_status"] = "UNKNOWN" if sample["issues"] else reading["acquisition_status"]
        if "observation_snapshot" in reading:
            reading["observation_snapshot"]["context"] = {
                "episode_id": sample["episode_id"], "control_tick": control_tick, "substep": substep,
                "origin_step": self.origin[0], "origin_time_s": self.origin[1],
                "issues": deepcopy(sample["issues"]), "acquisition_status": sample["acquisition_status"]}
        self._sample_count += 1
        if self.retain_samples:
            self._samples.append(deepcopy(sample))
        return deepcopy(sample)

    def export(self):
        result = _json({"schema": "gate5-jaw-raw-contact-diagnostic/v1", "diagnostic_only": True,
                        "task_id": TASK_ID, "episode_id": self.episode_id,
                        "origin_step": self.origin[0], "origin_time_s": self.origin[1],
                        "coverage": self.reader.coverage, "raw_semantics": RAW_SEMANTICS,
                        "setup": getattr(self.reader, "setup_evidence", None),
                        "samples": self._samples, "storage_issues": self._issues,
                        "sample_capacity": self.sample_capacity, "local_header_capacity": self.reader.max_headers,
                        "local_contact_capacity": self.reader.max_contacts,
                        "station_predicate": "NOT_IMPLEMENTED", "native_delivery_verified": False,
                        "completeness_proven": False})
        if not self.retain_samples:
            result.update(sample_storage="external_compact_contact_references", sample_count=self._sample_count)
        if self.reader.callback_collector is not None:
            result["collector_transport"] = self.reader.collector_transport
            if self.retain_samples:
                result["callback_lifecycle"] = self.reader.callback_collector.export_lifecycle()
            else:
                result["callback_lifecycle_storage"] = "external_compact_contact_references"
        json.dumps(result, allow_nan=False)
        return result

    def close(self):
        """Unsubscribe this diagnostic's error observer only; no scene/app close."""
        self.reader.close()
