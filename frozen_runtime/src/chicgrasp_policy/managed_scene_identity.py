"""Source-pinned identity checks for the existing Isaac Lab USD-context route.

Ordinary import is native-free. This does not construct, attach, repair, or advance
a scene. Native cache identity is used for USD stages; Python-owned object
continuity is used only for the retained environment/assets/tensor wrappers.
"""
from __future__ import annotations

from copy import deepcopy
import math
import time


MODE = "isaaclab_usd_context"
ENV_ROOT = "/World/envs/env_0"
ROBOT_ROOT = ENV_ROOT + "/Robot"
CHICKEN_BODY = ENV_ROOT + "/Chicken/raw_chicken_visual/raw_chicken_visual_mesh"
REQUIRED_LINKS = tuple(ROBOT_ROOT + "/GripperRig/" + name for name in
                      ("gripper_base_link", "jaw_1_link", "jaw_2_link", "jaw_3_link", "jaw_4_link"))
EXPLICIT_SOURCE = "omni.physx.get_physx_simulation_interface().get_attached_stage()"


def _type(value):
    return type(value).__module__ + "." + type(value).__qualname__


def _error(exc):
    try:
        message = str(exc)
    except Exception:
        message = "unprintable_exception"
    return _type(exc) + ":" + message


def _json(value, seen=None):
    """Copy diagnostics without executing arbitrary repr or retaining native objects."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    seen = set() if seen is None else seen
    if type(value) in (dict, list, tuple):
        if id(value) in seen:
            return {"unserializable_type": _type(value), "reason": "cycle"}
        seen.add(id(value))
        if type(value) is dict:
            result = {key: _json(item, seen) for key, item in value.items() if type(key) is str}
            if len(result) != len(value):
                result["nonstring_keys_omitted"] = True
        else:
            result = [_json(item, seen) for item in value]
        seen.remove(id(value))
        return result
    return {"unserializable_type": _type(value)}


def _positive_id(value):
    return type(value) is int and value > 0


class ManagedSceneIdentity:
    """Retain construction objects once; validate their continuing associations.

    Sources and private installed accesses are recorded in the Cycle32 source
    map. In particular, both assets retain ``_physics_sim_view``; only robot.data
    retains that member. Actor data ``_root_physx_view`` is a weakref proxy and
    must not be compared using Python ``is``.
    """

    def __init__(self, env, stage, *, cache, current_usd_stage, current_managed_stage,
                 current_simulation_view, explicit_simulation, diagnostics, mode=MODE):
        self.env, self.stage = env, stage
        self._cache = cache
        self._current_usd_stage = current_usd_stage
        self._current_managed_stage = current_managed_stage
        self._current_simulation_view = current_simulation_view
        self._explicit_simulation = explicit_simulation
        self.mode = mode
        self._diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        self._diagnostics.setdefault("records", [])
        self._diagnostics.setdefault("errors", [])
        self._context = _json(self._diagnostics.get("context", {}))
        if isinstance(self._context, dict):
            self._diagnostics["errors"].extend("execution_provenance:" + str(error)
                                               for error in self._context.get("errors", []))
        self._construction_errors = []

        def retain(name, fn):
            try:
                return fn()
            except Exception as exc:
                self._construction_errors.append(name + ":" + _error(exc))
                return None

        self._sim = retain("sim", lambda: env.sim)
        self._scene = retain("scene", lambda: env.scene)
        self._physics_context = retain("physics_context", lambda: self._sim.get_physics_context())
        self._physics_prim = retain("physics_prim", lambda: self._physics_context._physics_scene.GetPrim())
        self._robot = retain("robot", lambda: self._scene["robot"])
        self._chicken = retain("chicken", lambda: self._scene["chicken"])
        self._simulation_view = retain("simulation_view", current_simulation_view)
        self._robot_view = retain("robot_view", lambda: self._robot.root_physx_view)
        self._chicken_view = retain("chicken_view", lambda: self._chicken.root_physx_view)
        self._cache_id = retain("cache_id", lambda: cache.GetId(stage))
        self.stage_id = retain("cache_integer", lambda: self._cache_id.ToLongInt())
        roots = retain("robot_roots", lambda: list(self._robot_view.prim_paths))
        self._anchor = {
            "stage_cache_id": self.stage_id,
            "physics_scene_path": retain("configured_physics_scene", lambda: env.cfg.sim.physics_prim_path),
            "robot_articulation_path": roots[0] if roots and len(roots) == 1 else None,
            "robot_link_paths": retain("robot_link_paths", lambda: [list(x) for x in self._robot_view.link_paths]),
            "robot_body_names": retain("robot_body_names", lambda: list(self._robot.body_names)),
        }
        self._anchor = _json(self._anchor)
        self.validate("managed_bind", emit=True)

    def _explicit_diagnostic(self, phase):
        result = {"source": EXPLICIT_SOURCE, "phase": phase, "status": "observed",
                  "value": None, "type": None, "exception": None}
        try:
            value = self._explicit_simulation.get_attached_stage()
            result.update(value=_json(value), type=_type(value))
        except Exception as exc:
            result.update(status="query_exception", exception=_error(exc))
        return result

    def _publish(self, record):
        # Diagnostic persistence is separate from the primary validation result.
        try:
            self._diagnostics["records"].append(deepcopy(record))
            sink = self._diagnostics.get("persist")
            if sink is not None:
                sink(deepcopy(record))
        except Exception as exc:
            self._diagnostics["errors"].append("managed_scene_diagnostic:" + _error(exc))

    def validate(self, phase, *, reader=None, emit=False):
        observations, errors = {}, list(self._construction_errors)

        def observe(name, fn):
            try:
                return fn()
            except Exception as exc:
                errors.append(name + ":" + _error(exc))
                return None

        def stage_id(name, stage):
            return observe(name, lambda: self._cache.GetId(stage).ToLongInt())

        sim = observe("env.sim", lambda: self.env.sim)
        scene = observe("env.scene", lambda: self.env.scene)
        physics_context = observe("physics_context", lambda: sim.get_physics_context())
        selected_prim = observe("selected_physics_prim", lambda: physics_context._physics_scene.GetPrim())
        robot = observe("robot", lambda: scene["robot"])
        chicken = observe("chicken", lambda: scene["chicken"])
        simulation_view = observe("manager_simulation_view", self._current_simulation_view)
        usd_stage = observe("current_usd_stage", self._current_usd_stage)
        managed_stage = observe("current_managed_stage", self._current_managed_stage)
        cached_stage = observe("cache_lookup", lambda: self._cache.Find(self._cache_id))
        observations["construction"] = {
            "requested_mode": self.mode,
            "create_stage_in_memory": observe("create_stage_in_memory", lambda: self.env.cfg.sim.create_stage_in_memory),
            "sim_create_stage_in_memory": observe("sim_create_stage_in_memory", lambda: sim.cfg.create_stage_in_memory),
        }
        observations["continuity"] = {
            "env_sim": sim is self._sim and sim is not None,
            "physics_context": physics_context is self._physics_context and physics_context is not None,
            "env_scene": scene is self._scene and scene is not None,
            "robot_asset": robot is self._robot and robot is not None,
            "chicken_asset": chicken is self._chicken and chicken is not None,
            "manager_simulation_view": simulation_view is self._simulation_view and simulation_view is not None,
            "robot_simulation_view": observe("robot_simulation_view", lambda: robot._physics_sim_view is self._simulation_view),
            "chicken_simulation_view": observe("chicken_simulation_view", lambda: chicken._physics_sim_view is self._simulation_view),
            "robot_data_simulation_view": observe("robot_data_simulation_view", lambda: robot.data._physics_sim_view is self._simulation_view),
            "robot_view": observe("robot_view", lambda: robot.root_physx_view is self._robot_view and self._robot_view is not None),
            "chicken_view": observe("chicken_view", lambda: chicken.root_physx_view is self._chicken_view and self._chicken_view is not None),
        }
        observations["cache"] = {
            "contains": observe("cache_contains", lambda: self._cache.Contains(self._cache_id)),
            "stage_ids": {
                "constructed": stage_id("constructed_cache", self.stage),
                "current_usd": stage_id("current_usd_cache", usd_stage),
                "current_managed": stage_id("current_managed_cache", managed_stage),
                "simulation_initial": observe("simulation_initial_cache", lambda: self._cache.GetId(sim.get_initial_stage()).ToLongInt()),
                "simulation_retained": observe("simulation_retained_cache", lambda: self._cache.GetId(sim.stage).ToLongInt()),
                "cache_lookup": stage_id("lookup_cache", cached_stage),
                "retained_physics_scene": observe("retained_scene_cache", lambda: self._cache.GetId(self._physics_prim.GetStage()).ToLongInt()),
                "selected_physics_scene": observe("selected_scene_cache", lambda: self._cache.GetId(selected_prim.GetStage()).ToLongInt()),
                "robot_asset": observe("robot_stage_cache", lambda: self._cache.GetId(robot.stage).ToLongInt()),
                "chicken_asset": observe("chicken_stage_cache", lambda: self._cache.GetId(chicken.stage).ToLongInt()),
                "env_scene": observe("scene_stage_cache", lambda: self._cache.GetId(scene.stage).ToLongInt()),
            },
        }
        observations["physics_scene"] = {
            "configured_path": observe("configured_path", lambda: self.env.cfg.sim.physics_prim_path),
            "sim_configured_path": observe("sim_configured_path", lambda: sim.cfg.physics_prim_path),
            "context_path": observe("context_path", lambda: physics_context.prim_path),
            "retained_path": observe("retained_scene_path", lambda: str(self._physics_prim.GetPath())),
            "selected_path": observe("selected_scene_path", lambda: str(selected_prim.GetPath())),
            "retained_valid": observe("retained_scene_valid", lambda: self._physics_prim.IsValid()),
            "selected_valid": observe("selected_scene_valid", lambda: selected_prim.IsValid()),
            "retained_type": observe("retained_scene_type", lambda: self._physics_prim.GetTypeName()),
            "selected_type": observe("selected_scene_type", lambda: selected_prim.GetTypeName()),
        }
        observations["native"] = {
            "simulation_valid": observe("simulation_valid", lambda: self._simulation_view.is_valid),
            "robot_check": observe("robot_check", lambda: self._robot_view.check()),
            "chicken_check": observe("chicken_check", lambda: self._chicken_view.check()),
            "robot_count": observe("robot_count", lambda: self._robot_view.count),
            "chicken_count": observe("chicken_count", lambda: self._chicken_view.count),
            "robot_initialized": observe("robot_initialized", lambda: robot.is_initialized),
            "chicken_initialized": observe("chicken_initialized", lambda: chicken.is_initialized),
            "robot_roots": observe("robot_roots", lambda: list(self._robot_view.prim_paths)),
            "robot_link_paths": observe("robot_link_paths", lambda: [list(x) for x in self._robot_view.link_paths]),
            "robot_body_names": observe("robot_body_names", lambda: list(robot.body_names)),
            "chicken_paths": observe("chicken_paths", lambda: list(self._chicken_view.prim_paths)),
        }
        observations["reader"] = {"supplied": reader is not None}
        if reader is not None:
            observations["reader"].update({
                "env": observe("reader_env", lambda: reader.env is self.env),
                "robot_asset": observe("reader_robot", lambda: reader.robot is self._robot),
                "chicken_asset": observe("reader_chicken", lambda: reader.chicken is self._chicken),
                "robot_view": observe("reader_robot_view", lambda: reader.robot_view is self._robot_view),
                "chicken_view": observe("reader_chicken_view", lambda: reader.chicken_view is self._chicken_view),
                "body_index": observe("reader_body_index", lambda: reader.body_index),
                "selected_path": observe("reader_selected_path", lambda: self._robot_view.link_paths[0][reader.body_index]),
            })
        observations = _json(observations)
        native, physics = observations["native"], observations["physics_scene"]
        roots, links, names = native["robot_roots"], native["robot_link_paths"], native["robot_body_names"]
        mapping_ok = (type(links) is list and len(links) == 1 and type(names) is list
                      and len(links[0]) == len(names) and all(type(x) is str for x in names)
                      and len(set(names)) == len(names))
        if mapping_ok:
            mapping_ok = all(links[0].count(path) == 1 and names[links[0].index(path)] == path.rsplit("/", 1)[1]
                             for path in REQUIRED_LINKS)
        read = observations["reader"]
        reader_ok = (reader is None or (
            all(read[key] is True for key in ("env", "robot_asset", "chicken_asset", "robot_view", "chicken_view"))
            and type(read["body_index"]) is int and read["body_index"] >= 0
            and read["selected_path"] == REQUIRED_LINKS[0]))
        checks = {
            "supported_mode": (self.mode == MODE and observations["construction"]["create_stage_in_memory"] is False
                               and observations["construction"]["sim_create_stage_in_memory"] is False),
            "construction_anchor": _positive_id(self.stage_id),
            "owned_object_continuity": all(x is True for x in observations["continuity"].values()),
            "cache_association": (observations["cache"]["contains"] is True and _positive_id(self.stage_id)
                                  and all(type(x) is int and x == self.stage_id for x in observations["cache"]["stage_ids"].values())),
            "physics_scene_association": (type(self._anchor["physics_scene_path"]) is str
                and self._anchor["physics_scene_path"].startswith("/")
                and all(physics[key] == self._anchor["physics_scene_path"] for key in
                        ("configured_path", "sim_configured_path", "context_path", "retained_path", "selected_path"))
                and physics["retained_valid"] is True and physics["selected_valid"] is True
                and physics["retained_type"] == "PhysicsScene" and physics["selected_type"] == "PhysicsScene"),
            "native_validity": all(native[key] is True for key in
                ("simulation_valid", "robot_check", "chicken_check", "robot_initialized", "chicken_initialized")),
            "native_counts": type(native["robot_count"]) is int and native["robot_count"] == 1
                and type(native["chicken_count"]) is int and native["chicken_count"] == 1,
            "native_actor_mapping": (type(roots) is list and len(roots) == 1 and type(roots[0]) is str
                and (roots[0] == ROBOT_ROOT or roots[0].startswith(ROBOT_ROOT + "/"))
                and roots[0] == self._anchor["robot_articulation_path"]
                and mapping_ok and links == self._anchor["robot_link_paths"]
                and names == self._anchor["robot_body_names"] and native["chicken_paths"] == [CHICKEN_BODY]),
            "reader_association": reader_ok,
            "required_observations_complete": not errors,
        }
        issues = [name for name, passed in checks.items() if not passed]
        record = _json({
            "guard": "managed_scene_identity", "sequence": len(self._diagnostics["records"]),
            "monotonic_ns": time.monotonic_ns(), "context": self._context,
            "phase": phase, "mode": self.mode, "anchor": self._anchor,
            "observations": observations, "checks": checks, "acquisition_errors": errors,
            "issues": issues, "verdict": "REJECT" if issues else "PASS",
        })
        if emit or issues:
            record["observations"]["explicit_simulation"] = self._explicit_diagnostic(phase)
            self._publish(record)
        else:
            record["observations"]["explicit_simulation"] = {
                "source": EXPLICIT_SOURCE, "phase": phase, "status": "not_sampled_at_continuity_boundary",
                "value": None, "type": None, "exception": None}
        continuity = self._diagnostics.setdefault("continuity", {"validations": 0, "passes": 0, "rejections": 0})
        continuity["validations"] += 1
        continuity["rejections" if issues else "passes"] += 1
        continuity["latest"] = deepcopy(record)
        if issues:
            raise ValueError("managed_scene_identity_invalid:" + ",".join(issues))
        return deepcopy(record)


def bind_managed_scene(env, stage, diagnostics):
    """Explicit deferred binding; use existing managed/native objects only."""
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import UsdUtils

    class ExplicitSimulationDiagnostic:
        def get_attached_stage(self):
            # Acquisition belongs to this diagnostic too; its exception must
            # not prevent collection of authoritative managed observations.
            import omni.physx
            return omni.physx.get_physx_simulation_interface().get_attached_stage()

    return ManagedSceneIdentity(
        env, stage, cache=UsdUtils.StageCache.Get(),
        current_usd_stage=lambda: omni.usd.get_context().get_stage(),
        current_managed_stage=get_current_stage,
        current_simulation_view=SimulationManager.get_physics_sim_view,
        explicit_simulation=ExplicitSimulationDiagnostic(), diagnostics=diagnostics)
