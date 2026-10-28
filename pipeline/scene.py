"""Shared scene source loading and identity checks; no simulator dependencies."""
import hashlib
import json
import math
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
BASE_SCENE_SHA256 = "4f40d26e122c99ea5cb980edea4a391b5335a99fe060b84cacd40305374c0d5f"
CALLBACKS = ("describe_scene", "validate_position", "configure_scene", "build_scene", "evaluate_episode")
BRIDGE_FILES = ("pipeline/scene.py", "pipeline/recording.py")


def _file_record(relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Scene inputs must name repository-relative files")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Scene inputs must stay inside the repository")
    path = (ROOT / path).resolve(strict=True)
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError("Scene input is not a file inside the repository")
    data = path.read_bytes()
    return path, data, {"path": path.relative_to(ROOT.resolve()).as_posix(),
                        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _json_copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def load_scene_setup(experiment, *, expected=None):
    path, data, source = _file_record(experiment.get("scene_setup", "scene_setup.py"))
    if expected is not None:
        if source != expected.get("source"):
            raise ValueError("Scene setup source changed since the plan or approval")
        for record in [*expected.get("assets", []), *expected.get("bridge_files", [])]:
            if _file_record(record["path"])[2] != record:
                raise ValueError("Scene setup dependency changed: " + record["path"])
    module = types.ModuleType("_research_scene_setup")
    module.__file__ = str(path)
    # Execute exactly the bytes hashed above, without writing import caches.
    exec(compile(data, str(path), "exec"), module.__dict__)
    for name in CALLBACKS:
        if not callable(getattr(module, name, None)):
            raise ValueError("Scene setup must define " + name)
    options = _json_copy(experiment.get("scene_options", {}))
    if not isinstance(options, dict):
        raise ValueError("scene_options must be an object")
    definition = _json_copy(module.describe_scene(options))
    if (not isinstance(definition, dict) or not isinstance(definition.get("id"), str)
            or not definition["id"] or definition.get("kind") not in ("table", "custom")):
        raise ValueError("describe_scene must return a nonempty id and table/custom kind")
    inputs = getattr(module, "ASSET_FILES", [])
    if not isinstance(inputs, (list, tuple)) or any(not isinstance(p, str) for p in inputs):
        raise ValueError("ASSET_FILES must list repository-relative input files")
    assets = [_file_record(p)[2] for p in inputs]
    if len({p["path"] for p in assets}) != len(assets):
        raise ValueError("Duplicate scene asset input")
    identity = {"schema": "research-scene-setup/v1",
                "base_scene_sha256": experiment.get("scene_sha256", BASE_SCENE_SHA256),
                "source": source, "options": options, "definition": definition,
                "assets": sorted(assets, key=lambda x: x["path"]),
                "bridge_files": [_file_record(p)[2] for p in BRIDGE_FILES]}
    if expected is not None and identity != expected:
        raise ValueError("Scene setup changed since the plan or approval")
    return module, identity


def scene_setup_identity(experiment):
    return load_scene_setup(experiment)[1]


def authenticate_scene_setup(identity):
    if not isinstance(identity, dict) or identity.get("schema") != "research-scene-setup/v1":
        raise ValueError("Missing scene_setup_identity")
    source = identity.get("source", {})
    if not isinstance(source, dict) or not isinstance(source.get("path"), str):
        raise ValueError("Invalid scene setup source")
    module, actual = load_scene_setup({"scene_setup": source["path"],
                                     "scene_options": identity.get("options"),
                                     "scene_sha256": identity.get("base_scene_sha256")}, expected=identity)
    if actual != identity:
        raise ValueError("Scene setup changed since the plan or approval")
    if actual["base_scene_sha256"] != BASE_SCENE_SHA256:
        raise ValueError("The frozen base scene identity changed")
    return module, actual


def validate_scene_position(experiment, position):
    xyz = position.get("authored_xyz_m")
    quat = position.get("authored_quaternion_wxyz", [1, 0, 0, 0])
    for vector, size, label in ((xyz, 3, "XYZ"), (quat, 4, "quaternion")):
        if (not isinstance(vector, (list, tuple)) or len(vector) != size
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in vector)):
            raise ValueError("Position requires finite " + label)
    if not math.isclose(sum(x*x for x in quat), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Chicken orientation must be a unit quaternion in wxyz order")
    module, identity = load_scene_setup(experiment)
    module.validate_position(_json_copy(position), _json_copy(identity["options"]))


"""Generic scene callback orchestration; no built-in task physics.

Native imports occur only when an explicitly configured custom scene is built.
"""
from copy import deepcopy


def install_scene_builder(cfg, session, module, identity, position, *, event_factory=None):
    """Register one prestartup callback before first reset in all three modes."""
    options = deepcopy(identity["options"])
    module.configure_scene(cfg, deepcopy(position), options)
    custom = identity["definition"]["kind"] == "custom"
    session.research_scene_build = {"status": "NOT_REQUESTED", "custom": custom}
    if not custom:
        # Default callbacks are no-ops; no event, replication or physics change.
        return
    if event_factory is None:
        from isaaclab.managers import EventTermCfg
        event_factory = EventTermCfg
    before = cfg.scene.replicate_physics
    cfg.scene.replicate_physics = False  # Required by IsaacLab's prestartup gate.
    session.research_scene_build = dict(status="PENDING", custom=True,
        replicate_physics_before=before, replicate_physics_after=False)
    calls = 0
    def before_physics(env, env_ids):
        nonlocal calls
        calls += 1
        if calls != 1:
            raise RuntimeError("Custom scene builder must run exactly once before startup")
        stage = env.sim.get_initial_stage()
        session.research_scene_build.update(status="BUILDING", phase="prestartup_before_initial_sim_reset")
        try:
            result = module.build_scene(stage, deepcopy(position), deepcopy(options))
            session.research_scene_build.update(status="BUILT", result=result, calls=calls)
        except Exception as exc:
            session.research_scene_build.update(status="FAILED", error=type(exc).__name__+":"+str(exc))
            raise
    if hasattr(cfg.events, "research_scene_prestartup"):
        raise ValueError("Custom prestartup hook already exists")
    cfg.events.research_scene_prestartup = event_factory(func=before_physics, mode="prestartup")


def validate_custom_build(session):
    result = session.research_scene_build
    if (result.get("status") != "BUILT" or result.get("calls") != 1 or
            result.get("replicate_physics_after") is not False):
        raise RuntimeError("Custom scene was not built before initial physics reset")
    return deepcopy(result)


"""GUI controls for one scripted preview. Importing this module starts nothing."""
from copy import deepcopy
import time


class PreviewStopped(RuntimeError):
    """The owner stopped a preview before all actions completed."""


def validate_preview_configuration(actual, baseline):
    expected = deepcopy(baseline)
    expected.update(has_gui=True, has_gui_raw=True, offscreen_render=False)
    if actual != expected:
        raise ValueError("preview_physics_configuration_changed")
    return {name: {"recording": baseline.get(name), "preview": actual[name]}
            for name in ("has_gui", "has_gui_raw", "offscreen_render")}


def create_controls():
    import omni.ui as ui
    controls = dict(start=False, stop=False, finish=False, window=None, status=None,
                    start_button=None, stop_button=None, finish_button=None)
    try:
        controls["window"] = ui.Window("Scripted demonstration", width=430, height=225, visible=True)
        def request(name):
            controls[name] = True
        with controls["window"].frame:
            with ui.VStack(spacing=8):
                controls["status"] = ui.Label("Ready: 200 actions, 20 seconds of simulation.", word_wrap=True)
                ui.Label("Use the camera menu to switch views. Navigate in Perspective only.\n"
                         "Do not use the timeline play/pause buttons.", word_wrap=True)
                controls["start_button"] = ui.Button("Start demonstration", height=30,
                    clicked_fn=lambda: request("start"))
                controls["stop_button"] = ui.Button("Stop preview", height=30,
                    clicked_fn=lambda: request("stop"))
                controls["finish_button"] = ui.Button("Finish preview", height=30, enabled=False,
                    clicked_fn=lambda: request("finish"))
        return controls
    except Exception:
        if controls["window"] is not None:
            controls["window"].destroy()
        raise


class PreviewSession:
    """Only scheduled steps advance physics; idle renders keep Fabric current."""
    def __init__(self, runtime, cameras, *, seconds=600, controls_factory=None,
                 monotonic=None, sleep=None):
        self.runtime, self.cameras, self.seconds = runtime, cameras, seconds
        self.now, self.sleep = monotonic or time.monotonic, sleep or time.sleep
        self.controls = (controls_factory or create_controls)()
        self.original_step, self.ticks, self.started_at = None, 0, None
        self.record = dict(status="READY", expected_actions=200, expected_control_ticks=1200,
                           simulation_seconds=20, training_data_created=False, video_created=False,
                           real_time_pacing="at_most_1x; rendering may run slower", ui_errors=[],
                           idle_mode="render_only; no physics steps or timeline transitions")
        try:
            from omni.kit.viewport.utility import get_active_viewport
            viewport = get_active_viewport()
            if viewport is None:
                raise RuntimeError("no active viewport; select a camera using its menu")
            viewport.camera_path = cameras.paths["LEFT_HD"]
        except Exception as exc:
            self.record["ui_errors"].append(type(exc).__name__ + ": " + str(exc))

    def _label(self, text):
        if self.controls.get("status") is not None:
            self.controls["status"].text = text

    def _enable(self, name, value):
        if self.controls.get(name) is not None:
            self.controls[name].enabled = value

    def _idle_wait(self, key, *, completed=False):
        # Keep the initialized rendering pipeline alive. IsaacLab sim.render()
        # forwards Fabric, disables app-driven physics around app.update(),
        # then restores that setting. No env.step() runs in this wait.
        clock = tuple(self.runtime.clock())
        end = min(self.now() + self.seconds,
                  getattr(self.runtime, "media_deadline_monotonic", float("inf")) - 5)
        while not self.controls[key]:
            if self.controls["stop"]:
                if completed:
                    return "FINISH_AFTER_COMPLETE"
                raise PreviewStopped("stopped_before_demonstration")
            if not self.runtime.app.is_running():
                if completed:
                    return "WINDOW_CLOSED_AFTER_COMPLETE"
                raise PreviewStopped("window_closed_before_demonstration")
            if self.now() >= end:
                if completed:
                    return "REVIEW_TIME_LIMIT"
                raise PreviewStopped("start_wait_expired; no_actions_executed")
            if not self.runtime.env.sim.is_playing():
                raise PreviewStopped("timeline_paused; use preview_controls_instead")
            self.runtime.env.sim.render()
            if tuple(self.runtime.clock()) != clock:
                raise RuntimeError("preview_physics_advanced_during_idle_wait")
            self.sleep(.01)
        if self.controls["stop"] and not completed:
            raise PreviewStopped("stopped_before_demonstration")
        return "FINISH_BUTTON" if completed else "START_BUTTON"

    def wait_for_start(self):
        self.record["start_reason"] = self._idle_wait("start")
        self.cameras.before_render()
        if not self.runtime.env.sim.is_playing():
            raise PreviewStopped("timeline_paused; use preview_controls_instead")
        self.started_at = self.now()
        self.record["status"] = "RUNNING"
        self._enable("start_button", False)
        self._label("Playing the scripted demonstration. Stop preview ends it early.")

    def wrap_step(self):
        if self.original_step is not None or self.started_at is None:
            raise RuntimeError("preview_step_wrapper_requires_one_start")
        self.original_step = self.runtime.step
        def step(action):
            if self.controls["stop"] or not self.runtime.app.is_running():
                raise PreviewStopped("stopped_during_demonstration")
            if not self.runtime.env.sim.is_playing():
                raise PreviewStopped("timeline_paused; use preview_controls_instead")
            self.cameras.before_render()
            tick_started = self.now()
            flags = self.original_step(action)
            self.ticks += 1
            self.record["completed_control_ticks"] = self.ticks
            # One existing control tick is 1/60 second, with two native substeps.
            delay = 1 / 60 - (self.now() - tick_started)
            if delay > 0:
                self.sleep(delay)
            return flags
        self.runtime.step = step

    def finished(self):
        if self.ticks != 1200:
            raise RuntimeError("preview_requires_all_1200_control_ticks")
        self.record.update(status="COMPLETED", completed_actions=200,
                           playback_wall_seconds=self.now() - self.started_at)
        self._label("Demonstration finished: 200/200 actions. Click Finish preview to close.")
        self._enable("stop_button", False)
        self._enable("finish_button", True)
        self.record["close_reason"] = self._idle_wait("finish", completed=True)

    def close(self):
        if self.original_step is not None:
            self.runtime.step = self.original_step
        if self.record["status"] != "COMPLETED":
            self.record["status"] = "INCOMPLETE"
        if self.controls["window"] is not None:
            try:
                self.controls["window"].destroy()
            except Exception as exc:
                self.record["ui_errors"].append("cleanup: " + type(exc).__name__ + ": " + str(exc))
