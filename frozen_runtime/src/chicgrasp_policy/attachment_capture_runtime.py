"""Explicit native adapter for the capture runner; ordinary import is inert.

The recorder hook runs after the existing sim.step, not from a physics callback.
Only ``RuntimeSession.initialize`` crosses the native import/startup boundary.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import math
import time
import uuid

import numpy as np

from . import interface as policy_api
from .lift_event_pose_evidence import TASK_ID, _json
from .render_profile import SHARED_EXPERIENCE


APP_CONFIGURATION = {
    "headless": True, "enable_cameras": True, "device": "cuda:0",
    "fast_shutdown": True, "livestream": 0, "experience": SHARED_EXPERIENCE,
    # AppLauncher uses this flag to keep viewport rendering active headlessly.
    # It does not create a video recorder; RGB still comes from the two sensors.
    "video": True,
}

PRODUCTION_GPU = "PRODUCTION_GPU"
CPU_BACKEND_DIAGNOSTIC = "CPU_BACKEND_DIAGNOSTIC"
CPU_STARTUP_SHUTDOWN_ONLY = "CPU_STARTUP_SHUTDOWN_ONLY"
CPU_EVIDENCE_FIRST_CLOSE = "CPU_EVIDENCE_FIRST_CLOSE/v1"
EARLY_CPU_PROFILE = "cpu-early-profile/v1"


def _unobserved_cpu_backend():
    """Requested settings never stand in for an observed backend."""
    return {"schema": "cpu-backend-diagnostic/v1", "backend_mode": CPU_BACKEND_DIAGNOSTIC,
            "requested_configuration": {"simulation_device": "cpu",
                                        "application_configuration": deepcopy(APP_CONFIGURATION)},
            "effective_configuration": {}, "read_errors": [],
            "validation_issues": ["backend_not_observed"],
            "verification": "CPU_CONFIGURATION_UNAVAILABLE",
            "observation_phase": "not_observed", "monotonic_ns": None,
            "physics_scene_identity": None}


def _observe_cpu_backend(env, settings, scene_identity):
    """Read the existing configuration boundary once; never change engine state.

    CPU is a conjunction of supported independent operands, not the manager's
    fallback device string alone. Optional renderer metadata is not physics.
    All failures are retained so the caller can persist them before rejection.
    """
    diagnostic = _unobserved_cpu_backend()
    diagnostic.update(observation_phase="after_initial_reset_before_candidate",
                      monotonic_ns=time.monotonic_ns(),
                      physics_scene_identity=deepcopy(scene_identity))
    errors = diagnostic["read_errors"]

    def read(field, getter):
        try:
            value = getter()
            if value is None or type(value) in (str, int, float, bool):
                return value
            return {"unavailable_metadata_type": type(value).__name__,
                    "representation": repr(value)[:512]}
        except Exception as exc:
            errors.append({"field": field, "exception_type": type(exc).__name__,
                           "message": str(exc)[:512]})
            return None

    try:
        physics = env.sim.get_physics_context()
    except Exception as exc:
        errors.append({"field": "physics_context", "exception_type": type(exc).__name__,
                       "message": str(exc)[:512]})
        physics = None
    raw_has_gui = read("has_gui_raw", lambda: env.sim.has_gui())
    result = {
        "task_id": TASK_ID, "cfg_class": read("cfg_class", lambda: type(env.cfg).__name__),
        "num_envs": read("num_envs", lambda: env.num_envs),
        "device": read("device", lambda: env.device),
        "simulation_device": read("simulation_device", lambda: env.sim.device),
        "physics_device": read("physics_device", lambda: physics.device),
        "gpu_sim": read("gpu_sim", lambda: physics.use_gpu_sim),
        "gpu_pipeline": read("gpu_pipeline", lambda: physics.use_gpu_pipeline),
        "gpu_dynamics": read("gpu_dynamics", lambda: physics.is_gpu_dynamics_enabled()),
        "broadphase": read("broadphase", lambda: physics.get_broadphase_type()),
        "physics_dt_s": read("physics_dt_s", lambda: env.physics_dt),
        "control_dt_s": read("control_dt_s", lambda: env.step_dt),
        "decimation": read("decimation", lambda: env.cfg.decimation),
        "render_interval": read("render_interval", lambda: env.cfg.sim.render_interval),
        "action_dimension": read("action_dimension", lambda: env.action_manager.total_action_dim),
        "cameras_enabled": read("cameras_enabled", lambda: settings.get("/isaaclab/cameras_enabled")),
        "offscreen_render": read("offscreen_render", lambda: settings.get("/isaaclab/render/offscreen")),
        "rtx_sensors": read("rtx_sensors", lambda: env.sim.has_rtx_sensors()),
        "fast_shutdown": read("fast_shutdown", lambda: settings.get("/app/fastShutdown")),
        # Preserve the existing settings-OR interpretation, including None.
        "has_gui": bool(raw_has_gui), "has_gui_raw": raw_has_gui,
        "configured_seed": read("configured_seed", lambda: env.cfg.seed),
        "requested_simulation_device": read("requested_simulation_device", lambda: env.cfg.sim.device),
        "suppress_readback": read("suppress_readback", lambda: settings.get("/physics/suppressReadback")),
        "renderer_active_gpu": read("renderer_active_gpu", lambda: settings.get("/renderer/activeGpu")),
        "backend_mode": CPU_BACKEND_DIAGNOSTIC,
    }
    cpu_operands = {"device": "cpu", "simulation_device": "cpu", "physics_device": "cpu",
                    "gpu_sim": False, "gpu_pipeline": False, "gpu_dynamics": False,
                    "broadphase": "MBP", "suppress_readback": False}

    def agrees(field, wanted):
        value = result.get(field)
        return type(value) is type(wanted) and value == wanted

    gpu_operands = {"device": "cuda:0", "simulation_device": "cuda:0", "physics_device": "cuda:0",
                    "gpu_sim": True, "gpu_pipeline": True, "gpu_dynamics": True,
                    "broadphase": "GPU", "suppress_readback": True}
    result["physics_backend"] = (
        "CPU" if all(agrees(k, v) for k, v in cpu_operands.items()) else
        "GPU" if all(agrees(k, v) for k, v in gpu_operands.items()) else "UNAVAILABLE")
    expected = dict(cpu_operands, cfg_class="ChicgraspDynamicChickenEnvCfg", num_envs=1,
                    decimation=2, render_interval=2, action_dimension=5,
                    cameras_enabled=True, offscreen_render=True, rtx_sensors=True,
                    fast_shutdown=True, has_gui=False, configured_seed=42,
                    requested_simulation_device="cpu")
    issues = [key for key, wanted in expected.items() if not agrees(key, wanted)]
    for field, wanted in (("physics_dt_s", 1 / 120), ("control_dt_s", 1 / 60)):
        value = result[field]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not math.isclose(value, wanted, rel_tol=0, abs_tol=1e-12)):
            issues.append(field)
    # A GUI getter exception must not become an apparently observed False.
    issues.extend(item["field"] for item in errors
                  if item["field"] != "renderer_active_gpu" and item["field"] not in issues)
    if not isinstance(scene_identity, dict) or not scene_identity:
        issues.append("physics_scene_identity")
    contradicted = any(field in result and result[field] is not None for field in issues
                       if field not in {item["field"] for item in errors})
    diagnostic.update(effective_configuration=deepcopy(_json(result)), validation_issues=issues,
        verification=("CPU_CONFIGURATION_CONTRADICTED" if contradicted else
                      "CPU_CONFIGURATION_UNAVAILABLE" if issues else "CPU_CONFIGURATION_VERIFIED"))
    return diagnostic


def _copy_tensor(value):
    return value.detach().cpu().numpy().copy()


def make_recorder_configuration(RecorderTerm, RecorderTermCfg, RecorderManagerBaseCfg,
                                DatasetExportMode, session):
    """Use the public recorder term integration, with no second dataset writer."""
    class CaptureBoundaryRecorder(RecorderTerm):
        def record_post_physics_decimation_step(self):
            if session.armed:
                session.scheduled_substeps += 1
                session._boundary()
            return None, None

        def record_post_step(self):
            if session.armed:
                flags = session._flags(self._env.reset_terminated, self._env.reset_time_outs)
                session._lifecycle(deepcopy(flags))
                if any(flags.values()):
                    raise RuntimeError("capture_termination_before_automatic_reset")
            return None, None

        def record_pre_reset(self, env_ids):
            if (getattr(session, "terminal_policy", None) == CPU_EVIDENCE_FIRST_CLOSE
                    and session.initialization.get("initial_reset_returned")):
                session.observation_phase = "REAL_RESET"
            if session.callback_collector is not None:
                session.callback_collector.new_epoch("ordinary_recorder_pre_reset")
            if session.armed:
                session._pre_reset()
                raise RuntimeError("capture_unexpected_reset_rejected_before_reset")
            session.initialization["pre_reset_observations"] += 1
            return None, None

    cfg = RecorderManagerBaseCfg()
    cfg.dataset_export_mode = DatasetExportMode.EXPORT_NONE
    cfg.export_in_record_pre_reset = False
    cfg.export_in_close = False
    cfg.capture_boundary = RecorderTermCfg(class_type=CaptureBoundaryRecorder)
    return cfg


def effective_configuration(env, settings, *, stage0_mode=None):
    """Require native physics getters as well as configured CUDA/camera values."""
    physics = env.sim.get_physics_context()
    raw_has_gui = env.sim.has_gui()
    result = {
        "task_id": TASK_ID, "cfg_class": type(env.cfg).__name__,
        "num_envs": int(env.num_envs), "device": str(env.device),
        "simulation_device": str(env.sim.device), "physics_device": str(physics.device),
        "gpu_sim": physics.use_gpu_sim, "gpu_pipeline": physics.use_gpu_pipeline,
        "gpu_dynamics": physics.is_gpu_dynamics_enabled(),
        "broadphase": str(physics.get_broadphase_type()),
        "physics_dt_s": float(env.physics_dt), "control_dt_s": float(env.step_dt),
        "decimation": int(env.cfg.decimation), "render_interval": int(env.cfg.sim.render_interval),
        "action_dimension": int(env.action_manager.total_action_dim),
        "cameras_enabled": settings.get("/isaaclab/cameras_enabled"),
        "offscreen_render": settings.get("/isaaclab/render/offscreen"),
        "rtx_sensors": env.sim.has_rtx_sensors(),
        "fast_shutdown": settings.get("/app/fastShutdown"),
        # SimulationContext returns the raw settings OR; missing XR can yield None.
        "has_gui": bool(raw_has_gui), "has_gui_raw": raw_has_gui,
        "configured_seed": env.cfg.seed,
    }
    expected = {
        "cfg_class": "ChicgraspDynamicChickenEnvCfg", "num_envs": 1,
        "device": "cuda:0", "simulation_device": "cuda:0", "physics_device": "cuda:0",
        "gpu_sim": True, "gpu_pipeline": True, "gpu_dynamics": True, "broadphase": "GPU",
        "decimation": 2, "render_interval": 2, "action_dimension": 5,
        "cameras_enabled": True, "offscreen_render": stage0_mode != "TUNE_GUI", "rtx_sensors": True,
        "fast_shutdown": True, "has_gui": stage0_mode == "TUNE_GUI", "configured_seed": 42,
    }
    issues = [name for name, wanted in expected.items()
              if result[name] != wanted or (isinstance(wanted, bool) and result[name] is not wanted)]
    if not math.isclose(result["physics_dt_s"], 1 / 120, rel_tol=0, abs_tol=1e-12):
        issues.append("physics_dt_s")
    if not math.isclose(result["control_dt_s"], 1 / 60, rel_tol=0, abs_tol=1e-12):
        issues.append("control_dt_s")
    if issues:
        raise ValueError("effective_runtime_configuration_rejected:" + ",".join(issues))
    return result


def verify_existing_reporting(stage, coverage, PhysxSchema):
    """No post-start reporting mutation: the frozen robot spawn already enables it."""
    for actor in coverage["jaws"]:
        report = PhysxSchema.PhysxContactReportAPI(stage.GetPrimAtPath(actor))
        if not report:
            raise ValueError("existing_contact_report_API_missing:" + actor)
        threshold = report.GetThresholdAttr().Get()
        if isinstance(threshold, bool) or threshold != 0.0:
            raise ValueError("existing_contact_report_threshold_not_zero:" + actor)
        if report.GetReportPairsRel().GetTargets():
            raise ValueError("existing_contact_report_pair_restriction:" + actor)


class _NativeDependencies:
    """Imports are deliberately deferred until after the single AppLauncher."""
    def launch_app(self, *, stage0_mode=None):
        import os
        import sys
        # Keep the same explicit Windows DLL exposure used by the successful
        # installed-Python probe; retain handles for the application lifetime.
        self.dll_handles = [os.add_dll_directory(path) for path in
                            ("C:/isaacsim", "C:/isaacsim/kit", "C:/isaacsim/kit/kernel/plugins")]
        from isaaclab.app import AppLauncher
        original_argv, original_stdout, original_stderr = sys.argv, sys.stdout, sys.stderr
        configuration = deepcopy(APP_CONFIGURATION)
        if stage0_mode is not None:
            if stage0_mode not in ("TUNE_GUI", "CHECK_HEADLESS"):
                raise ValueError("unsupported_stage0_mode")
            configuration["headless"] = stage0_mode != "TUNE_GUI"
        self.launch_details = {"caller_argv": list(original_argv),
                               "application_argv": [original_argv[0], "--info"],
                               "application_configuration": configuration}
        # SimulationApp forwards sys.argv to Kit. Do not forward our capture
        # CLI paths/options as Kit settings. --info retains startup streams and
        # disables AppLauncher's temporary devnull redirection. Restoration also
        # covers a failed constructor before AppLauncher restores its streams.
        try:
            sys.argv = list(self.launch_details["application_argv"])
            self.launcher = AppLauncher(deepcopy(configuration))
            # Retain AppLauncher's resolved path, not only our requested profile.
            self.launch_details["resolved_experience"] = self.launcher._sim_experience_file
        finally:
            sys.argv, sys.stdout, sys.stderr = original_argv, original_stdout, original_stderr
        return self.launcher.app

    def create_callback_collector(self, session):
        # Called only after AppLauncher returned, before gym construction/reset.
        # Acquiring this interface does not require final task tensor views.
        from omni.physx import get_physx_simulation_interface
        from .callback_contact_collector import CallbackContactCollector
        return CallbackContactCollector(get_physx_simulation_interface())

    def construct_environment(self, session):
        import sys
        import gymnasium as gym
        import torch
        import carb.settings
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from isaaclab.managers import RecorderTerm, RecorderTermCfg
        from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, DatasetExportMode

        project_source = Path(__file__).resolve().parents[2] / "source" / "chicgrasp_isaac"
        if str(project_source) not in sys.path:
            sys.path.insert(0, str(project_source))
        import chicgrasp_isaac  # noqa: F401 - exact registered task

        self.torch, self.settings = torch, carb.settings.get_settings()
        cfg = load_cfg_from_registry(TASK_ID, "env_cfg_entry_point")
        if (type(cfg).__name__ != "ChicgraspDynamicChickenEnvCfg"
                or cfg.scene.num_envs != 1 or cfg.sim.device != "cuda:0"):
            raise ValueError("registered_dynamic_configuration_required_no_override")
        # Validate the original production task before the explicit diagnostic
        # device override below. No other physics/camera/controller override.
        cfg.seed = session.seed
        if getattr(session, "translated_point", None) is not None:
            from .translated_script import configure_start
            configure_start(cfg, session)
        if getattr(session, "r4_live_binding", None) is not None:
            from .r4_live_binding import configure_start
            configure_start(cfg, session)
        prior = getattr(cfg, "recorders", None)
        if prior is not None:
            settings = {"dataset_file_handler_class_type", "dataset_export_dir_path", "dataset_filename",
                        "dataset_export_mode", "export_in_record_pre_reset", "export_in_close"}
            if any(value is not None for key, value in vars(prior).items() if key not in settings):
                raise ValueError("existing_recorder_terms_must_not_be_replaced")
        cfg.recorders = make_recorder_configuration(
            RecorderTerm, RecorderTermCfg, RecorderManagerBaseCfg, DatasetExportMode, session)
        if session.observation_mode:
            from .attachment_observation_control import add_observation_control
            # New diagnostic primitives only, during normal scene construction.
            # The registered task and all production settings remain untouched.
            add_observation_control(cfg)
        if getattr(session, "backend_mode", PRODUCTION_GPU) == CPU_BACKEND_DIAGNOSTIC:
            # Installed Lab SimulationCfg.device selects CPU dynamics, MBP and
            # host readback. Keep the populated physics configuration and the
            # unchanged application graphics configuration; no manual toggles.
            cfg.sim.device = "cpu"
        if hasattr(session, "record_phase"):
            session.record_phase("environment_configuration", "available",
                qualification="CONFIGURED_NOT_EFFECTIVE", cfg_class=type(cfg).__name__,
                configured_simulation_device=cfg.sim.device, configured_seed=cfg.seed,
                configured_num_envs=cfg.scene.num_envs,
                requested_application_configuration=deepcopy(self.launch_details["application_configuration"]))
        if getattr(session, "terminal_policy", None) == CPU_EVIDENCE_FIRST_CLOSE:
            from isaaclab.managers import ObservationManager
            from chicgrasp_isaac.tasks.manager_based.chicgrasp_isaac.mdp.policy_interface import jaw_state_dimensions
            from .observation_metadata import cpu_observation_metadata
            # Config-local adapter, not an installed manager/global replacement.
            # The source-qualified sizing call alone may avoid reading jaw state.
            with cpu_observation_metadata(cfg, ObservationManager,
                    dimension_func=jaw_state_dimensions,
                    phase_provider=lambda: session.observation_phase,
                    sink=session.record_policy_guard):
                return gym.make(TASK_ID, cfg=cfg)
        if session.stage0_mode or getattr(session, "stage1_demo", False) or getattr(session, "stage3_learned", False) or getattr(session, "translated_point", None) is not None:
            from isaaclab.envs import manager_based_env
            from .stage0_light_dump import observe_construction, write_inventory
            # Observe scene completion before the existing constructor reset.
            # No setting/physics mutation; restore the imported symbol even if
            # construction or durable diagnostic persistence fails.
            with observe_construction(manager_based_env, lambda scene:
                    write_inventory(scene, session.output_dir, session.stage0_mode or "CHECK_HEADLESS")):
                return gym.make(TASK_ID, cfg=cfg)
        return gym.make(TASK_ID, cfg=cfg)

    def prepare_reporting(self, env, stage):
        from pxr import PhysxSchema
        from . import jaw_contact_diagnostic as jaw
        coverage = jaw.load_contract_map()
        verify_existing_reporting(stage, coverage, PhysxSchema)
        setup = jaw.prepare_reporting(stage, task_id=TASK_ID, num_envs=env.num_envs)
        if any(item.get("authored") for item in setup["properties"]):
            raise ValueError("unexpected_post_initialization_reporting_mutation")
        return setup

    def capture_scene_identity(self, session):
        from .managed_scene_identity import bind_managed_scene
        diagnostic = session.initialization["stage_identity_diagnostics"]
        diagnostic.setdefault("continuity", {"validations": 0, "passes": 0, "rejections": 0})
        context = dict(session.execution_context, task_id=TASK_ID,
            stage_source="env.sim.get_initial_stage() retained after environment construction, before initial gym reset")
        return bind_managed_scene(session.env, session.stage,
            diagnostics=dict(diagnostic, context=context, persist=session.guard_record_sink))

    def bind_measurements(self, session, setup):
        from .lift_event_pose_evidence import NativePoseReader
        from . import jaw_contact_diagnostic as jaw
        session.pose_reader = NativePoseReader(session.env, task_id=TASK_ID,
                                               scene_identity=session.scene_identity)
        diagnostic = session.initialization["stage_identity_diagnostics"]
        context = dict(session.execution_context, task_id=TASK_ID, lifecycle_phase="initialization",
            stage_source="env.sim.get_initial_stage() retained before initial gym_env.reset(seed=42)",
            candidate_started=session.initialization["candidate_started"],
            control_ticks=session.control_ticks, scheduled_substeps=session.scheduled_substeps,
            completed_policy_actions=0, attachment_authoring_attempts=0,
            owner_bound=session.owner is not None)
        reader = jaw.bind_runtime(session.env, session.stage, task_id=TASK_ID, setup=setup,
            diagnostics=dict(diagnostic, context=context, persist=session.guard_record_sink),
            scene_identity=session.scene_identity,
            **({"callback_collector": session.callback_collector}
               if session.callback_collector is not None else {}))
        # Preserve this handle even if diagnostic construction fails.
        session.contact_reader = reader
        if diagnostic["errors"]:
            raise ValueError("stage_identity_diagnostic_incomplete")
        session.coverage = deepcopy(reader.coverage)
        session.origin_clock = tuple(session.pose_reader.clock_identity())
        session.contact_collector = jaw.JawContactDiagnostic(
            reader, episode_id=session.episode_id,
            origin_step=session.origin_clock[0], origin_time=session.origin_clock[1],
            retain_samples=getattr(session, "contact_writer", None) is None)

    def bind_geometric_measurements(self, session):
        """Existing measured pose reader only; no contact interface or reporting gate."""
        from .lift_event_pose_evidence import NativePoseReader
        session.pose_reader = NativePoseReader(session.env, task_id=TASK_ID,
                                               scene_identity=session.scene_identity)
        session.origin_clock = tuple(session.pose_reader.clock_identity())

    def bind_observation(self, session):
        from .attachment_observation_control import bind_observation_control, observation_control_spec
        session.control_reader = bind_observation_control(session.env, session.stage, session.scene_identity)
        spec = observation_control_spec()
        session.contact_reader.configure_observation(spec["body_path"], spec["support_path"])
        session.observation_configuration = {
            "control": deepcopy(session.control_reader.configuration),
            "production_geometry": collision_description_context(session.stage, session.coverage),
            "native_drop_accounting": "NATIVE_DROP_ACCOUNTING_UNAVAILABLE",
            "cooked_surfaces": {"status": "UNAVAILABLE", "reason": "no existing qualified cooked-surface reader"},
            "scene_change": "isolated observation control added; not numerically identical to prior campaign"}
        if session.callback_collector is not None:
            session.observation_configuration["contact_transport"] = session.contact_transport

    def read_observation(self, session, clock):
        from .attachment_observation_control import read_articulation_observations
        return {"control": session.control_reader.read(),
                "articulation": read_articulation_observations(session.env, clock)}

    def flags(self, terminated, truncated):
        return {"terminated": bool(self.torch.any(terminated).item()),
                "truncated": bool(self.torch.any(truncated).item())}

    def action_tensor(self, action, device):
        return self.torch.as_tensor(action.reshape(1, 5), dtype=self.torch.float32, device=device)


class RuntimeSession:
    """Inert handle retained by the caller across partial initialization failures.

    Main owns evidence/writers and must save their complete state before close_app.
    No retry occurs here. Native owner binding uses the explicit capture policy in
    the runner, then assigns ``owner`` for ordered managed cleanup.
    """
    def __init__(self, output_dir, *, seed=42, episode_id=None, deps=None, observation_mode=False,
                 contact_transport="polling-contact/v1", backend_mode=PRODUCTION_GPU,
                 diagnostic_mode=None, terminal_policy=None, geometric_mode=False, stage0_mode=None):
        if type(seed) is not int or seed != 42:
            raise ValueError("declared_seed_42_required")
        self.output_dir, self.seed = Path(output_dir), seed
        if type(observation_mode) is not bool:
            raise TypeError("observation_mode_must_be_boolean")
        self.observation_mode = observation_mode
        if type(geometric_mode) is not bool or (geometric_mode and (
                observation_mode or backend_mode != PRODUCTION_GPU or diagnostic_mode is not None
                or terminal_policy is not None or contact_transport != "polling-contact/v1")):
            raise ValueError("geometric_mode_requires_unmodified_GPU_no_contact_or_control_profile")
        self.geometric_mode = geometric_mode
        if stage0_mode not in (None, "TUNE_GUI", "CHECK_HEADLESS") or (stage0_mode is not None and not geometric_mode):
            raise ValueError("stage0_requires_explicit_geometric_mode")
        self.stage0_mode = stage0_mode
        if contact_transport not in ("polling-contact/v1", "callback-contact/v1"):
            raise ValueError("unsupported_contact_transport")
        if contact_transport == "callback-contact/v1" and not observation_mode:
            raise ValueError("callback_contact_requires_observation_mode")
        if backend_mode not in (PRODUCTION_GPU, CPU_BACKEND_DIAGNOSTIC):
            raise ValueError("unsupported_backend_mode")
        if backend_mode == CPU_BACKEND_DIAGNOSTIC and (
                not observation_mode or contact_transport != "callback-contact/v1"):
            raise ValueError("cpu_backend_diagnostic_requires_callback_observation")
        self.backend_mode = backend_mode
        if diagnostic_mode not in (None, CPU_STARTUP_SHUTDOWN_ONLY):
            raise ValueError("unsupported_diagnostic_mode")
        if diagnostic_mode is not None and backend_mode != CPU_BACKEND_DIAGNOSTIC:
            raise ValueError("startup_only_requires_cpu_callback_observation")
        self.diagnostic_mode = diagnostic_mode
        if terminal_policy not in (None, CPU_EVIDENCE_FIRST_CLOSE):
            raise ValueError("unsupported_terminal_policy")
        if terminal_policy is not None and (backend_mode != CPU_BACKEND_DIAGNOSTIC or diagnostic_mode is not None):
            raise ValueError("cpu_terminal_policy_requires_cpu_campaign_not_startup_trial")
        self.terminal_policy = terminal_policy
        self.observation_phase = "CONSTRUCTION"
        self.policy_guard_sink = None
        self.policy_guard_sequence = 0
        self.policy_guard_errors = []
        self.terminal_scene_close_result = None
        self._terminal_scene_close_attempted = False
        self.startup_guard_records, self.startup_backend_records, self.startup_diagnostic_errors = [], [], []
        self.startup_guard_sink = None
        self.early_cpu_profile = {"schema": EARLY_CPU_PROFILE, "status": "NOT_REACHED",
            "reader_calls": 0, "capture_phase": "BEFORE_INITIAL_RESET", "reference": None}
        self.early_profile_sink = None
        self.early_profile_errors = []
        self.backend_diagnostic = (_unobserved_cpu_backend()
                                   if backend_mode == CPU_BACKEND_DIAGNOSTIC else None)
        self.contact_transport = contact_transport
        # Assigned by the runner before initialization. Never used by callbacks;
        # persistence occurs only at the existing post-step/terminal boundaries.
        self.contact_writer = None
        self.contact_storage_schema = None
        self.callback_collector = None
        self.callback_initialization_routing = None
        self._step_hook = self._step_hook_sim = None
        self._step_original = self._step_original_instance = None
        self._step_had_instance_attribute = False
        self._callback_runtime = {"events": [], "errors": [], "locally_omitted_metadata": 0,
            "intervals_started": 0, "intervals_completed": 0,
            "source": "existing env.sim.step(render=False) call; returns after core simulate/fetch_results; recorder follows before render",
            "native_timestamp_claim": False, "metadata_limit": 256}
        self.control_reader = None
        self.observation_configuration = None
        self.pre_candidate_record = None
        self.episode_id = episode_id or ("attachment_capture_" + uuid.uuid4().hex)
        self.deps = deps
        self.app = self.gym_env = self.env = self.stage = None
        self.scene_identity = None
        self.owner = self.pose_reader = self.contact_reader = self.contact_collector = None
        self.origin_clock, self.coverage, self.effective_config = None, None, None
        self.armed, self.scheduled_substeps, self.control_ticks = False, 0, 0
        self._boundary = self._lifecycle = self._pre_reset = None
        self._initialize_attempted = self._scene_close_attempted = self._app_close_attempted = False
        self.scene_close_result = None
        self.execution_context, self.guard_record_sink = {}, None
        self.phase_record_sink = None
        self.initialization = {"application_attempts": 0, "application_returned": False,
                               "environment_returned": False, "initial_reset_returned": False,
                               "pre_reset_observations": 0, "candidate_started": False,
                               "stage_identity_diagnostics": {"records": [], "errors": []}}

    @staticmethod
    def failure_details(exc):
        """Copy only the operands already carried by the exception; no state reads."""
        details = {"error": type(exc).__name__ + ":" + str(exc)}
        for name in ("jaw_state_diagnostic", "jaw_state_diagnostic_error"):
            try:
                value = getattr(exc, name, None)
                if value is not None:
                    details[name] = json.loads(json.dumps(_json(value), allow_nan=False))
            except Exception as error:
                details["jaw_state_diagnostic_error"] = type(error).__name__ + ":" + str(error)
        return details

    def record_policy_guard(self, record):
        """Persist one copied phase record, never retain all real frames in RAM."""
        try:
            if self.policy_guard_sink is None:
                raise RuntimeError("cpu_policy_guard_sink_required")
            copied = json.loads(json.dumps(_json(record), allow_nan=False))
            copied.update(sequence=self.policy_guard_sequence, episode_id=self.episode_id)
            self.policy_guard_sink(copied)
            self.policy_guard_sequence += 1
        except Exception as exc:
            if len(self.policy_guard_errors) < 16:
                self.policy_guard_errors.append(type(exc).__name__ + ":" + str(exc))
            raise

    def record_phase(self, phase, event, **details):
        """Best-effort evidence at existing Python boundaries, never a native guard."""
        try:
            if event == "failed" and RuntimeSession._early_cpu_profile_selected(self):
                details["early_cpu_profile"] = deepcopy(self.early_cpu_profile)
            record = json.loads(json.dumps(_json(dict(phase=phase, event=event,
                monotonic_ns=time.monotonic_ns(), **details)), allow_nan=False))
            self.initialization.setdefault("phase_events", []).append(record)
            if self.phase_record_sink is not None:
                self.phase_record_sink(deepcopy(record))
        except Exception as exc:
            self.initialization.setdefault("phase_evidence_errors", []).append(
                type(exc).__name__ + ":" + str(exc))

    @property
    def effective_configuration(self):
        return deepcopy(self.effective_config)

    def _early_cpu_profile_selected(self):
        return (getattr(self, "backend_mode", None) == CPU_BACKEND_DIAGNOSTIC
                and getattr(self, "observation_mode", False)
                and getattr(self, "diagnostic_mode", None) is None)

    def capture_early_cpu_profile(self):
        """One existing partial read, persisted before reset; never qualifies CPU."""
        if not self._early_cpu_profile_selected():
            return
        if self.early_cpu_profile["reader_calls"] != 0:
            raise RuntimeError("early_cpu_profile_read_not_repeatable")
        from .startup_backend_diagnostics import observe_existing_backend
        self.early_cpu_profile["reader_calls"] = 1
        self.record_phase("early_cpu_profile", "requested", capture_phase="BEFORE_INITIAL_RESET")
        started = time.monotonic_ns()
        snapshot, read_error = None, None
        try:
            snapshot = observe_existing_backend(self.env, phase="BEFORE_INITIAL_RESET")
        except Exception as exc:
            read_error = {"type": type(exc).__name__, "message": str(exc)}
        returned = time.monotonic_ns()
        status = "READ_UNAVAILABLE" if read_error is not None else "CAPTURED"
        self.early_cpu_profile["status"] = status
        try:
            # Strict JSON copying preserves the reader's values and timestamps;
            # serialization and disk errors are required-evidence failures.
            envelope = json.loads(json.dumps({"schema": EARLY_CPU_PROFILE,
                "episode_id": self.episode_id, "capture_phase": "BEFORE_INITIAL_RESET",
                "reader_calls": 1, "status": status, "capture_started_ns": started,
                "capture_returned_ns": returned, "snapshot": snapshot,
                "read_error": read_error}, allow_nan=False))
            if not callable(self.early_profile_sink):
                raise RuntimeError("early_cpu_profile_sink_required")
            reference = self.early_profile_sink(envelope)
            if not isinstance(reference, dict):
                raise TypeError("early_cpu_profile_persisted_reference_required")
            self.early_cpu_profile["reference"] = json.loads(json.dumps(reference, allow_nan=False))
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            self.early_cpu_profile.update(status="PERSISTENCE_FAILED", reference=None, error=error)
            self.early_profile_errors.append(deepcopy(error))
            self.record_phase("early_cpu_profile", "persistence_failed",
                              early_cpu_profile=deepcopy(self.early_cpu_profile))
        else:
            self.record_phase("early_cpu_profile", "persisted",
                              early_cpu_profile=deepcopy(self.early_cpu_profile))

    def initialize(self):
        if self.terminal_policy == CPU_EVIDENCE_FIRST_CLOSE and not callable(self.policy_guard_sink):
            raise RuntimeError("cpu_policy_guard_sink_required_before_initialization")
        if self._initialize_attempted:
            raise RuntimeError("runtime_initialization_not_repeatable")
        self._initialize_attempted = True
        self.deps = self.deps if self.deps is not None else _NativeDependencies()
        self.initialization["start_wall_monotonic_ns"] = time.monotonic_ns()
        self.initialization["application_attempts"] += 1
        self.record_phase("application_launch", "requested")
        try:
            self.app = (self.deps.launch_app(stage0_mode=self.stage0_mode)
                        if self.stage0_mode else self.deps.launch_app())
        except BaseException as exc:
            self.record_phase("application_launch", "failed", **self.failure_details(exc))
            raise
        finally:
            self.initialization["launch"] = deepcopy(getattr(self.deps, "launch_details", None))
        self.initialization["application_returned"] = True
        self.record_phase("application_launch", "returned", app_handle_available=self.app is not None)
        if self.contact_transport == "callback-contact/v1":
            self._callback_note("subscription_setup_requested_before_environment_construction")
            # Retain the object before starting: even setup failure has an owner
            # whose observed partial lifecycle can be exported and closed.
            self.callback_collector = self.deps.create_callback_collector(self)
            self.callback_collector.start()
            self._callback_note("subscription_setup_returned_before_environment_construction")
        self.record_phase("environment_construction", "requested")
        try:
            if self.diagnostic_mode == CPU_STARTUP_SHUTDOWN_ONLY:
                from .jaw_state_diagnostics import startup_guard_observation
                def receive(record):
                    self.startup_guard_records.append(deepcopy(record))
                    if self.startup_guard_sink is not None:
                        self.startup_guard_sink(deepcopy(record))
                with startup_guard_observation(receive, {
                        "diagnostic_mode": self.diagnostic_mode, "episode_id": self.episode_id,
                        "construction_generation": 1,
                        "subscription_generation": 1 if self.callback_collector is not None else None,
                        "constructor_internal_resets": "SOURCE_DEFINED_NOT_COUNTED",
                        "later_harness_resets": 0}, self.startup_diagnostic_errors.append) as observations:
                    try:
                        self.gym_env = self.deps.construct_environment(self)
                    finally:
                        for error in observations["errors"]:
                            if error not in self.startup_diagnostic_errors:
                                self.startup_diagnostic_errors.append(deepcopy(error))
            else:
                self.gym_env = self.deps.construct_environment(self)
        except BaseException as exc:
            self.record_phase("environment_construction", "failed", **self.failure_details(exc))
            raise
        self.record_phase("environment_construction", "returned",
                          environment_handle_available=self.gym_env is not None)
        self.env = self.gym_env.unwrapped
        if self.diagnostic_mode == CPU_STARTUP_SHUTDOWN_ONLY:
            self.initialization["environment_returned"] = True
            from .jaw_state_diagnostics import observe_existing_backend
            try:
                self.startup_backend_records.append(observe_existing_backend(self.env, phase="CONSTRUCTOR_RETURNED"))
            except Exception as exc:
                self.startup_diagnostic_errors.append({"phase": "returned_backend_recording",
                    "error": type(exc).__name__ + ":" + str(exc)})
            self.initialization["end_wall_monotonic_ns"] = time.monotonic_ns()
            return self  # No later harness reset, scene binding, measurement or candidate preparation.
        self.stage = self.env.sim.get_initial_stage()
        self.initialization["environment_returned"] = True
        # Native dependencies always supply this construction anchor. Older
        # explicitly substituted offline dependencies retain their own route.
        if hasattr(self.deps, "capture_scene_identity"):
            self.scene_identity = self.deps.capture_scene_identity(self)
        self.capture_early_cpu_profile()
        self.record_phase("initial_reset", "requested")
        self.observation_phase = "INITIAL_RESET"
        try:
            self.gym_env.reset(seed=self.seed)
        except BaseException as exc:
            self.record_phase("initial_reset", "failed", **self.failure_details(exc))
            raise
        self.initialization["initial_reset_returned"] = True
        self.record_phase("initial_reset", "returned")
        if self._early_cpu_profile_selected() and self.early_cpu_profile["status"] == "PERSISTENCE_FAILED":
            raise RuntimeError("required_early_cpu_profile_evidence_failed")
        self.observation_phase = "PRE_CANDIDATE"
        if self.scene_identity is not None:
            self.scene_identity.validate("after_initial_reset", emit=True)
        if self.callback_collector is not None:
            if self.scene_identity is None:
                raise ValueError("callback_contact_requires_managed_scene_anchor")
            self.callback_collector.associate(self.scene_identity.stage_id,
                                             deepcopy(self.scene_identity._anchor))
            self._callback_note("managed_association_after_initial_reset")
        if self.backend_mode == CPU_BACKEND_DIAGNOSTIC:
            self.backend_diagnostic = _observe_cpu_backend(
                self.env, self.deps.settings, getattr(self.scene_identity, "_anchor", None))
            self.backend_diagnostic["effective_configuration"]["contact_transport"] = self.contact_transport
            self.effective_config = deepcopy(self.backend_diagnostic["effective_configuration"])
            if self.backend_diagnostic["verification"] != "CPU_CONFIGURATION_VERIFIED":
                raise ValueError("effective_cpu_configuration_rejected:" +
                                 ",".join(self.backend_diagnostic["validation_issues"]))
        else:
            self.effective_config = effective_configuration(self.env, self.deps.settings,
                                                           stage0_mode=self.stage0_mode)
        if self.callback_collector is not None:
            self.effective_config["contact_transport"] = self.contact_transport
        if self.geometric_mode:
            self.deps.bind_geometric_measurements(self)
            self.initialization["end_wall_monotonic_ns"] = time.monotonic_ns()
            self.initialization["candidate_origin_clock"] = list(self.origin_clock)
            return self
        setup = self.deps.prepare_reporting(self.env, self.stage)
        self.initialization["reporting_setup"] = deepcopy(setup)
        if self.callback_collector is not None:
            self._callback_note("existing_report_settings_verified_after_initial_reset",
                                setup_property_count=len(setup.get("properties", [])))
        self.deps.bind_measurements(self, setup)
        if self.observation_mode:
            self.deps.bind_observation(self)
            if self.backend_mode == CPU_BACKEND_DIAGNOSTIC:
                self.observation_configuration["backend_mode"] = self.backend_mode
        if self.callback_collector is not None:
            self.callback_initialization_routing = self.contact_reader.decode_initialization_evidence(
                self.callback_collector.export_lifecycle())
            self._install_callback_step_hook()
        self.initialization["end_wall_monotonic_ns"] = time.monotonic_ns()
        self.initialization["candidate_origin_clock"] = list(self.origin_clock)
        return self

    def set_hooks(self, boundary, lifecycle, pre_reset):
        if not self.initialization["initial_reset_returned"] or self.pose_reader is None or self.armed:
            raise RuntimeError("hooks_require_initialized_unarmed_session")
        if not all(callable(item) for item in (boundary, lifecycle, pre_reset)):
            raise TypeError("all_capture_hooks_required")
        self._boundary, self._lifecycle, self._pre_reset = boundary, lifecycle, pre_reset
        if self.callback_collector is not None:
            self.callback_collector.new_epoch("candidate_hooks_armed_after_initialization")
            self._callback_note("candidate_hooks_armed", clock=list(self.clock()))
        self.armed = True

    def _callback_note(self, event, **fields):
        """Small setup/failure records only; per-step records live in the collector."""
        record = dict(event=event, monotonic_ns=time.monotonic_ns(), **fields)
        target = self._callback_runtime["events"]
        if len(target) < self._callback_runtime["metadata_limit"]:
            target.append(_json(record))
        else:
            self._callback_runtime["locally_omitted_metadata"] += 1

    def _callback_error(self, operation, exc):
        item = {"operation": operation[:512], "error": (type(exc).__name__ + ":" + str(exc))[:512]}
        errors = self._callback_runtime["errors"]
        if len(errors) < self._callback_runtime["metadata_limit"]:
            errors.append(item)
        else:
            self._callback_runtime["locally_omitted_metadata"] += 1

    def _install_callback_step_hook(self):
        """Bookkeep the existing Python call; never call simulate/fetch ourselves.

        Installed ManagerBasedRLEnv calls sim.step(render=False), then the
        recorder, then render/scene.update. The retained Lab method calls core
        step, whose physics-only path calls simulate and blocking fetch_results.
        This local interval label is not a native callback substep timestamp.
        """
        sim = self.env.sim
        original = sim.step
        if not callable(original) or self._step_hook is not None:
            raise RuntimeError("callback_step_hook_requires_original_once")
        self._step_had_instance_attribute = "step" in vars(sim)
        self._step_original_instance = vars(sim).get("step")
        self._step_original = original

        def observed_step(*args, **kwargs):
            if not self.armed:
                return original(*args, **kwargs)
            explicit_no_render = (not args and set(kwargs) == {"render"}
                                  and kwargs["render"] is False)
            label = {"control_tick": self.control_ticks,
                     "scheduled_substeps_before": self.scheduled_substeps,
                     "source": "existing_env_sim_step_call",
                     "explicit_render_false": explicit_no_render}
            try:
                self.callback_collector.begin_interval(self.clock(), label,
                    phase="candidate" if explicit_no_render else "unassigned")
                self._callback_runtime["intervals_started"] += 1
            except BaseException as exc:
                self._callback_error("begin_existing_step_interval", exc)
                raise
            try:
                result = original(*args, **kwargs)
            except BaseException:
                # Do not replace an original simulation exception with a
                # bookkeeping exception. Both remain independently visible.
                try:
                    self.callback_collector.complete_interval(self.clock(), success=False)
                    self._callback_runtime["intervals_completed"] += 1
                except BaseException as exc:
                    self._callback_error("complete_failed_existing_step_interval", exc)
                raise
            try:
                self.callback_collector.complete_interval(self.clock(), success=explicit_no_render)
                self._callback_runtime["intervals_completed"] += 1
            except BaseException as exc:
                self._callback_error("complete_existing_step_interval", exc)
                raise
            return result

        sim.step = observed_step
        self._step_hook, self._step_hook_sim = observed_step, sim
        if sim.step is not observed_step:
            raise RuntimeError("callback_step_hook_assignment_not_retained")
        self._callback_note("existing_step_hook_installed_after_reset_and_binding")

    def _restore_callback_step_hook(self):
        if self._step_hook is None:
            return
        sim = self._step_hook_sim
        if sim.step is not self._step_hook:
            raise RuntimeError("callback_step_hook_replaced_no_foreign_overwrite")
        if self._step_had_instance_attribute:
            sim.step = self._step_original_instance
        else:
            delattr(sim, "step")
        self._step_hook = self._step_hook_sim = None
        self._callback_note("existing_step_hook_restored")

    def export_contact_transport(self, scope="before_cleanup"):
        """Copied setup/initialization evidence, including partial setup failures."""
        exported = deepcopy(_json({"transport": self.contact_transport,
            "collector": (self.callback_collector.export_lifecycle()
                          if self.callback_collector is not None else None),
            "initialization_routing": self.callback_initialization_routing,
            "runtime": self._callback_runtime}))
        writer = getattr(self, "contact_writer", None)
        if writer is None:
            return exported
        from .compact_contact_replay import pack_transport
        from .compact_contact_storage import SCHEMA
        packed = pack_transport(exported, self.coverage,
            getattr(self.contact_reader, "observation_control", None), writer=writer)
        return {"schema": SCHEMA, "scope": scope,
                "reference": writer.write_lifecycle(scope, packed)}

    def _flags(self, terminated, truncated):
        return self.deps.flags(terminated, truncated)

    def step(self, action):
        if not self.armed or self._scene_close_attempted:
            raise RuntimeError("step_requires_armed_live_capture")
        action = np.asarray(action, dtype=np.float32)
        policy_api.validate_policy_action(action)
        if not np.all((action[3:] == 0) | (action[3:] == 1)):
            raise ValueError("binary_jaw_commands_required")
        previous = self.scheduled_substeps
        self.initialization["candidate_started"] = True
        if getattr(self, "terminal_policy", None) == CPU_EVIDENCE_FIRST_CLOSE:
            self.observation_phase = "CANDIDATE"
        result = self.gym_env.step(self.deps.action_tensor(action, self.env.device))
        flags = self._flags(result[2], result[3])
        if any(flags.values()):
            raise RuntimeError("unexpected_terminal_return_no_continuation")
        if self.scheduled_substeps != previous + 2:
            raise RuntimeError("exactly_two_observed_scheduled_substeps_required")
        self.control_ticks += 1
        return flags

    def observation(self):
        if getattr(self, "terminal_policy", None) == CPU_EVIDENCE_FIRST_CLOSE:
            self.observation_phase = "CANDIDATE" if self.initialization["candidate_started"] else "PRE_CANDIDATE"
        raw = self.env.observation_manager.compute_group("policy", update_history=False)
        if not isinstance(raw, dict) or tuple(raw) != tuple(policy_api.POLICY_OBSERVATION_KEYS):
            raise RuntimeError("runtime_policy_observation_keys_changed")
        mapping = {key: _copy_tensor(raw[key][0]) for key in policy_api.POLICY_OBSERVATION_KEYS}
        return policy_api.validate_policy_observation(mapping)

    observe_policy = observation

    def commanded_jaw_metadata(self, observed_bits):
        """Copy the accepted action term state without issuing or predicting an action."""
        before = self.clock()
        term = self.env.action_manager.get_term("policy_action")
        bits = _copy_tensor(term.commanded_jaw_state)
        if (bits.shape != (1, 2) or bits.dtype != np.dtype(np.uint8)
                or not np.all((bits == 0) | (bits == 1))):
            raise ValueError("invalid_commanded_jaw_state")
        if not np.array_equal(bits[0], observed_bits):
            raise ValueError("policy_jaw_state_disagrees_with_accepted_command")
        if self.clock() != before:
            raise ValueError("commanded_jaw_acquisition_clock_changed")
        return dict(semantics=policy_api.JAW_STATE_SEMANTICS, bits=bits[0].tolist(),
            dtype="uint8", order=["LEFT", "RIGHT"], acquisition_native_clock=list(before),
            source="env.action_manager.get_term('policy_action').commanded_jaw_state",
            command_issue_time_s=None, command_issue_time_status="NOT_PERSISTED")

    def camera_frames(self):
        return {name: int(self.env.scene[name].frame[0].item())
                for name in ("camera_left", "camera_right")}

    def camera_metadata(self):
        """Read provenance of the existing acquisition, never request another image.

        Installed SensorBase.update advances _timestamp; its buffer update sets
        _timestamp_last_update. These seconds are since sensor reset, NOT native
        simulation time. Camera.frame increments with that sensor update.
        """
        result = {}
        for name in ("camera_left", "camera_right"):
            sensor = self.env.scene[name]
            if len(sensor._view.prims) != 1:
                raise ValueError("single_resolved_camera_prim_required:" + name)
            item = {"prim_path": str(sensor._view.prims[0].GetPath()),
                    "configured_prim_expression": sensor.cfg.prim_path,
                    "identity_source": "Camera._view.prims[0].GetPath()",
                    "source": "existing policy RGB acquisition",
                    "sensor_time_s": None, "sensor_time_status": "UNAVAILABLE",
                    "sensor_time_source": "isaaclab.sensors.sensor_base.SensorBase._timestamp_last_update",
                    "sensor_time_domain": "seconds_since_sensor_reset", "frame_source": "Camera.frame",
                    "frame": int(sensor.frame[0].item())}
            try:
                value = float(sensor._timestamp_last_update[0].item())
                if not math.isfinite(value) or value < 0:
                    raise ValueError("invalid_sensor_timestamp")
                item.update(sensor_time_s=value, sensor_time_status="OBSERVED")
            except Exception as exc:
                item["sensor_time_unavailable_reason"] = type(exc).__name__ + ":" + str(exc)
            result[name] = item
        return result

    def read_observation(self):
        """Only additional reads at existing boundaries; never a grasp gate."""
        if not self.observation_mode:
            raise RuntimeError("observation_mode_required")
        before = self.clock()
        result = {"clock_before": list(before), "clock_after": None, "issues": []}
        try:
            result.update(self.deps.read_observation(self, before))
        except Exception as exc:
            result["issues"].append("observation_read_unavailable:" + type(exc).__name__ + ":" + str(exc))
        result["clock_after"] = list(self.clock())
        if tuple(result["clock_after"]) != before:
            result["issues"].append("observation_clock_changed")
        return deepcopy(_json(result))

    def pre_candidate_observation(self, pose):
        geometric = getattr(self, "geometric_mode", False)
        if (not (self.observation_mode or geometric) or self.pre_candidate_record is not None
                or not self.initialization["initial_reset_returned"]
                or self.initialization["candidate_started"]):
            raise RuntimeError("pre_candidate_requires_normal_reset_before_first_command_once")
        record = {"kind": "PRE_CANDIDATE", "episode_id": self.episode_id,
                  "action_frame": None, "control_tick": None, "substep": None,
                  "indices_status": "NOT_APPLICABLE_NO_COMMAND_EXECUTED", "native_clock": list(self.clock()),
                  "wall_monotonic_ns": time.monotonic_ns(), "pose": deepcopy(pose),
                  **({"jaws": self.read_jaws()} if geometric else {"observation": self.read_observation()}),
                  "reset_provenance": {"normal_initial_reset_returned": True,
                       "pre_reset_observations": self.initialization["pre_reset_observations"],
                       "candidate_started": False, "settling_established": False},
                  "managed_stage": deepcopy(self.scene_identity._anchor) if self.scene_identity is not None else None}
        self.pre_candidate_record = deepcopy(_json(record))
        return deepcopy(self.pre_candidate_record)

    def read_jaws(self):
        """Four native prismatic positions, no velocity/force/contact acquisition."""
        before = self.clock()
        robot = self.env.scene["robot"]
        names = list(robot.joint_names)
        wanted = [f"PrismaticJoint{j}" for j in range(1, 5)]
        if any(names.count(name) != 1 for name in wanted):
            raise ValueError("four_unique_measured_jaw_names_required")
        positions = _copy_tensor(robot.root_physx_view.get_dof_positions())
        if positions.shape != (1, len(names)) or not np.isfinite(positions).all():
            raise ValueError("invalid_native_jaw_positions")
        after = self.clock()
        if after != before:
            raise ValueError("jaw_clock_changed")
        return {"positions_m": positions[0, [names.index(n) for n in wanted]].tolist(),
                "names": wanted, "clock_before": list(before), "clock_after": list(after),
                "units": "m", "source": "robot.root_physx_view.get_dof_positions"}

    def clock(self):
        return tuple(self.pose_reader.clock_identity())

    def measure(self):
        return self.pose_reader.read()

    def close_scene(self, context=None, *, defer_terminal=False):
        if defer_terminal and getattr(self, "terminal_policy", None) != CPU_EVIDENCE_FIRST_CLOSE:
            raise ValueError("deferred_terminal_close_requires_explicit_cpu_policy")
        if self._scene_close_attempted:
            return deepcopy(self.scene_close_result)
        self._scene_close_attempted = True
        self.armed = False
        result = {"attempts": [], "errors": [], "owner": None, "managed_environment_closed": False}
        if defer_terminal:
            result["terminal_environment_deferred"] = self.gym_env is not None
        self.scene_close_result = result
        operations = []
        if self.scene_identity is not None:
            operations.append(("managed_identity_pre_cleanup",
                               lambda: self.scene_identity.validate("pre_cleanup", emit=True)))
        if self.owner is not None:
            operations.append(("owner", lambda: self.owner.close(context)))
        if self.contact_collector is not None:
            operations.append(("contact_collector", self.contact_collector.close))
        elif self.contact_reader is not None:
            operations.append(("contact_reader", self.contact_reader.close))
        if self._step_hook is not None:
            operations.append(("callback_step_hook", self._restore_callback_step_hook))
        if self.callback_collector is not None:
            operations.append(("callback_subscription", self.callback_collector.close))
        if self.env is not None and hasattr(self.env, "recorder_manager"):
            operations.append(("recorder_manager", self.env.recorder_manager.close))
        if self.gym_env is not None and not defer_terminal:
            operations.append(("environment", self.gym_env.close))
        for name, operation in operations:
            result["attempts"].append(name)
            self.record_phase("cleanup", "requested", operation=name)
            try:
                value = operation()
                if name == "owner":
                    result["owner"] = value
                if name == "environment":
                    result["managed_environment_closed"] = self.env._is_closed is True
                    if not result["managed_environment_closed"]:
                        raise RuntimeError("managed_environment_close_not_confirmed")
                self.record_phase("cleanup", "returned", operation=name)
            except BaseException as exc:
                result["errors"].append({"operation": name, "error": type(exc).__name__ + ":" + str(exc)})
                self.record_phase("cleanup", "failed", operation=name, **self.failure_details(exc))
        return deepcopy(result)

    def close_terminal_scene(self):
        """CPU policy only: unchanged environment close after evidence is finalized.

        Owned-joint, subscription and recorder cleanup already ran in close_scene.
        This separate result cannot mutate its finalized pre-terminal snapshot.
        """
        if self.terminal_policy != CPU_EVIDENCE_FIRST_CLOSE:
            raise ValueError("terminal_scene_close_requires_explicit_cpu_policy")
        if not self._scene_close_attempted or self._terminal_scene_close_attempted:
            raise RuntimeError("terminal_scene_close_requires_prior_cleanup_once")
        if "terminal_environment_deferred" not in self.scene_close_result:
            raise RuntimeError("terminal_scene_close_requires_deferred_cleanup")
        self._terminal_scene_close_attempted = True
        result = {"environment_handle_available": self.gym_env is not None,
                  "managed_environment_closed": False, "errors": []}
        self.terminal_scene_close_result = result
        if self.gym_env is not None:
            self.record_phase("terminal_environment_close", "requested", environment_handle_available=True)
            try:
                self.gym_env.close()
                result["managed_environment_closed"] = self.env._is_closed is True
                if not result["managed_environment_closed"]:
                    raise RuntimeError("managed_environment_close_not_confirmed")
                self.record_phase("terminal_environment_close", "returned", managed_environment_closed=True)
            except BaseException as exc:
                result["errors"].append(type(exc).__name__ + ":" + str(exc))
                self.record_phase("terminal_environment_close", "failed", **self.failure_details(exc))
        return deepcopy(result)

    def close_app(self):
        if self._app_close_attempted:
            raise RuntimeError("application_close_not_repeatable")
        self._app_close_attempted = True
        if self.app is not None:
            self.record_phase("application_close", "requested", app_handle_available=True,
                              wait_for_replicator=True, skip_cleanup=False)
            # The caller already completed all required file writes. Retain
            # Replicator completion as well; this can terminate without return.
            try:
                self.app.close(wait_for_replicator=True, skip_cleanup=False)
            except Exception as exc:
                self.record_phase("application_close", "failed", **self.failure_details(exc))
                raise
            else:
                if self.diagnostic_mode == CPU_STARTUP_SHUTDOWN_ONLY:
                    self.record_phase("CLOSE_RETURNED", "returned")
                self.record_phase("application_close", "returned")


def collision_description_context(stage, coverage):
    """Authored descriptions only; no mesh vertices, cooking or runtime writes."""
    result = {"representation": "authored_USD_not_native_cooked_surface", "colliders": [], "issues": []}
    paths = set()
    for jaw in coverage.get("jaws", {}).values():
        paths.update(jaw.get("enabled_colliders", []))
        paths.update(jaw.get("allowed_chicken_colliders", []))
        paths.update(jaw.get("forbidden_chicken_colliders", []))
    for path in sorted(paths):
        item = {"path": path, "attributes": {}, "issues": []}
        try:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise ValueError("collision_prim_unavailable")
            item["shape_type"] = str(prim.GetTypeName())
            item["parent_path"] = str(prim.GetParent().GetPath())
            for name in ("physics:collisionEnabled", "physics:approximation", "physxCollision:contactOffset",
                         "physxCollision:restOffset", "size", "radius", "height", "axis", "extent", "xformOpOrder"):
                attr = prim.GetAttribute(name)
                if attr and attr.HasAuthoredValueOpinion():
                    value = attr.Get()
                    try:
                        value = value if isinstance(value, (str, bool, int, float)) else list(value)
                    except TypeError:
                        value = str(value)
                    item["attributes"][name] = _json(value)
            # Local transform stack, including parents. These are descriptions,
            # not measured/cooked geometry; retain exact USD text/order.
            item["authored_transform_chain"] = []
            ancestor = prim
            while ancestor.IsValid() and str(ancestor.GetPath()) != "/":
                item["authored_transform_chain"].append({"path": str(ancestor.GetPath()),
                    "attributes": [{"name": str(a.GetName()), "value": str(a.Get())}
                        for a in ancestor.GetAttributes()
                        if str(a.GetName()).startswith("xformOp:") or str(a.GetName()) == "xformOpOrder"]})
                ancestor = ancestor.GetParent()
        except Exception as exc:
            item["issues"].append(type(exc).__name__ + ":" + str(exc))
            result["issues"].append("collider_description_unavailable:" + path)
        result["colliders"].append(item)
    return result
