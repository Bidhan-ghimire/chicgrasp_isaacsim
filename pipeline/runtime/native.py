"""New research runs using the frozen native scheduler; imports are inert.

This is not the historical HC dispatcher or its acceptance checker. Only run()
crosses the explicit native boundary. Frozen source files are never rewritten.
"""
from copy import deepcopy
import ast
import functools
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import textwrap
import time

from pipeline.runtime.host import FROZEN, ROOT, SOURCE_COMMIT, load_json, preflight, sha256, write_json
import pipeline.runtime.host as host

REQUEST_SCHEMA = "research-native-request/v1"
RESULT_SCHEMA = "research-run/v1"


def load_host_config(path):
    path = Path(path).resolve(strict=True)
    config = load_json(path)
    for key in ("isaac_sim", "isaac_lab", "policy_support"):
        raw = config.get(key)
        if not isinstance(raw, str) or not raw:
            raise ValueError("missing_host_path:" + key)
        import os
        target = Path(os.path.expandvars(raw)).expanduser()
        config[key] = str((path.parent / target).resolve() if not target.is_absolute() else target.resolve())
    return config


def _finite_vector(value, size, name):
    if not isinstance(value, list) or len(value) != size or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError("invalid_" + name)
    return [float(x) for x in value]


def validate_request(value, directory):
    value = deepcopy(value)
    if value.get("schema") != REQUEST_SCHEMA:
        raise ValueError("research_request_schema")
    if value.get("mode") not in ("scene", "scripted", "preview", "live"):
        raise ValueError("research_mode")
    if value.get("scene_profile") != "attempt005":
        raise ValueError("only_attempt005_supported; alternative scenes require a separate dependency adapter")
    if type(value.get("environment_seed")) is not int or value["environment_seed"] != 42:
        raise ValueError("only_environment_seed_42_supported")
    position = value.get("position", {})
    if not isinstance(position.get("id"), str) or not position["id"]:
        raise ValueError("position_id_required")
    position["authored_xyz_m"] = _finite_vector(position.get("authored_xyz_m"), 3, "position")
    position["authored_quaternion_wxyz"] = _finite_vector(
        position.get("authored_quaternion_wxyz", [1, 0, 0, 0]), 4, "quaternion")
    if not math.isclose(sum(x*x for x in position["authored_quaternion_wxyz"]), 1., rel_tol=0, abs_tol=1e-6):
        raise ValueError("position_quaternion_must_be_unit_wxyz")
    from pipeline.scene import authenticate_scene_setup
    module, identity = authenticate_scene_setup(value.get("scene_setup_identity"))
    module.validate_position(deepcopy(position), deepcopy(identity["options"]))
    value["scene_setup_identity"] = identity
    for key in (("script", "checkpoint") if value["mode"] == "live" else
                ("script",) if value["mode"] in ("scripted", "preview") else ()):
        record = value.get(key, {})
        if not isinstance(record.get("path"), str) or not record["path"]:
            raise ValueError(key + "_path_required")
        digest = record.get("sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(key + "_sha256_required")
        target = Path(record["path"])
        record["path"] = str((Path(directory) / target).resolve() if not target.is_absolute() else target.resolve())
        if "bytes" in record and (type(record["bytes"]) is not int or record["bytes"] < 1):
            raise ValueError(key + "_byte_count")
    if value.get("evaluation_seed") is None and value["mode"] != "live":
        value["evaluation_seed"] = 15345  # Unused in scene/scripted execution.
    seed = value.setdefault("evaluation_seed", 15345)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("evaluation_seed_must_be_uint32")
    seconds = value.setdefault("review_seconds", 600)
    if type(seconds) not in (int, float) or not 0 < seconds <= 1100:
        raise ValueError("review_seconds_must_be_positive_at_most_1100")
    if "deadline_unix" in value and (type(value["deadline_unix"]) not in (int, float)
                                     or not math.isfinite(value["deadline_unix"])):
        raise ValueError("invalid_parent_deadline")
    video = value.setdefault("video", {"enabled": False})
    if not isinstance(video, dict) or type(video.get("enabled")) is not bool:
        raise ValueError("explicit_video_enabled_boolean_required")
    if video["enabled"] and value["mode"] == "scene":
        raise ValueError("scene_review_does_not_record_a_video_take")
    if video["enabled"] and value["mode"] == "preview":
        raise ValueError("scripted_preview_does_not_encode_video_or_create_training_data")
    return value


def authenticate_artifact(record):
    path = Path(record["path"]).resolve(strict=True)
    if not path.is_file() or ("bytes" in record and path.stat().st_size != record["bytes"]):
        raise ValueError("input_size_changed:" + str(path))
    digest = sha256(path)
    if digest != record["sha256"]:
        raise ValueError("input_sha256_changed:" + str(path))
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def load_script(record):
    """Only array parsing, not a policy, simulator or prediction."""
    identity = authenticate_artifact(record)
    import numpy as np
    from chicgrasp_policy.interface import validate_policy_action
    with Path(identity["path"]).open("rb") as stream:
        actions = np.load(stream, allow_pickle=False)
    if sha256(identity["path"]) != identity["sha256"]:
        raise ValueError("script_changed_during_array_load")
    if actions.shape != (200, 5) or actions.dtype != np.dtype("float32"):
        raise ValueError("first_script_protocol_requires_exactly_200_float32_5D_actions; no_padding_or_truncation")
    if not np.isfinite(actions).all() or not np.isin(actions[:, 3:], [0, 1]).all():
        raise ValueError("script_requires_finite_actions_and_binary_jaws")
    for action in actions:
        validate_policy_action(action)
    actions.flags.writeable = False
    return actions, identity


def configure_research(cfg, session, position, ground, *, scene_module=None, scene_identity=None):
    """Resolve shared pose first; selected builder configures before construction."""
    if cfg.scene.ground.spawn.usd_path != host.ORIGINAL_GROUND_URL:
        raise ValueError("unexpected_original_ground_url")
    before = dict(xyz_m=list(cfg.scene.chicken.init_state.pos),
                  quaternion_wxyz=list(cfg.scene.chicken.init_state.rot))
    cfg.scene.ground.spawn.usd_path = ground["path"]
    cfg.scene.chicken.init_state.pos = tuple(position["authored_xyz_m"])
    cfg.scene.chicken.init_state.rot = tuple(position["authored_quaternion_wxyz"])
    session.research_start_configuration = dict(before=before,
        after=dict(xyz_m=list(cfg.scene.chicken.init_state.pos),
                   quaternion_wxyz=list(cfg.scene.chicken.init_state.rot)),
        ground_path=ground["path"], robot_reset_configuration_changed=False,
        changed_fields=["scene.chicken.init_state.pos", "scene.chicken.init_state.rot", "scene.ground.spawn.usd_path"])
    if scene_module is not None:
        robot_before = repr(cfg.scene.robot.init_state)
        from pipeline.scene import install_scene_builder
        protected_before = {name:repr(getattr(cfg, name)) for name in ("actions", "observations")}
        install_scene_builder(cfg, session, scene_module, scene_identity, position)
        if protected_before != {name:repr(getattr(cfg, name)) for name in protected_before}:
            raise ValueError("scene_builder_changed_policy_action_or_observation_contract")
        session.research_start_configuration.update(scene_setup_identity=deepcopy(scene_identity),
            after=dict(xyz_m=list(cfg.scene.chicken.init_state.pos), quaternion_wxyz=list(cfg.scene.chicken.init_state.rot)),
            custom_scene=scene_identity["definition"]["kind"] == "custom",
            robot_reset_configuration_changed=repr(cfg.scene.robot.init_state) != robot_before,
            robot_reset_configuration_before=robot_before,
            robot_reset_configuration_after=repr(cfg.scene.robot.init_state),
            requested_pose=deepcopy(position), configuration_callback_invoked=True)


def configuration_hook_tree(source):
    """Insert the placement callback after the one existing cfg.seed assignment."""
    tree = ast.parse(textwrap.dedent(source))
    hits = 0

    class Insert(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal hits
            if (len(node.targets) == 1 and ast.dump(node.targets[0], include_attributes=False) ==
                    ast.dump(ast.parse("cfg.seed = session.seed").body[0].targets[0], include_attributes=False)
                    and ast.dump(node.value, include_attributes=False) ==
                    ast.dump(ast.parse("session.seed", mode="eval").body, include_attributes=False)):
                hits += 1
                callback = ast.parse("self.configure_research(cfg, session)").body[0]
                return [node, ast.copy_location(callback, node)]
            return node

    tree = Insert().visit(tree)
    if hits != 1:
        raise ValueError("registered_configuration_hook_count:" + str(hits))
    return ast.fix_missing_locations(tree)


def _hook_configuration(function):
    scope = dict(function.__globals__)
    exec(compile(configuration_hook_tree(inspect.getsource(function)), inspect.getsourcefile(function), "exec"), scope)
    return functools.update_wrapper(scope[function.__name__], function)


def install_host(config, request):
    """Host-only adaptations, without the old checkpoint/HC binding installer."""
    resources = host.verify_sdk_resources(config)
    ground = host.verify_bundled_ground()
    from chicgrasp_policy import render_profile, grey_studio_check, dome_recovery
    from chicgrasp_policy import attachment_capture_runtime as native, attachment_owner
    experience = resources["experience"]
    original_renderer = render_profile.validate_renderer
    render_profile.SHARED_EXPERIENCE = experience["path"]
    native.SHARED_EXPERIENCE = experience["path"]
    native.APP_CONFIGURATION["experience"] = experience["path"]

    def validate_renderer(record, baseline=None):
        return original_renderer(host.relocate_renderer_record(record, experience),
            None if baseline is None else host.relocate_renderer_record(baseline, experience))

    render_profile.validate_renderer = validate_renderer
    sim = Path(config["isaac_sim"])
    dlls = {"C:/isaacsim": sim.as_posix(), "C:/isaacsim/kit": (sim / "kit").as_posix(),
            "C:/isaacsim/kit/kernel/plugins": (sim / "kit/kernel/plugins").as_posix()}
    native._NativeDependencies.launch_app = host.relocate_function(native._NativeDependencies.launch_app, dlls)
    versions = {"C:/isaacsim/VERSION": resources["sim_version"]["path"],
                "C:/robotics/IsaacLab/VERSION": resources["lab_version"]["path"]}
    attachment_owner._installed_identity = host.relocate_function(attachment_owner._installed_identity, versions)
    attachment_owner.PINNED_RUNTIME["binding_path"] = str(Path(resources["physx_binding"]["path"]))
    shader = resources["ground_shader"]
    shader_property = grey_studio_check.SHADER + ".info:mdl:sourceAsset"
    original_grey, original_dome = grey_studio_check.validate_report, dome_recovery.apply_overrides

    def validate_report(report, calibration):
        return original_grey(host.relocate_shader_report(report, shader, shader_property), calibration)

    def apply_overrides(stage, *, edit_context=None):
        original_dome(stage, edit_context=edit_context)
        from pxr import Sdf, Usd
        attr = stage.GetPrimAtPath(grey_studio_check.SHADER).GetAttribute("info:mdl:sourceAsset")
        if not attr.IsValid() or host.normalized_path(attr.Get().path) != host.normalized_path(shader["original_path"]):
            raise ValueError("ground_shader_original_asset_path_changed")
        if host.normalized_path(shader["path"]) != host.normalized_path(shader["original_path"]):
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                if not attr.Set(Sdf.AssetPath(shader["path"])):
                    raise ValueError("ground_shader_host_alias_write_failed")

    grey_studio_check.validate_report, dome_recovery.apply_overrides = validate_report, apply_overrides

    class ResearchDependencies(native._NativeDependencies):
        construct_environment = _hook_configuration(native._NativeDependencies.construct_environment)

        def configure_research(self, cfg, session):
            from pipeline.scene import authenticate_scene_setup
            module, identity = authenticate_scene_setup(request["scene_setup_identity"])
            return configure_research(cfg, session, request["position"], ground,
                scene_module=module, scene_identity=identity)

    source = FROZEN / "scripts/policy/attachment_assisted_lift_record.py"
    spec = importlib.util.spec_from_file_location("research_original_recorder", source)
    recorder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = recorder
    spec.loader.exec_module(recorder)
    return recorder, native, ResearchDependencies(), dict(schema="research-host-adapter/v1",
        frozen_files_modified=False, old_hc_bindings_used=False, resources=resources,
        ground=ground, configuration_hook="after cfg.seed = session.seed",
        literal_replacements={"native_launch": dlls, "owner_identity": versions},
        adaptations=["host SDK paths", "authenticated shader path alias", "authored research position", "bundled unchanged ground"])


class ScriptController:
    """Explicit 200-action research script; no fallback, padding or model."""
    total_actions = 200

    def __init__(self, actions, writer, identity):
        self.actions, self.writer, self.identity, self.next_index = actions, writer, identity, 0

    @staticmethod
    def indices(index):
        return dict(global_action=index, script_action=index,
            setup_action=index if index < 50 else None, training_action=index-50 if index >= 50 else None,
            learned_action=None)

    def select(self, index, observation, sensor):
        if type(index) is not int or index != self.next_index or not 0 <= index < 200:
            raise ValueError("research_script_sequence")
        action = self.actions[index].copy()
        self.writer.action(dict(global_action=index, action=action, source="research-script/v1",
            source_action=index, script_identity=self.identity,
            acquisition_native_clock=sensor["native_clock"], control_tick_start=6*index))
        self.writer.command_event("SELECTED_VALIDATED", index,
            observation_references=[f"rgb/{index:06d}.json"], scheduled_boundary=[6*index, 0])
        self.next_index += 1
        return action

    def summary(self, completed):
        return dict(source="research-script/v1", selected_actions=self.next_index,
            completed_actions=completed, scripted_actions=self.next_index, learned_actions=0,
            script_identity=self.identity)


def episode_arrays(capture, runtime, *, np, interface, episode_io):
    count = len(capture.executed_actions)
    if (not capture.completed or count != len(capture.observations) or count != len(capture.diagnostics)
            or len(capture.rows) != 12*count or capture.lifecycle_events or count < 1):
        raise ValueError("incomplete_or_reset_capture_not_training_episode")
    arrays = {"timestamp_ns": np.arange(count, dtype=np.int64)*episode_io.TIMESTAMP_PERIOD_NS}
    for key in interface.POLICY_OBSERVATION_KEYS:
        arrays[key] = np.stack([observation.as_mapping()[key] for observation in capture.observations])
    arrays["action"] = np.stack(capture.executed_actions)
    metadata = dict(schema_version=episode_io.COMMANDED_EPISODE_SCHEMA_VERSION,
        episode_id=runtime.episode_id, task_id="ChicGrasp-Research-Pipeline-v1", seed=runtime.seed,
        num_policy_steps=count, policy_hz=10, control_hz=60, control_ticks_per_policy_action=6,
        timestamp_origin=episode_io.TIMESTAMP_ORIGIN,
        observation_action_alignment=episode_io.OBSERVATION_ACTION_ALIGNMENT,
        jaw_state_semantics=interface.JAW_STATE_SEMANTICS)
    return arrays, metadata


def finalize_episode(capture, runtime, output):
    import numpy as np
    from chicgrasp_policy import episode_io, interface
    arrays, metadata = episode_arrays(capture, runtime, np=np, interface=interface, episode_io=episode_io)
    destination = output / "episode"
    episode_io.write_episode(destination, arrays, episode_end=metadata, diagnostic_frames=capture.diagnostics)
    reloaded = episode_io.load_policy_episode(destination)
    if any(not np.array_equal(arrays[key], reloaded[key]) for key in episode_io.POLICY_ARRAY_KEYS):
        raise ValueError("research_episode_roundtrip_mismatch")
    manifest = destination / "manifest.json"
    binding = output / "episode_binding.json"
    write_json(binding, dict(schema="research-episode-binding/v1",
        scene_setup_identity=deepcopy(runtime.research_scene_setup_identity),
        episode_manifest=dict(bytes=manifest.stat().st_size, sha256=sha256(manifest))))
    return dict(path=str(destination), schema_version=2, policy_steps=len(arrays["action"]),
        roundtrip_equal=True, manifest_sha256=sha256(manifest),
        binding=dict(path=str(binding.resolve()), bytes=binding.stat().st_size, sha256=sha256(binding)))


def create_scene_cameras(runtime, output, *, creator=None, camera_factory=None, pause=True):
    """Add selectable presentation camera prims, without encoding or stepping."""
    from camera_presets.apply_cameras import create_demo_cameras, PRESET_DIR
    creator = creator or create_demo_cameras
    if pause:
        runtime.env.sim.pause()
    before = (tuple(runtime.clock()), deepcopy(runtime.camera_frames()), deepcopy(runtime.effective_configuration))
    created = []
    def factory(**kwargs):
        if camera_factory is None:
            from isaacsim.sensors.camera import Camera
            camera = Camera(**kwargs)
        else:
            camera = camera_factory(**kwargs)
        created.append(camera)
        return camera
    try:
        cameras = creator(runtime.stage, enabled=True, camera_factory=factory)
        cameras.before_render()  # Read-only policy-camera and massless-render checks.
        after = (tuple(runtime.clock()), deepcopy(runtime.camera_frames()), deepcopy(runtime.effective_configuration))
        if after != before:
            raise ValueError("scene_camera_creation_changed_physics_clock_or_policy_camera_state")
        receipt = dict(schema="research-scene-cameras/v1", created_paths=deepcopy(cameras.paths),
            configs=deepcopy(cameras.configs), policy_cameras_before=deepcopy(cameras.policy_before),
            preset_validation="INITIAL_CONFIGURATION_ONLY", camera_edits_saved=False,
            policy_cameras_unchanged=True, massless_render_only=True, encoding_started=False,
            preset_identities={name:dict(bytes=(PRESET_DIR/name).stat().st_size, sha256=sha256(PRESET_DIR/name))
                for name in ("HAND_DEMO_ONLY.json", "HD_TWINS_DEMO_ONLY.json")}, viewport_errors=[])
        # USD camera prims remain selectable even if the optional viewport API is absent.
        try:
            from omni.kit.viewport.utility import get_active_viewport
            viewport = get_active_viewport()
            if viewport is None:
                raise RuntimeError("no_active_viewport; select HAND through any viewport camera menu")
            # Navigate with the editor camera; named preset views remain selectable.
            viewport.camera_path = "/OmniverseKit_Persp"
        except Exception as exc:
            receipt["viewport_errors"].append(type(exc).__name__ + ":" + str(exc))
        write_json(output / "scene_cameras.json", receipt)
        return cameras, created, receipt
    except BaseException:
        close_scene_cameras(created)
        raise


def close_scene_cameras(created):
    errors = []
    for camera in reversed(created):
        for operation in ("pause", "destroy"):
            try:
                getattr(camera, operation)()
            except Exception as exc:
                errors.append(operation + ":" + type(exc).__name__ + ":" + str(exc))
    return errors


def create_inspection_controls():
    """Scene-only UI; a click requests the existing orderly cleanup path."""
    controls = dict(finish_requested=False, window=None, errors=[])
    try:
        import omni.ui as ui
        controls["window"] = ui.Window("Scene inspection", width=360, height=140, visible=True)
        def finish():
            controls["finish_requested"] = True
        with controls["window"].frame:
            with ui.VStack(spacing=8):
                ui.Label("Move around freely. Keep physics paused.", word_wrap=True)
                ui.Label("Finish here to save the result and close Isaac Sim.", word_wrap=True)
                ui.Button("Finish inspection", height=38, clicked_fn=finish)
    except Exception as exc:
        controls["errors"].append(type(exc).__name__ + ": " + str(exc))
    return controls


def paused_scene_review(runtime, attempt, seconds, cameras=None, *, controls_factory=None):
    """Allow temporary view edits; finish cleanly before app shutdown."""
    runtime.env.sim.pause()
    clock = tuple(runtime.clock())
    controls = (controls_factory or create_inspection_controls)()
    end = time.monotonic()+seconds
    try:
        while (runtime.app.is_running() and time.monotonic() < end
               and not controls["finish_requested"] and not (attempt / "CLOSE_SCENE").exists()):
            runtime.app.update()
            if not runtime.app.is_running():
                break
            if tuple(runtime.clock()) != clock:
                raise RuntimeError("scene_review_physics_advanced; no_policy_or_candidate_authorized_in_scene_mode")
            # Navigation changes are temporary; filming guards stay out of review.
            time.sleep(.01)
    finally:
        if controls["window"] is not None:
            try:
                controls["window"].destroy()
            except Exception as exc:
                controls["errors"].append("UI cleanup: " + type(exc).__name__ + ": " + str(exc))
    return dict(status="SCENE_REVIEW_ENDED", approved=False, physics_paused=True,
        clock=list(clock), candidate_actions=0, camera_edits_saved=False,
        camera_navigation="temporary_inspection_only", controls_errors=controls["errors"],
        reason="FINISH_BUTTON" if controls["finish_requested"] else
               "CLOSE_SCENE" if (attempt / "CLOSE_SCENE").exists() else
               "WINDOW_CLOSED" if not runtime.app.is_running() else "REVIEW_TIME_LIMIT")


def attach_media(runtime, output, options):
    if not options["enabled"]:
        return None
    from pipeline.video import MediaRecorder
    recorder = MediaRecorder(runtime, output, ffmpeg_path=options["ffmpeg_path"],
        enabled=True, crf=options.get("crf", 16), fps=options.get("fps", 30))
    original_step = runtime.step

    def step(action):
        flags = original_step(action)
        recorder.after_substep(runtime.scheduled_substeps-1)
        return flags

    runtime.step = step
    return recorder


def validate_scene_evaluation(value, *, custom_scene):
    if not isinstance(value, dict) or value.get("status") not in ("UNSCORED", "PASS", "FAIL"):
        raise ValueError("scene_evaluator_status_required")
    if type(value.get("training_eligible")) is not bool:
        raise ValueError("scene_evaluator_explicit_training_eligibility_required")
    if custom_scene and value.get("label") in ("PIPELINE_VERTICAL_SLICE_PASS_WITH_TEMPORARY_ATTACHMENT",
            "PHYSICAL_GRASP_PASS", "G1_PASS"):
        raise ValueError("custom_scene_cannot_inherit_original_or_physical_success_label")
    if value["status"] == "FAIL" and value["training_eligible"]:
        raise ValueError("failed_custom_task_cannot_be_training_eligible")
    return deepcopy(value)


def run(config, request, attempt):
    """One owned child invocation; parent owns deadline and process adjudication."""
    attempt = Path(attempt).resolve()
    request = validate_request(request, attempt)
    from pipeline.scene import authenticate_scene_setup
    scene_module, scene_setup_identity = authenticate_scene_setup(request["scene_setup_identity"])
    custom_scene = scene_setup_identity["definition"]["kind"] == "custom"
    attempt.mkdir(parents=True, exist_ok=True)
    destination = attempt / "research_result.json"
    if destination.exists():
        raise FileExistsError(destination)
    output = attempt / "capture/experiment"
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema=RESULT_SCHEMA, mode=request["mode"], completion="FAILED",
        acceptance_eligible=False, original_hc_evidence=False,
        environment_seed=42, scene_profile="attempt005", position=request["position"],
        scene_setup_identity=scene_setup_identity, scene_kind=scene_setup_identity["definition"]["kind"],
        provenance=dict(source_commit=SOURCE_COMMIT, executed_commit=None,
            provenance_kind="new_research_run", runtime_snapshot_sha256=sha256(ROOT / "manifests/CORE_INPUTS.json"),
            wrapper_identity={"pipeline/runtime/native.py": sha256(ROOT / "pipeline/runtime/native.py")},
            requested_scene_config_sha256=request.get("scene_sha256"),
            script_source_sha256=request.get("script_source_sha256")),
        objective_metrics=None, objective_evaluator=("custom-scene/"+scene_setup_identity["definition"]["id"] if custom_scene
            else "frozen attachment_capture history evaluator in native child"), scene_evaluation=None,
        independent_process_checker=False, training_eligible=False,
        episode=None, media=None, failures=[], cleanup=None,
        process_completion="PARENT_OBSERVATION_REQUIRED")
    runtime = capture = writer = controller = media = core = preview = None
    scene_camera_handles = []
    phase, actions, script_identity = "preflight", None, None
    try:
        checked = preflight(config, include_checkpoint=False)
        write_json(attempt / "native_preflight.json", checked)
        if checked["status"] != "PASS":
            raise ValueError("frozen_or_host_preflight_failed:" + repr(checked["errors"]))
        if str(FROZEN / "src") not in sys.path:
            sys.path.insert(0, str(FROZEN / "src"))
        if request["mode"] != "scene":
            actions, script_identity = load_script(request["script"])
            result["script_identity"] = script_identity
        if request["mode"] == "live":
            result["checkpoint_identity"] = authenticate_artifact(request["checkpoint"])
            if request["checkpoint"].get("artifact_path") is not None:
                artifact = Path(request["checkpoint"]["artifact_path"]).resolve(strict=True)
                if artifact != Path(request["checkpoint"]["path"]).parent / "artifact.json":
                    raise ValueError("checkpoint_artifact_not_adjacent_expected_file")
                result["checkpoint_artifact_identity"] = authenticate_artifact(
                    dict(path=str(artifact), sha256=request["checkpoint"]["artifact_sha256"]))
                if load_json(artifact).get("scene_setup_identity") != scene_setup_identity:
                    raise ValueError("checkpoint_scene_setup_identity_mismatch")
            else:
                raise ValueError("checkpoint_scene_artifact_required")
            from pipeline.models import inspect_checkpoint
            result["checkpoint_metadata"] = inspect_checkpoint(request["checkpoint"]["path"])
            if (result["checkpoint_metadata"].get("training_window_start") != 50
                    or result["checkpoint_metadata"].get("live_compatible") is not True):
                raise ValueError("native_live_requires_explicitly_compatible_training_window_start_50")
        if request["video"]["enabled"]:
            from pipeline.video import validate_tools
            ffmpeg, ffprobe = validate_tools(request["video"].get("ffmpeg_path"))
            request["video"]["ffmpeg_path"] = str(ffmpeg)
            result["media_tools"] = dict(ffmpeg=str(ffmpeg), ffprobe=str(ffprobe))
        core, native, deps, receipt = install_host(config, request)
        write_json(attempt / "host_adapter.json", receipt)
        from chicgrasp_policy import stage1_demo, geometric_capture, attachment_owner
        from chicgrasp_policy.attachment_observation_evidence import IncrementalRGBWriter
        from chicgrasp_policy.stage3_policy import PolicyWriter, LiveController
        profile = stage1_demo.approved_scaled_profile()
        runtime = native.RuntimeSession(output, seed=42, episode_id="research_" + attempt.name,
            deps=deps, geometric_mode=True,
            **({"stage0_mode": "TUNE_GUI"} if request["mode"] in ("scene", "preview") else {}))
        runtime.execution_context = dict(result["provenance"], attempt_id=attempt.name)
        runtime.translated_point = runtime.r4_live_binding = None
        runtime.stage1_demo = request["mode"] in ("scripted", "preview")
        runtime.stage3_learned = request["mode"] == "live"
        runtime.research_scene_setup_identity = deepcopy(scene_setup_identity)
        runtime.stage3_witness = not custom_scene
        runtime.envelope_v2 = request["mode"] == "live" and not custom_scene
        runtime.envelope_protocol = "v3c" if runtime.envelope_v2 else "v2"
        runtime.witness_run_type = "HARNESS"
        runtime.admission_policy = "side-opening-progress/v1"
        runtime.media_deadline_monotonic = time.monotonic() + max(0.,
            request.get("deadline_unix", time.time()+1195)-time.time())
        runtime.media_scene_binding = dict(profile_id=profile["profile_id"],
            scene_profile_sha256=stage1_demo.SCALED_PROFILE_SHA256,
            runtime_snapshot_sha256=result["provenance"]["runtime_snapshot_sha256"],
            asset_root_identities=deepcopy(profile["asset_root_identities"]), chicken_version="original",
            scene_setup_identity=deepcopy(scene_setup_identity), custom_scene=custom_scene,
            attachment_mode="DISABLED" if custom_scene else "ORIGINAL_TEMPORARY_ATTACHMENT")
        def append_phase(record):
            with (attempt / "runtime_phases.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False)+"\n")
                stream.flush()
        runtime.phase_record_sink = append_phase
        runtime.guard_record_sink = append_phase
        phase = "initialize"
        runtime.initialize()
        result["start_configuration"] = runtime.research_start_configuration
        if custom_scene:
            from pipeline.scene import validate_custom_build
            result["scene_build"] = validate_custom_build(runtime)
            write_json(output / "custom_scene_build.json", result["scene_build"])
        if request["mode"] == "scene":
            phase = "scene_cameras"
            scene_cameras, scene_camera_handles, camera_receipt = create_scene_cameras(runtime, output)
            result["scene_cameras"] = camera_receipt
            print(json.dumps(dict(phase="SCENE_READY", camera_paths=scene_cameras.paths,
                physics_paused=True, approved=False)), flush=True)
            phase = "paused_scene_review"
            result["scene_review"] = paused_scene_review(runtime, attempt, request["review_seconds"], scene_cameras)
            result["completion"] = "COMPLETED"
        else:
            if request["mode"] == "preview":
                from pipeline.scene import PreviewSession, validate_preview_configuration
                if not custom_scene:
                    result["preview_gui_configuration"] = validate_preview_configuration(
                        runtime.effective_configuration, profile["physical_configuration"])
                phase = "preview_cameras"
                cameras, scene_camera_handles, result["scene_cameras"] = create_scene_cameras(runtime, output, pause=False)
                preview = PreviewSession(runtime, cameras, seconds=request["review_seconds"])
                result["preview"] = preview.record
                result["provenance"]["wrapper_identity"]["pipeline/scene.py"] = sha256(ROOT / "pipeline/scene.py")
                print(json.dumps(dict(phase="PREVIEW_READY", camera_paths=cameras.paths,
                    action_count=200, simulation_seconds=20, training_data_created=False,
                    instruction="Click Start demonstration, then Finish preview when done.")), flush=True)
                phase = "preview_wait_for_start"
                preview.wait_for_start()
                preview.wrap_step()
            elif not custom_scene:
                if runtime.effective_configuration != profile["physical_configuration"]:
                    raise ValueError("attempt005_effective_physics_changed")
                stage1_demo.save_visuals(runtime, output, "before_candidate", scene_profile=True)
            else:
                write_json(output / "custom_runtime_before_candidate.json", dict(
                    scene_setup_identity=scene_setup_identity, effective_configuration=runtime.effective_configuration,
                    attachment_mode="DISABLED", original_hc_evidence=False))
            rgb = IncrementalRGBWriter(output / "rgb", **({"envelope_v2": True} if runtime.envelope_v2 else {}))
            if request["mode"] in ("scripted", "preview"):
                class ResearchScriptWriter(PolicyWriter):
                    prefix, command_schema = "research_script", "research-script-command-lifecycle/v1"
                    def observation_indices(self, index):
                        return {k:v for k,v in ScriptController.indices(index).items() if k != "global_action"}
                writer = ResearchScriptWriter(rgb, output)
                controller = ScriptController(actions, writer, script_identity)
            else:
                writer = PolicyWriter(rgb, output)
                from pipeline.models import load_predictor
                phase = "load_research_checkpoint"
                predictor = load_predictor(request["checkpoint"]["path"], device="cuda:0",
                    evaluation_seed=request["evaluation_seed"])
                controller = LiveController(predictor, writer, evaluation_seed=request["evaluation_seed"],
                    setup_actions=actions[:50].copy(), checkpoint_sha256=result["checkpoint_identity"]["sha256"], envelope_v2=not custom_scene)
            if custom_scene:
                from pipeline.recording import CustomCapture
                phase = "custom_capture_binding"
                if runtime.owner is not None:
                    raise ValueError("custom_scene_must_not_bind_gripper_attachment")
                capture = CustomCapture(core, runtime, writer, controller)
            else:
                phase = "owner_binding"
                runtime.owner = attachment_owner.bind_runtime(runtime.env, runtime.stage, task_id=core.TASK_ID,
                    capture_policy=True, episode_id=runtime.episode_id, origin_step=runtime.origin_clock[0],
                    origin_time=runtime.origin_clock[1], scene_identity=runtime.scene_identity)
                runtime.owner.journal.select_geometric_admission(runtime.admission_policy, run_type="HARNESS",
                    envelope_v2=runtime.envelope_v2, protocol=runtime.envelope_protocol)
                capture = geometric_capture.make_capture(core, runtime, writer, action_source=controller)
            media = attach_media(runtime, output / "media", request["video"])
            phase = "candidate"
            result["objective_metrics"] = capture.run()
            if request["mode"] in ("scripted", "preview"):
                # The frozen generic controller otherwise labels this field as
                # checkpoint commands. Only provenance changes, not its values.
                result["objective_metrics"]["requested_tcp_source"] = "research_script_commands_robot_base_metres"
            if request["mode"] != "preview":
                phase = "save_complete_episode"
                result["episode"] = finalize_episode(capture, runtime, output)
            phase = "scene_evaluation"
            evaluation = scene_module.evaluate_episode(capture, deepcopy(result["objective_metrics"]),
                deepcopy(scene_setup_identity["options"]))
            result["scene_evaluation"] = validate_scene_evaluation(evaluation, custom_scene=custom_scene)
            technically_complete = bool(capture.completed and not capture.lifecycle_events and
                not getattr(capture, "recording_errors", []) and len(capture.executed_actions) == 200 and
                len(capture.rows) == 2400) if custom_scene else bool(capture.completed)
            result["training_eligible"] = (request["mode"] == "scripted" and technically_complete and
                evaluation["training_eligible"] is True)
            if preview is not None:
                if not technically_complete or len(capture.executed_actions) != 200:
                    raise ValueError("preview_script_incomplete")
                phase = "preview_finished"
                preview.finished()
            elif not custom_scene:
                stage1_demo.save_visuals(runtime, output, "after_candidate", scene_profile=True)
            else:
                result["custom_task_status"] = evaluation["status"]
                write_json(output / "custom_runtime_after_candidate.json", dict(
                    scene_setup_identity=scene_setup_identity, effective_configuration=runtime.effective_configuration,
                    attachment_mode="DISABLED", original_hc_evidence=False))
            result["completion"] = "COMPLETED"
    except Exception as exc:
        result["failures"].append(dict(phase=phase, type=type(exc).__name__, message=str(exc)))
        result["training_eligible"] = False
    finally:
        if preview is not None:
            try:
                preview.close()
            except Exception as exc:
                result["failures"].append(dict(phase="preview_cleanup", type=type(exc).__name__, message=str(exc)))
        if scene_camera_handles:
            scene_errors = close_scene_cameras(scene_camera_handles)
            result["scene_camera_cleanup"] = dict(errors=scene_errors, attempted=True)
            if scene_errors:
                result["failures"].append(dict(phase="scene_camera_cleanup", type="SceneCameraCleanupErrors",
                    message=repr(scene_errors)))
        if media is not None:
            try:
                result["media"] = media.finish()
                if result["media"].get("status") != "PASS":
                    result["failures"].append(dict(phase="media_finish", type="MediaFailure",
                        message=repr(result["media"].get("errors", result["media"]))))
            except Exception as exc:
                result["failures"].append(dict(phase="media_finish", type=type(exc).__name__, message=str(exc)))
        if core is not None:
            try:
                core.write_json(output / "measurements.json", dict(schema="research-measurements/v1",
                    rows=capture.rows if capture else [], sensor_samples=capture.sensor_samples if capture else [],
                    lifecycle_events=capture.lifecycle_events if capture else []))
                if runtime is not None and runtime.owner is not None:
                    core.write_json(output / "owner_before_cleanup.json", runtime.owner.export())
                if writer is not None:
                    writer.finish(bool(capture and capture.completed), [x["message"] for x in result["failures"]])
            except Exception as exc:
                result["failures"].append(dict(phase="save_raw_capture", type=type(exc).__name__, message=str(exc)))
        result["counts"] = dict(completed_actions=len(capture.executed_actions) if capture else 0,
            selected_actions=controller.next_index if controller else 0,
            observed_substeps=len(capture.rows) if capture else 0,
            completed_control_ticks=capture.control_ticks if capture else 0)
        if capture is not None and getattr(capture, "recording_errors", []):
            result["failures"].append(dict(phase="recording", type="CaptureRecordingErrors",
                message=repr(capture.recording_errors)))
        if runtime is not None:
            result["initialization"] = deepcopy(runtime.initialization)
            if runtime.initialization.get("phase_evidence_errors"):
                result["failures"].append(dict(phase="phase_evidence", type="PhaseEvidenceErrors",
                    message=repr(runtime.initialization["phase_evidence_errors"])))
            result["effective_configuration"] = runtime.effective_configuration
            try:
                result["cleanup"] = runtime.close_scene(capture.context if capture else None)
                if result["cleanup"].get("errors"):
                    result["failures"].append(dict(phase="cleanup", type="CleanupErrors", message=repr(result["cleanup"]["errors"])))
                if result["cleanup"].get("owner"):
                    issues = core.owner_cleanup_issues(result["cleanup"]["owner"])
                    if issues:
                        result["failures"].append(dict(phase="owner_cleanup", type="CleanupErrors", message=repr(issues)))
            except Exception as exc:
                result["failures"].append(dict(phase="cleanup", type=type(exc).__name__, message=str(exc)))
        if result["failures"]:
            result["completion"], result["training_eligible"] = "FAILED", False
        result["application_close"] = "REQUESTED_AFTER_DURABLE_RESULT; parent_process_receipt_required" if runtime else "NOT_STARTED"
        # Save before SimulationApp.close, which can terminate without returning.
        if core is not None:
            core.write_json(destination, result)
        else:
            write_json(destination, result)
        if runtime is not None:
            runtime.close_app()
            write_json(attempt / "application_close_returned.json", {"returned": True, "time_unix": time.time()})
    return result


"""Explicit native child entry. --help does not import simulator or models."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def native_main(argv=None):
    parser = argparse.ArgumentParser(description="Execute one separately authorized research run; not original HC evidence.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--attempt", required=True, type=Path)
    args = parser.parse_args(argv)
    from pipeline.runtime.host import load_json
    from pipeline.settings import load_host
    request_path = args.request.resolve(strict=True)
    request = validate_request(load_json(request_path), request_path.parent)
    result = run(load_host(args.config), request, args.attempt)
    return 0 if result["completion"] == "COMPLETED" else 2

if __name__ == "__main__":
    raise SystemExit(native_main())
