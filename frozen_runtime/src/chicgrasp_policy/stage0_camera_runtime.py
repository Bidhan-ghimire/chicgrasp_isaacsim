"""Stage0 diagnostic camera UI/checkpoints; native imports are deferred until use.

The scheduled physics callback stays blocked throughout GUI tuning. Its only
refresh is SimulationContext.render, followed by zero-dt sensor refreshes.
Every native/UI operation is checked against the same frozen physics counters.
"""
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

CAMERAS = {"camera_left": "/World/envs/env_0/PolicyCameraLeft",
           "camera_right": "/World/envs/env_0/PolicyCameraRight"}
CHECKPOINTS = ((0, 1), (839, 1), (1199, 1))
CONFIGURATION_PATH = Path(__file__).resolve().parents[2] / "config/policy_camera_calibration.json"


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _descriptor(path, root):
    raw = Path(path).read_bytes()
    return dict(path=Path(path).relative_to(root).as_posix(), bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())


def _rgb(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.array(value, copy=True)
    if result.shape != (240, 320, 3) or result.dtype != np.dtype(np.uint8):
        raise ValueError("stage0_sensor_requires_240x320x3_uint8_RGB")
    return result


class _NativeDependencies:
    """Installed API boundary, constructed only during an explicitly selected run."""
    def __init__(self, session):
        from pxr import Usd, UsdGeom, UsdLux
        self.Usd, self.UsdGeom, self.UsdLux = Usd, UsdGeom, UsdLux
        self.session = session
        self.window = None
        self.providers = {}
        self.status_label = None
        self.camera_locks = {}

    def render(self):
        self.session.env.sim.render()

    def is_running(self):
        return self.session.app.is_running()

    def read_rgb(self, name):
        sensor = self.session.env.scene[name]
        sensor.update(0.0, force_recompute=True)
        return _rgb(sensor.data.output["rgb"][0])

    def sensor_metadata(self):
        return self.session.camera_metadata()

    def snapshot_renderer(self):
        from .render_profile import snapshot_renderer
        return snapshot_renderer(self.session)

    def snapshot_render_diagnostic(self):
        from .stage0_render_diagnostic import snapshot
        return snapshot(self.session)

    def snapshot_lights(self):
        from .stage0_light_dump import snapshot
        return snapshot(self.session.env.scene, phase="GUI_EXPORT_READBACK")

    def snapshot_frozen_poses(self):
        reading = self.session.measure()
        if not reading["valid"]:
            raise ValueError("dome_native_pose_read_invalid")
        def plain(value):
            if hasattr(value, "tolist"):
                return value.tolist()
            if isinstance(value, dict):
                return {k: plain(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [plain(v) for v in value]
            return value
        # All native robot-link/root and chicken poses, not only requested TCP.
        return plain(reading["raw_native_xyzw"])

    def snapshot_grey_studio(self, calibration):
        # Reuse the one constructor-time native read. No second scene scan,
        # render or dependency load is introduced at this scheduled boundary.
        return json.loads((Path(self.session.output_dir) / "grey_studio_check.json").read_text(encoding="utf-8"))

    def snapshot_visuals(self):
        stage = self.session.stage
        cache = self.UsdGeom.XformCache(self.Usd.TimeCode.Default())
        def matrix(value):
            return [[float(value[row][column]) for column in range(4)] for row in range(4)]
        def transform(prim):
            parent = prim.GetParent()
            local = self.UsdGeom.Xformable(prim)
            if not prim.IsValid() or not parent.IsValid() or local.GetResetXformStack():
                raise ValueError("stage0_unusable_or_reset_transform_stack")
            return dict(prim_path=str(prim.GetPath()), parent_path=str(parent.GetPath()),
                parent_world_matrix=matrix(cache.GetLocalToWorldTransform(parent)),
                local_matrix=matrix(local.GetLocalTransformation()),
                world_matrix=matrix(cache.GetLocalToWorldTransform(prim)))
        cameras = {}
        for name, path in CAMERAS.items():
            sensor = self.session.env.scene[name]
            if len(sensor._view.prims) != 1 or str(sensor._view.prims[0].GetPath()) != path:
                raise ValueError("stage0_actual_recording_camera_identity:" + name)
            prim = stage.GetPrimAtPath(path)
            camera = self.UsdGeom.Camera(prim)
            if not camera:
                raise ValueError("stage0_recording_prim_is_not_camera:" + path)
            item = transform(prim)
            item["effective_optics"] = dict(projection=str(camera.GetProjectionAttr().Get()),
                focal_length=float(camera.GetFocalLengthAttr().Get()),
                horizontal_aperture=float(camera.GetHorizontalApertureAttr().Get()),
                vertical_aperture=float(camera.GetVerticalApertureAttr().Get()),
                horizontal_aperture_offset=float(camera.GetHorizontalApertureOffsetAttr().Get()),
                vertical_aperture_offset=float(camera.GetVerticalApertureOffsetAttr().Get()),
                focus_distance=float(camera.GetFocusDistanceAttr().Get()), f_stop=float(camera.GetFStopAttr().Get()),
                clipping_range=[float(value) for value in camera.GetClippingRangeAttr().Get()],
                width=int(sensor.cfg.width), height=int(sensor.cfg.height), data_types=list(sensor.cfg.data_types))
            cameras[path.rsplit("/", 1)[1]] = item
        light_prim = stage.GetPrimAtPath("/World/DomeLight")
        if not light_prim.IsValid() or light_prim.GetTypeName() != "DomeLight":
            raise ValueError("stage0_existing_DomeLight_required")
        light, api = self.UsdLux.DomeLight(light_prim), self.UsdLux.LightAPI(light_prim)
        texture = light.GetTextureFileAttr().Get()
        texture = None if texture is None or not texture.path else str(texture.path)
        properties = dict(color=[float(v) for v in api.GetColorAttr().Get()],
            intensity=float(api.GetIntensityAttr().Get()), exposure=float(api.GetExposureAttr().Get()),
            normalize=bool(api.GetNormalizeAttr().Get()),
            enable_color_temperature=bool(api.GetEnableColorTemperatureAttr().Get()),
            color_temperature=float(api.GetColorTemperatureAttr().Get()), texture_file=texture,
            texture_format=str(light.GetTextureFormatAttr().Get()),
            visible_in_primary_ray=light_prim.GetAttribute("visibleInPrimaryRay").Get())
        return dict(cameras=cameras, light=dict(transform(light_prim), type="DomeLight", properties=properties))

    def select_camera(self, name):
        from omni.kit.viewport.utility import get_active_viewport
        prim = self.session.stage.GetPrimAtPath(CAMERAS[name])
        if not prim.IsValid() or prim.GetTypeName() != "Camera":
            raise ValueError("stage0_actual_recording_camera_missing")
        lock = prim.GetAttribute("omni:kit:cameraLock")
        if not lock.IsValid() or type(lock.Get()) is not bool:
            raise ValueError("stage0_recording_camera_lock_unavailable")
        self.camera_locks.setdefault(CAMERAS[name], lock.Get())
        if not lock.Set(False) or lock.Get() is not False:
            raise ValueError("stage0_recording_camera_unlock_failed")
        viewport = get_active_viewport()
        if viewport is None:
            raise ValueError("stage0_active_viewport_unavailable")
        viewport.camera_path = CAMERAS[name]
        if str(viewport.camera_path) != CAMERAS[name]:
            raise ValueError("stage0_viewport_did_not_select_recording_camera")
        return str(viewport.camera_path)

    def set_light(self, name, value):
        prim = self.session.stage.GetPrimAtPath("/World/DomeLight")
        if not prim.IsValid() or prim.GetTypeName() != "DomeLight":
            raise ValueError("stage0_existing_DomeLight_required")
        if name not in ("intensity", "exposure") or not math.isfinite(value) or (name == "intensity" and value < 0):
            raise ValueError("stage0_only_finite_Dome_intensity_and_exposure_controls")
        attribute = prim.GetAttribute("inputs:" + name)
        if not attribute.IsValid() or not attribute.Set(float(value)):
            raise ValueError("stage0_Dome_property_write_failed:" + name)

    def open_ui(self, adapter, snapshot):
        import omni.ui as ui
        self.window = ui.Window("Gate 6 Stage 0: recording cameras", width=700, height=420)
        self.window.set_visibility_changed_fn(lambda visible: None if visible else adapter.request_quit())
        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label("Physics frozen at 839/1. Select a recording camera, then fly it in the viewport.")
                ui.Label("Do not press Play or edit the robot/chicken. Use Export settings, then Quit.")
                with ui.HStack(height=30):
                    ui.Button("Select left camera", clicked_fn=lambda: adapter.select_camera("camera_left"))
                    ui.Button("Select right camera", clicked_fn=lambda: adapter.select_camera("camera_right"))
                    ui.Button("Export settings", clicked_fn=adapter.export)
                    ui.Button("Quit", clicked_fn=adapter.request_quit)
                with ui.HStack(height=32):
                    for name in ("intensity", "exposure"):
                        ui.Label("Dome " + name, width=105)
                        model = ui.SimpleFloatModel(snapshot["light"]["properties"][name])
                        ui.FloatField(model, width=180)
                        model.add_value_changed_fn(lambda model, name=name: adapter.set_light(name, model.as_float))
                with ui.HStack(height=250):
                    for name in CAMERAS:
                        with ui.VStack():
                            ui.Label(name + " actual sensor RGB", height=20)
                            provider = self.providers[name] = ui.ByteImageProvider()
                            ui.ImageWithProvider(provider, width=320, height=240,
                                fill_policy=ui.IwpFillPolicy.IWP_STRETCH)
                self.status_label = ui.Label("No export yet. Exports remain unapproved until headless owner review.")

    def show_previews(self, images):
        for name, pixels in images.items():
            rgba = np.empty((240, 320, 4), dtype=np.uint8)
            rgba[:, :, :3], rgba[:, :, 3] = pixels, 255
            self.providers[name].set_bytes_data(rgba.reshape(-1).tolist(), [320, 240])

    def set_status(self, message):
        if self.status_label is not None:
            self.status_label.text = message

    def close_ui(self):
        if self.window is not None:
            self.window.set_visibility_changed_fn(lambda visible: None)
            self.window.visible = False
        for path, value in self.camera_locks.items():
            lock = self.session.stage.GetPrimAtPath(path).GetAttribute("omni:kit:cameraLock")
            if not lock.Set(value) or lock.Get() is not value:
                raise ValueError("stage0_recording_camera_lock_restore_failed")


class Stage0CameraRuntime:
    def __init__(self, session, output_dir, mode, *, dependencies=None, recovery_enabled=True):
        if mode not in ("TUNE_GUI", "CHECK_HEADLESS"):
            raise ValueError("explicit_stage0_camera_mode_required")
        self.session, self.output_dir, self.mode = session, Path(output_dir), mode
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.dependencies = dependencies
        self.recovery_enabled = recovery_enabled
        self.baseline_config = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
        self.capture = None
        self.initial_snapshot = None
        self.frozen_state = None
        self.quit_requested = False
        self._last_gui_preview = None
        self._finished = None
        self.report = dict(schema="gate6-stage0-camera-runtime/v2", mode=mode, run_type="HARNESS",
            approval_status="NOT_APPROVED", status="INCOMPLETE", checkpoints=[], pause=None,
            exports=[], errors=[], guard_samples=[], viewport_selections=[],
            render_diagnostics=[],
            baseline_configuration=deepcopy(self.baseline_config))

    def _state(self):
        capture = self.capture
        state = dict(native_clock=list(self.session.clock()),
            scheduled_substeps=self.session.scheduled_substeps, runtime_control_ticks=self.session.control_ticks,
            capture_control_ticks=capture.control_ticks, capture_substep=capture.substep,
            capture_rows=len(capture.rows))
        if self.recovery_enabled and self.dependencies is not None:
            state["native_scene_poses"] = self.dependencies.snapshot_frozen_poses()
        return state

    def _guard(self, operation, callback):
        before = self._state()
        if before != self.frozen_state:
            raise RuntimeError("stage0_frozen_boundary_changed_before:" + operation)
        entry = dict(operation=operation, before=before, after=None, wall_monotonic_ns=time.monotonic_ns())
        self.report["guard_samples"].append(entry)
        try:
            return callback()
        finally:
            entry["after"] = self._state()
            if entry["after"] != before:
                raise RuntimeError("stage0_physics_or_scheduled_counter_advanced:" + operation)

    def _initialize(self):
        from .stage0_visuals import validate_export
        if self.dependencies is None:
            self.dependencies = _NativeDependencies(self.session)
        self.initial_snapshot = self._guard("initial_visual_snapshot", self.dependencies.snapshot_visuals)
        validate_export(self.initial_snapshot, self.baseline_config, initial_snapshot=self.initial_snapshot)
        _write_json(self.output_dir / "initial_snapshot.json", self.initial_snapshot)
        self.report["initial_snapshot"] = _descriptor(self.output_dir / "initial_snapshot.json", self.output_dir)
        self.report["renderer"] = self._guard("initial_renderer_snapshot", self.dependencies.snapshot_renderer)
        from .render_profile import validate_renderer
        validate_renderer(self.report["renderer"])
        # Persist failures as well as successes, before any checkpoint RGB.
        # This is a read-only composition check, not an appearance verdict.
        from .grey_studio_check import validate_report
        scene_check = self._guard("grey_studio_scene_check",
            lambda: self.dependencies.snapshot_grey_studio(self.baseline_config))
        _write_json(self.output_dir / "grey_studio_check.json", scene_check)
        self.report["grey_studio_check"] = _descriptor(
            self.output_dir / "grey_studio_check.json", self.output_dir)
        if scene_check.get("phase") != "BEFORE_INITIAL_RESET_AND_POLICY_RGB":
            raise ValueError("grey_studio_early_check_phase")
        validate_report(scene_check, self.baseline_config)

    def _images(self, *, render=True):
        from .render_profile import validate_renderer
        previous = deepcopy(self._guard("sensor_metadata_before_acquisition", self.dependencies.sensor_metadata))
        if render:
            self._guard("render_only", self.dependencies.render)
        images = {name: _rgb(self._guard("camera_update_zero_dt:" + name,
                    lambda name=name: self.dependencies.read_rgb(name))) for name in CAMERAS}
        metadata = self._guard("sensor_metadata", self.dependencies.sensor_metadata)
        renderer = self._guard("renderer_readback", self.dependencies.snapshot_renderer)
        self._last_renderer = validate_renderer(renderer, self.report["renderer"])
        if set(metadata) != set(CAMERAS):
            raise ValueError("stage0_both_sensor_provenance_records_required")
        for name, expected in CAMERAS.items():
            if metadata[name].get("prim_path") != expected:
                raise ValueError("stage0_sensor_provenance_identity:" + name)
            if type(metadata[name].get("frame")) is not int or metadata[name]["frame"] < 0:
                raise ValueError("stage0_sensor_frame_missing:" + name)
            if metadata[name]["frame"] <= previous[name]["frame"]:
                raise ValueError("stage0_sensor_frame_not_fresh:" + name)
            metadata[name]["source"] = "stage0_render_only_sensor_update/v1"
            metadata[name]["refresh_dt_s"] = 0.0
            metadata[name]["previous_frame"] = previous[name]["frame"]
            metadata[name]["acquisition_returned_monotonic_ns"] = time.monotonic_ns()
        return images, metadata

    def _save_recovery_pair(self, context, variant):
        from PIL import Image
        images, metadata = self._images(render=False)
        snapshot = self._guard("recovery_visuals:" + variant, self.dependencies.snapshot_visuals)
        if snapshot != self.initial_snapshot:
            raise ValueError("dome_recovery_camera_or_light_configuration_changed")
        record = dict(variant=variant, context=deepcopy(context), frozen_state=self._state(),
            sensor_metadata=metadata, acquisition_monotonic_ns=time.monotonic_ns(),
            readiness="render returned; forced zero-dt annotator read returned 240x320 RGB; "
                      "independent renderer convergence status UNKNOWN", images={})
        for name, pixels in images.items():
            path = self.output_dir / (variant + "_" + name + ".png")
            with path.open("xb") as stream:
                Image.fromarray(pixels).save(stream, format="PNG")
            record["images"][name] = _descriptor(path, self.output_dir)
        self._verify_dome_lights(variant)
        return record

    def _verify_dome_lights(self, label):
        from .dome_recovery import validate_lights
        lights = self._guard("dome_lights:" + label, self.dependencies.snapshot_lights)
        lights["phase"] = "POST_RESET_BEFORE_CAPTURE:" + label
        _write_json(self.output_dir / (label + "_lights.json"), lights)
        validate_lights(lights)

    def _checkpoint(self, context):
        from PIL import Image
        from .stage0_visuals import validate_export
        if self.recovery_enabled:
            self._verify_dome_lights("tick%04d" % context["control_tick"])
        self._render_diagnostic(context, "before_render")
        images, metadata = self._images()
        self._render_diagnostic(context, "after_render")
        snapshot = self._guard("checkpoint_visual_snapshot", self.dependencies.snapshot_visuals)
        validate_export(snapshot, self.baseline_config, initial_snapshot=self.initial_snapshot)
        if snapshot != self.initial_snapshot:
            raise ValueError("stage0_headless_visual_configuration_changed")
        record = dict(context=deepcopy(context), native_clock=list(self.session.clock()),
            frozen_state=deepcopy(self.frozen_state), sensor_metadata=metadata, images={},
            renderer=deepcopy(self._last_renderer))
        prefix = f"tick{context['control_tick']:04d}_substep{context['substep']}"
        for name, pixels in images.items():
            path = self.output_dir / (prefix + "_" + name + ".png")
            with path.open("xb") as stream:
                Image.fromarray(pixels).save(stream, format="PNG")
                stream.flush(); os.fsync(stream.fileno())
            with Image.open(path) as saved:
                if saved.mode != "RGB" or not np.array_equal(np.array(saved), pixels):
                    raise ValueError("stage0_lossless_RGB_verification_failed")
            record["images"][name] = dict(_descriptor(path, self.output_dir), shape=[240,320,3],
                dtype="uint8", encoding="PNG_lossless", pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
        snapshot_path = self.output_dir / (prefix + "_visuals.json")
        _write_json(snapshot_path, snapshot)
        record["visual_snapshot"] = _descriptor(snapshot_path, self.output_dir)
        self.report["checkpoints"].append(record)

    def _render_diagnostic(self, context, phase):
        """Separate passive sidecar, inside the existing frozen-clock guard."""
        prefix = f"tick{context['control_tick']:04d}_substep{context['substep']}"
        operation = "render_diagnostic:" + prefix + ":" + phase
        observed = self._guard(operation, self.dependencies.snapshot_render_diagnostic)
        path = self.output_dir / (prefix + "_" + phase + "_diagnostic.json")
        payload = dict(context=deepcopy(context), phase=phase,
                       frozen_state=self._state(), observation=observed)
        _write_json(path, payload)
        self.report["render_diagnostics"].append(dict(context=deepcopy(context),
            phase=phase, operation=operation, frozen_state=self._state(),
            artifact=_descriptor(path, self.output_dir)))

    def select_camera(self, name):
        try:
            if self.mode != "TUNE_GUI" or self.report["pause"] is None or name not in CAMERAS:
                raise ValueError("stage0_unknown_recording_camera")
            path = self._guard("select_recording_camera:" + name,
                              lambda: self.dependencies.select_camera(name))
            if path != CAMERAS[name]:
                raise ValueError("stage0_viewport_readback_disagreement")
            self.report["viewport_selections"].append(dict(name=name, prim_path=path,
                frozen_state=self._state(), wall_monotonic_ns=time.monotonic_ns()))
            return path
        except Exception as exc:
            self._error(exc)
            raise

    def set_light(self, name, value):
        try:
            if self.mode != "TUNE_GUI" or self.report["pause"] is None:
                raise ValueError("stage0_light_control_requires_frozen_GUI")
            if name not in ("intensity", "exposure") or not math.isfinite(float(value)) or (name == "intensity" and value < 0):
                raise ValueError("stage0_only_finite_Dome_intensity_and_exposure_controls")
            self._guard("edit_existing_Dome:" + name, lambda: self.dependencies.set_light(name, float(value)))
        except Exception as exc:
            self._error(exc)
            raise

    def export(self):
        try:
            if self.mode != "TUNE_GUI" or self.report["pause"] is None:
                raise ValueError("stage0_export_requires_frozen_GUI")
            return self._guard("export_visual_configuration", self._export)
        except Exception as exc:
            self._error(exc)
            raise

    def _export(self):
        from .stage0_visuals import validate_export
        snapshot = self.dependencies.snapshot_visuals()
        configuration = validate_export(snapshot, self.baseline_config, initial_snapshot=self.initial_snapshot)
        index = len(self.report["exports"])
        config_path = self.output_dir / f"export_{index:03d}_configuration.json"
        snapshot_path = self.output_dir / f"export_{index:03d}_snapshot.json"
        _write_json(snapshot_path, snapshot)
        _write_json(config_path, configuration)
        light_path = self.output_dir / f"export_{index:03d}_lights_gui_displayed.json"
        _write_json(light_path, dict(timing="export_time_readback_not_prior_display_acquisition",
            context=deepcopy(self.report["pause"]["context"]), frozen_state=self._state(),
            wall_monotonic_ns=time.monotonic_ns(), observation=self.dependencies.snapshot_lights()))
        record = dict(configuration=_descriptor(config_path, self.output_dir),
            snapshot=_descriptor(snapshot_path, self.output_dir), frozen_state=self._state(),
            approval_status="NOT_APPROVED", wall_monotonic_ns=time.monotonic_ns())
        record["lights_at_export"] = _descriptor(light_path, self.output_dir)
        record["displayed_preview"] = self._export_displayed_preview(index)
        self.report["exports"].append(record)
        self.dependencies.set_status("Exported " + str(config_path))
        return record

    def _export_displayed_preview(self, index):
        """Persist the last displayed acquisition, never refresh sensors on export."""
        if self._last_gui_preview is None:
            return dict(status="UNAVAILABLE", reason="no_completed_GUI_sensor_preview_before_export",
                        images={})
        from PIL import Image
        preview = self._last_gui_preview
        record = deepcopy(preview["provenance"])
        record["images"] = {}
        for name, pixels in preview["images"].items():
            path = self.output_dir / f"export_{index:03d}_displayed_{name}.png"
            with path.open("xb") as stream:
                Image.fromarray(pixels).save(stream, format="PNG")
                stream.flush(); os.fsync(stream.fileno())
            with Image.open(path) as saved:
                if saved.mode != "RGB" or not np.array_equal(np.array(saved), pixels):
                    raise ValueError("stage0_lossless_GUI_RGB_verification_failed")
            record["images"][name] = dict(_descriptor(path, self.output_dir), shape=[240,320,3],
                dtype="uint8", encoding="PNG_lossless", pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
        return record

    def _show_gui_preview(self, images, metadata, context):
        # Copy before handing buffers to the UI. Publication occurs only after
        # both previews were supplied successfully; no extra native acquisition.
        copied = {name: _rgb(pixels) for name, pixels in images.items()}
        provenance = dict(status="AVAILABLE", source="stage0_last_displayed_GUI_sensor_RGB/v1",
            context=deepcopy(context), frozen_state=deepcopy(self.frozen_state),
            native_clock=list(self.frozen_state["native_clock"]), sensor_metadata=deepcopy(metadata),
            renderer=deepcopy(self._last_renderer), preview_iteration=self.report["pause"]["iterations"]+1,
            display_requested_monotonic_ns=time.monotonic_ns(),
            export_snapshot_alignment="last_displayed_acquisition_not_necessarily_export_snapshot")
        self._guard("sensor_RGB_preview", lambda: self.dependencies.show_previews(images))
        provenance["display_returned_monotonic_ns"] = time.monotonic_ns()
        self._last_gui_preview = dict(images=copied, provenance=provenance)

    def request_quit(self):
        self.quit_requested = True

    def _error(self, exc):
        message = type(exc).__name__ + ":" + str(exc)
        if message not in self.report["errors"]:
            self.report["errors"].append(message)

    def boundary(self, context, capture):
        context = asdict(context) if is_dataclass(context) else dict(context)
        position = (context["control_tick"], context["substep"])
        if position not in CHECKPOINTS:
            return False
        self.capture = capture
        if self.dependencies is None:
            self.dependencies = _NativeDependencies(self.session)
        self.frozen_state = self._state()
        try:
            if self.initial_snapshot is None:
                if position != CHECKPOINTS[0]:
                    raise ValueError("stage0_initial_0_1_boundary_missing")
                self._initialize()
            if self.mode == "CHECK_HEADLESS":
                expected = CHECKPOINTS[len(self.report["checkpoints"])]
                if position != expected:
                    raise ValueError("stage0_checkpoint_order")
                self._checkpoint(context)
                if self.recovery_enabled and position == (0, 1):
                    from .dome_recovery import compare
                    compare(self, context)
                return False
            if position != (839, 1):
                return False
            self.report["pause"] = dict(context=context, start=deepcopy(self.frozen_state), end=None,
                method="blocked_scheduled_callback_render_only", iterations=0, explicit_quit=False)
            snapshot = self._guard("pause_visual_snapshot", self.dependencies.snapshot_visuals)
            self._guard("open_ui", lambda: self.dependencies.open_ui(self, snapshot))
            self.select_camera("camera_left")
            while not self.quit_requested:
                if not self.dependencies.is_running():
                    raise RuntimeError("stage0_application_closed_without_explicit_quit")
                images, metadata = self._images()
                self._show_gui_preview(images, metadata, context)
                self.report["pause"]["iterations"] += 1
                if self.report["errors"]:
                    raise RuntimeError("stage0_UI_operation_failed")
            self.report["pause"].update(end=self._state(), explicit_quit=True)
            self._guard("close_ui", self.dependencies.close_ui)
            return True
        except Exception as exc:
            self._error(exc)
            raise

    def finish(self):
        """Persist before native cleanup. This report never constitutes approval."""
        if self._finished is not None:
            return deepcopy(self._finished)
        if self.report["errors"]:
            status = "FAIL"
        elif self.mode == "CHECK_HEADLESS" and len(self.report["checkpoints"]) == 3:
            status = "HEADLESS_CHECKPOINTS_COMPLETE"
        elif self.report["pause"] and self.report["pause"]["explicit_quit"]:
            status = "GUI_QUIT_EXPORTED" if self.report["exports"] else "GUI_QUIT_NO_EXPORT"
        else:
            status = "INCOMPLETE"
        self.report["status"] = status
        _write_json(self.output_dir / "report.json", self.report)
        self._finished = deepcopy(self.report)
        return deepcopy(self._finished)
