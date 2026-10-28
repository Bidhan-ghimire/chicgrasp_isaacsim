"""Stage0 owner-authorized light isolation and bounded render-only comparison.

No native imports at module import. The shared scene callback owns the overrides;
this module never edits the exported rig or the robot/chicken asset files.
"""
from copy import deepcopy
import json
from pathlib import Path
import time

INACTIVE_PATHS = ("/World/DomeLight", "/World/envs/env_0/Chicken/env_light")
RIG_DOME = "/Environment/Grey_Studio/DomeLight"
PROFILE = {"schema": "stage0-single-dome/v1", "inactive_paths": list(INACTIVE_PATHS),
           "startup_refresh": "NONE", "startup_additional_renders": 0}


def apply_overrides(stage, *, edit_context=None):
    """After scene spawning, before context attachment/initial reset rendering."""
    if edit_context is None:
        from pxr import Usd
        edit_context = Usd.EditContext
    layer = stage.GetRootLayer()
    if not layer.anonymous:
        raise ValueError("dome_overrides_require_task_owned_anonymous_scene_layer")
    for path in INACTIVE_PATHS:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or prim.GetTypeName() != "DomeLight":
            raise ValueError("dome_override_target_missing_or_not_dome:" + path)
        if not prim.GetParent().IsActive():
            raise ValueError("dome_override_parent_inactive:" + path)
    with edit_context(stage, layer):
        for path in INACTIVE_PATHS:
            if not stage.GetPrimAtPath(path).SetActive(False):
                raise ValueError("dome_override_write_failed:" + path)
    if any(stage.GetPrimAtPath(path).IsActive() for path in INACTIVE_PATHS):
        raise ValueError("dome_override_not_composed")


def validate_lights(inventory):
    lights = {item["path"]: item for item in inventory["lights"]}
    for path in INACTIVE_PATHS:
        if path not in lights or lights[path]["active"] is not False:
            raise ValueError("dome_inactive_override_missing:" + path)
    rig = lights.get(RIG_DOME, {})
    if (rig.get("active") is not True or
            rig.get("computed_visibility", {}).get("value") != "inherited"):
        raise ValueError("grey_studio_dome_not_effectively_visible")
    # Texture absence is valid; intensity/material/transforms are verified by
    # the existing exact-export checker, not guessed from a desired appearance.
    return inventory


def wait_decision(directory, name, *, timeout_s=180):
    """Hold the existing callback, without app updates, stepping or resetting."""
    path = Path(directory) / (name + "_decision.json")
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("dome_review_decision_missing:" + str(path))
        time.sleep(.2)
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("action") not in ("RESUME", "WARMUP") or not decision.get("reason"):
        raise ValueError("dome_review_decision_invalid")
    return decision


def compare(runtime, context):
    """A already uses normal checkpoint preparation; C adds exactly 30 renders."""
    from .stage0_camera_runtime import _write_json
    report = dict(schema="stage0-dome-comparison/v1", profile=deepcopy(PROFILE),
                  physics_boundary=deepcopy(runtime.frozen_state),
                  A=deepcopy(runtime.report["checkpoints"][0]),
                  B={"status": "NOT_NEEDED"}, C={"status": "NOT_NEEDED"},
                  transitions=[], acceptance_eligible=False)
    path = runtime.output_dir / "dome_A.json"
    _write_json(path, report)
    decision = runtime._guard("A_inspection_wait", lambda: wait_decision(runtime.output_dir, "A"))
    report["A_decision"] = decision
    if decision["action"] == "WARMUP":
        # Installed handler is stage-context/global and rewrites visibility of
        # all lights. Its applicability to the two sensor products is unproven.
        # Do not impersonate a headless viewport or merely toggle its global flag.
        report["B"] = dict(status="NOT_TESTED", reason=
            "Installed _set_lighting_mode uses context-wide session-layer VisibilityEdit and "
            "/rtx/useViewLightingMode; equivalent sensor-render-product scope not established.",
            source="omni.kit.viewport.menubar.lighting-107.3.1+107.3.0/actions.py:96-256")
        report["C"] = dict(status="TESTED", additional_updates=30, frames=[])
        for count in range(1, 31):
            runtime._guard("warmup_render:" + str(count), runtime.dependencies.render)
            if count in (1, 10, 30):
                record = runtime._save_recovery_pair(context, "C_%02d" % count)
                record["additional_render_updates"] = count
                report["C"]["frames"].append(record)
                _write_json(runtime.output_dir / ("dome_C_%02d.json" % count), record)
        report["C_decision"] = runtime._guard("C_inspection_wait",
            lambda: wait_decision(runtime.output_dir, "C"))
        if report["C_decision"]["action"] != "RESUME":
            raise ValueError("dome_comparison_cannot_repeat_warmup")
    _write_json(runtime.output_dir / "dome_comparison.json", report)
    runtime.report["dome_comparison"] = report
    return report


def check_comparison(directory, report):
    """Independent disk/frame/pose verification, including C's exact render count."""
    import hashlib
    from PIL import Image
    comparison = report.get("dome_comparison")
    if comparison is None:
        return  # Historical reports remain readable; they prove no recovery.
    if comparison["profile"] != PROFILE or comparison["transitions"]:
        raise ValueError("dome_profile_or_unapproved_transition")
    if comparison["A"] != report["checkpoints"][0]:
        raise ValueError("dome_A_not_initial_checkpoint")
    boundary = comparison["physics_boundary"]
    if "native_scene_poses" not in boundary or not boundary["native_scene_poses"]:
        raise ValueError("dome_native_pose_guard_missing")
    if comparison["B"]["status"] not in ("NOT_NEEDED", "NOT_TESTED"):
        raise ValueError("dome_refresh_not_supported_by_this_revision")
    frames = comparison["C"].get("frames", [])
    renders = [g for g in report["guard_samples"] if g["operation"].startswith("warmup_render:")]
    if comparison["C"]["status"] == "TESTED":
        if ([f["additional_render_updates"] for f in frames] != [1,10,30] or
                [g["operation"] for g in renders] != ["warmup_render:"+str(i) for i in range(1,31)]):
            raise ValueError("dome_warmup_count")
    elif frames or renders:
        raise ValueError("dome_unreported_warmup")
    previous = comparison["A"]["sensor_metadata"]
    for frame in frames:
        if frame["frozen_state"] != boundary:
            raise ValueError("dome_comparison_pose_or_clock_changed")
        for name in ("camera_left", "camera_right"):
            sensor = frame["sensor_metadata"][name]
            if sensor["frame"] <= previous[name]["frame"]:
                raise ValueError("dome_stale_comparison_frame")
            descriptor = frame["images"][name]
            path = (Path(directory) / descriptor["path"]).resolve()
            if not path.is_relative_to(Path(directory).resolve()):
                raise ValueError("dome_image_path_escape")
            raw = path.read_bytes()
            if len(raw) != descriptor["bytes"] or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
                raise ValueError("dome_image_integrity")
            with Image.open(path) as image:
                if image.mode != "RGB" or image.size != (320,240):
                    raise ValueError("dome_image_shape")
        previous = frame["sensor_metadata"]
    for guard in renders:
        if guard["before"] != boundary or guard["after"] != boundary:
            raise ValueError("dome_warmup_pose_or_clock_changed")
    for tick in (0,839,1199):
        validate_lights(json.loads((Path(directory)/('tick%04d_lights.json'%tick)).read_text()))
