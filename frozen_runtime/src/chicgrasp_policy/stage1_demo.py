"""Stage1 recording connection and independent checks using existing episode APIs.

No alternate loader, native replay, training, physics or renderer setting writes.
"""
from copy import deepcopy
import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import episode_io
from .attachment_capture import frozen_candidate_actions
from .interface import POLICY_OBSERVATION_KEYS
from .lift_event_pose_evidence import _json

PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT / "config/policy_camera_calibration.json"
APPROVAL = PROJECT / "docs/grants/GATE6_STAGE0_VISUAL_APPROVAL_20261012.json"
SCALED_PROFILE = PROJECT / "docs/plans/CHICGRASP_SCALED_TABLE_AND_COVERAGE/attempt005/SCENE_PROFILE.json"
SCALED_PROFILE_SHA256 = "4f40d26e122c99ea5cb980edea4a391b5335a99fe060b84cacd40305374c0d5f"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(_json(value), stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def append(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(_json(value), allow_nan=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class DemoWriter:
    """Persist the same acquired fields before the associated action executes."""
    def __init__(self, rgb, output):
        self.rgb, self.output = rgb, Path(output)
        self.actions = frozen_candidate_actions()
        for name in ("policy_rows", "substeps"):
            with (self.output / ("stage1_" + name + ".jsonl")).open("x") as stream:
                stream.flush()
                os.fsync(stream.fileno())

    def write_pre_candidate(self, mapping, metadata):
        return self.rgb.write_pre_candidate(mapping, metadata)

    def write_bundle(self, action_frame, mapping, metadata):
        bundle = self.rgb.write_bundle(action_frame, mapping, metadata)
        append(self.output / "stage1_policy_rows.jsonl", dict(action_frame=action_frame,
            timestamp_ns=metadata["policy_timestamp_ns"],
            arm_joint_pos=np.asarray(mapping["arm_joint_pos"]),
            jaw_state=np.asarray(mapping["jaw_state"]),
            action=self.actions[action_frame], acquisition=deepcopy(metadata),
            rgb_bundle=f"rgb/{action_frame:06d}.json"))
        return bundle

    def write_substep(self, row):
        append(self.output / "stage1_substeps.jsonl", row)

    def finish(self, completed, primary_errors):
        return self.rgb.finish(completed, primary_errors)


def approved_scaled_profile():
    """Bind the specific owner-approved Grant2 publication using existing identities."""
    from .geometric_capture_check import identity
    if identity(SCALED_PROFILE)["sha256"] != SCALED_PROFILE_SHA256:
        raise ValueError("scaled_owner_profile_identity")
    profile = load(SCALED_PROFILE)
    if (profile["owner_review"] != "APPROVED" or
            profile["profile_id"] != "scaled-table-path-b-attempt005" or
            profile["executed_commit"] != "f90cf37e7495bf7429e391e050a216ff28d7be93"):
        raise ValueError("scaled_owner_profile_approval")
    for item in [profile["calibration_identity"], profile["physics_source_identity"],
                 *profile["asset_root_identities"]]:
        if identity(PROJECT/item["path"]) != {k:item[k] for k in ("bytes", "sha256")}:
            raise ValueError("scaled_owner_source_identity:"+item["path"])
    if load(CONFIG)["scaled_scene_approval"] != profile["approval"]:
        raise ValueError("scaled_owner_configuration_approval")
    return profile


def save_visuals(runtime, output, phase, *, dependencies=None, scene_profile=False):
    """Read only, without additional acquisition, render, reset or stepping."""
    from .stage0_camera_runtime import _NativeDependencies
    from .grey_studio_check import snapshot
    from .dome_recovery import PROFILE
    deps = dependencies or _NativeDependencies(runtime)
    calibration, approval = load(CONFIG), load(APPROVAL)
    before = list(runtime.clock())
    record = dict(phase=phase, clock_before=before,
        config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        calibration=calibration, approval=approval, startup_profile=deepcopy(PROFILE),
        visuals=deps.snapshot_visuals(), renderer=deps.snapshot_renderer(),
        lights=deps.snapshot_lights(), render_diagnostic=deps.snapshot_render_diagnostic(),
        grey_studio=snapshot(runtime.stage, calibration))
    record["clock_after"] = list(runtime.clock())
    if scene_profile:
        profile = approved_scaled_profile()
        if runtime.effective_configuration != profile["physical_configuration"]:
            raise ValueError("scaled_effective_configuration")
        record["scene_profile"] = dict(path=SCALED_PROFILE.relative_to(PROJECT).as_posix(),
            sha256=SCALED_PROFILE_SHA256, profile_id=profile["profile_id"])
    write(Path(output) / ("stage1_visuals_" + phase + ".json"), record)
    validate_visuals(record)
    return record


def validate_visuals(record):
    from .stage0_visuals import validate_export, config_snapshot
    from .render_profile import validate_renderer
    from .grey_studio_check import validate_report
    from .dome_recovery import validate_lights, PROFILE
    if record["clock_before"] != record["clock_after"]:
        raise ValueError("stage1_visual_read_advanced_physics")
    calibration, approval = record["calibration"], record["approval"]
    if record.get("scene_profile") is not None:
        profile = approved_scaled_profile()
        if (record["scene_profile"] != dict(path=SCALED_PROFILE.relative_to(PROJECT).as_posix(),
                sha256=SCALED_PROFILE_SHA256, profile_id=profile["profile_id"]) or
                record["config_sha256"] != profile["calibration_identity"]["sha256"] or
                calibration != load(CONFIG) or approval != load(APPROVAL) or
                record["startup_profile"] != PROFILE or
                record["visuals"]["light"] != profile["native_final_visuals"]["light"]):
            raise ValueError("scaled_approved_configuration_identity")
        renderer = profile["renderer_and_exposure_readback"]
        initial_visuals = profile["native_final_visuals"]
    elif (record["config_sha256"] != approval["approved_config_sha256"]
            or calibration != load(CONFIG) or approval != load(APPROVAL)
            or record["startup_profile"] != PROFILE
            or calibration["calibration_status"] != "STAGE0_OWNER_APPROVED"):
        raise ValueError("stage1_approved_configuration_identity")
    else:
        renderer = approval["renderer"]
        initial_visuals = record["visuals"]
    # Keep the existing native/config transform representation check below;
    # recomposed quaternion matrices need not be byte-identical across starts.
    # The frozen native optics and parent transforms remain exact in validate_export.
    validate_export(record["visuals"], calibration, initial_snapshot=initial_visuals)
    expected = config_snapshot(calibration)
    for name in expected["cameras"]:
        for key in ("local_matrix", "world_matrix", "parent_world_matrix"):
            if not np.allclose(record["visuals"]["cameras"][name][key],
                               expected["cameras"][name][key], atol=2e-7, rtol=0):
                raise ValueError("stage1_camera_transform:" + name + ":" + key)
    validate_renderer(record["renderer"], renderer)
    validate_lights(record["lights"])
    if record["lights"].get("status") != "COMPLETE" or not approved_lights_match(record["lights"]["lights"], approval):
        raise ValueError("stage1_approved_lights_changed_or_incomplete")
    validate_report(record["grey_studio"], calibration)


def approved_lights_for_worktree(approval):
    """Permit only the identical copied inactive texture's resolved path relocation."""
    expected = deepcopy(approval["approved_lights"])
    texture = approval["inactive_dome_texture"]
    target = (PROJECT / texture["relative_path"]).resolve()
    if (not target.is_relative_to(PROJECT) or target.stat().st_size != texture["bytes"]
            or hashlib.sha256(target.read_bytes()).hexdigest() != texture["sha256"]):
        raise ValueError("stage1_copied_texture_identity")
    item = next(v for v in expected if v["path"] == "/World/envs/env_0/Chicken/env_light")
    value = item["attributes"]["inputs:texture:file"]["value"]["value"]
    call = ast.parse(value["representation"], mode="eval").body
    if not isinstance(call, ast.Call) or ast.unparse(call.func) != "Sdf.AssetPath" or len(call.args) != 2 or call.keywords:
        raise ValueError("stage1_approved_texture_representation")
    authored, resolved = [ast.literal_eval(arg) for arg in call.args]
    if authored != texture["authored_path"] or Path(resolved) != Path(texture["source_path"]):
        raise ValueError("stage1_approved_texture_binding")
    value["representation"] = f"Sdf.AssetPath({authored!r}, {str(target)!r})"
    return expected


def approved_lights_match(recorded, approval):
    """Resolve only the frozen inactive texture alias; preserve all other facts."""
    expected = approved_lights_for_worktree(approval)
    if recorded == expected:
        return True
    actual = deepcopy(recorded)
    light_path = "/World/envs/env_0/Chicken/env_light"
    items = [item for item in actual if item.get("path") == light_path]
    if len(items) != 1:
        return False
    try:
        value = items[0]["attributes"]["inputs:texture:file"]["value"]["value"]
        call = ast.parse(value["representation"], mode="eval").body
        if (not isinstance(call, ast.Call) or ast.unparse(call.func) != "Sdf.AssetPath"
                or len(call.args) != 2 or call.keywords):
            return False
        authored, resolved = [ast.literal_eval(arg) for arg in call.args]
        texture = approval["inactive_dome_texture"]
        target = (PROJECT / texture["relative_path"]).resolve(strict=True)
        if (not isinstance(authored, str) or not isinstance(resolved, str)
                or authored != texture["authored_path"] or not Path(resolved).is_absolute()
                or Path(resolved).resolve(strict=True) != target
                or not Path(resolved).samefile(target)):
            return False
        wanted = next(item for item in expected if item["path"] == light_path)
        value["representation"] = wanted["attributes"]["inputs:texture:file"]["value"]["value"]["representation"]
    except (KeyError, TypeError, ValueError, SyntaxError, OSError):
        return False
    return actual == expected


def finalize_episode(capture, output):
    arrays, metadata, diagnostics = capture.episode()
    root = episode_io.write_episode(Path(output) / "episode", arrays,
        episode_end=metadata, diagnostic_frames=diagnostics)
    reloaded = episode_io.load_policy_episode(root)
    for name in episode_io.POLICY_ARRAY_KEYS:
        if not np.array_equal(arrays[name], reloaded[name]):
            raise ValueError("stage1_episode_roundtrip:" + name)
    return dict(path="episode", rows=len(reloaded["action"]),
                schema=metadata["schema_version"], roundtrip="PASS")


def check_startup_attempt(root, result, process, invocation, inherited_issues):
    """Explicit HARNESS startup only; it can never produce demo acceptance."""
    from .geometric_capture_check import identity
    root = Path(root)
    output = root / "capture/experiment"
    issues, process_issues = list(inherited_issues), []
    try:
        if (result.get("run_type") != "HARNESS" or process.get("run_type") != "HARNESS"
                or process.get("stage1_demo") is not True or result.get("acceptance_eligible") is not False
                or result.get("candidate_started") is not False or result.get("attachment_authoring_attempted") is not False
                or result.get("script_status") != "PASS"
                or result.get("experiment_result") != dict(status="NOT_EVALUABLE", reason="STAGE1_STARTUP_ONLY", acceptance_eligible=False)
                or result.get("counts") != dict(completed_control_ticks=0,observed_substeps=0,completed_policy_actions=0)
                or any(result.get(k) for k in ("primary_errors","recording_errors","cleanup_errors"))):
            raise ValueError("stage1_startup_scope_or_saved_failure")
        if (output / "episode").exists() or list((output / "rgb").glob("*.png")):
            raise ValueError("stage1_startup_must_not_record_demo")
        for filename in ("stage1_policy_rows.jsonl", "stage1_substeps.jsonl"):
            if (output / filename).read_bytes():
                raise ValueError("stage1_startup_command_or_substep_record")
        record = load(output / "stage1_visuals_before_candidate.json")
        validate_visuals(record)
        measurement = load(output / "measurements.json")
        if (measurement["rows"] or measurement["sensor_samples"] or measurement["lifecycle_events"]
                or record["clock_before"] != measurement["origin_clock"]):
            raise ValueError("stage1_startup_not_pre_candidate")
        events = [json.loads(line) for line in (root / "capture/structured/events.jsonl").read_text().splitlines()]
        if (any(e["kind"] == "CANDIDATE_REQUESTED" for e in events)
                or sum(e["kind"] == "STAGE1_STARTUP_CHECKED_NO_CANDIDATE" for e in events) != 1
                or [e["kind"] for e in events[-2:]] != ["PRE_CLOSE_EVIDENCE_COMPLETE","CLOSE_REQUESTED"]
                or any(e["manifest"] != identity(output/"artifact_manifest.json") for e in events[-2:])):
            raise ValueError("stage1_startup_journal")
        for field, relative in (("stdout","stdout.bin"),("stderr","stderr.bin"),("completed_journal","structured/events.jsonl")):
            item = process[field]
            path = root / "capture" / relative
            if {k:item[k] for k in ("bytes","sha256")} != identity(path) or Path(item["path"]).resolve() != path.resolve():
                raise ValueError("stage1_startup_stream:"+field)
        if (process["invocation_sha256"] != identity(root/"invocation.json")["sha256"]
                or result["execution_provenance"]["executed_commit"] != invocation["executed_commit"]):
            raise ValueError("stage1_startup_execution_identity")
    except Exception as exc:
        issues.append(type(exc).__name__+":"+str(exc))
    expected = dict(exit_code=0, root_exit_verified=True, tree_empty_verified=True, capture_complete=True,
        forced_termination=False, timed_out=False, parent_termination_requested=False, kill_on_job_close_used=False,
        kill_on_job_close_possible=False, startup_containment_used=False, autonomous_tree_exit=True,
        job_assigned_before_resume=True, root_resumed=True, job_disposed_after_observation=True,
        job_members_before_disposal=0, absolute_deadline_ms=1200000, parent_launch_boundary_ms=0,
        termination_and_drain_cap_ms=5000)
    for key, wanted in expected.items():
        if type(process.get(key)) is not type(wanted) or process[key] != wanted:
            process_issues.append("process:"+key)
    for key in ("root_exit_observed_ms","job_empty_observed_ms","drain_completed_observed_ms"):
        if type(process.get(key)) is not int or not 0 <= process[key] <= 1200000:
            process_issues.append("deadline:"+key)
    if process.get("containment_errors") != []:
        process_issues.append("containment_errors")
    return dict(schema="stage1-startup-check/v1", status="HARNESS_COMPLETE" if not issues and not process_issues else "FAIL",
        CAPTURE_INTEGRITY="PASS" if not issues else "FAIL", PROCESS_COMPLETION="PASS" if not process_issues else "NOT_CONFIRMED",
        DEMO_INTEGRITY="NOT_RECORDED", END_STATE_RESULT="NOT_EVALUABLE", acceptance_eligible=False,
        candidate_commands=0, issues=issues, process_issues=process_issues, native_launch_authority="NONE")


def check_demo(output):
    """Recompute demo integrity from saved arrays, acquisitions and PNG pixels."""
    from PIL import Image
    from .overfit_data import make_training_windows
    output = Path(output)
    issues, details = [], {}
    try:
        arrays = episode_io.load_policy_episode(output / "episode")
        metadata = episode_io.load_episode_end(output / "episode")
        measurement = load(output / "measurements.json")
        rows = [json.loads(line) for line in (output / "stage1_policy_rows.jsonl").read_text().splitlines()]
        substeps = [json.loads(line) for line in (output / "stage1_substeps.jsonl").read_text().splitlines()]
        if len(rows) != 200 or len(substeps) != 2400 or substeps != measurement["rows"]:
            raise ValueError("stage1_incremental_rows_incomplete_or_disagree")
        if metadata["episode_id"] != measurement["episode_id"] or metadata["num_policy_steps"] != 200:
            raise ValueError("stage1_episode_identity_or_count")
        if not np.array_equal(arrays["action"], frozen_candidate_actions()):
            raise ValueError("stage1_candidate_changed")
        if len(measurement["sensor_samples"]) != 200:
            raise ValueError("stage1_acquisition_count")
        for i, row in enumerate(rows):
            sensor = measurement["sensor_samples"][i]
            if (row["action_frame"] != i or row["timestamp_ns"] != int(arrays["timestamp_ns"][i])
                    or row["acquisition"] != sensor or row["rgb_bundle"] != f"rgb/{i:06d}.json"):
                raise ValueError("stage1_pre_action_provenance:" + str(i))
            for name in ("arm_joint_pos", "jaw_state", "action"):
                if not np.array_equal(arrays[name][i], np.asarray(row[name], dtype=arrays[name].dtype)):
                    raise ValueError("stage1_incremental_array_disagreement:" + name + ":" + str(i))
            expected_jaw = [1, 1] if i == 0 else arrays["action"][i-1, 3:]
            if not np.array_equal(arrays["jaw_state"][i], expected_jaw) or row["jaw_state"] != sensor["jaw_state"]["bits"]:
                raise ValueError("stage1_jaw_not_already_active:" + str(i))
            bundle = load(output / row["rgb_bundle"])
            if bundle["metadata"] != sensor:
                raise ValueError("stage1_bundle_acquisition_disagreement:" + str(i))
            for camera in ("camera_left", "camera_right"):
                with Image.open(output / "rgb" / f"{i:06d}_{camera}.png") as image:
                    pixels = np.asarray(image)
                if not np.array_equal(pixels, arrays[camera + "_rgb"][i]):
                    raise ValueError("stage1_image_array_disagreement:" + camera + ":" + str(i))
        if not np.all(arrays["jaw_state"][140:] == 0):
            raise ValueError("stage1_lift_hold_not_commanded_closed")
        windows = make_training_windows(arrays)
        indices = np.arange(-1, 196)[:, None] + np.arange(16)
        valid = (indices >= 0) & (indices < 200)
        if (tuple(windows["obs"]) != POLICY_OBSERVATION_KEYS
                or windows["action"].shape != (197, 16, 5)
                or not np.array_equal(windows["valid_mask"], valid)
                or not np.array_equal(windows["source_indices"], np.where(valid, indices, -1))
                or not np.array_equal(windows["observation_indices"], indices[:, :2].clip(0, 199))
                or not np.array_equal(windows["action"], arrays["action"][indices.clip(0, 199)])):
            raise ValueError("stage1_unchanged_loader_windows_or_mask")
        for name in POLICY_OBSERVATION_KEYS:
            expected = arrays[name][indices[:, :2].clip(0, 199)]
            if name.endswith("_rgb"):
                expected = expected.transpose(0, 1, 4, 2, 3).astype(np.float32) / np.float32(255)
            if windows["obs"][name].dtype != np.float32 or not np.array_equal(windows["obs"][name], expected.astype(np.float32)):
                raise ValueError("stage1_window_observation:" + name)
        for phase, clock in (("before_candidate", measurement["origin_clock"]),
                             ("after_candidate", measurement["rows"][-1]["pose"]["clock_after"])):
            record = load(output / ("stage1_visuals_" + phase + ".json"))
            validate_visuals(record)
            if record["phase"] != phase or record["clock_before"] != clock:
                raise ValueError("stage1_visual_clock:" + phase)
        details = dict(policy_rows=200, substep_rows=2400, rgb_bundles=201, rgb_images=402,
            training_windows=197, observation_history=2, prediction_horizon=16,
            valid_target_slots=int(valid.sum()), startup_observation_indices=[0, 0],
            loaders=["episode_io.load_policy_episode", "overfit_data.make_training_windows"],
            free_space_validator="UNCHANGED_NOT_APPLICABLE", training_executed=False,
            native_replay_executed=False)
    except Exception as exc:
        issues.append(type(exc).__name__ + ":" + str(exc))
    return dict(status="PASS" if not issues else "FAIL", issues=issues, **details)
