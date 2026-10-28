"""Attachment-free research recording, separate from original HC scoring.

Scheduler provenance: frozen_runtime/scripts/policy/attachment_assisted_lift_record.py
CandidateCapture.run (238-323), and geometric_capture.py boundary clock/command
checks. Retains 10 Hz observations, six 60 Hz control ticks and two scheduled
120 Hz physics substeps per tick. No scene writes, extra steps, or owner binding.
Native imports are deferred; no policy/model/Kit starts occur on import.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math
import sys
import time


@dataclass(frozen=True)
class Context:
    episode_id: str
    control_tick: int
    substep: int


class CustomCapture:
    def __init__(self, core, runtime, writer, action_source, *, dependencies=None):
        if runtime.owner is not None:
            raise ValueError("Custom capture cannot bind a gripper attachment owner")
        if getattr(action_source, "total_actions", None) != 200:
            raise ValueError("Custom recording currently requires exactly 200 actions")
        if dependencies is None:
            import numpy as np
            from chicgrasp_policy import interface
        else:
            np, interface = dependencies
        self.np, self.interface = np, interface
        self.core, self.runtime, self.rgb_writer, self.action_source = core, runtime, writer, action_source
        self.rows, self.command_frames, self.sensor_samples, self.diagnostics = [], [], [], []
        self.observations, self.executed_actions, self.lifecycle_events, self.recording_errors = [], [], [], []
        self.tick, self.substep, self.control_ticks = -1, 0, 0
        self.context, self.current_action = None, None
        self.last_clock = tuple(runtime.origin_clock)
        self.phase, self.completed = "initialized", False

    def lifecycle(self, flags):
        if any(flags.values()):
            if self.rows:
                self.rows[-1]["runtime"].update(flags)
            self.lifecycle_events.append(dict(context=asdict(self.context) if self.context else None,
                flags=deepcopy(flags), before_automatic_reset=True))
            raise self.core.CaptureFailure("unexpected_termination_or_truncation_before_reset")

    def pre_reset(self):
        self.lifecycle_events.append(dict(context=asdict(self.context) if self.context else None,
            resetting=True, before_automatic_reset=True))
        raise self.core.CaptureFailure("unexpected_reset_before_reset")

    def boundary(self):
        t, s = self.tick, self.substep
        if not (0 <= t < 1200 and s in (0, 1)) or len(self.rows) != 2*t+s:
            raise self.core.CaptureFailure("extra_or_out_of_order_scheduled_boundary")
        self.context = Context(self.runtime.episode_id, t, s)
        clock = tuple(self.runtime.clock())
        row = dict(context=asdict(self.context), wall_monotonic_ns=time.monotonic_ns(),
            pose=None, jaws=None, commands=self.current_action[3:].tolist(),
            attachment=dict(status="DISABLED_CUSTOM_SCENE", gripper_attachment=False),
            runtime=dict(resetting=False, truncated=False, terminated=False, invalid=False), issues=[])
        self.rows.append(row)
        try:
            if self.runtime.owner is not None:
                raise self.core.CaptureFailure("custom_gripper_attachment_owner_unexpected")
            if (type(clock[0]) is not int or clock[0] != self.runtime.origin_clock[0]+len(self.rows)
                    or isinstance(clock[1], bool) or not math.isfinite(clock[1])
                    or clock[1] <= self.last_clock[1]):
                raise self.core.CaptureFailure("missing_stale_or_nonfinite_scheduled_clock")
            self.last_clock = clock
            row["pose"] = self.core._valid_pose(self.runtime.measure(), clock)
            row["jaws"] = self.runtime.read_jaws()
            if row["jaws"]["clock_before"] != list(clock) or row["jaws"]["clock_after"] != list(clock):
                raise self.core.CaptureFailure("jaw_measurement_not_current")
            if t % 6 == 0 and s == 0:
                command = self.core.RuntimeSession.commanded_jaw_metadata(self.runtime, self.current_action[3:])
                if command.get("bits") != self.current_action[3:].tolist():
                    raise self.core.CaptureFailure("custom_command_not_accepted")
                row["accepted_command"] = command
                self.rgb_writer.command_event("ACCEPTANCE_OBSERVED", t//6,
                    scheduled_boundary=[t, s], accepted_command=deepcopy(command),
                    meaning="accepted_action_term_observed_after_scheduled_substep; exact_issue_time_unknown")
                self.command_frames.append(dict(action_frame=t//6, commands=row["commands"],
                    sample=dict(episode_id=self.runtime.episode_id, control_tick=t, substep=s,
                        physics_step=clock[0], simulation_time_s=clock[1]),
                    source="executed_custom_command_first_post_substep/v1"))
            self.substep += 1
        except Exception as exc:
            row["runtime"]["invalid"] = True
            row["issues"].append(type(exc).__name__+":"+str(exc))
            raise
        finally:
            primary = sys.exc_info()[1]
            try:
                self.rgb_writer.write_substep(row)
            except Exception as exc:
                self.recording_errors.append(type(exc).__name__+":"+str(exc))
                if primary is None:
                    raise

    def run(self):
        if self.phase != "initialized":
            raise self.core.CaptureFailure("custom_candidate_cannot_retry")
        self.runtime.set_hooks(self.boundary, self.lifecycle, self.pre_reset)
        self.phase = "pre_candidate"
        prior_target = None
        for index in range(200):
            observation = self.interface.validate_policy_observation(self.runtime.observation())
            before = self.core._valid_pose(self.runtime.measure(), self.runtime.clock())
            sensor = dict(action_frame=index, native_clock=list(self.runtime.clock()),
                sensor_frames=self.runtime.camera_frames(), wall_monotonic_ns=time.monotonic_ns(),
                policy_timestamp_ns=index*100_000_000, control_tick=6*index,
                camera_metadata=self.runtime.camera_metadata())
            jaw = self.core.RuntimeSession.commanded_jaw_metadata(self.runtime, observation.jaw_state)
            if index == 0:
                if jaw["bits"] != [1, 1]:
                    raise self.core.CaptureFailure("initial_jaw_state_not_reset_open")
                jaw["preceding_command"] = dict(kind="reset_open", action_frame=None,
                    control_tick=None, substep=None, observed_native_clock=None, observed_wall_monotonic_ns=None)
            else:
                prior = self.rows[-1]
                if jaw["bits"] != self.command_frames[-1]["commands"]:
                    raise self.core.CaptureFailure("jaw_state_not_preceding_accepted_command")
                jaw["preceding_command"] = dict(kind="accepted_control_action", action_frame=index-1,
                    control_tick=prior["context"]["control_tick"], substep=prior["context"]["substep"],
                    observed_native_clock=deepcopy(prior["pose"]["clock_after"]),
                    observed_wall_monotonic_ns=prior["wall_monotonic_ns"])
            sensor["jaw_state"] = jaw
            if index == 0:
                pre = self.runtime.pre_candidate_observation(before)
                sensor["shared_acquisition_action_frame"] = 0
                pre_metadata = dict(deepcopy(sensor), kind="PRE_CANDIDATE", action_frame=None,
                    control_tick=None, substep=None, policy_timestamp_ns=None)
                pre["rgb_metadata"] = pre_metadata
                self.core.write_json(self.runtime.output_dir/"pre_candidate.json", pre)
                self.rgb_writer.write_pre_candidate(observation.as_mapping(), pre_metadata)
            self.sensor_samples.append(sensor)
            self.rgb_writer.write_bundle(index, observation.as_mapping(), sensor)
            action = self.action_source.select(index, observation, sensor)
            self.interface.validate_policy_action(action)
            if tuple(self.runtime.clock()) != tuple(sensor["native_clock"]):
                raise self.core.CaptureFailure("policy_inference_advanced_native_physics")
            self.phase, self.current_action = "candidate", action.copy()
            for local_tick in range(6):
                self.tick, self.substep = 6*index+local_tick, 0
                if local_tick == 0:
                    self.rgb_writer.command_event("DISPATCH_REQUESTED", index,
                        scheduled_boundary=[self.tick, 0], native_clock=list(self.runtime.clock()),
                        meaning="about_to_call_runtime_step; acceptance_not_yet_observed")
                flags = self.runtime.step(self.current_action)
                self.lifecycle(flags)
                if self.substep != 2:
                    raise self.core.CaptureFailure("scheduler_did_not_deliver_exactly_two_substeps")
                self.control_ticks += 1
            after = self.core._valid_pose(self.rows[-1]["pose"])
            a, b = before["poses"], after["poses"]
            previous = self.np.asarray(a["tcp_base_wxyz"][:3]) if prior_target is None else prior_target
            self.diagnostics.append(self.interface.DiagnosticFrame(
                requested_tcp_pose_base_wxyz=tuple(action[:3])+self.interface.FIXED_TCP_QUATERNION_BASE_WXYZ,
                actual_tcp_pose_base_wxyz=tuple(a["tcp_base_wxyz"]),
                chicken_pose_world_wxyz=tuple(a["chicken_actor_world_wxyz"]),
                requested_tcp_movement_m=tuple(action[:3]-previous),
                actual_tcp_movement_m=tuple(self.np.asarray(b["tcp_base_wxyz"][:3])-a["tcp_base_wxyz"][:3]),
                chicken_movement_m=tuple(self.np.asarray(b["chicken_actor_world_wxyz"][:3])-a["chicken_actor_world_wxyz"][:3]),
                attachment_state=False))
            self.observations.append(observation)
            self.executed_actions.append(action.copy())
            self.rgb_writer.command_event("COMPLETED", index, scheduled_boundary=[self.tick, 1],
                native_clock=list(self.runtime.clock()))
            prior_target = action[:3].copy()
        self.completed, self.phase = True, "candidate_complete"
        self.diagnostics[-1] = replace(self.diagnostics[-1], outcome_label="CUSTOM_SCENE_TASK_NOT_SCORED")
        return dict(schema="custom-scene-recording/v1", status="UNSCORED", success=None,
            label="CUSTOM_SCENE_TASK_NOT_SCORED", technical_completion="COMPLETE",
            actions=200, control_ticks=self.control_ticks, observed_substeps=len(self.rows),
            gripper_attachment="DISABLED", unassisted_grasp_established=False,
            original_hc_evidence=False)
