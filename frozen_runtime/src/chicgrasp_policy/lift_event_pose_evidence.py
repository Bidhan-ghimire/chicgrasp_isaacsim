"""Callable Gate 5 diagnostic evidence only; no control, attachment or lift grading.

The future owner brackets its real operation with request_activation and
creation_outcome, and forwards every lifecycle event. Nothing installs callbacks.
NativePoseReader uses the current native getters underlying Isaac Lab actor poses,
not action-boundary caches. Ordinary import has no Isaac/Gym/Omniverse dependency.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
import math
import numbers

import numpy as np

from .interface import (POLICY_TCP_LOCAL_POSITION_M, POLICY_TCP_PARENT_BODY,
                        POLICY_TCP_LOCAL_QUATERNION_WXYZ)


TASK_ID = "Isaac-ChicGrasp-GraspLift-DynamicChicken-IK-Abs-v0"
ENV_ROOT = "/World/envs/env_0"
GRIPPER_BODY = ENV_ROOT + "/Robot/GripperRig/gripper_base_link"
CHICKEN_BODY = ENV_ROOT + "/Chicken/raw_chicken_visual/raw_chicken_visual_mesh"
STATIONS = ("negX_shank", "posX_shank")
ENDPOINTS = {"baseline": (839, 1, 1680), "result": (1199, 1, 2400)}
SYNTHETIC_ELIGIBILITY = "SYNTHETIC_ELIGIBILITY_FOR_NATIVE_LIFECYCLE_ONLY"


def _copy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def _json(value):
    """Explicit invalid metadata, never a substituted zero or success value."""
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return {"unavailable_metadata_type": type(value).__name__, "representation": repr(value)}


def _integer(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def _finite(value):
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def _invalid_metadata(value):
    if isinstance(value, np.ndarray):
        return _invalid_metadata(value.tolist())
    if isinstance(value, np.generic):
        return _invalid_metadata(value.item())
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(not isinstance(k, str) or _invalid_metadata(v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return any(_invalid_metadata(v) for v in value)
    return value is not None and not isinstance(value, (str, bool, int))


def _pose(xyzw):
    raw = np.asarray(xyzw, dtype=np.float64)
    if raw.shape != (7,) or not np.all(np.isfinite(raw)):
        raise ValueError("missing_or_nonfinite_actor_pose")
    q = raw[[6, 3, 4, 5]]
    if not np.linalg.norm(q) > 0:
        raise ValueError("zero_actor_quaternion")
    return np.r_[raw[:3], q]


def _rotation(q):
    # Normalize only the derived rotation; raw measured quaternions remain saved.
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _relative(parent, child):
    a = parent[3:] / np.linalg.norm(parent[3:])
    b = child[3:] / np.linalg.norm(child[3:])
    # conjugate(a) * b, the same T_parent^-1 T_child convention as Lab.
    w, u = a[0], -a[1:]
    v = b[1:]
    q = np.r_[w*b[0] - np.dot(u, v), w*v + b[0]*u + np.cross(u, v)]
    return np.r_[_rotation(a).T @ (child[:3]-parent[:3]), q]


class NativePoseReader:
    """Bind existing initialized assets only. No spawning, settings or stepping.

    Sources: Lab articulation_data.py:470-484,583-600; rigid_object_data.py:
    128-142; Gate 2 _resolve_runtime_state. Existing global view uses world roots.
    """

    def __init__(self, env, *, task_id, scene_identity=None):
        if task_id != TASK_ID or env.num_envs != 1:
            raise ValueError("exact_dynamic_task_and_one_environment_required")
        if tuple(POLICY_TCP_LOCAL_QUATERNION_WXYZ) != (1.0, 0.0, 0.0, 0.0):
            raise ValueError("source_tcp_local_orientation_changed")
        self.env = env
        self.scene_identity = scene_identity
        self.robot = env.scene["robot"]
        self.chicken = env.scene["chicken"]
        ids, names = self.robot.find_bodies(POLICY_TCP_PARENT_BODY, preserve_order=True)
        if list(names) != [POLICY_TCP_PARENT_BODY] or len(ids) != 1:
            raise ValueError("ambiguous_tcp_body")
        self.body_index = ids[0]
        self.robot_view = self.robot.root_physx_view
        self.chicken_view = self.chicken.root_physx_view
        self._bound_identities = self._check_identity()

    def _check_identity(self):
        if self.scene_identity is not None:
            self.scene_identity.validate("pose_reader_association", reader=self)
        paths = self.robot_view.link_paths
        roots = self.robot_view.prim_paths
        if (len(roots) != 1 or not (roots[0] == ENV_ROOT + "/Robot" or roots[0].startswith(ENV_ROOT + "/Robot/"))
                or len(paths) != 1 or paths[0][self.body_index] != GRIPPER_BODY
                or list(self.chicken_view.prim_paths) != [CHICKEN_BODY]):
            raise ValueError("runtime_body_identity_changed_or_wrong")
        identities = {"robot_articulation": roots[0], "gripper_actor": GRIPPER_BODY,
                      "chicken_actor": CHICKEN_BODY}
        if hasattr(self, "_bound_identities") and identities != self._bound_identities:
            raise ValueError("bound_runtime_body_identity_changed")
        return identities

    def clock_identity(self):
        return (self.env.sim.current_time_step_index, self.env.sim.current_time)

    def read(self):
        result = {"clock_before": None, "clock_after": None, "raw_native_xyzw": {},
                  "poses": {}, "issues": [], "identities": {},
                  "units": "m", "pose_convention": "position_xyz_quaternion_wxyz",
                  "source": "native_get_root_transforms/get_link_transforms/get_transforms",
                  "tcp_local_position_m": list(POLICY_TCP_LOCAL_POSITION_M)}
        try:
            result["clock_before"] = self.clock_identity()
            result["identities"] = self._check_identity()
            # The same non-stepping refresh used by Lab's body_link_pose_w.
            self.robot.data._physics_sim_view.update_articulations_kinematic()
            raw = result["raw_native_xyzw"]
            raw["robot_root_world"] = _copy(self.robot_view.get_root_transforms())
            raw["robot_links_world"] = _copy(self.robot_view.get_link_transforms())
            raw["chicken_actor_world"] = _copy(self.chicken_view.get_transforms())
            self._check_identity()
            if (raw["robot_root_world"].shape != (1, 7)
                    or raw["chicken_actor_world"].shape != (1, 7)
                    or raw["robot_links_world"].shape != (1, len(self.robot_view.link_paths[0]), 7)):
                raise ValueError("unexpected_native_pose_buffer_shape")
            root = _pose(raw["robot_root_world"][0])
            gripper = _pose(raw["robot_links_world"][0, self.body_index])
            chicken = _pose(raw["chicken_actor_world"][0])
            tcp = gripper.copy()
            tcp[:3] += _rotation(gripper[3:]) @ np.asarray(POLICY_TCP_LOCAL_POSITION_M)
            result["poses"] = {"robot_root_world_wxyz": root,
                               "gripper_world_wxyz": gripper,
                               "chicken_actor_world_wxyz": chicken,
                               "tcp_world_wxyz": tcp,
                               "tcp_base_wxyz": _relative(root, tcp),
                               "chicken_in_gripper_wxyz": _relative(gripper, chicken)}
        except Exception as exc:
            result["issues"].append(f"pose_acquisition_error:{type(exc).__name__}:{exc}")
        try:
            result["clock_after"] = self.clock_identity()
        except Exception as exc:
            result["issues"].append(f"native_clock_unavailable:{exc}")
        if result["clock_before"] != result["clock_after"]:
            result["issues"].append("native_clock_changed_during_acquisition")
        for clock in (result["clock_before"], result["clock_after"]):
            if (not isinstance(clock, (list, tuple)) or len(clock) != 2
                    or not _integer(clock[0]) or not _finite(clock[1])):
                result["issues"].append("missing_or_nonfinite_native_clock")
                break
        result["valid"] = not result["issues"]
        return deepcopy(result)


@dataclass(frozen=True)
class Context:
    episode_id: str
    control_tick: int
    substep: int


@dataclass(frozen=True)
class DiagnosticContext:
    """Caller-observed session/epoch identity, never executed control history.

    The caller must observe reset/stage continuity and invalidate on loss of it.
    These identifiers are not native callback fields or proof of unseen history.
    """
    session_id: str
    stage_id: int
    native_epoch: str
    boundary_id: str


@dataclass(frozen=True)
class _DiagnosticSession:
    context: DiagnosticContext
    boundary_clock: tuple
    native_origin: tuple | None
    native_origin_source: str | None


def _diagnostic_context_valid(context):
    return (type(context) is DiagnosticContext
            and type(context.stage_id) is int and context.stage_id > 0
            and all(isinstance(value, str) and value.strip() for value in
                    (context.session_id, context.native_epoch, context.boundary_id)))


def _diagnostic_clock_valid(clock):
    return (isinstance(clock, (tuple, list)) and len(clock) == 2
            and _integer(clock[0]) and clock[0] >= 0
            and _finite(clock[1]) and clock[1] >= 0)


class LiftEventPoseEvidence:
    """One episode, append-only producer journal; not a physical acceptance rule.

    Sequence IDs order operations even when native clocks match. The caller must
    not perform unreported state-changing operations between these calls.
    """

    def __init__(self, reader, *, episode_id, origin_step, origin_time):
        if hasattr(self, "_records"):
            raise ValueError("producer_session_cannot_be_reinitialized")
        if not episode_id or not _integer(origin_step) or origin_step < 0 or not _finite(origin_time):
            raise ValueError("invalid_episode_origin")
        self._start(reader, episode_id, origin_step, origin_time)

    def _start(self, reader, episode_id, origin_step, origin_time, diagnostic=None):
        self.reader, self.episode_id = reader, episode_id
        self.origin_step, self.origin_time = origin_step, origin_time
        self._diagnostic_session = diagnostic
        self._capture_policy = False
        self._geometric_admission = False
        self._envelope_v2 = False
        self._envelope_protocol = "v2"
        self._event_close = None
        self._capture_actions = 200
        self._records, self._endpoints, self._requests, self._outcomes = [], {}, {}, {}
        self._invalidated = False
        self._last_clock = diagnostic.boundary_clock if diagnostic else (origin_step, origin_time)

    @classmethod
    def for_diagnostic(cls, reader, *, context, native_origin=None, native_origin_source=None):
        """Select frozen lifecycle provenance before any event is recorded.

        Sample the current native clock, without poses or processing. An earlier
        native origin is optional and must be independently observed by caller;
        unknown origin stays None, not inferred from a nominal timestep.
        """
        if not _diagnostic_context_valid(context):
            raise ValueError("invalid_diagnostic_context")
        clock = reader.clock_identity()
        if not _diagnostic_clock_valid(clock):
            raise ValueError("invalid_diagnostic_boundary_clock")
        if native_origin is None:
            if native_origin_source is not None:
                raise ValueError("diagnostic_origin_source_without_observation")
        else:
            if (not _diagnostic_clock_valid(native_origin)
                    or not isinstance(native_origin_source, str) or not native_origin_source.strip()
                    or native_origin[0] > clock[0] or native_origin[1] > clock[1]
                    or (native_origin[0] == clock[0] and native_origin[1] != clock[1])
                    or (native_origin[0] < clock[0] and native_origin[1] >= clock[1])):
                raise ValueError("invalid_diagnostic_native_origin")
            native_origin = tuple(native_origin)
        session = _DiagnosticSession(context, tuple(clock), native_origin, native_origin_source)
        instance = cls.__new__(cls)
        instance._start(reader, None, None, None, session)
        return instance

    @property
    def provenance_domain(self):
        return "diagnostic" if self._diagnostic_session is not None else ("capture" if self._capture_policy else "episode")

    @classmethod
    def for_capture(cls, reader, *, episode_id, origin_step, origin_time):
        """Explicit real-episode scheduled-step policy; never synthetic eligibility."""
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("real_episode_identity_required")
        instance = cls(reader, episode_id=episode_id, origin_step=origin_step, origin_time=origin_time)
        instance._capture_policy = True
        return instance

    @property
    def capture_policy(self):
        return self._capture_policy

    def select_geometric_admission(self, source="geometric-eligibility/v1", *, run_type="HARNESS", envelope_v2=False, protocol="v2"):
        """Select the contact-free admission contract before any owner event.

        This changes admission only. The owner's authoring, release and exactly
        next scheduled substep confirmation implementation are not replaced.
        """
        if not self._capture_policy or self._records or self._geometric_admission:
            raise ValueError("geometric_admission_requires_unused_capture_session")
        if source not in ("geometric-eligibility/v1", "side-opening-progress/v1"):
            raise ValueError("unknown_geometric_admission")
        if run_type not in ("HARNESS", "EVIDENCE"):
            raise ValueError("explicit_geometric_run_type_required")
        if envelope_v2 and (source != "side-opening-progress/v1" or run_type != "HARNESS"):
            raise ValueError("v2_requires_explicit_learned_witness_HARNESS")
        if protocol not in ("v2", "v3c") or (protocol == "v3c" and not envelope_v2):
            raise ValueError("invalid_event_protocol_route")
        self._envelope_protocol = protocol
        self._envelope_v2 = envelope_v2
        self._admission_run_type = run_type
        self._admission_source = source
        self._geometric_admission = True

    def _event_decision(self):
        from .envelope_schedule import event_schedule
        return divmod(event_schedule(self._event_close, protocol=self._envelope_protocol)["decision"], 2)

    def bind_event_close(self, close_action):
        if not self._envelope_v2 or self._event_close is not None or type(close_action) is not int or not 0 <= close_action <= 199 or self._requests:
            raise ValueError("event_anchor_requires_first_accepted_close")
        self._event_close = close_action
        self._capture_actions = max(200, close_action+80)

    @property
    def diagnostic_context(self):
        return self._diagnostic_session.context if self._diagnostic_session else None

    def validate_owner_stage(self, stage_id):
        if self._diagnostic_session and stage_id != self.diagnostic_context.stage_id:
            raise ValueError("diagnostic_stage_does_not_match_bound_owner")

    def activation_phase_issues(self, context):
        if self._diagnostic_session:
            return []  # No episode phase or invented baseline index in this domain.
        if self._geometric_admission and self._envelope_v2:
            end = None if self._event_close is None else self._event_decision()
            return [] if type(context) is Context and (context.control_tick, context.substep) == end and "baseline" not in self._endpoints else ["v2_request_requires_exact_event_decision"]
        if self._geometric_admission:
            return ([] if type(context) is Context and
                    (context.control_tick, context.substep) == (839, 1) and
                    "baseline" not in self._endpoints else
                    ["geometric_request_requires_exact_839_1_boundary"])
        if self._capture_policy and (context.control_tick, context.substep) > (839, 0):
            return ["request_too_late_for_next_substep_confirmation"]
        if ((context.control_tick, context.substep) > (839, 1) or "baseline" in self._endpoints):
            return ["activation_not_before_baseline"]
        return []

    def _clock_issues(self, context, clock):
        if self._capture_policy and type(context) is not Context:
            return ["real_episode_context_required"]
        if self._diagnostic_session:
            issues = []
            if not _diagnostic_context_valid(context):
                issues.append("invalid_or_mixed_diagnostic_context")
            elif context != self.diagnostic_context:
                issues.append("diagnostic_session_stage_epoch_or_boundary_mismatch")
            if not _diagnostic_clock_valid(clock):
                issues.append("invalid_diagnostic_native_clock")
            elif tuple(clock) != self._diagnostic_session.boundary_clock:
                issues.append("diagnostic_native_clock_changed_or_stale")
            return issues
        if isinstance(context, DiagnosticContext):
            return ["diagnostic_context_in_episode_domain"]
        issues = []
        if context.episode_id != self.episode_id:
            issues.append("wrong_episode")
        if (not _integer(context.control_tick) or not 0 <= context.control_tick < 6*self._capture_actions
                or not _integer(context.substep) or context.substep not in (0, 1)):
            issues.append("invalid_control_tick_or_substep")
        if (not isinstance(clock, (tuple, list)) or len(clock) != 2
                or not _integer(clock[0]) or not _finite(clock[1])):
            return issues + ["missing_or_nonfinite_native_clock"]
        if not issues and clock[0] != self.origin_step + 2*context.control_tick + context.substep + 1:
            issues.append("stale_or_wrong_native_step")
        if clock[0] < self._last_clock[0] or clock[1] < self._last_clock[1]:
            issues.append("native_clock_regression")
        if clock[0] > self._last_clock[0] and clock[1] <= self._last_clock[1]:
            issues.append("stale_native_time")
        if clock[0] == self._last_clock[0] and clock[1] != self._last_clock[1]:
            issues.append("contradictory_native_time")
        # Actual native time is retained, not compared with nominal dt multiplication.
        return issues

    def _append(self, kind, context, *, pose=False, **payload):
        if pose:
            reading = self.reader.read()
            clock = reading["clock_after"]
            issues = list(reading["issues"])
            issues.extend(self._clock_issues(context, reading["clock_before"]))
            if self._capture_policy:
                # Validate required measurements without altering reader math or
                # replacing invalid values. The independent evaluator repeats it.
                for name in ("gripper_world_wxyz", "chicken_actor_world_wxyz",
                             "chicken_in_gripper_wxyz", "tcp_world_wxyz"):
                    try:
                        value = np.asarray(reading["poses"][name], dtype=float)
                        norm = np.linalg.norm(value[3:])
                        if value.shape != (7,) or not np.isfinite(value).all() or not np.isfinite(norm) or norm <= 0:
                            raise ValueError("invalid_pose")
                    except (KeyError, TypeError, ValueError):
                        issues.append("invalid_required_capture_pose:" + name)
        else:
            reading = None
            try:
                clock = self.reader.clock_identity()
            except Exception:
                clock = None
            issues = []
        issues.extend(self._clock_issues(context, clock))
        # Capture contact identity is usable even when unused force diagnostics
        # are unavailable. Its required fields are validated separately below.
        metadata = ({k: v for k, v in payload.items() if k != "eligibility"}
                    if self._capture_policy else payload)
        if _invalid_metadata(metadata):
            issues.append("nonfinite_or_unavailable_metadata")
        if self._invalidated:
            issues.append("lifecycle_invalidated")
        context_record = (asdict(context) if (not self._diagnostic_session and not self._capture_policy) or is_dataclass(context)
                          else _json(context))
        record = {"sequence": len(self._records), "kind": kind, "context": context_record,
                  "native_clock": clock, "pose_reading": reading, "issues": list(dict.fromkeys(issues)),
                  **deepcopy(payload)}
        if self._diagnostic_session:
            record.update(provenance_domain="diagnostic", lifecycle_only=True,
                          control_frame_history=None, physical_eligibility_established=False)
        elif self._capture_policy:
            record.update(provenance_domain="capture", confirmation_policy="next_scheduled_substep")
        self._records.append(record)
        if not issues:
            self._last_clock = clock
        return record

    def capture_endpoint(self, name, context):
        if self._diagnostic_session or isinstance(context, DiagnosticContext):
            raise ValueError("diagnostic_provenance_cannot_capture_episode_endpoint")
        if name not in ENDPOINTS:
            raise ValueError("unknown_endpoint")
        if name in self._endpoints:
            raise ValueError("duplicate_endpoint_original_preserved")
        record = self._append("endpoint", context, pose=True, endpoint=name)
        tick, substep, relative_step = ENDPOINTS[name]
        if (context.control_tick, context.substep) != (tick, substep):
            record["issues"].append("wrong_endpoint_boundary")
        record["required_relative_physics_step"] = relative_step
        self._endpoints[name] = record["sequence"]
        return record["sequence"]

    def _station_evidence_issues(self, evidence):
        """Classify structural station evidence; leave raw input in the journal."""
        if self._diagnostic_session:
            contacts = evidence.get("contacts") if isinstance(evidence, dict) else None
            valid = (isinstance(contacts, list) and len(contacts) == len(STATIONS)
                     and all(isinstance(c, dict) and set(c) == {"station", "valid", "source"}
                             and isinstance(c.get("station"), str) and type(c.get("valid")) is bool
                             and c.get("source") == SYNTHETIC_ELIGIBILITY for c in contacts))
            if valid:
                valid = sorted(c["station"] for c in contacts) == sorted(STATIONS)
            return [] if valid else [{"reason": "invalid_synthetic_station_fixture",
                "location": "eligibility.contacts", "source": SYNTHETIC_ELIGIBILITY,
                "issue": "two_distinct_labelled_synthetic_station_values_required"}]
        details = []
        contacts = evidence.get("contacts") if isinstance(evidence, dict) else None

        def issue(reason, location, explanation, record=None):
            details.append({"reason": reason, "location": location,
                            "source": record.get("source") if isinstance(record, dict) else None,
                            "issue": explanation})

        malformed = "malformed_station_evidence"
        missing = "missing_required_station_contacts"
        if contacts is None:
            issue(missing, "eligibility.contacts", "required_station_collection_absent")
            return details
        if not isinstance(contacts, list):
            issue(malformed, "eligibility.contacts", "station_collection_not_a_list")
            return details
        for index, contact in enumerate(contacts):
            path = f"eligibility.contacts[{index}]"
            if not isinstance(contact, dict):
                issue(malformed, path, "station_record_not_an_object")
                continue
            if not isinstance(contact.get("station"), str):
                issue(malformed, path+".station", "station_identifier_not_a_string", contact)
            if type(contact.get("valid")) is not bool:
                issue(malformed, path+".valid", "contact_validity_not_a_boolean", contact)
            if not isinstance(contact.get("source"), str) or not contact["source"].strip():
                issue(malformed, path+".source", "contact_source_unavailable", contact)
            sample = contact.get("sample")
            if not isinstance(sample, dict):
                issue(malformed, path+".sample", "sample_provenance_not_an_object", contact)
            else:
                if not isinstance(sample.get("episode_id"), str) or not sample["episode_id"].strip():
                    issue(malformed, path+".sample.episode_id", "sample_episode_unavailable", contact)
                for name in ("control_tick", "substep", "physics_step"):
                    if not _integer(sample.get(name)):
                        issue(malformed, path+".sample."+name, "sample_identity_not_an_integer", contact)
                if not _finite(sample.get("simulation_time_s")):
                    issue(malformed, path+".sample.simulation_time_s", "sample_time_unavailable_or_nonfinite", contact)
        if details:
            return details
        stations = [c["station"] for c in contacts]
        if len(stations) != len(set(stations)):
            issue(malformed, "eligibility.contacts", "duplicate_station_records")
        if len(stations) > len(STATIONS):
            issue(malformed, "eligibility.contacts", "unexpected_station_record_count")
        if not details and set(stations) != set(STATIONS):
            issue(missing, "eligibility.contacts", "absent_required_stations:" + ",".join(
                station for station in STATIONS if station not in stations))
        return details

    def _eligibility_issues(self, evidence, context, clock):
        if self._capture_policy:
            return self.capture_eligibility_issues(evidence, context, clock)
        if self._diagnostic_session:
            return self._diagnostic_eligibility_issues(evidence, context, clock)
        if isinstance(context, DiagnosticContext):
            return ["diagnostic_context_in_episode_domain"]
        issues = []
        if (not _integer(context.control_tick) or not isinstance(clock, (list, tuple))
                or len(clock) != 2 or not _integer(clock[0]) or not _finite(clock[1])):
            return ["invalid_request_context_or_clock"]
        if not isinstance(evidence, dict):
            return ["missing_eligibility_evidence"]

        def provenance(item, *, current=False):
            if not isinstance(item, dict) or not isinstance(item.get("source"), str) or not item["source"].strip():
                return False
            s = item.get("sample", {})
            if not isinstance(s, dict):
                return False
            tick, sub = s.get("control_tick"), s.get("substep")
            step, timestamp = s.get("physics_step"), s.get("simulation_time_s")
            return (s.get("episode_id") == self.episode_id and _integer(tick) and 0 <= tick <= 1199
                    and _integer(sub) and sub in (0, 1) and _integer(step)
                    and step == self.origin_step + 2*tick + sub + 1 and _finite(timestamp)
                    and clock is not None and step <= clock[0] and self.origin_time <= timestamp <= clock[1]
                    and (not current or (step, timestamp) == tuple(clock)))

        frames = evidence.get("command_frames")
        if not isinstance(frames, list) or len(frames) != 3:
            issues.append("missing_three_command_frames")
        else:
            expected = list(range(context.control_tick//6-2, context.control_tick//6+1))
            indices = [f.get("action_frame") if isinstance(f, dict) else None for f in frames]
            if indices != expected or not all(_integer(i) for i in indices):
                issues.append("commands_not_three_distinct_consecutive_current_frames")
            for f in frames:
                if not provenance(f) or f.get("action_frame") != f.get("sample", {}).get("control_tick", -6)//6:
                    issues.append("missing_or_invalid_command_provenance")
                commands = f.get("commands") if isinstance(f, dict) else None
                if not isinstance(commands, (list, tuple)) or len(commands) != 2 or not all(_finite(c) and c == 0 for c in commands):
                    issues.append("both_jaw_commands_not_closed")
        contacts = evidence.get("contacts")
        station_issues = self._station_evidence_issues(evidence)
        if station_issues:
            issues.extend(detail["reason"] for detail in station_issues)
        else:
            for c in contacts:
                if not provenance(c, current=True):
                    issues.append("missing_or_stale_contact_provenance")
                if c.get("valid") is not True:
                    issues.append("passive_contact_not_valid")
        runtime = evidence.get("runtime")
        if not provenance(runtime, current=True):
            issues.append("missing_or_stale_runtime_provenance")
        if not isinstance(runtime, dict) or any(runtime.get(k) is not False for k in ("resetting", "truncated", "terminated", "invalid")):
            issues.append("runtime_conditions_not_valid")
        return list(dict.fromkeys(issues))

    def _diagnostic_eligibility_issues(self, evidence, context, clock):
        issues = self._clock_issues(context, clock)
        if self._invalidated:
            issues.append("lifecycle_invalidated")
        required = {"label", "session_id", "boundary_id", "control_frame_history",
                    "contacts", "jaw_commands", "runtime"}
        if not isinstance(evidence, dict) or set(evidence) != required:
            return issues + ["invalid_diagnostic_eligibility_schema_no_execution_history"]
        if _invalid_metadata(evidence):
            issues.append("nonfinite_or_unavailable_metadata")
        if (evidence["label"] != SYNTHETIC_ELIGIBILITY
                or evidence["session_id"] != self.diagnostic_context.session_id
                or evidence["boundary_id"] != self.diagnostic_context.boundary_id
                or evidence["control_frame_history"] is not None):
            issues.append("invalid_diagnostic_fixture_identity_or_history")
        station_issues = self._station_evidence_issues(evidence)
        issues.extend(d["reason"] for d in station_issues)
        if not station_issues and any(c["valid"] is not True for c in evidence["contacts"]):
            issues.append("synthetic_station_fixture_not_valid")
        jaws = evidence["jaw_commands"]
        values = jaws.get("values") if isinstance(jaws, dict) else None
        if (not isinstance(jaws, dict) or set(jaws) != {"source", "values"}
                or jaws.get("source") != SYNTHETIC_ELIGIBILITY
                or not isinstance(values, (list, tuple)) or len(values) != 2
                or not all(_finite(v) and v == 0 for v in values)):
            issues.append("invalid_synthetic_jaw_fixture")
        runtime = evidence["runtime"]
        flags = ("resetting", "truncated", "terminated", "invalid")
        if (not isinstance(runtime, dict) or set(runtime) != {"source", *flags}
                or not isinstance(runtime.get("source"), str) or not runtime["source"].strip()
                or any(runtime.get(k) is not False for k in flags)):
            issues.append("diagnostic_runtime_conditions_not_valid")
        return list(dict.fromkeys(issues))

    def capture_eligibility_issues(self, evidence, context, clock, *, require_contact=True):
        """Validate real command history and current bilateral collider evidence.

        The caller's predicate has resolved current collider/body identities;
        this boundary validates its status and synchronized provenance, not force.
        """
        if not self._capture_policy:
            raise ValueError("explicit_capture_policy_required")
        if self._geometric_admission:
            return self._geometric_eligibility_issues(evidence, context, clock)
        issues = list(self._clock_issues(context, clock))
        if issues or type(context) is not Context:
            return issues
        if self._invalidated:
            issues.append("lifecycle_invalidated")
        if not isinstance(evidence, dict) or "label" in evidence or "contacts" in evidence:
            return issues + ["real_capture_eligibility_required_no_diagnostic_or_station_fallback"]

        def provenance(item, *, current=False):
            if (not isinstance(item, dict) or not isinstance(item.get("source"), str)
                    or not item["source"].strip() or item["source"] == SYNTHETIC_ELIGIBILITY):
                return False
            sample = item.get("sample")
            if not isinstance(sample, dict):
                return False
            tick, sub = sample.get("control_tick"), sample.get("substep")
            step, timestamp = sample.get("physics_step"), sample.get("simulation_time_s")
            return (sample.get("episode_id") == self.episode_id
                    and _integer(tick) and 0 <= tick <= 1199 and _integer(sub) and sub in (0, 1)
                    and _integer(step) and step == self.origin_step + 2*tick + sub + 1
                    and _finite(timestamp) and self.origin_time < timestamp <= clock[1]
                    and step <= clock[0] and (not current or
                        ((tick, sub) == (context.control_tick, context.substep)
                         and (step, timestamp) == tuple(clock))))

        frames = evidence.get("command_frames")
        expected = list(range(context.control_tick//6-2, context.control_tick//6+1))
        if not isinstance(frames, list) or len(frames) != 3:
            issues.append("missing_three_command_frames")
        else:
            ids = [frame.get("action_frame") if isinstance(frame, dict) else None for frame in frames]
            if ids != expected or not all(_integer(i) and i >= 0 for i in ids):
                issues.append("commands_not_three_distinct_consecutive_current_frames")
            for frame in frames:
                if not provenance(frame) or frame.get("action_frame") != frame.get("sample", {}).get("control_tick", -6)//6:
                    issues.append("missing_or_invalid_command_provenance")
                commands = frame.get("commands") if isinstance(frame, dict) else None
                if (not isinstance(commands, (tuple, list)) or len(commands) != 2
                        or not all(_finite(v) and v == 0 for v in commands)):
                    issues.append("both_jaw_commands_not_closed")
        contact = evidence.get("bilateral_contact")
        if not provenance(contact, current=True):
            issues.append("missing_or_stale_bilateral_contact_provenance")
        if not isinstance(contact, dict):
            issues.append("required_bilateral_contact_unavailable")
        else:
            status, left, right = contact.get("status"), contact.get("left"), contact.get("right")
            if (status not in ("TRUE", "FALSE") or type(left) is not bool or type(right) is not bool
                    or (status == "TRUE") != (left and right) or contact.get("issues") != []):
                issues.append("required_bilateral_contact_unavailable")
            elif require_contact and status != "TRUE":
                issues.append("bilateral_contact_false")
        runtime = evidence.get("runtime")
        if not provenance(runtime, current=True):
            issues.append("missing_or_stale_runtime_provenance")
        if not isinstance(runtime, dict) or any(runtime.get(k) is not False
                for k in ("resetting", "truncated", "terminated", "invalid")):
            issues.append("runtime_conditions_not_valid")
        return list(dict.fromkeys(issues))

    def _geometric_eligibility_issues(self, evidence, context, clock):
        """Recompute fixed geometric eligibility; never synthesize contact."""
        from .geometric_admission import evaluate_history
        issues = list(self._clock_issues(context, clock))
        if issues or type(context) is not Context:
            return issues
        issues.extend(self.activation_phase_issues(context))
        if self._invalidated:
            issues.append("lifecycle_invalidated")
        if not isinstance(evidence, dict) or set(evidence) != {"geometric_eligibility", "runtime"}:
            return issues + ["explicit_geometric_eligibility_required_no_contact_fallback"]
        claimed = evidence["geometric_eligibility"]
        if not isinstance(claimed, dict):
            return issues + ["geometric_eligibility_unavailable"]
        try:
            if self._admission_source == "side-opening-progress/v1":
                from .side_opening_admission import evaluate
                reproduced = evaluate(claimed.get("history"), claimed.get("transitions"), run_type=self._admission_run_type,
                    expected_end=self._event_decision() if self._envelope_v2 else (839, 1))
            else:
                reproduced = evaluate_history(claimed.get("history"))
        except (TypeError, ValueError, KeyError, OverflowError):
            return issues + ["geometric_history_unavailable"]
        if reproduced != claimed:
            issues.append("geometric_eligibility_replay_disagreement")
        if reproduced.get("status") != "TRUE" or reproduced.get("issues") != []:
            issues.append("geometric_eligibility_not_true")
        expected = {"episode_id": self.episode_id, "control_tick": context.control_tick,
                    "substep": context.substep, "physics_step": clock[0], "simulation_time_s": clock[1]}
        if claimed.get("source") != self._admission_source or claimed.get("sample") != expected:
            issues.append("geometric_eligibility_current_provenance_mismatch")
        runtime = evidence["runtime"]
        if (not isinstance(runtime, dict) or runtime.get("sample") != expected or
                not isinstance(runtime.get("source"), str) or not runtime["source"].strip() or
                runtime["source"] == SYNTHETIC_ELIGIBILITY or
                any(runtime.get(key) is not False for key in ("resetting", "truncated", "terminated", "invalid"))):
            issues.append("runtime_conditions_not_valid")
        return list(dict.fromkeys(issues))

    def record_pending(self, request_id, context, joint_path):
        if not self._capture_policy or request_id not in self._requests:
            raise ValueError("capture_request_required")
        return self._append("attachment_pending", context, request_id=request_id,
                            status="PENDING", joint_path=joint_path)["sequence"]

    def capture_observation(self, context):
        if not self._capture_policy:
            raise ValueError("explicit_capture_policy_required")
        return self._append("scheduled_observation", context, pose=True)["sequence"]

    def record_capture_failure(self, request_id, context, cause):
        """Close an early failed request without inventing a scheduled outcome."""
        if not self._capture_policy or request_id not in self._requests:
            raise ValueError("capture_request_required")
        if request_id in self._outcomes:
            return self._outcomes[request_id]
        record = self._append("attachment_request_failure", context, pose=True,
                              request_id=request_id, cause=cause, scheduled_confirmation=False)
        self._outcomes[request_id] = record["sequence"]
        return record["sequence"]

    def record(self, sequence):
        """Copy one retained record without repeatedly copying the whole episode."""
        return _json(deepcopy(self._records[sequence]))

    def _scheduled_outcome(self, request_id, context, outcome, source, *, relationship_used, details):
        if request_id in self._outcomes:
            raise ValueError("duplicate_creation_outcome_original_preserved")
        record = self._append("scheduled_attachment_outcome", context, pose=True,
            request_id=request_id, reported_outcome=outcome, source=source,
            relationship_used=relationship_used, details=details)
        issues = record["issues"]
        request = self._requests.get(request_id)
        if request is None:
            issues.append("missing_activation_request")
        else:
            if request["issues"] or request["eligibility_issues"]:
                issues.append("invalid_pre_creation_pose_or_provenance")
            before, after = request["context"], record["context"]
            a, b = request["native_clock"], record["native_clock"]
            ordered = (isinstance(before, dict) and isinstance(after, dict)
                and all(_integer(item.get(name)) for item in (before, after)
                        for name in ("control_tick", "substep"))
                and 2*after["control_tick"] + after["substep"] == 2*before["control_tick"] + before["substep"] + 1
                and _diagnostic_clock_valid(a) and _diagnostic_clock_valid(b)
                and b[0] == a[0] + 1 and b[1] > a[1])
            if not ordered:
                issues.append("confirmation_not_exactly_next_scheduled_substep")
        if (not isinstance(source, str) or not source.strip() or outcome != "confirmed"):
            issues.append("scheduled_confirmation_unconfirmed")
        if (not isinstance(relationship_used, dict) or relationship_used.get("parent") != GRIPPER_BODY
                or relationship_used.get("child") != CHICKEN_BODY
                or relationship_used.get("convention") != "chicken_in_gripper_position_xyz_quaternion_wxyz_m"):
            issues.append("missing_or_wrong_attachment_relationship")
        else:
            try:
                raw = np.asarray(relationship_used["pose"], dtype=float)
                _pose(raw[[0, 1, 2, 4, 5, 6, 3]])
                if raw.shape != (7,):
                    raise ValueError("shape")
            except (KeyError, ValueError, TypeError, IndexError):
                issues.append("invalid_attachment_relationship_pose")
        record["relative_change"] = None
        if request is not None:
            try:
                pre = np.asarray(request["pose_reading"]["poses"]["chicken_in_gripper_wxyz"], dtype=float)
                post = np.asarray(record["pose_reading"]["poses"]["chicken_in_gripper_wxyz"], dtype=float)
                if (pre.shape != (7,) or post.shape != (7,) or not np.isfinite(pre).all()
                        or not np.isfinite(post).all() or not np.linalg.norm(pre[3:]) > 0
                        or not np.linalg.norm(post[3:]) > 0):
                    raise ValueError("invalid_relative_measurement")
                distance = float(np.linalg.norm(post[:3] - pre[:3]))
                q = _relative(np.r_[np.zeros(3), pre[3:]], np.r_[np.zeros(3), post[3:]])[3:]
                angle = math.degrees(2*math.atan2(np.linalg.norm(q[1:]), abs(q[0])))
                record["relative_change"] = {"translation_m": distance, "rotation_degrees": angle,
                    "translation_limit_m_inclusive": 0.01, "rotation_limit_degrees_inclusive": 5.0}
                if distance > 0.01 or angle > 5.0:
                    issues.append("excessive_scheduled_relative_change")
            except (KeyError, TypeError, ValueError):
                issues.append("missing_pre_or_post_creation_pose")
        record["scheduled_confirmation"] = outcome == "confirmed" and not issues
        record["confirmed_creation"] = False  # No independent native callback claim.
        self._outcomes[request_id] = record["sequence"]
        return record["sequence"]

    def request_activation(self, context, eligibility):
        record = self._append("activation_request", context, pose=True, eligibility=eligibility)
        record["station_evidence_issues"] = [] if self._capture_policy else self._station_evidence_issues(eligibility)
        record["eligibility_issues"] = self._eligibility_issues(eligibility, context, record["native_clock"])
        if self._geometric_admission and not record["eligibility_issues"]:
            # Match the gate's current geometric input to the owner's actual
            # fresh pre-authoring pose read, not merely a claimed clock.
            last = eligibility["geometric_eligibility"]["history"][-1]
            poses = _json((record.get("pose_reading") or {}).get("poses", {}))
            if (last.get("chicken_world_wxyz") != poses.get("chicken_actor_world_wxyz") or
                    last.get("tcp_world_wxyz") != poses.get("tcp_world_wxyz")):
                record["eligibility_issues"].append("geometric_current_pose_disagrees_with_owner_reading")
        record["confirmed_creation"] = False
        self._requests[record["sequence"]] = record
        return record["sequence"]

    def creation_outcome(self, request_id, context, outcome, source, *, relationship_used=None, details=None):
        if self._capture_policy:
            return self._scheduled_outcome(request_id, context, outcome, source,
                relationship_used=relationship_used, details=details)
        if request_id in self._outcomes:
            raise ValueError("duplicate_creation_outcome_original_preserved")
        record = self._append("creation_outcome", context, pose=True, request_id=request_id,
                              reported_outcome=outcome, source=source,
                              relationship_used=relationship_used, details=details)
        request = self._requests.get(request_id)
        gaps = record["issues"]
        if request is None:
            gaps.append("missing_activation_request")
        else:
            if request["sequence"] != record["sequence"]-1 or request["native_clock"] != record["native_clock"]:
                gaps.append("creation_not_immediately_bracketed")
            if request["issues"]:
                gaps.append("invalid_pre_creation_pose_or_provenance")
            if request["eligibility_issues"]:
                gaps.append("failed_or_missing_eligibility")
        record["confirmed_creation"] = outcome == "confirmed" and isinstance(source, str) and bool(source.strip())
        if not record["confirmed_creation"]:
            gaps.append("creation_unconfirmed")
        if (not isinstance(relationship_used, dict) or relationship_used.get("parent") != GRIPPER_BODY
                or relationship_used.get("child") != CHICKEN_BODY
                or relationship_used.get("convention") != "chicken_in_gripper_position_xyz_quaternion_wxyz_m"):
            gaps.append("missing_or_wrong_attachment_relationship")
        else:
            try:
                # Validate without changing the owner-reported raw relationship.
                rel = np.asarray(relationship_used["pose"], dtype=np.float64)
                _pose(rel[[0, 1, 2, 4, 5, 6, 3]])
                if rel.shape != (7,):
                    raise ValueError("shape")
            except (ValueError, TypeError, KeyError, IndexError):
                gaps.append("invalid_attachment_relationship_pose")
        record["activation_discontinuity"] = None
        if request is not None:
            try:
                pre = request["pose_reading"]["poses"]["chicken_actor_world_wxyz"]
                post = record["pose_reading"]["poses"]["chicken_actor_world_wxyz"]
                distance = float(np.linalg.norm(post[:3]-pre[:3]))
                a, b = pre[3:]/np.linalg.norm(pre[3:]), post[3:]/np.linalg.norm(post[3:])
                # atan2 avoids adding a pose/discontinuity tolerance at the boundary.
                relative = _relative(np.r_[np.zeros(3), a], np.r_[np.zeros(3), b])[3:]
                angle = math.degrees(2*math.atan2(np.linalg.norm(relative[1:]), abs(relative[0])))
                record["activation_discontinuity"] = {"chicken_translation_m": distance,
                    "chicken_rotation_degrees": angle, "translation_limit_m_strict": 0.0005,
                    "rotation_limit_degrees_strict": 0.5,
                    "within_existing_limits": distance < 0.0005 and angle < 0.5}
                if not distance < 0.0005 or not angle < 0.5:
                    gaps.append("excessive_activation_discontinuity")
            except (KeyError, TypeError, ValueError):
                gaps.append("missing_pre_or_post_creation_pose")
        self._outcomes[request_id] = record["sequence"]
        return record["sequence"]

    def record_release(self, context, cause, source):
        record = self._append("release", context, cause=cause, source=source)
        if not isinstance(cause, str) or not cause.strip() or not isinstance(source, str) or not source.strip():
            record["issues"].append("missing_release_cause_or_source")
        if (self._diagnostic_session or self._capture_policy) and record["issues"]:
            # Keep evidence, but do not let the owner's reporting call certify
            # cleanup with invalid diagnostic provenance (including a repeat).
            raise ValueError("invalid_diagnostic_release_evidence:" + ",".join(record["issues"]))
        return record["sequence"]

    def record_lifecycle(self, context, event, source):
        if event not in ("reset", "truncation", "termination", "teardown", "invalid"):
            raise ValueError("unknown_lifecycle_event")
        record = self._append(event, context, source=source)
        if not isinstance(source, str) or not source.strip():
            record["issues"].append("missing_lifecycle_source")
        self._invalidated = True
        if (self._diagnostic_session or self._capture_policy) and record["issues"]:
            raise ValueError("invalid_diagnostic_lifecycle_evidence:" + ",".join(record["issues"]))
        return record["sequence"]

    def export(self):
        """Availability/preconditions only. Never emits rise, settled or lift PASS."""
        if self._capture_policy:
            return _json(deepcopy({"provenance_domain": "capture", "episode_id": self.episode_id,
                "origin_step": self.origin_step, "origin_time_s": self.origin_time,
                "confirmation_policy": "next_scheduled_substep", "records": self._records,
                "endpoints": self._endpoints, "policy_input": False, "lift_result": None,
                "integration": "explicit_capture_policy", "synthetic_eligibility_allowed": False,
                **({"admission_policy": self._admission_source} if self._geometric_admission else {}),
                **(dict(envelope_v2=True,first_accepted_close=self._event_close,total_actions=self._capture_actions) if self._envelope_v2 else {}),
                **(dict(envelope_protocol="v3c") if self._envelope_protocol == "v3c" else {})}))
        if self._diagnostic_session:
            session = self._diagnostic_session
            return _json(deepcopy({"provenance_domain": "diagnostic", "lifecycle_only": True,
                "session_id": session.context.session_id, "diagnostic_context": asdict(session.context),
                "observation_boundary": {"id": session.context.boundary_id,
                                         "native_clock": session.boundary_clock},
                "native_clock_origin": (None if session.native_origin is None else
                    {"clock": session.native_origin, "source": session.native_origin_source}),
                "native_clock_source": "env.sim.current_time_step_index/current_time",
                "epoch_identity_source": "caller_observed_stage_reset_epoch",
                "command_frame_history": None, "episode_id": None,
                "origin_step": None, "origin_time_s": None, "records": self._records,
                "endpoints": {}, "evidence_qualification_issues": ["diagnostic_not_episode_evidence"],
                "event_endpoint_preconditions_available": False, "episode_capture_eligible": False,
                "physical_eligibility_established": False, "baseline_settling_qualified": None,
                "lift_result": None, "table_clear_rule": None, "policy_input": False,
                "integration": "lifecycle_only_not_connected_to_capture"}))
        issues = []
        endpoint_records = {n: self._records[i] for n, i in self._endpoints.items()}
        for name in ENDPOINTS:
            if name not in endpoint_records:
                issues.append("missing_"+name+"_endpoint")
            elif endpoint_records[name]["issues"]:
                issues.append("invalid_"+name+"_endpoint")
        baseline = endpoint_records.get("baseline")
        result = endpoint_records.get("result")
        confirmations = [r for r in self._records if r["kind"] == "creation_outcome" and r["confirmed_creation"]]
        if not confirmations:
            issues.append("missing_confirmed_creation")
        if baseline is not None:
            prior = [r for r in confirmations if r["sequence"] < baseline["sequence"] and not r["issues"]]
            if not prior:
                issues.append("no_valid_creation_and_post_pose_before_baseline")
            if any(r["sequence"] > baseline["sequence"] for r in confirmations):
                issues.append("creation_after_baseline")
            # A failed/ambiguous attempt may have changed a body too: never let it
            # disappear just because it did not supply a confirmed outcome.
            end = result["sequence"] if result is not None else len(self._records)
            if any(r["kind"] in ("activation_request", "creation_outcome")
                   for r in self._records[baseline["sequence"]+1:end+1]):
                issues.append("attachment_operation_inside_endpoint_interval")
            if prior:
                start = prior[-1]["sequence"]
                end = result["sequence"] if result is not None else len(self._records)
                if any(r["kind"] in ("release", "reset", "truncation", "termination", "invalid", "teardown")
                       for r in self._records[start+1:end+1]):
                    issues.append("attachment_or_lifecycle_interruption")
        if baseline is not None and result is not None and result["sequence"] <= baseline["sequence"]:
            issues.append("endpoint_order_invalid")
        # A teardown after the result is retained but not mistaken for an in-window event.
        return _json(deepcopy({"episode_id": self.episode_id, "origin_step": self.origin_step,
            "origin_time_s": self.origin_time, "records": self._records,
            "endpoints": self._endpoints, "evidence_qualification_issues": issues,
            "event_endpoint_preconditions_available": not issues,
            "baseline_settling_qualified": None, "lift_result": None,
            "table_clear_rule": None, "policy_input": False,
            "integration": "callable_only_not_connected_to_capture"}))
