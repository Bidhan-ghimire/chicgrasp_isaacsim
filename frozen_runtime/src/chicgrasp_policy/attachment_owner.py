"""Isolated, build-pinned external FixedJoint owner; no application ownership.

Cycle 21 accepts Cycle 20's saved synthetic CPU observations for this offline
implementation only. Native registration is not physical holding or grasp
success. The station predicate and capture caller remain external dependencies.
Native imports occur only in explicit bind_runtime; ordinary import is inert.
"""
from copy import deepcopy
from uuid import uuid4

import numpy as np

from .lift_event_pose_evidence import (
    CHICKEN_BODY, GRIPPER_BODY, TASK_ID, LiftEventPoseEvidence, NativePoseReader,
    _invalid_metadata, _json,
)


from dataclasses import asdict, dataclass


PINNED_RUNTIME = {
    "isaac_sim": "5.1.0-rc.19+release.26219.9c81211b.gl",
    "isaac_lab": "2.3.2",
    "omni_physx": "107.3.26",
    "integration_build": "107.3.26+release.47132.3a61992c.gl",
    "native_python": "3.11.13",
    "binding_path": r"C:\isaacsim\extscache\omni.physx-107.3.26+107.3.3.wx64.r.cp311.u353\omni\physx\bindings\_physx.cp311-win_amd64.pyd",
    "binding_bytes": 1311840,
    "binding_sha256": "7fd515caa211084c1b4b2d57fbaba3fda232c280981ceab122143e4220b0265e",
}

# Separate empirical categories, not a provider enum or uniqueness assertion.
_EMPIRICAL_PROVENANCE = {
    "non_native_fixture": False,
    "creation_category": 6,
    "destruction_category": 6,
    "scope": "one external FixedJoint; synthetic CPU/MBP scene; stop_callback_when_sim_stopped=False",
    "cycle20_report_sha256": "730bef4a2501dd8cc179eac4bc15e8c06d1a57fd22bf9042bb75988c6aa3049d",
    "cycle20_journal_sha256": "4d6d027726035a117e99851f135be85a04b6d5bbf316817284facd05b2863c18",
    "cycle20_reconciliation_sha256": "47b5a2ea7774eaa601cc655d5ddc9e9e4e3fbb4af610cd035084e7d34b9d0e90",
    "creation_timing": "observed inside creation flush, available at immediate checkpoint",
    "destruction_timing": "observed inside RemovePrim before removal flush, available at immediate checkpoint",
    "offline_acceptance": "Cycle21 accepts authenticated observations; Cycle20 remains historically BLOCKED",
    "production_articulation_backend_validated": False,
    "physical_holding_tested": False,
    "projection_applicability": "UNRESOLVED",
    "projection_attribute_authored": False,
    "open_issues": ["SHUTDOWN_PROCESS_EXIT_UNRESOLVED", "USDRT_GLOBAL_TIME_WARNING_UNRESOLVED"],
}


@dataclass
class OwnedDescription:
    path: str
    token: str
    stage_id: int
    prim: object = None
    expected: object = None


class NativeRuntime:
    """Concrete existing-stage boundary; no app ownership or implicit processing.

    Installed omni.physx 107.3.26 _physx.pyi:1348-1365 supplies flush/stage;
    :1886-1898,1926-1933 supplies object observer and unsubscribe. USD calls
    follow the authenticated Cycle20 author/snapshot/remove implementation.
    """

    def __init__(self, stage, *, task_id, num_envs, modules, physics, simulation, identity,
                 scene_identity=None):
        self.stage, self.task_id, self.num_envs = stage, task_id, num_envs
        self.modules, self.physics, self.simulation = modules, physics, simulation
        self.identity = deepcopy(identity)
        self.scene_identity = scene_identity
        self.creation_category = 6
        self.destruction_category = 6
        self.provenance = {**deepcopy(_EMPIRICAL_PROVENANCE), "runtime": deepcopy(identity)}
        self._stage_id, self._root_layer = None, None
        self._body_prims = None

    def current_stage_id(self):
        if self.scene_identity is not None:
            self.scene_identity.validate("owner_stage_association")
            # USD cache domain for this source-qualified managed construction;
            # never represented as a matching IPhysxSimulation attachment ID.
            return self.scene_identity.stage_id
        # Actual attached stage, not a field invented in the callback payload.
        value = self.simulation.get_attached_stage()
        if type(value) is not int or value <= 0:
            raise ValueError("native_stage_identity_unavailable")
        return value

    def _stage_guard(self):
        if self.stage is None:
            raise ValueError("existing_stage_required")
        cache_id = self.modules.UsdUtils.StageCache.Get().GetId(self.stage).ToLongInt()
        current_id = self.current_stage_id()
        layer = self.stage.GetRootLayer()
        if (cache_id <= 0 or cache_id != current_id
                or (self._stage_id is not None and cache_id != self._stage_id)
                or (self._root_layer is not None and layer != self._root_layer)
                or self.stage.GetEditTarget().GetLayer() != layer):
            raise ValueError("stage_cache_attachment_or_edit_layer_changed")
        return current_id, layer

    def validate(self):
        if self.task_id != TASK_ID or type(self.num_envs) is not int or self.num_envs != 1:
            raise ValueError("exact_dynamic_task_and_one_environment_required")
        if self.identity != PINNED_RUNTIME:
            raise ValueError("incompatible_empirical_runtime_identity")
        stage_id, layer = self._stage_guard()
        m = self.modules
        if (m.UsdGeom.GetStageMetersPerUnit(self.stage) != 1.0
                or m.UsdPhysics.GetStageKilogramsPerUnit(self.stage) != 1.0
                or str(m.UsdGeom.GetStageUpAxis(self.stage)) != "Z"):
            raise ValueError("metres_kilograms_z_up_required")
        envs = self.stage.GetPrimAtPath("/World/envs")
        if not envs or [str(p.GetPath()) for p in envs.GetChildren()] != ["/World/envs/env_0"]:
            raise ValueError("exact_one_environment_namespace_required")
        bodies = []
        for path in (GRIPPER_BODY, CHICKEN_BODY):
            prim = self.stage.GetPrimAtPath(path)
            if not prim or not prim.HasAPI(m.UsdPhysics.RigidBodyAPI):
                raise ValueError("expected_rigid_actor_missing:" + path)
            actor = m.UsdPhysics.RigidBodyAPI(prim)
            if (actor.GetRigidBodyEnabledAttr().Get() is not True
                    or actor.GetKinematicEnabledAttr().Get() is not False):
                raise ValueError("expected_enabled_dynamic_actor:" + path)
            bodies.append(prim)
        if self._body_prims is not None and bodies != self._body_prims:
            raise ValueError("bound_actor_description_replaced")
        self._body_prims = bodies
        self._stage_id, self._root_layer = stage_id, layer
        return stage_id

    def decode(self, encoded_path):
        return str(self.modules.PhysicsSchemaTools.intToSdfPath(encoded_path))

    def subscribe(self, created, destroyed, all_destroyed):
        self.validate()
        return self.physics.subscribe_object_changed_notifications(
            created, destroyed, all_destroyed, stop_callback_when_sim_stopped=False)

    def unsubscribe(self, subscription):
        self.physics.unsubscribe_object_change_notifications(subscription)

    def exists(self, path):
        return bool(self.stage.GetPrimAtPath(path))

    def _owned_prim(self, owned):
        stage_id, _ = self._stage_guard()
        prim = self.stage.GetPrimAtPath(owned.path)
        if (stage_id != owned.stage_id or owned.prim is None or not prim
                or prim != owned.prim or prim.GetStage() != self.stage
                or not prim.IsA(self.modules.UsdPhysics.FixedJoint)
                or prim.GetCustomDataByKey("chicgraspAttachmentOwner") != owned.token):
            raise ValueError("owned_joint_lost_replaced_or_wrong_stage")
        return prim

    def verify(self, owned, complete=True):
        prim = self._owned_prim(owned)
        if not complete:
            # Partial cleanup cannot depend on complete authoring or live actors.
            partial = {"path": owned.path, "token": owned.token, "stage_id": owned.stage_id,
                       "type": str(prim.GetTypeName()), "partial_ownership_verified": True,
                       "authored_attributes": {}, "relationships": {}}
            for attr in prim.GetAuthoredAttributes():
                name = str(attr.GetName())
                try:
                    value = attr.Get()
                    if name.startswith("physics:localPos"):
                        value = [float(v) for v in value]
                    elif name.startswith("physics:localRot"):
                        value = [float(value.GetReal()), *[float(v) for v in value.GetImaginary()]]
                    partial["authored_attributes"][name] = value
                except Exception as exc:
                    partial["authored_attributes"][name] = {"unavailable": repr(exc)}
            for name in ("physics:body0", "physics:body1"):
                try:
                    partial["relationships"][name] = [str(v) for v in prim.GetRelationship(name).GetTargets()]
                except Exception as exc:
                    partial["relationships"][name] = {"unavailable": repr(exc)}
            return deepcopy(partial)
        self.validate()
        joint = self.modules.UsdPhysics.FixedJoint(prim)
        values = {}
        names = ("physics:collisionEnabled", "physics:excludeFromArticulation", "physics:breakForce",
                 "physics:breakTorque", "physics:jointEnabled", "physics:localPos0", "physics:localPos1",
                 "physics:localRot0", "physics:localRot1")
        for name in names:
            attr = prim.GetAttribute(name)
            value = attr.Get()
            if name.startswith("physics:localPos"):
                value = [float(v) for v in value]
            elif name.startswith("physics:localRot"):
                value = [float(value.GetReal()), *[float(v) for v in value.GetImaginary()]]
            values[name] = {"value": value, "authored": attr.HasAuthoredValueOpinion()}
        snapshot = {"path": owned.path, "token": owned.token, "stage_id": owned.stage_id,
                    "type": str(prim.GetTypeName()),
                    "body0": [str(x) for x in joint.GetBody0Rel().GetTargets()],
                    "body1": [str(x) for x in joint.GetBody1Rel().GetTargets()],
                    "attributes": values,
                    "applied_schemas": list(prim.GetAppliedSchemas()),
                    "authored_attribute_names": sorted(str(a.GetName()) for a in prim.GetAuthoredAttributes())}
        if (snapshot["body0"] != [GRIPPER_BODY] or snapshot["body1"] != [CHICKEN_BODY]
                or snapshot["applied_schemas"] or set(snapshot["authored_attribute_names"]) != set(names)
                or not all(x["authored"] for x in values.values())
                or any(values[name]["value"] is not True for name in ("physics:collisionEnabled", "physics:excludeFromArticulation"))
                or any(values[name]["value"] != float("inf") for name in ("physics:breakForce", "physics:breakTorque"))
                or values["physics:localPos1"]["value"] != [0.0, 0.0, 0.0]
                or values["physics:localRot1"]["value"] != [1.0, 0.0, 0.0, 0.0]):
            raise ValueError("owned_joint_properties_changed_or_incomplete")
        if owned.expected is not None and snapshot != owned.expected:
            raise ValueError("owned_joint_readback_does_not_match_authored_description")
        return deepcopy(snapshot)

    def author(self, owned, anchor):
        if self.validate() != owned.stage_id:
            raise ValueError("operation_stage_mismatch")
        suffix = owned.path.removeprefix("/World/ChicGraspAttachment_")
        if (not owned.path.startswith("/World/ChicGraspAttachment_") or not suffix or "/" in suffix
                or not isinstance(owned.token, str) or not owned.token or owned.prim is not None
                or owned.expected is not None or self.exists(owned.path)):
            raise ValueError("initially_absent_unique_owned_path_required")
        raw = np.asarray(anchor, dtype=np.float64)
        if raw.shape != (7,) or not np.isfinite(raw).all() or not np.linalg.norm(raw[3:]) > 0:
            raise ValueError("finite_measured_relative_pose_required")
        m = self.modules
        pos = m.Gf.Vec3f(*raw[:3])
        rot = m.Gf.Quatf(float(raw[3]), m.Gf.Vec3f(*raw[4:]))
        expected_pos = [float(x) for x in pos]
        expected_rot = [float(rot.GetReal()), *[float(x) for x in rot.GetImaginary()]]
        if not np.isfinite(expected_pos + expected_rot).all() or not np.linalg.norm(expected_rot) > 0:
            raise ValueError("finite_float32_relative_anchor_required")
        joint = m.UsdPhysics.FixedJoint.Define(self.stage, owned.path)
        # Retain the actual handle immediately for any later partial failure.
        owned.prim = joint.GetPrim()
        disabled = joint.CreateJointEnabledAttr(False)
        # Setter return types alone do not prove a disabled, owned description.
        # Stop before any body/anchor authoring if either prerequisite failed.
        if not disabled or not disabled.HasAuthoredValueOpinion() or disabled.Get() is not False:
            raise ValueError("initial_disabled_state_not_verified")
        owned.prim.SetCustomDataByKey("chicgraspAttachmentOwner", owned.token)
        self._owned_prim(owned)
        joint.GetBody0Rel().SetTargets([m.Sdf.Path(GRIPPER_BODY)])
        joint.GetBody1Rel().SetTargets([m.Sdf.Path(CHICKEN_BODY)])
        joint.CreateLocalPos0Attr(pos)
        joint.CreateLocalRot0Attr(rot)
        joint.CreateLocalPos1Attr(m.Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot1Attr(m.Gf.Quatf(1, m.Gf.Vec3f(0, 0, 0)))
        joint.CreateCollisionEnabledAttr(True)
        joint.CreateExcludeFromArticulationAttr(True)
        joint.CreateBreakForceAttr(float("inf"))
        joint.CreateBreakTorqueAttr(float("inf"))
        snapshot = self.verify(owned)
        if (snapshot["attributes"]["physics:jointEnabled"]["value"] is not False
                or snapshot["attributes"]["physics:localPos0"]["value"] != expected_pos
                or snapshot["attributes"]["physics:localRot0"]["value"] != expected_rot):
            raise ValueError("disabled_complete_float32_authoring_readback_mismatch")
        owned.expected = deepcopy(snapshot)
        return snapshot

    def enable(self, owned):
        self.verify(owned)
        # Only expected enabled value changes; all other authoring stays frozen.
        expected = deepcopy(owned.expected)
        if expected is None or expected["attributes"]["physics:jointEnabled"]["value"] is not False:
            raise ValueError("complete_disabled_description_required")
        expected["attributes"]["physics:jointEnabled"]["value"] = True
        self.modules.UsdPhysics.FixedJoint(owned.prim).GetJointEnabledAttr().Set(True)
        owned.expected = expected
        return self.verify(owned)

    def flush(self):
        self._stage_guard()
        # Exactly one processing call. Owner controls the operation budget.
        self.simulation.flush_changes()

    def remove(self, owned):
        self.verify(owned, complete=owned.expected is not None)
        if not self.stage.RemovePrim(owned.path):
            raise RuntimeError("owned_usd_removal_returned_false")
        if self.exists(owned.path):
            raise RuntimeError("owned_usd_path_not_absent_after_removal")
        return True


def _installed_identity(binding):
    """Read exact loaded-module/file identities; no introspection experiment."""
    import hashlib
    from pathlib import Path
    import platform
    import tomllib

    path = Path(binding.__file__).resolve()
    data = path.read_bytes()
    root = path.parents[3]
    package = tomllib.loads((root / "config/extension.toml").read_text(encoding="utf-8-sig"))
    generated = tomllib.loads((root / "config/extension.gen.toml").read_text(encoding="utf-8-sig"))
    return {
        "isaac_sim": Path("C:/isaacsim/VERSION").read_text(encoding="utf-8-sig").strip(),
        "isaac_lab": Path("C:/robotics/IsaacLab/VERSION").read_text(encoding="utf-8-sig").strip(),
        "omni_physx": package["package"]["version"],
        "integration_build": generated["package"]["publish"]["buildNumber"],
        "native_python": platform.python_version(),
        "binding_path": str(path), "binding_bytes": len(data),
        "binding_sha256": hashlib.sha256(data).hexdigest(),
    }


def bind_runtime(env, stage, *, task_id, episode_id=None, origin_step=None, origin_time=None,
                 diagnostic_context=None, native_origin=None, native_origin_source=None, capture_policy=False,
                 scene_identity=None):
    """Future explicit binding to an existing task; never starts an application."""
    if type(capture_policy) is not bool or (capture_policy and diagnostic_context is not None):
        raise ValueError("explicit_unmixed_capture_policy_required")
    if diagnostic_context is not None:
        if any(v is not None for v in (episode_id, origin_step, origin_time)):
            raise ValueError("episode_and_diagnostic_construction_cannot_mix")
    elif native_origin is not None or native_origin_source is not None:
        raise ValueError("diagnostic_origin_requires_explicit_diagnostic_context")
    if task_id != TASK_ID or type(env.num_envs) is not int or env.num_envs != 1:
        raise ValueError("exact_dynamic_task_and_one_environment_required")
    if scene_identity is not None and (not capture_policy or scene_identity.env is not env):
        raise ValueError("managed_identity_requires_its_capture_environment")
    from types import SimpleNamespace
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdUtils, PhysicsSchemaTools
    from omni.physx import get_physx_interface, get_physx_simulation_interface
    from omni.physx.bindings import _physx

    identity = _installed_identity(_physx)
    if identity != PINNED_RUNTIME:
        raise ValueError("incompatible_empirical_runtime_identity")
    runtime = NativeRuntime(
        stage, task_id=task_id, num_envs=env.num_envs,
        modules=SimpleNamespace(Gf=Gf, Sdf=Sdf, UsdGeom=UsdGeom, UsdPhysics=UsdPhysics,
                                UsdUtils=UsdUtils, PhysicsSchemaTools=PhysicsSchemaTools),
        physics=get_physx_interface(), simulation=get_physx_simulation_interface(), identity=identity,
        scene_identity=scene_identity)
    runtime.validate()
    if scene_identity is not None:
        scene_identity.validate("owner_bind", emit=True)
    reader = NativePoseReader(env, task_id=task_id, scene_identity=scene_identity)
    if diagnostic_context is None:
        factory = LiftEventPoseEvidence.for_capture if capture_policy else LiftEventPoseEvidence
        journal = factory(reader, episode_id=episode_id, origin_step=origin_step, origin_time=origin_time)
    else:
        journal = LiftEventPoseEvidence.for_diagnostic(reader, context=diagnostic_context,
            native_origin=native_origin, native_origin_source=native_origin_source)
    return AttachmentOwner(runtime, journal)


class AttachmentOwner:
    """One owner/episode with synchronous, non-pumping operation checkpoints.

    The callback only appends diagnostics. No lock spans a native boundary;
    reentrant public mutations are rejected. Local generation/stage/clock are
    correlation context, never represented as native callback payload fields.
    """

    def __init__(self, runtime, journal):
        if not isinstance(journal, LiftEventPoseEvidence) or not isinstance(journal.reader, NativePoseReader):
            raise TypeError("accepted_event_and_native_pose_producer_required")
        self.runtime, self.journal = runtime, journal
        self.owner_id = uuid4().hex
        self._stage_id = runtime.validate()
        self.journal.validate_owner_stage(self._stage_id)
        self._operations, self._callbacks = [], []
        self._active = None
        self._owned = None
        self._native_id = None
        self._generation = 0
        self._held = False
        self._closed = False
        self._busy = False
        self._failed = False
        self._release_result = None
        self._last_completed_receipt = 0
        self._pending = None
        self._capture_requested = False
        self._capture_state = "UNATTACHED"
        self._capture_request_context = self._capture_confirmation_context = None
        self._capture_request_clock = self._capture_confirmation_clock = None
        self._capture_request_relationship = None
        self._capture_path = None
        self._capture_verified = False
        self._subscription = None
        self._subscription_issue = None
        try:
            self._subscription = runtime.subscribe(
                lambda p, i, t: self._receive("creation", p, i, t),
                lambda p, i, t: self._receive("destruction", p, i, t),
                lambda: self._receive("all_objects_destroyed", None, None, None),
            )
        except Exception as exc:
            if not journal.capture_policy:
                raise
            self._subscription_issue = repr(exc)

    @property
    def held(self):
        # An unexpected native loss immediately revokes presentation as held,
        # without mutating the scene or retroactively modifying an operation.
        return self._held and not self._unexpected_loss()

    def _unexpected_loss(self):
        if self.journal.capture_policy:
            return False  # Scheduled USD verification, not callback timing/categories.
        if self._owned is None:
            return False
        return any(c["kind"] == "all_objects_destroyed" or
                   (c["kind"] in ("creation", "destruction") and
                    c.get("decoded_path") == self._owned.path)
                   for c in self._callbacks[self._last_completed_receipt:])

    def _receive(self, kind, path, object_id, category):
        active = self._active
        record = {"receipt": len(self._callbacks), "kind": kind,
                  "native": ({} if kind == "all_objects_destroyed" else
                             {"path": path, "object_id": object_id, "physx_type": category}),
                  "local": {"owner_id": self.owner_id,
                            "generation": active["generation"] if active else None,
                            "operation_id": active["id"] if active else None,
                            "phase": active["kind"] if active else None},
                  "decoded_path": None, "issues": []}
        try:
            record["local"]["stage_id"] = self.runtime.current_stage_id()
            record["local"]["clock"] = tuple(self.journal.reader.clock_identity())
            if kind != "all_objects_destroyed":
                # The binding specifies integer identifiers, not an object-ID
                # range or a sentinel value. Decode paths and correlate opaque
                # IDs; do not confuse an ID's numeric value with its category.
                if not all(type(value) is int for value in (path, object_id, category)):
                    raise ValueError("invalid_native_notification_fields")
                record["decoded_path"] = str(self.runtime.decode(path))
        except Exception as exc:
            record["issues"].append(type(exc).__name__ + ": " + str(exc))
        self._callbacks.append(deepcopy(record))

    def _begin(self, kind, **fields):
        if self._busy:
            raise RuntimeError("reentrant_owner_operation_rejected")
        self._busy = True
        op = {"id": len(self._operations), "kind": kind, "generation": self._generation,
              "owner_id": self.owner_id, "issues": [], **fields}
        if self.journal.provenance_domain == "diagnostic":
            op.update(provenance_domain="diagnostic", lifecycle_only=True,
                      diagnostic_context=asdict(self.journal.diagnostic_context), control_frame_history=None)
        self._operations.append(op)
        return op

    def _finish(self, op):
        self._active = None
        self._busy = False
        op["completed"] = True
        return _json(deepcopy(op))

    def _arm(self, op, clock):
        op.update(receipt_boundary=len(self._callbacks), stage_id=self._stage_id,
                  checkpoint_clock=deepcopy(clock), path=self._owned.path, token=self._owned.token)
        self._active = op

    def _checkpoint(self, op, kind, expected_id=None):
        """No waits or additional processing: decide on receipts already copied."""
        self._active = None
        op["receipt_end"] = len(self._callbacks)
        op["callback_indices"] = list(range(op["receipt_boundary"], len(self._callbacks)))
        records = self._callbacks[op["receipt_boundary"]:]
        matches = []
        category = getattr(self.runtime, kind + "_category")
        for rec in records:
            if rec["kind"] == "all_objects_destroyed":
                op["issues"].append("scene_wide_destruction_is_not_individual_confirmation")
                continue
            if rec["decoded_path"] != op["path"]:
                continue  # retained unrelated report, not a candidate confirmation
            raw, local = rec["native"], rec["local"]
            valid = (rec["kind"] == kind and not rec["issues"]
                     and local.get("stage_id") == self._stage_id
                     and local.get("operation_id") == op["id"]
                     and local.get("generation") == op["generation"]
                     and local.get("clock") == tuple(op["checkpoint_clock"])
                     and raw["physx_type"] == category
                     and (expected_id is None or raw["object_id"] == expected_id))
            if valid:
                matches.append(rec)
            else:
                op["issues"].append("owned_path_notification_mismatch")
        op["matching_native_receipts"] = [r["receipt"] for r in matches]
        if len(matches) != 1:
            op["issues"].append("missing_native_confirmation" if not matches else "duplicate_native_confirmation")
        try:
            if tuple(self.journal.reader.clock_identity()) != tuple(op["checkpoint_clock"]):
                op["issues"].append("native_clock_changed_during_operation")
            if self.runtime.validate() != self._stage_id:
                op["issues"].append("stage_changed_during_operation")
        except Exception as exc:
            op["issues"].append("checkpoint_validation: " + repr(exc))
        op["native_observation_count"] = len(matches)
        return matches

    def _record(self, sequence):
        return self.journal.record(sequence)

    def snapshot(self):
        """Last scheduled USD verification, not a new native call or hold claim."""
        if not self.journal.capture_policy:
            raise ValueError("explicit_capture_policy_required")
        return _json(deepcopy({"status": self._capture_state, "joint_path": self._capture_path,
            "generation": None if self._capture_state == "UNATTACHED" else self._generation,
            "owned_usd_verified": self._capture_verified,
            "request_context": self._capture_request_context,
            "confirmation_context": self._capture_confirmation_context,
            "request_native_clock": self._capture_request_clock,
            "confirmation_native_clock": self._capture_confirmation_clock,
            "request_relationship": self._capture_request_relationship,
            "held": self._held, "confirmation_basis": "scheduled_step_owned_USD_and_relative_pose"}))

    def _capture_failure(self, op, context, cause):
        self._held = False
        self._failed = True
        if self._pending is not None:
            op.setdefault("request_id", self._pending["request_id"])
        self._pending = None
        self._active = None
        if "request_id" in op:
            try:
                op["failure_evidence_id"] = self.journal.record_capture_failure(op["request_id"], context, cause)
                failure = self._record(op["failure_evidence_id"])
                op["failure_pose_reading"] = deepcopy(failure["pose_reading"])
            except Exception as exc:
                op["issues"].append("capture_failure_evidence_error: " + repr(exc))
        if self._owned is not None:
            self._busy = False
            try:
                op["cleanup"] = self.release(context, cause)
            finally:
                self._busy = True
        self._capture_state = "FAILED"
        self._capture_verified = False
        op.update(self.snapshot())
        return self._finish(op)

    def _capture_request(self, context, eligibility):
        op = self._begin("scheduled_request", authored=False, processed=False,
                         activation_qualified=False, native_creation_observed=False)
        try:
            if self._closed or self._failed or self._capture_requested or self._owned is not None:
                op["issues"].append("capture_request_once_only")
                return self._capture_failure(op, context, "repeated_or_invalid_request")
            self._capture_requested = True
            self._generation += 1
            op["generation"] = self._generation
            request_id = self.journal.request_activation(context, eligibility)
            op["request_id"] = request_id
            request = self._record(request_id)
            self._capture_request_context = deepcopy(request["context"])
            self._capture_request_clock = deepcopy(request["native_clock"])
            op["pose_reading"] = deepcopy(request["pose_reading"])
            op["issues"].extend(request["issues"] + request["eligibility_issues"])
            if not op["issues"]:
                op["issues"].extend(self.journal.activation_phase_issues(context))
            if op["issues"]:
                return self._capture_failure(op, context, "invalid_capture_request")
            anchor = np.asarray(request["pose_reading"]["poses"]["chicken_in_gripper_wxyz"], dtype=np.float64)
            self._capture_request_relationship = anchor.tolist()
            token = self.owner_id + "_" + uuid4().hex
            self._owned = OwnedDescription("/World/ChicGraspAttachment_" + token, token, self._stage_id)
            self._capture_path = self._owned.path
            self._release_result = None
            self._arm(op, request["native_clock"])
            # These are the same reviewed author/enable/verify implementation,
            # not a second scene/joint implementation. No flush or advancement.
            op["disabled_description"] = self.runtime.author(self._owned, anchor)
            op["authored"] = True
            op["enabled_description"] = self.runtime.enable(self._owned)
            self.runtime.verify(self._owned)
            if tuple(self.journal.reader.clock_identity()) != tuple(request["native_clock"]):
                raise ValueError("native_clock_changed_during_request_authoring")
            attributes = op["enabled_description"]["attributes"]
            relation = {"parent": GRIPPER_BODY, "child": CHICKEN_BODY,
                "convention": "chicken_in_gripper_position_xyz_quaternion_wxyz_m",
                "pose": attributes["physics:localPos0"]["value"] + attributes["physics:localRot0"]["value"]}
            self._pending = {"request_id": request_id, "relationship": relation}
            pending_id = self.journal.record_pending(request_id, context, self._owned.path)
            op["pending_event_id"] = pending_id
            op["issues"].extend(self._record(pending_id)["issues"])
            if op["issues"]:
                return self._capture_failure(op, context, "invalid_pending_evidence")
            self._capture_state, self._capture_verified = "PENDING", True
            op["receipt_end"] = len(self._callbacks)
            op.update(self.snapshot())
            return self._finish(op)
        except Exception as exc:
            op["issues"].append("capture_request_error: " + repr(exc))
            if self._owned is not None:
                try:
                    op["partial_description"] = self.runtime.verify(self._owned, complete=False)
                except Exception as ownership_error:
                    op["partial_description_unavailable"] = repr(ownership_error)
            return self._capture_failure(op, context, "failed_capture_request")

    def confirm_pending(self, context):
        """Observe exactly the next scheduler-owned substep; never perform it."""
        if not self.journal.capture_policy:
            raise ValueError("explicit_capture_policy_required")
        op = self._begin("scheduled_confirmation", activation_qualified=False,
                         native_creation_observed=False, processed=False)
        try:
            if self._pending is None or self._closed or self._failed:
                op["issues"].append("no_pending_capture_request")
                return self._capture_failure(op, context, "confirmation_without_pending")
            pending = self._pending
            op["request_id"] = pending["request_id"]
            try:
                op["owned_description"] = self.runtime.verify(self._owned)
                self._capture_verified = True
            except Exception as exc:
                self._capture_verified = False
                op["issues"].append("owned_description_unavailable: " + repr(exc))
            outcome_id = self.journal.creation_outcome(pending["request_id"], context,
                "confirmed" if not op["issues"] else "unconfirmed",
                "attachment_owner:" + self.owner_id + ":scheduled_confirmation",
                relationship_used=pending["relationship"], details={"owner_operation_id": op["id"]})
            op["outcome_id"] = outcome_id
            outcome = self._record(outcome_id)
            op["pose_reading"] = deepcopy(outcome["pose_reading"])
            op["relative_change"] = deepcopy(outcome["relative_change"])
            op["issues"].extend(outcome["issues"])
            if op["issues"] or not outcome["scheduled_confirmation"]:
                return self._capture_failure(op, context, "failed_scheduled_confirmation")
            self._pending = None
            self._held = True
            self._capture_state = "CONFIRMED"
            self._capture_confirmation_context = deepcopy(outcome["context"])
            self._capture_confirmation_clock = deepcopy(outcome["native_clock"])
            op["activation_qualified"] = True
            op.update(self.snapshot())
            return self._finish(op)
        except Exception as exc:
            op["issues"].append("scheduled_confirmation_error: " + repr(exc))
            return self._capture_failure(op, context, "failed_scheduled_confirmation")

    def _capture_update(self, context, eligibility, jaw_commands):
        try:
            contact_evidence = _json(eligibility)
        except Exception as exc:
            contact_evidence = {"diagnostic_serialization_error": type(exc).__name__,
                                "unavailable_metadata_type": type(eligibility).__name__}
        op = self._begin("scheduled_observation", eligibility=contact_evidence)
        jaw_open = False
        try:
            event_id = self.journal.capture_observation(context)
            observation = self._record(event_id)
            op["event_id"] = event_id
            op["pose_reading"] = deepcopy(observation["pose_reading"])
            op["issues"].extend(observation["issues"])
            jaws = np.asarray(jaw_commands, dtype=float)
            if jaws.shape != (2,) or not np.isfinite(jaws).all() or not np.all((jaws == 0) | (jaws == 1)):
                raise ValueError("invalid_jaw_commands")
            jaw_open = bool(np.any(jaws == 1))
            contact = contact_evidence.get("bilateral_contact") if isinstance(contact_evidence, dict) else None
            op["contact_diagnostic"] = deepcopy(contact)
            if self._pending is not None:
                op["issues"].append("pending_request_requires_explicit_next_substep_confirmation")
            if self._owned is not None:
                try:
                    op["owned_description"] = self.runtime.verify(self._owned)
                    self._capture_verified = True
                    if tuple(self.journal.reader.clock_identity()) != tuple(observation["native_clock"]):
                        raise ValueError("native_clock_changed_during_scheduled_verification")
                except Exception:
                    self._capture_verified = False
                    raise
                # After authoring, contact (including unavailable/malformed
                # reporting) is diagnostic only. The actual pose/clock and owned
                # USD description above remain required at this boundary.
            if op["issues"]:
                return self._capture_failure(op, context, "invalid_required_measurements_or_owned_joint")
            if jaw_open and self._owned is not None:
                self._busy = False
                try:
                    op["release"] = self.release(context, "jaw_open")
                finally:
                    self._busy = True
                op["issues"].extend(op["release"]["issues"])
            op.update(self.snapshot())
            return self._finish(op)
        except Exception as exc:
            op["issues"].append("required_measurement_or_owned_joint_error: " + repr(exc))
            return self._capture_failure(op, context, "invalid_required_measurements_or_owned_joint")

    def activate(self, context, eligibility):
        if self.journal.capture_policy:
            return self._capture_request(context, eligibility)
        op = self._begin("activation", authored=False, processed=False,
                         native_creation_observed=False, activation_qualified=False)
        relationship = None
        try:
            if (self._closed or self._failed or self._owned is not None
                    or (self._release_result is not None and not self._release_result["cleanup_confirmed"])):
                op["issues"].append("closed_failed_or_existing_owner_resource")
                return self._finish(op)
            self._generation += 1
            op["generation"] = self._generation
            request_id = self.journal.request_activation(context, eligibility)
            op["request_id"] = request_id
            request = self._record(request_id)
            op["issues"].extend(request["issues"] + request["eligibility_issues"])
            if not op["issues"]:
                op["issues"].extend(self.journal.activation_phase_issues(context))
            if not op["issues"]:
                try:
                    if self.runtime.validate() != self._stage_id:
                        raise ValueError("stage_changed")
                    anchor = np.asarray(request["pose_reading"]["poses"]["chicken_in_gripper_wxyz"], dtype=np.float64)
                    if anchor.shape != (7,) or not np.all(np.isfinite(anchor)):
                        raise ValueError("invalid_measured_relative_pose")
                    relationship = {"parent": GRIPPER_BODY, "child": CHICKEN_BODY,
                                    "convention": "chicken_in_gripper_position_xyz_quaternion_wxyz_m",
                                    "pose": np.asarray(anchor, dtype=np.float32).tolist()}
                    op["measured_relative_pose_wxyz"] = anchor.tolist()
                    token = self.owner_id + "_" + uuid4().hex
                    self._owned = OwnedDescription("/World/ChicGraspAttachment_" + token, token, self._stage_id)
                    self._native_id = None
                    self._release_result = None
                    self._arm(op, request["native_clock"])
                    op["authoring_started"] = True
                    op["disabled_description"] = self.runtime.author(self._owned, anchor)
                    op["authored"] = True
                    op["enabled_description"] = self.runtime.enable(self._owned)
                    self.runtime.flush()  # exactly one; never pumped for confirmation
                    op["processed"] = True
                except Exception as exc:
                    op["issues"].append("author_or_process_error: " + repr(exc))
                    if self._owned is not None:
                        try:
                            op["partial_description"] = self.runtime.verify(self._owned, complete=False)
                        except Exception as ownership_error:
                            op["partial_description_unavailable"] = repr(ownership_error)
                if self._active is not None:
                    matches = self._checkpoint(op, "creation")
                    op["native_creation_observed"] = bool(matches)
                    if len(matches) == 1:
                        self._native_id = matches[0]["native"]["object_id"]
                    try:
                        self.runtime.verify(self._owned)
                    except Exception as exc:
                        op["issues"].append("description_validation: " + repr(exc))
            # Post-pose collection precedes ANY cleanup. Callback records never
            # insert shared journal entries between this pair.
            confirmed = op["native_creation_observed"] and not op["issues"]
            outcome_id = self.journal.creation_outcome(
                request_id, context, "confirmed" if confirmed else "unconfirmed",
                "attachment_owner:" + self.owner_id + ":operation:" + str(op["id"]),
                relationship_used=relationship,
                details={"owner_operation_id": op["id"], "native_observed": op["native_creation_observed"]},
            )
            op["outcome_id"] = outcome_id
            outcome = self._record(outcome_id)
            op["issues"].extend(outcome["issues"])
            op["activation_qualified"] = confirmed and not outcome["issues"]
            self._held = op["activation_qualified"]
        except Exception as exc:
            # Accepted reader normally reports unavailable readings itself;
            # preserve unexpected producer failure before bounded cleanup too.
            op["issues"].append("event_or_owner_error: " + repr(exc))
            self._held = False
        self._last_completed_receipt = op.get("receipt_end", len(self._callbacks))
        if self._held and self._unexpected_loss():
            op["issues"].append("owned_notification_after_checkpoint")
            op["activation_qualified"] = self._held = False
        self._active = None
        if not self._held and self._owned is not None:
            self._failed = True
            self._busy = False
            op["cleanup"] = self.release(context, "failed_activation_qualification")
            self._busy = True
        return self._finish(op)

    def release(self, context, cause):
        op = self._begin("release", cause=cause, usd_removed=False,
                         native_destruction_observed=False, cleanup_confirmed=False)
        self._held = False
        if self.journal.capture_policy:
            self._pending = None
            self._capture_verified = False
        try:
            if not isinstance(cause, str) or not cause.strip():
                raise ValueError("release_cause_required")
            if self._closed:
                op["issues"].append("observer_closed_cleanup_unavailable")
            elif self._release_result is not None:
                # No repeated flush/delete for an ambiguous prior removal.
                op.update(idempotent=bool(self._release_result["cleanup_confirmed"]),
                          cleanup_confirmed=bool(self._release_result["cleanup_confirmed"]),
                          prior_release_id=self._release_result["id"])
                if not op["cleanup_confirmed"]:
                    op["issues"].append("prior_cleanup_unconfirmed_no_retry")
            elif self._owned is None:
                # No operation has occurred: this is a no-op, not evidence of
                # a previously confirmed destruction or an idempotent release.
                op.update(idempotent=False, no_owned_resource=True, cleanup_required=False)
            else:
                try:
                    if self._unexpected_loss():
                        raise ValueError("unexpected_native_identity_change_ownership_unresolved")
                    self.runtime.verify(self._owned, complete=False)
                    if self._owned.expected is not None:
                        self.runtime.verify(self._owned)
                    clock = tuple(self.journal.reader.clock_identity())
                    # Invalid clocks cannot qualify evidence, but must not
                    # prevent the bounded removal of a verified owned prim.
                    op["issues"].extend(self.journal._clock_issues(context, clock))
                    self._arm(op, clock)  # MUST precede RemovePrim, not just flush
                    op["usd_removed"] = bool(self.runtime.remove(self._owned))
                    if not op["usd_removed"]:
                        op["issues"].append("usd_removal_not_confirmed")
                    if not self.journal.capture_policy:
                        self.runtime.flush()  # retain the during-RemovePrim receipts
                        op["processed"] = True
                except Exception as exc:
                    op["issues"].append("owned_removal_error: " + repr(exc))
                if self._active is not None and not self.journal.capture_policy:
                    matches = self._checkpoint(op, "destruction", self._native_id)
                    op["native_destruction_observed"] = bool(matches) and self._native_id is not None
                    if self._native_id is None:
                        op["issues"].append("native_object_identity_unavailable")
                    try:
                        if self.runtime.exists(self._owned.path):
                            op["issues"].append("owned_path_still_present_or_replaced")
                    except Exception as exc:
                        op["issues"].append("post_removal_path_unavailable: " + repr(exc))
                if self.journal.capture_policy:
                    op["receipt_end"] = len(self._callbacks)
                    op["callback_indices"] = list(range(op.get("receipt_boundary", len(self._callbacks)), len(self._callbacks)))
                    op["cleanup_basis"] = "verified_owned_USD_removal_no_extra_processing"
                op["cleanup_confirmed"] = (op["usd_removed"] and (self.journal.capture_policy or op["native_destruction_observed"])
                                           and not op["issues"])
                self._release_result = op
            op["release_event_id"] = self.journal.record_release(
                context, cause, "attachment_owner:" + self.owner_id + ":operation:" + str(op["id"]))
        except Exception as exc:
            op["issues"].append("release_evidence_error: " + repr(exc))
            op["cleanup_confirmed"] = False
        if self._release_result is not None:
            self._release_result = op
        if op["cleanup_confirmed"]:
            self._owned = None
            self._native_id = None
        self._last_completed_receipt = op.get("receipt_end", len(self._callbacks))
        if self.journal.capture_policy:
            if self._capture_requested:
                self._capture_state = "RELEASED" if op["cleanup_confirmed"] else "FAILED"
            op.update(self.snapshot())
        return self._finish(op)

    def update(self, context, eligibility, jaw_commands):
        """Fresh caller evidence every check; never a latched contact predicate."""
        if self.journal.capture_policy:
            return self._capture_update(context, eligibility, jaw_commands)
        if self._busy:
            raise RuntimeError("reentrant_owner_operation_rejected")
        issues = []
        jaw_open = False
        self._busy = True
        try:
            jaws = np.asarray(jaw_commands, dtype=float)
            if jaws.shape != (2,) or not np.all(np.isfinite(jaws)) or not np.all((jaws == 0) | (jaws == 1)):
                raise ValueError("invalid_jaw_commands")
            jaw_open = bool(np.any(jaws == 1))
            if not jaw_open:
                clock = self.journal.reader.clock_identity()
                issues.extend(self.journal._clock_issues(context, clock))
                issues.extend(self.journal._eligibility_issues(eligibility, context, clock))
                if _invalid_metadata(eligibility):
                    issues.append("nonfinite_or_unavailable_metadata")
                if self._owned is not None:
                    self.runtime.verify(self._owned)
                    if self._unexpected_loss():
                        issues.append("unexpected_native_identity_change")
        except Exception as exc:
            issues.append("required_evidence_or_ownership_unavailable: " + repr(exc))
        finally:
            self._busy = False
        if jaw_open:
            return self.release(context, "jaw_open")
        if issues:
            cause = "invalid_station_contact" if "passive_contact_not_valid" in issues else "required_evidence_unusable"
            result = self.release(context, cause)
            # The detailed fresh evidence/causes remain separate from release.
            op = self._begin("continued_eligibility", eligibility=deepcopy(eligibility),
                             validation_issues=issues, release_operation_id=result["id"])
            self._finish(op)
            return result
        op = self._begin("continued_eligibility", eligibility=deepcopy(eligibility),
                         validation_issues=[], held=self.held)
        return self._finish(op)

    def lifecycle(self, context, event):
        if self._busy:
            raise RuntimeError("reentrant_owner_operation_rejected")
        if event not in ("reset", "truncation", "termination", "teardown", "invalid"):
            raise ValueError("unknown_lifecycle_event")
        # Lifecycle invalidation cannot be undone by failure to record it.
        self._failed = True
        result = self.release(context, event)
        op = self._begin("lifecycle", event=event, release_operation_id=result["id"], recorded=False)
        try:
            op["event_id"] = self.journal.record_lifecycle(context, event, "attachment_owner:" + self.owner_id)
            op["recorded"] = True
        except Exception as exc:
            op["issues"].append("lifecycle_evidence_error: " + repr(exc))
        finally:
            self._finish(op)
        result["lifecycle_operation_id"] = op["id"]
        result["lifecycle_recorded"] = op["recorded"]
        result["lifecycle_issues"] = list(op["issues"])
        return result

    def close(self, context):
        if self._busy:
            raise RuntimeError("reentrant_owner_operation_rejected")
        if self._closed:
            return _json(deepcopy(self._close_result))
        cleanup, lifecycle_error = None, None
        try:
            cleanup = self.lifecycle(context, "teardown")
        except Exception as exc:
            lifecycle_error = repr(exc)
        op = self._begin("close", cleanup=cleanup, observer_unsubscribed=False)
        if lifecycle_error is not None:
            op["issues"].append("lifecycle_evidence_error: " + lifecycle_error)
        try:
            if self._subscription is not None:
                self.runtime.unsubscribe(self._subscription)
                op["observer_unsubscribed"] = True
            elif self.journal.capture_policy:
                op["observer_not_available"] = self._subscription_issue
        except Exception as exc:
            op["issues"].append("observer_cleanup_error: " + repr(exc))
        self._closed = True
        self._close_result = self._finish(op)
        return deepcopy(self._close_result)

    def export(self):
        return _json(deepcopy({"owner_id": self.owner_id, "held": self.held,
            **({"capture_policy": "next_scheduled_substep", "attachment": self.snapshot(),
                "optional_subscription_issue": self._subscription_issue} if self.journal.capture_policy else {}),
            **({"provenance_domain": "diagnostic", "lifecycle_only": True,
                "diagnostic_context": asdict(self.journal.diagnostic_context), "control_frame_history": None}
               if self.journal.provenance_domain == "diagnostic" else {}),
            "closed": self._closed, "operations": self._operations,
            "callbacks": self._callbacks, "event_evidence": self.journal.export(),
            "provenance": self.runtime.provenance,
            "projection_applicability": "UNRESOLVED_non_gating_no_property_authored",
            "physical_holding_tested": False, "policy_input": False,
            "capture_integration": "not_connected", "lift_result": None}))
