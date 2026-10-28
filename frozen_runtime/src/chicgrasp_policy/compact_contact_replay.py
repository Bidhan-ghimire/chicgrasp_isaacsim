"""Offline, one-boundary reconstruction of compact callback evidence.

Only recorded external observations supply the injected reader boundaries.
The saved decision is a comparison target, never a reader input. Native imports,
scene access and native calls are neither needed nor available here.
"""
from copy import deepcopy
import hashlib
import json
import struct
from types import SimpleNamespace

from . import jaw_contact_diagnostic as jaw
from .attachment_capture import bilateral_contact

CONTEXT_SCHEMA = "compact-contact-replay-context/v1"
INIT_SCHEMA = "compact-contact-initialization-routing/v1"


def _typed(value):
    if value is None:
        return ["none"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        return ["int", hex(value)]
    if type(value) is float:
        return ["f64", struct.pack(">d", value).hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, (tuple, list)):
        return ["sequence", [_typed(x) for x in value]]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return ["map", [[k, _typed(value[k])] for k in sorted(value)]]
    raise ValueError("unsupported_replay_context_type:" + type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(_typed(value), separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _intern(writer, value, kind):
    if writer is None:
        return {"inline": deepcopy(value)}
    return {"dictionary": writer.intern_dictionary(value, kind)}


def _resolve(value, reader):
    if not isinstance(value, dict) or set(value) not in ({"inline"}, {"dictionary"}):
        raise ValueError("invalid_replay_dictionary_reference")
    if "inline" in value:
        return deepcopy(value["inline"])
    if reader is None:
        raise ValueError("missing_replay_dictionary_reader")
    return reader.read_dictionary(value["dictionary"])


def _pack_calls(records, writer, kind):
    if not isinstance(records, list):
        raise ValueError("missing_observed_" + kind + "_calls")
    values, indexes, ids, keys = [], [], {}, []
    by_input, varying = {}, False
    for record in records:
        key = digest(record)
        if key not in ids:
            ids[key] = len(values)
            values.append(deepcopy(record))
            keys.append(key)
        indexes.append(ids[key])
        source = record.get("encoded") if kind == "decode" else [record.get("actor"), record.get("collider")]
        identity = digest(source)
        if identity in by_input and by_input[identity] != key:
            varying = True
        by_input[identity] = key
    # Stable native identities need one observation per distinct input, not a
    # path string or lookup trace for every duplicate header occurrence.
    pages = None
    if writer is None:
        values = [{"inline": value} for value in values]
    else:
        # Publish each newly observed mapping once, in bounded pages rather than
        # issuing one filesystem transaction per native path. The cache contains
        # identities/references, never retained callback payloads.
        if not hasattr(writer, "_contact_replay_mappings"):
            writer._contact_replay_mappings = {}
        cache = writer._contact_replay_mappings.setdefault(kind, {})
        fresh = [(key, value) for key, value in zip(keys, values) if key not in cache]
        for start in range(0, len(fresh), 256):
            batch = fresh[start:start+256]
            ref = writer.intern_dictionary([value for _, value in batch], kind + "-observed-page")
            for position, (key, _) in enumerate(batch):
                cache[key] = (ref, position)
        groups, order = {}, []
        for key in keys:
            ref, position = cache[key]
            path = ref["path"]
            if path not in groups:
                groups[path] = {"dictionary": ref, "positions": [], "keys": []}
                order.append(path)
            groups[path]["positions"].append(position)
            groups[path]["keys"].append(key)
        packed_keys = [key for path in order for key in groups[path]["keys"]]
        remap = {key: index for index, key in enumerate(packed_keys)}
        indexes = [remap[keys[index]] for index in indexes]
        pages = [{key: value for key, value in groups[path].items() if key != "keys"} for path in order]
    return {"mode": "ordered" if varying else "stable", "observations": values if pages is None else None,
            "observation_pages": pages,
            "order": indexes if varying else None, "call_count": len(records)}


class _Calls:
    def __init__(self, packed, reader, kind):
        self.packed, self.kind, self.index = packed, kind, 0
        if type(packed.get("call_count")) is not int or packed["call_count"] < 0:
            raise ValueError("invalid_recorded_call_count")
        references = packed["observations"]
        if packed.get("observation_pages") is not None:
            if references is not None:
                raise ValueError("ambiguous_observed_mapping_storage")
            self.values = []
            for page in packed["observation_pages"]:
                values = reader.read_dictionary(page["dictionary"])
                positions = page["positions"]
                if any(type(index) is not int or not 0 <= index < len(values) for index in positions):
                    raise ValueError("invalid_observed_mapping_page_index")
                self.values.extend(deepcopy(values[index]) for index in positions)
        else:
            self.values = [_resolve(v, reader) for v in references]
        self.lookup = {}
        for value in self.values:
            source = value.get("encoded") if kind == "decode" else [value.get("actor"), value.get("collider")]
            key = digest(source)
            if key in self.lookup and packed["mode"] == "stable":
                raise ValueError("contradictory_stable_observed_mapping")
            self.lookup[key] = value

    def __call__(self, *args):
        source = args[0] if self.kind == "decode" else list(args)
        if self.packed["mode"] == "ordered":
            if self.index >= len(self.packed["order"]):
                raise ValueError("missing_recorded_" + self.kind + "_call")
            record = self.values[self.packed["order"][self.index]]
            expected = record.get("encoded") if self.kind == "decode" else [record.get("actor"), record.get("collider")]
            if digest(source) != digest(expected):
                raise ValueError("recorded_" + self.kind + "_call_order_disagreement")
        elif self.packed["mode"] == "stable":
            if digest(source) not in self.lookup:
                raise ValueError("unmapped_recorded_" + self.kind + "_identity")
            record = self.lookup[digest(source)]
        else:
            raise ValueError("unsupported_observed_mapping_mode")
        self.index += 1
        if record.get("outcome") == "RAISE":
            name, message = record.get("exception_type"), record.get("message")
            if not isinstance(name, str) or not name or not isinstance(message, str):
                raise ValueError("malformed_recorded_native_exception")
            # Recreate only the exposed name/message, never execute a saved
            # class or import its native implementation.
            raise type(name, (Exception,), {})(message)
        if record.get("outcome") != "RETURN":
            raise ValueError("unavailable_recorded_" + self.kind + "_outcome")
        return deepcopy(record["value"])

    def finish(self):
        if self.index != self.packed["call_count"]:
            raise ValueError("recorded_" + self.kind + "_call_count_disagreement")


def pack_sample(sample, coverage, control, origin, writer=None):
    reading = sample["reading"]
    observed = reading["replay_observations"]
    return jaw._json({"schema": CONTEXT_SCHEMA,
        "sample": {key: deepcopy(sample[key]) for key in ("episode_id", "control_tick", "substep")},
        "origin": list(origin), "stage_id": reading["stage_id"],
        "clock_before": deepcopy(reading["clock_before"]), "clock_after": deepcopy(reading["clock_after"]),
        "health": deepcopy(reading["health"]), "chicken_pose": deepcopy(reading["chicken_pose"]),
        "physics_dt": observed["physics_dt"],
        "configuration": _intern(writer, jaw._json({"coverage": coverage, "control": control}), "contact-configuration"),
        "decode": _pack_calls(observed["decode"], writer, "decode"),
        "identity": _pack_calls(observed["identity"], writer, "identity")})


def compact_target(sample, predicate):
    reading = sample["reading"]
    reduction = reading["callback_reduction"]
    return {"schema": "compact-contact-live-target/v1", "predicate": deepcopy(predicate),
        "sample_issues": deepcopy(sample["issues"]),
        "reading_digest": digest(jaw._json({k: v for k, v in reading.items() if k != "replay_observations"})),
        "reduction_digest": digest(reduction),
        "reduction": {k: deepcopy(reduction[k]) for k in
            ("status", "left", "right", "active_jaw_ids", "issues", "negative_qualification", "ending_validation")}}


def reconstruct_sample(envelope, context, previous_state=None, *, reader=None):
    if context.get("schema") != CONTEXT_SCHEMA:
        raise ValueError("unsupported_compact_replay_context")
    config = _resolve(context["configuration"], reader)
    decode = _Calls(context["decode"], reader, "decode")
    identity = _Calls(context["identity"], reader, "identity")
    health_calls = iter((context["health"]["before"], context["health"]["after"]))
    clock = SimpleNamespace(current_time_step_index=context["clock_before"][0],
        current_time=context["clock_before"][1], get_physics_dt=lambda: context["physics_dt"])
    def health():
        return deepcopy(next(health_calls))
    health_count = 0
    def observed_health():
        nonlocal health_count
        result = health()
        health_count += 1
        if health_count == 2:
            clock.current_time_step_index, clock.current_time = context["clock_after"]
        return result
    probe = SimpleNamespace(stage_id=context["stage_id"], health=observed_health, identity_issues=identity)
    pose_value = deepcopy(context["chicken_pose"])
    if pose_value is not None:
        # NativePoseReader returns tuple clocks; persisted strict JSON contains
        # arrays. Restore that existing interface shape before its equality guard.
        for field in ("clock_before", "clock_after"):
            if isinstance(pose_value.get(field), list):
                pose_value[field] = tuple(pose_value[field])
    pose = None if pose_value is None else SimpleNamespace(read=lambda: deepcopy(pose_value))
    contact = jaw.NativeJawReader(None, clock, decode, probe, config["coverage"], pose_reader=pose,
        callback_collector=SimpleNamespace(take_boundary=lambda *args: deepcopy(envelope)))
    contact.observation_control = deepcopy(config["control"])
    contact._callback_pair_state = deepcopy(previous_state)
    reading = contact.read()
    decode.finish()
    identity.finish()
    sample = deepcopy(context["sample"])
    sample.update(reading=reading, issues=list(reading["issues"]))
    t, s = sample["control_tick"], sample["substep"]
    if type(t) is not int or not 0 <= t <= 1199 or type(s) is not int or s not in (0, 1):
        sample["issues"].append("invalid_control_tick_or_substep")
    elif context["clock_before"][0] != context["origin"][0] + 2*t + s + 1:
        sample["issues"].append("stale_or_wrong_native_step")
    sample["issues"] = list(dict.fromkeys(sample["issues"]))
    if sample["issues"]:
        reading["no_contacts_reported"] = None
    sample["acquisition_status"] = "UNKNOWN" if sample["issues"] else reading["acquisition_status"]
    if "observation_snapshot" in reading:
        reading["observation_snapshot"]["context"] = dict(context["sample"],
            origin_step=context["origin"][0], origin_time_s=context["origin"][1],
            issues=deepcopy(sample["issues"]), acquisition_status=sample["acquisition_status"])
    return sample


def pack_transport(export, coverage, control, writer=None):
    result = deepcopy(export)
    archive = result.get("initialization_routing")
    if archive is None or archive.get("schema") == INIT_SCHEMA:
        return result
    clean = {k: v for k, v in archive.items() if k != "decode_observations"}
    result["initialization_routing"] = {"schema": INIT_SCHEMA,
        "source_sequences": deepcopy(archive["source_sequences"]), "stage_id": archive["stage_id"],
        "configuration": _intern(writer, jaw._json({"coverage": coverage, "control": control}), "contact-configuration"),
        "decode": _pack_calls(archive["decode_observations"], writer, "decode"),
        "target_digest": digest(clean)}
    return result


def reconstruct_transport(export, *, reader=None):
    result = deepcopy(export)
    archive = result.get("initialization_routing")
    if archive is None or archive.get("schema") != INIT_SCHEMA:
        return result
    config = _resolve(archive["configuration"], reader)
    decode = _Calls(archive["decode"], reader, "decode")
    lookup = {r["sequence"]: r for r in result["collector"]["initialization_callbacks"]}
    originals = [lookup[sequence] for sequence in archive["source_sequences"]]
    contact = jaw.NativeJawReader(None, None, decode, SimpleNamespace(stage_id=archive["stage_id"]),
        config["coverage"], callback_collector=object())
    contact.observation_control = config["control"]
    restored = contact.decode_initialization_evidence({"initialization_callbacks": originals})
    restored.pop("decode_observations", None)
    decode.finish()
    if digest(restored) != archive["target_digest"]:
        raise ValueError("compact_initialization_reconstruction_disagreement")
    result["initialization_routing"] = restored
    return result


class CompactReplay:
    """One decoded boundary and the bounded event-state map, never a raw history."""
    def __init__(self, root, *, require_final=False):
        from .compact_contact_storage import CompactContactReader
        self.reader = CompactContactReader(root, require_final=require_final)
        self.state = None

    def read(self, compact, *, expected_predicate=None):
        from .compact_contact_storage import SCHEMA
        if compact.get("schema") != SCHEMA:
            raise ValueError("compact_boundary_reference_schema_mismatch")
        record = self.reader.read_boundary(compact["reference"])
        if record["envelope"] is None:
            context = record["context"]
            if context.get("schema") != "compact-contact-unavailable/v1":
                raise ValueError("missing_compact_original_boundary")
            sample = deepcopy(context["unavailable_sample"])
            reading = sample.get("reading", {})
            if ("callback_boundary" in reading or any(reading.get(key) for key in
                    ("headers", "contacts", "unassociated_contacts", "callback_decoded_headers"))
                    or any(reading.get("observation_snapshot", {}).get(key) for key in
                           ("raw_headers", "headers", "data"))):
                raise ValueError("unattributed_payload_without_callback_boundary")
            if context.get("acquisition") not in ("EXCEPTION", "NO_CALLBACK_BOUNDARY_ACQUIRED"):
                raise ValueError("unavailable_acquisition_provenance_missing")
            predicate = bilateral_contact(sample, {}, episode_id=context["sample"]["episode_id"],
                context=context["sample"], native_clock=context["native_clock"])
            if record["live_target"] != {"schema": "compact-contact-unavailable-target/v1", "predicate": predicate}:
                raise ValueError("compact_unavailable_decision_disagreement")
            if expected_predicate is not None and predicate != expected_predicate:
                raise ValueError("compact_live_decision_reconstruction_disagreement")
            return sample
        sample = reconstruct_sample(record["envelope"], record["context"], self.state, reader=self.reader)
        context = record["context"]
        coverage = _resolve(context["configuration"], self.reader)["coverage"]
        predicate = bilateral_contact(sample, coverage, episode_id=sample["episode_id"],
            context=context["sample"], native_clock=context["clock_after"])
        target = compact_target(sample, predicate)
        if target != record["live_target"] or expected_predicate is not None and predicate != expected_predicate:
            raise ValueError("compact_live_decision_reconstruction_disagreement")
        self.state = deepcopy(sample["reading"]["callback_reduction"]["state"])
        return jaw._json(sample)

    def lifecycle(self, compact):
        from .compact_contact_storage import SCHEMA
        if compact.get("schema", SCHEMA) != SCHEMA:
            raise ValueError("compact_lifecycle_reference_schema_mismatch")
        return reconstruct_transport(self.reader.read_lifecycle(compact["reference"]), reader=self.reader)


def storage_selected(result, measurements):
    """Explicit version dispatch; absence means the retained historical format."""
    from .compact_contact_storage import SCHEMA
    version = result.get("contact_storage_schema")
    if version != measurements.get("contact_storage_schema"):
        raise ValueError("contact_storage_schema_disagreement")
    if version not in (None, SCHEMA):
        raise ValueError("unsupported_contact_storage_schema")
    if version and result.get("contact_transport") != "callback-contact/v1":
        raise ValueError("compact_storage_requires_callback_transport")
    return version is not None


class SnapshotStream:
    """Repeatable, non-caching traversal for the unchanged control classifier."""
    def __init__(self, root, rows, *, require_final=False):
        self.root, self.rows, self.require_final = root, rows, require_final

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        replay = CompactReplay(self.root, require_final=self.require_final)
        for row in self.rows:
            yield replay.read(row["contact"], expected_predicate=row["predicate"])["reading"]["observation_snapshot"]
