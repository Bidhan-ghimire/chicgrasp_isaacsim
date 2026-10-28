"""Synchronous, lossless RGB sidecar for an incomplete observation candidate.

No simulator dependency, background worker, extra acquisition, or image history.
The append-only index references complete PNG/metadata files. A crash may leave
an unindexed file or torn final index line; validation reports either explicitly.
"""
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np


CAMERAS = ("camera_left", "camera_right")
SCHEMA = "attachment-observation-rgb/v1"


def _json_bytes(value):
    return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _load(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_json_key:" + key)
            result[key] = value
        return result
    def bad(value):
        raise ValueError("nonfinite_json:" + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def _identity(path):
    raw = path.read_bytes()
    return {"path": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _atomic(path, payload):
    """Publish a fresh same-directory hard link; never replace another file.

    A supported filesystem must provide atomic link creation. If unavailable,
    the operation fails visibly and preserves its .incomplete file, no fallback.
    fsync is an OS-facing request, not a power-loss durability guarantee.
    """
    pending = path.with_name(path.name + ".incomplete")
    with pending.open("xb") as stream:
        if stream.write(payload) != len(payload):
            raise OSError("short_file_write:" + path.name)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(pending, path)
    pending.unlink()
    return _identity(path)


def _integer(value):
    return type(value) is int


def _metadata(action_frame, value):
    # JSON roundtrip rejects opaque/NaN payloads and relinquishes caller ownership.
    result = _load(_json_bytes(value))
    if not isinstance(result, dict):
        raise ValueError("metadata_mapping_required")
    pre = action_frame is None
    if pre:
        if result.get("kind") != "PRE_CANDIDATE" or any(
                result.get(key, "missing") is not None
                for key in ("action_frame", "control_tick", "substep", "policy_timestamp_ns")):
            raise ValueError("pre_candidate_has_no_command_indices")
        if type(result.get("shared_acquisition_action_frame")) is not int or result["shared_acquisition_action_frame"] != 0:
            raise ValueError("pre_candidate_shared_acquisition_required")
    else:
        if not _integer(result.get("action_frame", action_frame)) or result.get("action_frame", action_frame) != action_frame:
            raise ValueError("metadata_action_frame_mismatch")
        result["action_frame"] = action_frame
        if result.get("kind", "ACTION") != "ACTION":
            raise ValueError("action_metadata_kind_mismatch")
        if not _integer(result.get("control_tick")) or result["control_tick"] != 6 * action_frame:
            raise ValueError("preaction_control_tick_required")
        if ("shared_acquisition_action_frame" in result
                and (action_frame != 0 or type(result["shared_acquisition_action_frame"]) is not int
                     or result["shared_acquisition_action_frame"] != 0)):
            raise ValueError("shared_acquisition_only_action_zero")
    clock = result.get("native_clock")
    if (not isinstance(clock, list) or len(clock) != 2 or not _integer(clock[0]) or clock[0] < 0
            or isinstance(clock[1], bool) or not isinstance(clock[1], (int, float))
            or not math.isfinite(clock[1]) or clock[1] < 0):
        raise ValueError("actual_native_clock_required")
    if (not _integer(result.get("wall_monotonic_ns")) or result["wall_monotonic_ns"] < 0
            or (not pre and (not _integer(result.get("policy_timestamp_ns"))
                            or result["policy_timestamp_ns"] != action_frame * 100_000_000))):
        raise ValueError("actual_wall_and_policy_clocks_required")
    frames, cameras = result.get("sensor_frames"), result.get("camera_metadata")
    if not isinstance(frames, dict) or set(frames) != set(CAMERAS):
        raise ValueError("both_sensor_frame_ids_required")
    if not isinstance(cameras, dict) or set(cameras) != set(CAMERAS):
        raise ValueError("both_camera_provenance_records_required")
    for name in CAMERAS:
        if not _integer(frames[name]) or frames[name] < 0:
            raise ValueError("invalid_sensor_frame:" + name)
        item = cameras[name]
        if (not isinstance(item, dict) or not isinstance(item.get("prim_path"), str)
                or not item["prim_path"].startswith("/") or not item.get("source")
                or not isinstance(item["source"], str) or not isinstance(item.get("sensor_time_source"), str)
                or not item["sensor_time_source"]):
            raise ValueError("camera_identity_and_sources_required:" + name)
        stamp = item.get("sensor_time_s")
        if item.get("sensor_time_status") == "UNAVAILABLE":
            if stamp is not None:
                raise ValueError("unavailable_camera_time_must_be_null:" + name)
        elif item.get("sensor_time_status") != "OBSERVED" or isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or stamp < 0:
            raise ValueError("camera_time_status_invalid:" + name)
    if cameras[CAMERAS[0]]["prim_path"] == cameras[CAMERAS[1]]["prim_path"]:
        raise ValueError("distinct_camera_identities_required")
    return result


def _advance(previous, current):
    if previous is None:
        if current.get("kind") != "PRE_CANDIDATE" and "shared_acquisition_action_frame" in current:
            raise ValueError("shared_acquisition_without_pre_candidate")
        return
    if previous.get("kind") == "PRE_CANDIDATE":
        if current.get("action_frame") != 0 or current.get("shared_acquisition_action_frame") != 0:
            raise ValueError("action_zero_shared_acquisition_required")
        for key in ("native_clock", "sensor_frames", "wall_monotonic_ns", "camera_metadata"):
            if current[key] != previous[key]:
                raise ValueError("pre_candidate_shared_provenance_mismatch:" + key)
        return
    if any(current["sensor_frames"][key] <= previous["sensor_frames"][key] for key in CAMERAS):
        raise ValueError("sensor_frame_not_advanced")
    if (current["native_clock"][0] <= previous["native_clock"][0]
            or current["native_clock"][1] <= previous["native_clock"][1]
            or current["wall_monotonic_ns"] <= previous["wall_monotonic_ns"]):
        raise ValueError("acquisition_clock_not_advanced")
    for name in CAMERAS:
        a, b = previous["camera_metadata"][name], current["camera_metadata"][name]
        if a["prim_path"] != b["prim_path"] or a["source"] != b["source"]:
            raise ValueError("camera_identity_changed:" + name)
        if a["sensor_time_status"] == b["sensor_time_status"] == "OBSERVED" and b["sensor_time_s"] < a["sensor_time_s"]:
            raise ValueError("camera_time_regressed:" + name)


class IncrementalRGBWriter:
    """Copy both images first, finish a bundle synchronously, then accept next.

    Only one bundle's pixels/PNG buffer are live. Metadata history is on disk;
    instance state is bounded independently of the number of acquired images.
    """
    def __init__(self, root, *, max_samples=200, image_shape=(240, 320, 3), envelope_v2=False):
        if not _integer(max_samples) or not 0 < max_samples <= (279 if envelope_v2 else 200):
            raise ValueError("bounded_sample_count_required")
        if (len(image_shape) != 3 or any(not _integer(n) or n <= 0 for n in image_shape)
                or image_shape[2] != 3):
            raise ValueError("positive_HWC_RGB_shape_required")
        self.envelope_v2 = envelope_v2
        self.root, self.max_samples, self.image_shape = Path(root), max_samples, tuple(image_shape)
        self.root.mkdir(parents=True, exist_ok=False)
        with (self.root / "index.jsonl").open("xb") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        self.acquired = self.completed = 0
        self.action_acquired = self.action_completed = 0
        self.pre_acquired = self.pre_completed = False
        self.errors, self.previous = [], None
        self.previous_pixel_hashes = None
        self.closed, self._final = False, None
        self.peak_copied_image_bytes = 0

    def set_event_bound(self, total_actions):
        if not self.envelope_v2 or self.closed or type(total_actions) is not int or not 200 <= total_actions <= 279 or total_actions < self.action_acquired:
            raise ValueError("invalid_event_image_bound")
        self.max_samples = total_actions

    def write_bundle(self, action_frame, mapping, metadata):
        if self.closed or self.errors:
            raise RuntimeError("writer_closed_or_failed_no_retry")
        if not _integer(action_frame) or action_frame != self.action_acquired or action_frame >= self.max_samples:
            raise ValueError("fresh_sequential_action_frame_required")
        self.action_acquired += 1
        return self._write(action_frame, mapping, metadata)

    def write_pre_candidate(self, mapping, metadata):
        """Persist the existing reset acquisition before any candidate command.

        This is an extra bundle, not an extra acquisition or fictional action.
        Action zero must declare and match that same acquisition exactly.
        """
        if self.closed or self.errors:
            raise RuntimeError("writer_closed_or_failed_no_retry")
        if self.acquired or self.pre_acquired:
            raise ValueError("pre_candidate_must_be_first_once")
        self.pre_acquired = True
        return self._write(None, mapping, metadata)

    def _write(self, action_frame, mapping, metadata):
        pre = action_frame is None
        kind = "PRE_CANDIDATE" if pre else "ACTION"
        prefix = "pre_candidate" if pre else f"{action_frame:06d}"
        self.acquired += 1
        stage = "copy"
        try:
            from PIL import Image  # available after normal Kit startup, no path repair
            # Do not retain or borrow either sensor's mutable source buffer.
            copied = {}
            for name in CAMERAS:
                image = np.asarray(mapping[name + "_rgb"])
                if image.shape != self.image_shape or image.dtype != np.dtype("uint8"):
                    raise ValueError("unchanged_uint8_HWC_RGB_required:" + name)
                copied[name] = image.copy(order="C")
            self.peak_copied_image_bytes = max(self.peak_copied_image_bytes, sum(a.nbytes for a in copied.values()))
            meta = _metadata(action_frame, metadata)
            _advance(self.previous, meta)
            pixel_hashes = {name: hashlib.sha256(pixels.tobytes()).hexdigest() for name, pixels in copied.items()}
            if self.previous is not None and self.previous.get("kind") == "PRE_CANDIDATE" and pixel_hashes != self.previous_pixel_hashes:
                raise ValueError("pre_candidate_shared_pixels_mismatch")
            bundle = {"schema": SCHEMA, "kind": kind, "action_frame": action_frame, "metadata": meta, "images": {}}
            for name in CAMERAS:
                stage = "image:" + name
                with io.BytesIO() as buffer:
                    Image.fromarray(copied[name]).save(buffer, format="PNG")
                    descriptor = _atomic(self.root / f"{prefix}_{name}.png", buffer.getvalue())
                descriptor.update(camera=name, channel_order="RGB", shape=list(self.image_shape), dtype="uint8",
                    encoding="PNG_lossless", pixel_sha256=pixel_hashes[name])
                bundle["images"][name] = descriptor
            stage = "metadata"
            descriptor = _atomic(self.root / f"{prefix}.json", _json_bytes(bundle))
            stage = "index"
            payload = _json_bytes({"kind": kind, "action_frame": action_frame, "metadata_file": descriptor})
            with (self.root / "index.jsonl").open("ab") as stream:
                if stream.write(payload) != len(payload):
                    raise OSError("short_index_write")
                stream.flush()
                os.fsync(stream.fileno())
            self.completed += 1
            if pre:
                self.pre_completed = True
            else:
                self.action_completed += 1
            self.previous = meta
            self.previous_pixel_hashes = pixel_hashes
            return deepcopy(bundle)
        except Exception as exc:
            self.errors.append({"kind": kind, "action_frame": action_frame, "stage": stage,
                                "error": type(exc).__name__ + ":" + str(exc)})
            raise

    def finish(self, completed, primary_errors):
        if self._final is not None:
            return deepcopy(self._final)
        self.closed = True
        if type(completed) is not bool or not isinstance(primary_errors, list) or any(not isinstance(e, str) for e in primary_errors):
            raise ValueError("explicit_candidate_completion_and_primary_errors_required")
        if completed and (self.action_completed != self.max_samples or self.pre_acquired != self.pre_completed):
            self.errors.append({"stage": "finish", "error": "complete_candidate_image_count_mismatch"})
        inspected = validate(self.root, expected_count=self.completed, require_manifest=False, envelope_v2=self.envelope_v2)
        manifest = {"schema": SCHEMA, "state": "COMPLETE" if completed and not self.errors and not primary_errors and not inspected["issues"] else "PARTIAL",
            "candidate_completed": completed, "acquired_bundles": self.acquired, "completed_bundles": self.completed,
            "pre_candidate_acquired": self.pre_acquired, "pre_candidate_completed": self.pre_completed,
            "acquired_action_bundles": self.action_acquired, "completed_action_bundles": self.action_completed,
            "max_samples": self.max_samples, "image_shape": list(self.image_shape), "encoding": "PNG_lossless",
            "primary_errors": deepcopy(primary_errors), "writer_errors": deepcopy(self.errors),
            "validation_issues": inspected["issues"], "orphan_files": inspected["orphan_files"],
            "index": _identity(self.root / "index.jsonl"),
            "durability": "exclusive incomplete file; flush/fsync; atomic same-directory hard-link publication; append index flush/fsync; no power-loss guarantee",
            "peak_copied_image_bytes": self.peak_copied_image_bytes}
        descriptor = _atomic(self.root / "manifest.json", _json_bytes(manifest))
        self._final = dict(manifest, manifest_file=descriptor)
        return deepcopy(self._final)


def validate(root, *, expected_count=None, sensors=None, pre_candidate=None, require_manifest=True, envelope_v2=False):
    """Read-only independent file/index/PNG/provenance checks; no native imports.

    PARTIAL with complete acquired bundles is valid evidence. Missing/torn final
    metadata remains an explicit issue while earlier indexed bundles are read.
    Identical pixel hashes across frames are permitted; provenance governs time.
    """
    root = Path(root)
    from PIL import Image
    issues, records, referenced = [], [], {"index.jsonl", "manifest.json"}
    previous = None
    previous_pixel_hashes = None
    has_pre = False
    manifest = None
    def fail(message):
        issues.append(message)
    def verified(descriptor, expected_name):
        if (not isinstance(descriptor, dict) or descriptor.get("path") != expected_name
                or Path(expected_name).name != expected_name or not _integer(descriptor.get("bytes"))
                or descriptor["bytes"] < 0 or not isinstance(descriptor.get("sha256"), str)):
            raise ValueError("invalid_file_descriptor:" + expected_name)
        path = root / expected_name
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing_or_linked_file:" + expected_name)
        raw = path.read_bytes()
        if len(raw) != descriptor["bytes"] or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
            raise ValueError("file_identity_mismatch:" + expected_name)
        referenced.add(expected_name)
        return raw
    try:
        if (root / "index.jsonl").is_symlink():
            raise ValueError("linked_index_rejected")
        index_raw = (root / "index.jsonl").read_bytes()
        lines = index_raw.splitlines(keepends=True)
        if len(lines) > (280 if envelope_v2 else 201):
            raise ValueError("index_exceeds_bound")
        for number, line in enumerate(lines):
            try:
                if not line.endswith(b"\n"):
                    raise ValueError("incomplete_index_line")
                entry = _load(line)
                pre = entry.get("kind") == "PRE_CANDIDATE"
                if pre:
                    if number != 0 or entry.get("action_frame", "missing") is not None:
                        raise ValueError("pre_candidate_index_must_be_first_without_action")
                    has_pre = True
                    action_frame, prefix, kind = None, "pre_candidate", "PRE_CANDIDATE"
                else:
                    action_frame = number - int(has_pre)
                    prefix, kind = f"{action_frame:06d}", "ACTION"
                    if (entry.get("kind", "ACTION") != "ACTION" or not 0 <= action_frame < (279 if envelope_v2 else 200)
                            or not _integer(entry.get("action_frame")) or entry["action_frame"] != action_frame):
                        raise ValueError("index_action_sequence_mismatch")
                bundle = _load(verified(entry["metadata_file"], f"{prefix}.json"))
                if (bundle.get("schema") != SCHEMA or bundle.get("kind", "ACTION") != kind
                        or bundle.get("action_frame", "missing") != action_frame
                        or (not pre and not _integer(bundle["action_frame"]))):
                    raise ValueError("bundle_identity_mismatch")
                meta = _metadata(action_frame, bundle["metadata"])
                _advance(previous, meta)
                if not isinstance(bundle.get("images"), dict) or set(bundle["images"]) != set(CAMERAS):
                    raise ValueError("both_images_required")
                for camera in CAMERAS:
                    descriptor = bundle["images"][camera]
                    raw = verified(descriptor, f"{prefix}_{camera}.png")
                    if (descriptor.get("camera") != camera or descriptor.get("channel_order") != "RGB"
                            or descriptor.get("dtype") != "uint8" or descriptor.get("encoding") != "PNG_lossless"):
                        raise ValueError("image_representation_mismatch")
                    with Image.open(io.BytesIO(raw)) as image:
                        if image.format != "PNG" or image.mode != "RGB":
                            raise ValueError("lossless_RGB_PNG_required")
                        pixels = np.array(image)
                    if list(pixels.shape) != descriptor.get("shape") or hashlib.sha256(pixels.tobytes()).hexdigest() != descriptor.get("pixel_sha256"):
                        raise ValueError("decoded_pixel_mismatch")
                pixel_hashes = {camera: bundle["images"][camera]["pixel_sha256"] for camera in CAMERAS}
                if previous is not None and previous.get("kind") == "PRE_CANDIDATE" and pixel_hashes != previous_pixel_hashes:
                    raise ValueError("pre_candidate_shared_pixels_mismatch")
                if pre and pre_candidate is not None:
                    if (not isinstance(pre_candidate, dict) or pre_candidate.get("rgb_metadata") != meta
                            or pre_candidate.get("kind") != "PRE_CANDIDATE"
                            or any(pre_candidate.get(key, "missing") is not None for key in ("action_frame", "control_tick", "substep"))
                            or pre_candidate.get("native_clock") != meta["native_clock"]
                            or not _integer(pre_candidate.get("wall_monotonic_ns"))
                            or pre_candidate["wall_monotonic_ns"] > meta["wall_monotonic_ns"]):
                        raise ValueError("pre_candidate_pose_rgb_provenance_mismatch")
                if not pre and sensors is not None:
                    sensor = sensors[action_frame]
                    for key in ("action_frame", "native_clock", "sensor_frames", "wall_monotonic_ns", "policy_timestamp_ns"):
                        if meta[key] != sensor.get(key):
                            raise ValueError("sensor_provenance_mismatch:" + key)
                    if "camera_metadata" in sensor and meta["camera_metadata"] != sensor["camera_metadata"]:
                        raise ValueError("camera_metadata_mismatch")
                    if meta.get("shared_acquisition_action_frame") != sensor.get("shared_acquisition_action_frame"):
                        raise ValueError("shared_acquisition_sensor_provenance_mismatch")
                previous = meta
                previous_pixel_hashes = pixel_hashes
                records.append({"kind": kind, "action_frame": action_frame,
                                "metadata_file": deepcopy(entry["metadata_file"]), "metadata": meta})
            except Exception as exc:
                fail(f"index[{number}]:{type(exc).__name__}:{exc}")
        if expected_count is not None and (not _integer(expected_count) or len(records) != expected_count):
            fail("expected_bundle_count_mismatch")
        pre_records = [record for record in records if record["kind"] == "PRE_CANDIDATE"]
        action_records = [record for record in records if record["kind"] == "ACTION"]
        if pre_candidate is not None and len(pre_records) != 1:
            fail("required_pre_candidate_bundle_missing")
        if sensors is not None and len(action_records) != len(sensors):
            fail("sensor_bundle_count_mismatch")
        path = root / "manifest.json"
        if path.exists():
            if path.is_symlink():
                raise ValueError("linked_manifest_rejected")
            manifest = _load(path.read_bytes())
            if manifest.get("schema") != SCHEMA or manifest.get("state") not in ("COMPLETE", "PARTIAL"):
                fail("invalid_manifest_schema_or_state")
            shape = manifest.get("image_shape")
            if (type(manifest.get("candidate_completed")) is not bool
                    or not _integer(manifest.get("max_samples")) or not 0 < manifest["max_samples"] <= (279 if envelope_v2 else 200)
                    or not _integer(manifest.get("acquired_bundles")) or not _integer(manifest.get("completed_bundles"))
                    or type(manifest.get("pre_candidate_acquired")) is not bool
                    or type(manifest.get("pre_candidate_completed")) is not bool
                    or not _integer(manifest.get("acquired_action_bundles"))
                    or not _integer(manifest.get("completed_action_bundles"))
                    or not 0 <= manifest["completed_action_bundles"] <= manifest["acquired_action_bundles"] <= manifest["max_samples"]
                    or not 0 <= manifest["completed_bundles"] <= manifest["acquired_bundles"] <= manifest["max_samples"] + int(manifest["pre_candidate_acquired"])
                    or manifest["completed_bundles"] != manifest["completed_action_bundles"] + int(manifest["pre_candidate_completed"])
                    or manifest["acquired_bundles"] != manifest["acquired_action_bundles"] + int(manifest["pre_candidate_acquired"])
                    or (manifest["pre_candidate_completed"] and not manifest["pre_candidate_acquired"])
                    or not isinstance(shape, list) or len(shape) != 3
                    or any(not _integer(n) or n <= 0 for n in shape) or shape[2] != 3
                    or manifest.get("encoding") != "PNG_lossless"
                    or not isinstance(manifest.get("primary_errors"), list)
                    or any(not isinstance(e, str) for e in manifest["primary_errors"])
                    or any(not isinstance(manifest.get(name), list) for name in ("writer_errors", "validation_issues", "orphan_files"))):
                fail("invalid_manifest_field_types_or_bounds")
            verified(manifest["index"], "index.jsonl")
            if manifest.get("completed_bundles") != len(records) or manifest.get("acquired_bundles") != len(lines):
                fail("manifest_bundle_counts_mismatch")
            if (manifest.get("completed_action_bundles") != len(action_records)
                    or manifest.get("pre_candidate_completed") is not bool(pre_records)):
                fail("manifest_pre_candidate_action_counts_mismatch")
            if manifest.get("writer_errors") or manifest.get("validation_issues") or manifest.get("orphan_files"):
                fail("saved_writer_or_persistence_failure")
            if manifest.get("state") == "COMPLETE" and (manifest.get("candidate_completed") is not True
                    or len(action_records) != manifest.get("max_samples") or manifest.get("primary_errors")):
                fail("false_complete_manifest")
            for record in records:
                bundle = _load((root / record["metadata_file"]["path"]).read_bytes())
                if any(item.get("shape") != manifest.get("image_shape") for item in bundle["images"].values()):
                    fail("manifest_image_shape_mismatch")
        elif require_manifest:
            fail("final_manifest_missing")
    except Exception as exc:
        fail(type(exc).__name__ + ":" + str(exc))
    actual = {str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file()} if root.exists() else set()
    orphans = sorted(actual - referenced)
    if orphans:
        fail("unindexed_or_incomplete_files")
    return {"status": "PASS" if not issues else "INCOMPLETE", "issues": list(dict.fromkeys(issues)),
            "completed_bundles": len(records), "records": records, "orphan_files": orphans,
            "pre_candidate_bundles": sum(record["kind"] == "PRE_CANDIDATE" for record in records),
            "action_bundles": sum(record["kind"] == "ACTION" for record in records),
            "state": manifest.get("state") if isinstance(manifest, dict) else "UNFINALIZED"}
