"""Lossless, incremental offline storage for already-copied contact callbacks.

This module neither imports simulation packages nor changes collector retention.
The byte limits are persistence failure bounds, never silent omission policies.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import uuid

SCHEMA = "gate5-compact-contact/v1"
HEADER_BYTES = 117
POINT_BYTES = 135
PREFIX_BYTES = 64
MAX_CHUNK_METADATA_BYTES = 16 * 1024**2
MAX_RECORD_BYTES = 2 * 1024**2
MAX_DICTIONARY_BYTES = 8 * 1024**2
MAX_INTEGER_BYTES = 4096
MAX_DICTIONARY_POOL_BYTES = 64 * 1024**2
MAX_INDEX_BYTES = 64 * 1024**2
MAX_INDEX_ENTRY_BYTES = 2048
MAX_MANIFEST_BYTES = MAX_RECORD_BYTES
MAX_CHUNK_HEADERS = 8192
MAX_CHUNK_POINTS = 32768
MAX_RECORD_OCCURRENCES = 512
MAX_CHUNK_BYTES = PREFIX_BYTES + MAX_CHUNK_HEADERS * HEADER_BYTES + MAX_CHUNK_POINTS * POINT_BYTES + MAX_CHUNK_METADATA_BYTES
HEADER_FIELDS = ("stage_id", "actor0", "actor1", "collider0", "collider1", "type",
                 "num_contact_data", "contact_data_offset", "num_friction_anchors_data",
                 "friction_anchors_offset", "proto_index0", "proto_index1", "source_header_index")
POINT_FIELDS = ("position_raw", "normal_raw", "impulse_raw")
POINT_SCALARS = ("separation_raw", "face_index0_raw", "face_index1_raw", "material0_raw",
                 "material1_raw", "source_data_index")
_MAGIC = b"G5CCBIN1"
_PREFIX = struct.Struct("<8sHHIIIII32s")
_CELL = struct.Struct("<BQ")
_LAYOUT = hashlib.sha256((repr(HEADER_FIELDS) + repr(POINT_FIELDS) + repr(POINT_SCALARS)
                          + ":u8tag,u64payload:little:v1").encode()).digest()
_MISSING = object()


class CompactStorageError(ValueError):
    """Required persistence or independently verified reconstruction failed."""


class _Occurrence:
    def __init__(self, ref):
        self.ref = ref


def _typed(value, depth=0):
    if depth > 64:
        raise CompactStorageError("metadata_depth_limit")
    if isinstance(value, _Occurrence):
        return ["r", value.ref]
    if value is None:
        return ["n"]
    if type(value) is bool:
        return ["b", value]
    if type(value) is int:
        if (value.bit_length() + 7) // 8 > MAX_INTEGER_BYTES:
            raise CompactStorageError("integer_representability_limit")
        return ["i", ("-" if value < 0 else "+") + format(abs(value), "x")]
    if type(value) is float:
        return ["f", struct.pack("<d", value).hex()]
    if type(value) is str:
        return ["s", value]
    if type(value) in (list, tuple):
        return ["l" if type(value) is list else "t", [_typed(v, depth + 1) for v in value]]
    if type(value) is dict:
        if not all(type(k) is str for k in value):
            raise CompactStorageError("metadata_nonstring_key")
        return ["d", [[k, _typed(v, depth + 1)] for k, v in value.items()]]
    raise CompactStorageError("unsupported_retained_type:" + type(value).__name__)


def _untyped(value, resolve=None, depth=0):
    if depth > 64 or not isinstance(value, list) or not value:
        raise CompactStorageError("invalid_typed_metadata")
    tag = value[0]
    try:
        if tag == "n" and len(value) == 1:
            return None
        if len(value) != 2:
            raise ValueError("tag_arity")
        item = value[1]
        if tag == "r" and isinstance(item, dict):
            return _Occurrence(item) if resolve is None else resolve(item)
        if tag == "b" and type(item) is bool:
            return item
        if tag == "i" and type(item) is str and item[:1] in ("+", "-"):
            if len(item) - 1 > MAX_INTEGER_BYTES * 2:
                raise ValueError("integer_representability_limit")
            result = int(item[1:], 16) * (-1 if item[0] == "-" else 1)
            if _typed(result) != value:
                raise ValueError("noncanonical_integer")
            return result
        if tag == "f" and type(item) is str and len(item) == 16:
            return struct.unpack("<d", bytes.fromhex(item))[0]
        if tag == "s" and type(item) is str:
            return item
        if tag in ("l", "t") and type(item) is list:
            result = [_untyped(v, resolve, depth + 1) for v in item]
            return result if tag == "l" else tuple(result)
        if tag == "d" and type(item) is list:
            result = {}
            for pair in item:
                if not isinstance(pair, list) or len(pair) != 2 or type(pair[0]) is not str or pair[0] in result:
                    raise ValueError("dictionary_entry")
                result[pair[0]] = _untyped(pair[1], resolve, depth + 1)
            return result
    except (TypeError, ValueError, OverflowError) as exc:
        raise CompactStorageError("invalid_typed_metadata:" + str(exc)) from exc
    raise CompactStorageError("invalid_typed_metadata_tag")


def _json_bytes(value, limit=None):
    encoder = json.JSONEncoder(ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    result = bytearray()
    for piece in encoder.iterencode(value):
        encoded = piece.encode("ascii")
        if limit is not None and len(result) + len(encoded) > limit:
            raise CompactStorageError("encoded_metadata_byte_limit")
        result.extend(encoded)
    return bytes(result)


def _load_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise CompactStorageError("duplicate_json_key")
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda v: (_ for _ in ()).throw(CompactStorageError("nonfinite_json")))
    except (ValueError, UnicodeError) as exc:
        raise CompactStorageError("invalid_json:" + str(exc)) from exc


def _cell(value, fallback):
    if value is _MISSING:
        tag, payload = 0, 0
    elif value is None:
        tag, payload = 4, 0
    elif type(value) is bool:
        tag, payload = 6 if value else 5, 0
    elif type(value) is int and 0 <= value <= 0xffffffffffffffff:
        tag, payload = 1, value
    elif type(value) is int and -(1 << 63) <= value < 0:
        tag, payload = 2, value & 0xffffffffffffffff
    elif type(value) is float:
        tag, payload = 3, struct.unpack("<Q", struct.pack("<d", value))[0]
    else:
        tag, payload = 7, len(fallback)
        fallback.append(_typed(value))
    return _CELL.pack(tag, payload)


def _read_cell(raw, offset, fallback):
    tag, payload = _CELL.unpack_from(raw, offset)
    if tag == 0 and payload == 0:
        return _MISSING
    if tag == 1:
        return payload
    if tag == 2 and payload >= (1 << 63):
        return payload - (1 << 64)
    if tag == 3:
        return struct.unpack("<d", struct.pack("<Q", payload))[0]
    if tag in (4, 5, 6) and payload == 0:
        return {4: None, 5: False, 6: True}[tag]
    if tag == 7 and payload < len(fallback):
        return _untyped(fallback[payload])
    raise CompactStorageError("invalid_scalar_tag_or_payload")


def encode_records(records):
    """Encode copied occurrences only; H117/P135 arrays plus exact typed extras."""
    headers = bytearray()
    points = bytearray()
    fallback, descriptors = [], []
    header_count = point_count = 0
    for record in records:
        if type(record) is not dict or type(record.get("raw_headers")) is not list or type(record.get("data")) is not list:
            raise CompactStorageError("invalid_occurrence_record")
        if (header_count + len(record["raw_headers"]) > MAX_CHUNK_HEADERS or
                point_count + len(record["data"]) > MAX_CHUNK_POINTS):
            raise CompactStorageError("chunk_array_byte_limit")
        h_extras, p_extras = [], []
        for item in record["raw_headers"]:
            if type(item) is not dict:
                raise CompactStorageError("invalid_copied_header")
            for key in HEADER_FIELDS:
                headers.extend(_cell(item.get(key, _MISSING), fallback))
            h_extras.append(_typed({k: v for k, v in item.items() if k not in HEADER_FIELDS}))
        for item in record["data"]:
            if type(item) is not dict:
                raise CompactStorageError("invalid_copied_point")
            extras = {k: v for k, v in item.items() if k not in POINT_FIELDS + POINT_SCALARS}
            for key in POINT_FIELDS:
                vector = item.get(key, _MISSING)
                if type(vector) is list and len(vector) == 3:
                    values = vector
                else:
                    values = [_MISSING] * 3
                    if vector is not _MISSING:
                        extras[key] = vector
                for value in values:
                    points.extend(_cell(value, fallback))
            for key in POINT_SCALARS:
                points.extend(_cell(item.get(key, _MISSING), fallback))
            p_extras.append(_typed(extras))
        nh, np = len(record["raw_headers"]), len(record["data"])
        descriptors.append({"h": [header_count, nh], "p": [point_count, np],
                            "meta": _typed({k: v for k, v in record.items() if k not in ("raw_headers", "data")}),
                            "hx": h_extras, "px": p_extras})
        header_count += nh
        point_count += np
    meta = _json_bytes({"records": descriptors, "fallback": fallback}, MAX_CHUNK_METADATA_BYTES)
    if len(meta) > MAX_CHUNK_METADATA_BYTES:
        raise CompactStorageError("chunk_metadata_byte_limit")
    prefix = _PREFIX.pack(_MAGIC, 1, 0x4c45, header_count, point_count, len(descriptors), len(meta), 0, _LAYOUT)
    return prefix + headers + points + meta


def decode_records(raw):
    if len(raw) < PREFIX_BYTES:
        raise CompactStorageError("truncated_chunk_prefix")
    magic, version, endian, nh, np, nr, nm, flags, layout = _PREFIX.unpack_from(raw)
    if (magic, version, endian, flags, layout) != (_MAGIC, 1, 0x4c45, 0, _LAYOUT):
        raise CompactStorageError("chunk_schema_endian_layout_mismatch")
    expected = PREFIX_BYTES + nh * HEADER_BYTES + np * POINT_BYTES + nm
    if (len(raw) != expected or nm > MAX_CHUNK_METADATA_BYTES or nh > MAX_CHUNK_HEADERS
            or np > MAX_CHUNK_POINTS):
        raise CompactStorageError("chunk_size_mismatch")
    po = PREFIX_BYTES + nh * HEADER_BYTES
    meta = _load_json(raw[po + np * POINT_BYTES:])
    if set(meta) != {"records", "fallback"} or len(meta["records"]) != nr:
        raise CompactStorageError("chunk_descriptor_mismatch")
    fallback = meta["fallback"]
    records, hi, pi = [], 0, 0
    for desc in meta["records"]:
        if set(desc) != {"h", "p", "meta", "hx", "px"} or desc["h"][0] != hi or desc["p"][0] != pi:
            raise CompactStorageError("chunk_offsets_mismatch")
        hc, pc = desc["h"][1], desc["p"][1]
        if type(hc) is not int or type(pc) is not int or min(hc, pc) < 0 or hi + hc > nh or pi + pc > np or len(desc["hx"]) != hc or len(desc["px"]) != pc:
            raise CompactStorageError("chunk_counts_mismatch")
        record = _untyped(desc["meta"])
        if type(record) is not dict or "raw_headers" in record or "data" in record:
            raise CompactStorageError("chunk_receipt_metadata")
        record["raw_headers"], record["data"] = [], []
        for i in range(hc):
            item = _untyped(desc["hx"][i])
            if type(item) is not dict or set(item).intersection(HEADER_FIELDS):
                raise CompactStorageError("header_extras_collision")
            for j, key in enumerate(HEADER_FIELDS):
                value = _read_cell(raw, PREFIX_BYTES + (hi + i) * HEADER_BYTES + j * 9, fallback)
                if value is not _MISSING:
                    item[key] = value
            record["raw_headers"].append(item)
        for i in range(pc):
            item = _untyped(desc["px"][i])
            if type(item) is not dict or set(item).intersection(POINT_SCALARS):
                raise CompactStorageError("point_extras_collision")
            values = [_read_cell(raw, po + (pi + i) * POINT_BYTES + j * 9, fallback) for j in range(15)]
            for j, key in enumerate(POINT_FIELDS):
                vector = values[j * 3:j * 3 + 3]
                if all(v is not _MISSING for v in vector):
                    if key in item:
                        raise CompactStorageError("point_vector_collision")
                    item[key] = vector
                elif not all(v is _MISSING for v in vector):
                    raise CompactStorageError("partial_missing_vector")
            for key, value in zip(POINT_SCALARS, values[9:]):
                if value is not _MISSING:
                    item[key] = value
            record["data"].append(item)
        records.append(record)
        hi += hc
        pi += pc
    if (hi, pi) != (nh, np):
        raise CompactStorageError("unclaimed_chunk_arrays")
    return records


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _file_identity(path):
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            count += len(block)
    return count, digest.hexdigest()


class CompactContactWriter:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        for name in ("chunks", "records", "dictionaries"):
            (self.root / name).mkdir()
        self.archive_id = uuid.uuid4().hex
        self._occurrences, self._dictionaries = {}, {}
        self._sequence = self._chunks = self._boundaries = self._lifecycles = self._dictionary_bytes = 0
        self._index_bytes = 0
        self._closed = False
        self._errors = []
        self._publish("archive.json", _json_bytes({"schema": SCHEMA, "archive_id": self.archive_id}), "archive")
        with (self.root / "index.jsonl").open("xb") as stream:
            stream.flush()
            os.fsync(stream.fileno())

    def _publish(self, relative, raw, kind):
        path = self.root / relative
        temporary = path.with_name(path.name + ".tmp")
        if path.exists() or temporary.exists():
            raise CompactStorageError("immutable_output_exists:" + relative)
        with temporary.open("xb") as stream:
            if stream.write(raw) != len(raw):
                raise OSError("short_file_write:" + relative)
            stream.flush()
            os.fsync(stream.fileno())
        if _file_identity(temporary) != (len(raw), _sha(raw)):
            raise OSError("published_file_write_verification:" + relative)
        os.rename(temporary, path)
        return {"schema": SCHEMA, "kind": kind, "path": relative, "bytes": len(raw), "sha256": _sha(raw)}

    def _index(self, event):
        event = dict(event, sequence=self._sequence, archive_id=self.archive_id, schema=SCHEMA)
        raw = _json_bytes(event, MAX_INDEX_ENTRY_BYTES - 1) + b"\n"
        if self._index_bytes + len(raw) > MAX_INDEX_BYTES:
            raise CompactStorageError("index_byte_limit")
        with (self.root / "index.jsonl").open("ab") as stream:
            if stream.write(raw) != len(raw):
                raise OSError("short_index_write")
            stream.flush()
            os.fsync(stream.fileno())
        with (self.root / "index.jsonl").open("rb") as stream:
            stream.seek(self._index_bytes)
            if stream.read(len(raw) + 1) != raw:
                raise OSError("index_append_write_verification")
        self._sequence += 1
        self._index_bytes += len(raw)

    def intern_dictionary(self, value, kind="context"):
        self._require_open()
        try:
            raw = _json_bytes({"schema": SCHEMA, "kind": kind, "value": _typed(value)}, MAX_DICTIONARY_BYTES)
            if len(raw) > MAX_DICTIONARY_BYTES:
                raise CompactStorageError("dictionary_byte_limit")
            digest = _sha(raw)
            if digest in self._dictionaries:
                return dict(self._dictionaries[digest])
            if self._dictionary_bytes + len(raw) > MAX_DICTIONARY_POOL_BYTES:
                raise CompactStorageError("dictionary_pool_byte_limit")
            ref = self._publish("dictionaries/" + digest + ".json", raw, "dictionary")
            self._index({"kind": "dictionary", "ref": ref})
            self._dictionaries[digest] = ref
            self._dictionary_bytes += len(raw)
            return dict(ref)
        except Exception as exc:
            self._record_error(exc)
            raise

    def _require_open(self):
        if self._closed:
            raise CompactStorageError("archive_already_finalized")

    def _record_error(self, exc):
        self._errors.append(type(exc).__name__ + ":" + str(exc))

    def _write(self, kind, value):
        self._require_open()
        new, pending = [], {}
        retained_headers = retained_points = occurrence_count = 0
        def extract(item):
            nonlocal retained_headers, retained_points, occurrence_count
            if type(item) is dict and "raw_headers" in item and "data" in item and "sequence" in item and "subscription_generation" in item:
                if type(item["raw_headers"]) is not list or type(item["data"]) is not list:
                    raise CompactStorageError("invalid_occurrence_arrays")
                retained_headers += len(item["raw_headers"])
                retained_points += len(item["data"])
                occurrence_count += 1
                if (retained_headers > MAX_CHUNK_HEADERS or retained_points > MAX_CHUNK_POINTS
                        or occurrence_count > MAX_RECORD_OCCURRENCES):
                    raise CompactStorageError("reconstructed_record_byte_limit")
                gen, seq = item["subscription_generation"], item["sequence"]
                if type(gen) is not int or type(seq) is not int or gen < 0 or seq < 0:
                    raise CompactStorageError("invalid_occurrence_identity")
                key = (gen, seq)
                fingerprint = _sha(encode_records([item]))
                if key in self._occurrences:
                    old, prior = self._occurrences[key]
                    if fingerprint != prior:
                        raise CompactStorageError("occurrence_mutated")
                    return _Occurrence(old)
                if key in pending:
                    ref, prior = pending[key]
                    if fingerprint != prior:
                        raise CompactStorageError("occurrence_mutated")
                    return _Occurrence(ref)
                ref = {"archive_id": self.archive_id, "generation": gen, "sequence": seq,
                       "chunk": None, "record_index": len(new), "fingerprint": fingerprint}
                pending[key] = (ref, fingerprint)
                new.append(item)
                return _Occurrence(ref)
            if type(item) is dict:
                return {k: extract(v) for k, v in item.items()}
            if type(item) is list:
                return [extract(v) for v in item]
            if type(item) is tuple:
                return tuple(extract(v) for v in item)
            return item
        try:
            shell = extract(value)
            chunk = None
            if new:
                chunk = self._publish("chunks/%08d.bin" % self._chunks, encode_records(new), "chunk")
                self._chunks += 1
                for ref, _ in pending.values():
                    ref["chunk"] = chunk
            raw = _json_bytes({"schema": SCHEMA, "kind": kind, "value": _typed(shell)}, MAX_RECORD_BYTES)
            if len(raw) > MAX_RECORD_BYTES:
                raise CompactStorageError("record_metadata_byte_limit")
            ref = self._publish("records/%08d.json" % self._sequence, raw, kind)
            self._index({"kind": kind, "ref": ref, "chunk": chunk})
            self._occurrences.update(pending)
            if kind == "boundary":
                self._boundaries += 1
            else:
                self._lifecycles += 1
            return dict(ref)
        except Exception as exc:
            self._record_error(exc)
            raise

    def write_boundary(self, envelope, context, live_target):
        return self._write("boundary", {"envelope": envelope, "context": context, "live_target": live_target})

    def write_lifecycle(self, scope, lifecycle):
        return self._write("lifecycle", {"scope": scope, "lifecycle": lifecycle})

    def finish(self, primary_errors=()):
        self._require_open()
        size, digest = _file_identity(self.root / "index.jsonl")
        manifest = {"schema": SCHEMA, "archive_id": self.archive_id, "index_bytes": size,
                    "index_sha256": digest, "entries": self._sequence, "chunks": self._chunks,
                    "boundaries": self._boundaries, "lifecycles": self._lifecycles,
                    "unique_occurrences": len(self._occurrences), "dictionaries": len(self._dictionaries),
                    "dictionary_bytes": self._dictionary_bytes, "primary_errors": list(primary_errors),
                    "storage_errors": list(self._errors), "complete": not primary_errors and not self._errors}
        raw = _json_bytes(manifest, MAX_MANIFEST_BYTES)
        self._publish("manifest.json", raw, "manifest")
        self._closed = True
        return manifest


class CompactContactReader:
    def __init__(self, root, *, require_final=False):
        self.root = Path(root)
        archive = _load_json((self.root / "archive.json").read_bytes())
        if archive.get("schema") != SCHEMA:
            raise CompactStorageError("archive_schema_mismatch")
        self.archive_id = archive["archive_id"]
        self._entries, self._refs, self._chunks, self._seen_occurrences = [], {}, {}, {}
        self._consumed_records = set()
        index_path = self.root / "index.jsonl"
        if index_path.stat().st_size > MAX_INDEX_BYTES:
            raise CompactStorageError("index_byte_limit")
        with index_path.open("rb") as stream:
            for line in stream:
                if len(line) > MAX_INDEX_ENTRY_BYTES or not line.endswith(b"\n"):
                    raise CompactStorageError("partial_index_record")
                entry = _load_json(line)
                if entry.get("schema") != SCHEMA or entry.get("archive_id") != self.archive_id or entry.get("sequence") != len(self._entries):
                    raise CompactStorageError("index_sequence_or_archive")
                if entry.get("kind") not in ("boundary", "lifecycle", "dictionary"):
                    raise CompactStorageError("index_entry_kind")
                ref = entry["ref"]
                if ref.get("schema") != SCHEMA or ref.get("kind") != entry["kind"]:
                    raise CompactStorageError("index_reference_kind")
                if ref["path"] in self._refs:
                    raise CompactStorageError("duplicate_index_reference")
                self._refs[ref["path"]] = ref
                if entry.get("chunk") is not None:
                    chunk = entry["chunk"]
                    if chunk["path"] in self._chunks:
                        raise CompactStorageError("duplicate_chunk_transaction")
                    self._chunks[chunk["path"]] = chunk
                self._entries.append(entry)
        manifest_path = self.root / "manifest.json"
        self.manifest = None
        if manifest_path.exists():
            if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
                raise CompactStorageError("manifest_byte_limit")
            self.manifest = _load_json(manifest_path.read_bytes())
            size, digest = _file_identity(index_path)
            if (self.manifest.get("schema"), self.manifest.get("archive_id"), self.manifest.get("index_bytes"),
                self.manifest.get("index_sha256"), self.manifest.get("entries")) != (SCHEMA, self.archive_id, size, digest, len(self._entries)):
                raise CompactStorageError("manifest_index_mismatch")
            if self.manifest.get("complete") is True and (self.manifest.get("storage_errors") or self.manifest.get("primary_errors")):
                raise CompactStorageError("manifest_false_completion")
            # A saved, permitted experimental miss may have primary errors while
            # its storage is complete.  It still needs exact file membership.
            if not self.manifest.get("storage_errors"):
                expected_counts = (sum(e["kind"] == "boundary" for e in self._entries),
                                   sum(e["kind"] == "lifecycle" for e in self._entries),
                                   sum(e["kind"] == "dictionary" for e in self._entries), len(self._chunks))
                actual_counts = tuple(self.manifest.get(k) for k in ("boundaries", "lifecycles", "dictionaries", "chunks"))
                if actual_counts != expected_counts:
                    raise CompactStorageError("manifest_false_completion")
                dictionary_bytes = sum(e["ref"]["bytes"] for e in self._entries if e["kind"] == "dictionary")
                if dictionary_bytes != self.manifest.get("dictionary_bytes") or dictionary_bytes > MAX_DICTIONARY_POOL_BYTES:
                    raise CompactStorageError("manifest_dictionary_byte_count")
                expected_paths = set(self._refs) | set(self._chunks) | {"archive.json", "index.jsonl", "manifest.json"}
                actual_paths = {p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()}
                if actual_paths != expected_paths:
                    raise CompactStorageError("archive_missing_or_unindexed_file")
        elif require_final:
            raise CompactStorageError("missing_final_manifest")

    def _read(self, ref, kind, *, chunk=False):
        registry = self._chunks if chunk else self._refs
        if not isinstance(ref, dict) or ref.get("schema") != SCHEMA or ref.get("kind") != kind or registry.get(ref.get("path")) != ref:
            raise CompactStorageError("unindexed_or_altered_reference")
        relative = ref["path"]
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts or "\\" in relative:
            raise CompactStorageError("invalid_reference_path")
        path = self.root / relative
        expected_directory = "chunks" if chunk else ("dictionaries" if kind == "dictionary" else "records")
        if (":" in relative or relative.split("/")[0] != expected_directory or len(relative.split("/")) != 2
                or not path.resolve().is_relative_to(self.root.resolve())):
            raise CompactStorageError("invalid_reference_path")
        size, digest = _file_identity(path)
        if (size, digest) != (ref["bytes"], ref["sha256"]):
            raise CompactStorageError("artifact_size_or_checksum")
        limit = MAX_DICTIONARY_BYTES if kind == "dictionary" else (MAX_CHUNK_BYTES if chunk else MAX_RECORD_BYTES)
        if limit is not None and size > limit:
            raise CompactStorageError("referenced_metadata_byte_limit")
        return path.read_bytes()

    def _record(self, ref, kind):
        cached_path, cached_records = None, None
        retained_headers = retained_points = occurrences = 0
        def resolve(occurrence):
            nonlocal cached_path, cached_records, retained_headers, retained_points, occurrences
            if occurrence.get("archive_id") != self.archive_id:
                raise CompactStorageError("occurrence_archive_mismatch")
            chunk = occurrence["chunk"]
            path = chunk["path"]
            if path != cached_path:
                # Only the most recently needed chunk is cached.  A forged
                # multi-chunk envelope cannot create a whole-archive cache.
                cached_records = decode_records(self._read(chunk, "chunk", chunk=True))
                cached_path = path
            index = occurrence["record_index"]
            if type(index) is not int or not 0 <= index < len(cached_records):
                raise CompactStorageError("occurrence_record_index")
            result = cached_records[index]
            retained_headers += len(result["raw_headers"])
            retained_points += len(result["data"])
            occurrences += 1
            if (retained_headers > MAX_CHUNK_HEADERS or retained_points > MAX_CHUNK_POINTS
                    or occurrences > MAX_RECORD_OCCURRENCES):
                raise CompactStorageError("reconstructed_record_byte_limit")
            if (result.get("subscription_generation"), result.get("sequence"), _sha(encode_records([result]))) != (
                    occurrence["generation"], occurrence["sequence"], occurrence["fingerprint"]):
                raise CompactStorageError("occurrence_identity_or_fingerprint")
            key = (occurrence["generation"], occurrence["sequence"])
            prior = self._seen_occurrences.setdefault(key, occurrence)
            if prior != occurrence:
                raise CompactStorageError("occurrence_address_redefined")
            return result
        record = _load_json(self._read(ref, kind))
        if record.get("schema") != SCHEMA or record.get("kind") != kind:
            raise CompactStorageError("record_schema_mismatch")
        result = _untyped(record["value"], resolve)
        self._consumed_records.add(ref["path"])
        return result

    def read_boundary(self, ref):
        return self._record(ref, "boundary")

    def read_lifecycle(self, ref):
        return self._record(ref, "lifecycle")["lifecycle"]

    def read_dictionary(self, ref):
        record = _load_json(self._read(ref, "dictionary"))
        if record.get("schema") != SCHEMA:
            raise CompactStorageError("dictionary_schema_mismatch")
        return _untyped(record["value"])

    def iter_boundaries(self):
        for entry in self._entries:
            if entry["kind"] == "boundary":
                yield dict(self.read_boundary(entry["ref"]), ref=dict(entry["ref"]))

    def verify_record_references(self, refs, *, kinds=("boundary", "lifecycle")):
        """Require exactly the saved small measurement/lifecycle reference set.

        No raw payload is reread.  Dictionary entries are deliberately excluded:
        setup may have interned a mapping before a truthful primary failure.
        """
        if not set(kinds).issubset({"boundary", "lifecycle"}):
            raise CompactStorageError("invalid_record_reference_kinds")
        expected = {e["ref"]["path"]: e["ref"] for e in self._entries if e["kind"] in kinds}
        actual = {}
        for ref in refs:
            if not isinstance(ref, dict) or ref.get("path") in actual or expected.get(ref.get("path")) != ref:
                raise CompactStorageError("record_reference_set_mismatch")
            actual[ref["path"]] = ref
        if actual != expected:
            raise CompactStorageError("record_reference_set_mismatch")
        return True

    def verify_consumed_records(self, *, kinds=("boundary", "lifecycle")):
        expected = {e["ref"]["path"] for e in self._entries if e["kind"] in kinds}
        if not expected.issubset(self._consumed_records):
            raise CompactStorageError("unconsumed_required_record")
        if (set(kinds) == {"boundary", "lifecycle"} and self.manifest is not None
                and not self.manifest.get("storage_errors")
                and len(self._seen_occurrences) != self.manifest.get("unique_occurrences")):
            raise CompactStorageError("manifest_occurrence_count")
        return True
