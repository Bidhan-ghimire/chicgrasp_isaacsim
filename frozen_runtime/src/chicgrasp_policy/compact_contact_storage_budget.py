"""Read-only, exact-byte planning for compact callback-contact persistence.

This is an uncompressed bounded-output estimate, not a disk reservation or
permission to launch. The once-only 50 percent margin covers uncertainty, not
unbounded logs, arbitrary malformed strings, or unrelated disk consumption.
"""
from pathlib import Path
import argparse
import datetime
import json
import os
import shutil

from . import compact_contact_storage as codec

BOUNDARIES = 2400
LANES = 2
HEADERS = 4096
POINTS = 16384
LIFECYCLE_RECORDS = 2
PNG_LIMIT = 512 * 1024
EPISODE_LIMIT = 128 * 1024**2
STATE_LIMIT = BOUNDARIES * 256 * 1024 + 1024**2
RGB_METADATA_LIMIT = 8 * 1024**2
REPORT_LIMIT = 4 * 1024**3
LOG_LIMIT = 16 * 1024**3
EXTERNAL_EVIDENCE_ALLOWANCE = 2 * 1024**3


def storage_model(*, allocation_unit=4096):
    if type(allocation_unit) is not int or allocation_unit <= 0:
        raise ValueError("positive_integer_allocation_unit_required")
    lane = HEADERS * codec.HEADER_BYTES + POINTS * codec.POINT_BYTES
    records = BOUNDARIES + LIFECYCLE_RECORDS
    chunk_max = codec.PREFIX_BYTES + LANES * lane + codec.MAX_CHUNK_METADATA_BYTES
    # An indexed event necessarily contains a64-character SHA256. This weak
    # lower bound deliberately overestimates file count; no extra retention cap.
    max_indexed_files = codec.MAX_INDEX_BYTES // 64
    # At most one new chunk per boundary/lifecycle, plus archive/index/manifest.
    # Noncompact budget allows4096 other files (402PNGs +metadata +reports).
    max_files = max_indexed_files + records + 3 + 4096
    lines = []
    def add(name, size, formula, source):
        lines.append(dict(name=name, bytes=size, formula=formula, source=source))
    add("candidate_raw_arrays", BOUNDARIES * LANES * lane,
        "2400 * 2 * (4096 * H + 16384 * P)", "actual binary H/P layout")
    add("initialization_and_terminal_raw_arrays", 2 * lane,
        "2 * (4096 * H + 16384 * P)",
        "one retained initialization queue plus undrained final outside queue; repeated lifecycle references deduplicated")
    add("chunk_prefixes", records * codec.PREFIX_BYTES, "2402 * prefix_bytes",
        "at most one new chunk per boundary or lifecycle write")
    add("bounded_chunk_metadata", records * codec.MAX_CHUNK_METADATA_BYTES,
        "2402 * MAX_CHUNK_METADATA_BYTES", "enforced typed metadata/error/receipt ceiling, not fixture average")
    add("boundary_and_lifecycle_records", records * codec.MAX_RECORD_BYTES,
        "2402 * MAX_RECORD_BYTES", "enforced decision/context/reference record ceiling")
    add("mapping_configuration_dictionary_pool", codec.MAX_DICTIONARY_POOL_BYTES,
        "MAX_DICTIONARY_POOL_BYTES", "enforced aggregate pool; includes all applicable versions")
    add("index_and_checksums", codec.MAX_INDEX_BYTES, "MAX_INDEX_BYTES",
        "enforced full index ceiling; file and chunk hashes already included")
    add("archive_and_manifest", codec.MAX_MANIFEST_BYTES + 4096,
        "MAX_MANIFEST_BYTES + 4096", "bounded final manifest plus archive header")
    add("small_state_measurements", STATE_LIMIT, "2400 * 256 KiB + 1 MiB",
        "retained numeric pose/load/clock/state rows; no raw or decoded contact arrays")
    add("lossless_rgb", 402 * PNG_LIMIT, "402 * 512 KiB",
        "201 bundles including shared-acquisition PRE; uncompressed-size allowance per240x320RGB PNG")
    add("complete_episode", EPISODE_LIMIT, "128 MiB",
        "unchanged full episode arrays including400RGB images; no compression saving required")
    add("rgb_metadata", RGB_METADATA_LIMIT, "8 MiB", "unchanged per-image/bundle/index records")
    add("configuration_owner_journal_results", REPORT_LIMIT, "4 GiB",
        "remaining small runtime reports/owner operations; excludes separately charged raw files/state/RGB")
    add("process_logs", LOG_LIMIT, "16 GiB", "bounded-output assumption; excess explicitly invalidates storage qualification")
    add("external_offline_evidence", EXTERNAL_EVIDENCE_ALLOWANCE, "2 GiB",
        "future offline reports/preflight reserve, not four candidate archives")
    add("peak_incomplete_publication", max(chunk_max, codec.MAX_RECORD_BYTES,
        codec.MAX_DICTIONARY_BYTES, EPISODE_LIMIT + 2 * PNG_LIMIT),
        "max(one binary chunk, one record, one dictionary, episode + two PNGs)",
        "sequential temporary-publication headroom; no whole-run duplicate")
    add("filesystem_slack_and_metadata", max_files * (allocation_unit - 1 + 1024),
        "max_files * (allocation_unit - 1 + 1024)",
        "cluster tail plus1KiB/file NTFS-record allowance; explicit bounded filesystem assumption")
    subtotal = sum(x["bytes"] for x in lines)
    margin = (subtotal + 1) // 2
    return dict(schema="compact-contact-storage-budget/v1", contact_schema=codec.SCHEMA,
        header_bytes=codec.HEADER_BYTES, point_bytes=codec.POINT_BYTES,
        prefix_bytes=codec.PREFIX_BYTES, allocation_unit_bytes=allocation_unit,
        planning_boundaries=BOUNDARIES, current_late_lanes=LANES,
        lifecycle_records=LIFECYCLE_RECORDS, maximum_chunk_bytes=chunk_max,
        maximum_planned_files=max_files, components=lines, subtotal_bytes=subtotal,
        margin_policy="50 percent of subtotal, once, rounded up to whole byte",
        margin_bytes=margin, required_bytes=subtotal+margin,
        required_decimal_GB=(subtotal+margin)/10**9,
        required_binary_GiB=(subtotal+margin)/1024**3,
        assumptions=["No compression or sparse-file savings.",
            "Actual native/drop omissions remain explicit; persistence bounds do not alter collector caps.",
            "Oversize typed/context/dictionary/index records fail required persistence, never truncate to successful records.",
            "Logs and non-contact allowances are checked from file sizes; exceeding them fails storage qualification.",
            "Filesystem1KiB/file overhead is an NTFS planning allowance, not measured whole-volume metadata attribution.",
            "Free space must be remeasured before each separately authorized reservation; prior attempts remain on disk.",
            "This estimate establishes neither capture performance, RAM feasibility, nor native correctness."])


def validate_output_allowances(attempt_root):
    """Check observable noncontact assumptions without reading large payloads.

    Called only for new compact packages by independent adjudication. Contact
    file limits and integrity are the compact reader's separate responsibility.
    """
    root = Path(attempt_root)
    totals = dict(state=0, rgb=0, episode=0, logs=0, other=0)
    file_count = 0
    issues = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if "contact_compact" in rel.parts:
            continue
        file_count += 1
        size = path.stat().st_size
        name = path.name.lower()
        if path.suffix.lower() == ".png":
            if size > PNG_LIMIT:
                issues.append("storage_png_byte_assumption_exceeded:" + rel.as_posix())
            totals["rgb"] += size
        elif name == "measurements.json":
            totals["state"] += size
        elif path.suffix.lower() == ".npz":
            totals["episode"] += size
        elif name in ("stdout.bin", "stderr.bin") or path.suffix.lower() == ".log":
            totals["logs"] += size
        else:
            totals["other"] += size
    if file_count > 4096:
        issues.append("storage_noncontact_file_count_assumption_exceeded")
    for key, limit in (("state", STATE_LIMIT), ("rgb", 402*PNG_LIMIT),
        ("episode", EPISODE_LIMIT), ("logs", LOG_LIMIT),
        ("other", REPORT_LIMIT + RGB_METADATA_LIMIT)):
        if totals[key] > limit:
            issues.append("storage_" + key + "_byte_assumption_exceeded")
    return issues


def filesystem_allocation_unit(path):
    """Read filesystem cluster size; this is an OS query, not simulation access."""
    if os.name != "nt":
        return os.statvfs(path).f_frsize
    import ctypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel.GetDiskFreeSpaceW
    pointer = ctypes.POINTER(ctypes.c_uint32)
    function.argtypes = [ctypes.c_wchar_p, pointer, pointer, pointer, pointer]
    function.restype = ctypes.c_int
    sectors, size, free, total = (ctypes.c_uint32() for _ in range(4))
    if not function(str(Path(path).anchor), ctypes.byref(sectors), ctypes.byref(size),
                    ctypes.byref(free), ctypes.byref(total)):
        raise ctypes.WinError(ctypes.get_last_error())
    return sectors.value * size.value


def capacity_snapshot(path):
    path = Path(path).resolve(strict=True)
    usage = shutil.disk_usage(path)
    result = storage_model(allocation_unit=filesystem_allocation_unit(path))
    result["capacity"] = dict(path=str(path), observed_utc=datetime.datetime.now(
        datetime.timezone.utc).isoformat(), total_bytes=usage.total,
        used_bytes=usage.used, available_bytes=usage.free,
        available_decimal_GB=usage.free/10**9, available_binary_GiB=usage.free/1024**3,
        sufficient=usage.free >= result["required_bytes"],
        surplus_bytes=usage.free-result["required_bytes"],
        scope="filesystem containing this path; not aggregate computer storage")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="existing future-output ancestor on the destination volume")
    args = parser.parse_args()
    print(json.dumps(capacity_snapshot(args.path), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
