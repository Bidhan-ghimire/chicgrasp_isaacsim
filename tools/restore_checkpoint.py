#!/usr/bin/env python3
"""Reassemble a byte-preserving checkpoint without loading a model.

Example:
  python restore_checkpoint.py --manifest checkpoint-parts.json --output /new/BEST.pt

Parts must be ordinary sibling files of the manifest. Existing output files are
never overwritten. Only the Python standard library is required.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile


SCHEMA = "byte-preserving-checkpoint-parts/v1"
BUFFER_BYTES = 8 * 1024 * 1024
HEX_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
WINDOWS_DEVICES = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


def byte_count(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def sha256_text(value, label):
    if not isinstance(value, str) or not HEX_SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a SHA256 hexadecimal digest")
    return value.lower()


def validate_manifest(manifest):
    value = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError(f"Expected manifest schema {SCHEMA}")
    expected_bytes = byte_count(value.get("bytes"), "checkpoint bytes")
    expected_sha = sha256_text(value.get("sha256"), "checkpoint sha256")
    parts = value.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Manifest must contain a nonempty parts list")
    validated = []
    seen = set()
    for index, item in enumerate(parts, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Part {index} must be an object")
        name = item.get("name")
        if (not isinstance(name, str) or not SAFE_NAME.fullmatch(name)
                or name.endswith(".") or name.split(".", 1)[0].upper() in WINDOWS_DEVICES):
            raise ValueError(f"Part {index} has an unsafe filename")
        # Reject case-only duplicates too, for portable behavior on Windows.
        if name.casefold() in seen:
            raise ValueError(f"Duplicate part filename: {name}")
        seen.add(name.casefold())
        size = byte_count(item.get("bytes"), f"{name} bytes")
        digest = sha256_text(item.get("sha256"), f"{name} sha256")
        part = manifest.parent / name
        if part.is_symlink() or not part.is_file():
            raise ValueError(f"Part must be an ordinary file: {name}")
        if part.resolve(strict=True).parent != manifest.parent:
            raise ValueError(f"Part is outside the manifest directory: {name}")
        if part.stat().st_size != size:
            raise ValueError(f"Part size does not match manifest: {name}")
        validated.append((part, size, digest))
    if sum(item[1] for item in validated) != expected_bytes:
        raise ValueError("Part sizes do not sum to checkpoint size")
    return validated, expected_bytes, expected_sha


def finalize_without_overwrite(temporary, destination):
    if os.name == "nt":
        # Windows rename fails if the destination exists, including a race.
        os.rename(temporary, destination)
    else:
        # POSIX rename would overwrite. An exclusive hard-link creation publishes
        # the completed file atomically and fails if the destination already exists.
        # Both files are in the same directory/filesystem. If hard links are not
        # supported, fail safely rather than fall back to an overwriting operation.
        os.link(temporary, destination)
        temporary.unlink()


def restore(manifest_path, output_path):
    manifest = manifest_path.expanduser().resolve(strict=True)
    if not manifest.is_file():
        raise ValueError("Manifest must be a file")
    parts, expected_bytes, expected_sha = validate_manifest(manifest)
    output = output_path.expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    parent = output.parent.resolve(strict=True)
    output = parent / output.name
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if any(output == part.resolve(strict=True) for part, _, _ in parts):
        raise ValueError("Output must not be an input part")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".checkpoint-restore-",
                                         suffix=".tmp", dir=parent, delete=False) as writer:
            temporary = Path(writer.name)
            whole_hash = hashlib.sha256()
            whole_bytes = 0
            for part, expected_part_bytes, expected_part_sha in parts:
                part_hash = hashlib.sha256()
                part_bytes = 0
                with part.open("rb") as reader:
                    while True:
                        block = reader.read(BUFFER_BYTES)
                        if not block:
                            break
                        part_bytes += len(block)
                        if part_bytes > expected_part_bytes:
                            raise ValueError(f"Part grew or exceeds expected size: {part.name}")
                        writer.write(block)
                        part_hash.update(block)
                        whole_hash.update(block)
                        whole_bytes += len(block)
                if part_bytes != expected_part_bytes or part_hash.hexdigest() != expected_part_sha:
                    raise ValueError(f"Part verification failed: {part.name}")
                print(f"Verified {part.name}: {part_bytes} bytes", flush=True)
            if whole_bytes != expected_bytes or whole_hash.hexdigest() != expected_sha:
                raise ValueError("Final checkpoint size or SHA256 does not match manifest")
            writer.flush()
            os.fsync(writer.fileno())
        finalize_without_overwrite(temporary, output)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"RESTORE_VERIFIED: {output}\nBytes: {expected_bytes}\nSHA256: {expected_sha}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="Explicit new output filename; manifest target_path is not used for writing")
    args = parser.parse_args()
    try:
        restore(args.manifest, args.output)
    except (OSError, ValueError, TypeError) as exc:
        print(f"RESTORE_FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
