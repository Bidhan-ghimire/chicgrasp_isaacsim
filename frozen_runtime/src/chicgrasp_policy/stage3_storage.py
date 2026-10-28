"""Observed slim payload plus policy records and fixed1GiB; no contact model."""
from pathlib import Path
import shutil

from .stage1_demo import load
from .geometric_capture_check import identity

PROJECT = Path(__file__).resolve().parents[2]
ESTIMATE = PROJECT / "docs/checkpoints/gate6_stage3/storage.json"


def estimate(stage1_attempt, stage3_fixture):
    """Use actual saved Stage1 imagery and serialized full-run Stage3 fixtures."""
    source, fixture = Path(stage1_attempt), Path(stage3_fixture)
    experiment = source / "capture/experiment"
    fixture_output = fixture / "capture/experiment"
    stage1_sizes = {}
    for path in experiment.rglob("*"):
        if path.is_file():
            relative = path.relative_to(experiment).as_posix()
            if relative.startswith("episode/") or relative in ("stage1_policy_rows.jsonl", "stage1_substeps.jsonl"):
                continue
            if "contact" in relative.lower():
                raise ValueError("source_must_be_slim")
            stage1_sizes[relative] = path.stat().st_size
    fixture_sizes = {p.relative_to(fixture_output).as_posix(): p.stat().st_size
                     for p in fixture_output.rglob("*") if p.is_file()}
    # Same categories use the larger observed value; Stage3-only streams add bytes.
    sizes = {name: max(stage1_sizes.get(name, 0), fixture_sizes.get(name, 0))
             for name in set(stage1_sizes) | set(fixture_sizes)}
    for name in ("stdout.bin", "stderr.bin", "structured/events.jsonl"):
        sizes["parent/"+name] = max((source/"capture"/name).stat().st_size,
                                    (fixture/"capture"/name).stat().st_size)
    sizes["parent_records_and_growth_allowance"] = 1 << 20
    payload = sum(sizes.values())
    return dict(schema="stage3-observed-storage/v1", estimated_payload_bytes=payload,
        fixed_margin_bytes=1 << 30, required_bytes=payload+(1 << 30),
        required_GiB=(payload+(1 << 30))/(1 << 30), components_bytes=dict(sorted(sizes.items())),
        policy_payload_bytes=sum(v for k,v in sizes.items() if k.startswith(("stage3_", "predictions/"))),
        source_attempt=str(source), fixture_attempt=str(fixture),
        source_manifest=identity(experiment/"artifact_manifest.json"),
        fixture_manifest=identity(fixture_output/"artifact_manifest.json"),
        counts=dict(substeps=2400, setup_actions=50, learned_actions=150, inferences=13,
                    rgb_bundles=201, rgb_images=402), contact_payload_bytes=0,
        limit="Sample-based estimate; future image and diagnostic sizes may differ. Fixed margin is additional.")


WITNESS_ESTIMATE = PROJECT / "docs/checkpoints/gate6_stage3/witness_storage.json"


def check_available(output, *, witness=False, envelope_v2=False):
    path = WITNESS_ESTIMATE if witness else ESTIMATE
    value = load(path)
    if value["fixed_margin_bytes"] != 1 << 30 or value["required_bytes"] != value["estimated_payload_bytes"]+(1 << 30):
        raise ValueError("stage3_storage_estimate_contract")
    if envelope_v2:
        value = envelope_estimate(value)
    free = shutil.disk_usage(Path(output).resolve()).free
    record = dict(estimate_sha256=identity(path)["sha256"], estimate=value,
                  free_bytes=free, output_filesystem=str(Path(output).resolve().anchor),
                  passed=free >= value["required_bytes"])
    return record


def envelope_estimate(value):
    from copy import deepcopy
    value=deepcopy(value)
    # Worst permitted279 actions, scaling observed payload, then fixed1GiB.
    value["estimated_payload_bytes"]=(value["estimated_payload_bytes"]*279+199)//200
    value["required_bytes"]=value["estimated_payload_bytes"]+(1<<30)
    value["required_GiB"]=value["required_bytes"]/(1<<30)
    value["envelope_v2_max_actions"]=279
    return value
