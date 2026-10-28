"""Shared camera-enabled startup profile and passive renderer provenance.

No native imports or setting writes. Matching a profile is not pixel parity;
actual GUI/headless images still require visual comparison and owner approval.
"""
import hashlib
import json
import math
from pathlib import Path
import re

SHARED_EXPERIENCE = "C:/robotics/IsaacLab/apps/isaaclab.python.rendering.kit"
SETTING_PATHS = (
    "/rtx/rendermode", "/rtx/background/source/type", "/rtx/background/source/color",
    "/rtx/domeLight/upperLowerStrategy", "/rtx/sceneDb/ambientLightIntensity",
    "/rtx/directLighting/enabled", "/rtx/directLighting/sampledLighting/enabled",
    "/rtx/indirectDiffuse/enabled", "/rtx/reflections/enabled", "/rtx/post/tonemap/op",
    "/rtx/post/tonemap/filmIso", "/rtx/post/tonemap/cm2Factor",
    "/rtx/post/histogram/enabled", "/app/vulkan",
    "/isaaclab/render/active_viewport", "/isaaclab/render/offscreen",
    "/rtx/useViewLightingMode", "/rtx/scenedb/skipMostLights",
    "/rtx/directLighting/domeLight/enabled",
)


def _value(value):
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _value(item)
        return
    raise ValueError("renderer_invalid_setting_value")


def validate_renderer(record, baseline=None):
    """Validate recorded readback and continuity, without inventing missing values."""
    if not isinstance(record, dict) or set(record) != {"schema", "experience", "settings"}:
        raise ValueError("renderer_record_keys")
    experience = record["experience"]
    if (record["schema"] != "shared-camera-renderer/v1" or not isinstance(experience, dict)
            or set(experience) != {"path", "bytes", "sha256"}
            or experience["path"] != SHARED_EXPERIENCE
            or type(experience["bytes"]) is not int or experience["bytes"] <= 0
            or not isinstance(experience["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", experience["sha256"]) is None):
        raise ValueError("renderer_experience_identity")
    if not isinstance(record["settings"], dict) or set(record["settings"]) != set(SETTING_PATHS):
        raise ValueError("renderer_setting_keys")
    for value in record["settings"].values():
        _value(value)
    if record["settings"]["/isaaclab/render/active_viewport"] is not True:
        raise ValueError("renderer_active_viewport_required")
    serialized = json.dumps(record, sort_keys=True, allow_nan=False)
    if baseline is not None:
        validate_renderer(baseline)
        if serialized != json.dumps(baseline, sort_keys=True, allow_nan=False):
            raise ValueError("renderer_changed")
    return json.loads(serialized)


def snapshot_renderer(session):
    """Read existing settings once each. None means unavailable, not a default."""
    resolved = session.initialization["launch"]["resolved_experience"]
    if not isinstance(resolved, str) or resolved.replace("\\", "/") != SHARED_EXPERIENCE:
        raise ValueError("renderer_resolved_experience_mismatch")
    raw = Path(SHARED_EXPERIENCE).read_bytes()
    settings = {path: session.env.sim.get_setting(path) for path in SETTING_PATHS}
    # Carb array values can be tuples; materialize strict JSON without changing pixels.
    settings = json.loads(json.dumps(settings, allow_nan=False))
    return validate_renderer(dict(schema="shared-camera-renderer/v1",
        experience=dict(path=SHARED_EXPERIENCE, bytes=len(raw),
                        sha256=hashlib.sha256(raw).hexdigest()), settings=settings))
