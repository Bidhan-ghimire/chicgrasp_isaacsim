"""Offline domain arithmetic and durable exploratory preview records. No native imports."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .attachment_capture import TARGETS
from .interface import TCP_X_RANGE_M, TCP_Y_RANGE_M, TCP_Z_RANGE_M, validate_policy_action

ROOT = Path(__file__).resolve().parents[2]
BASIS = ROOT / 'docs/plans/CHICGRASP_GENERALIZATION_DESIGN/DESIGN_BASIS.json'
CAMERAS = ('camera_left', 'camera_right')


def identity(path):
    data = Path(path).read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def rotation(wxyz):
    q = np.asarray(wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-5:
        raise ValueError('invalid_quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])










