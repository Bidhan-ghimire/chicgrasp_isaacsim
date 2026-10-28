"""Small JSON configuration and position-list validation. No simulator imports."""
from pathlib import Path
import hashlib
import json
import math
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'frozen_runtime'
SCENE_SHA256 = '4f40d26e122c99ea5cb980edea4a391b5335a99fe060b84cacd40305374c0d5f'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def repo_path(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()

def add_frozen_source():
    for path in (FROZEN / 'src', FROZEN / 'source' / 'chicgrasp_isaac'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

def load_host(path):
    path = Path(path).resolve()
    value = read(path)
    if value.get('schema') not in ('hc-reproduction-host/v1', 'research-host/v1'):
        raise ValueError('Host file must use research-host/v1 or hc-reproduction-host/v1.')
    for key in ('isaac_sim', 'isaac_lab', 'policy_support'):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError('Missing host path: ' + key)
        p = Path(value[key])
        value[key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    for key in ('training_python', 'ffmpeg', 'checkpoint'):
        if value.get(key):
            p = Path(value[key])
            value[key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    # Compatibility with read-only portable host validation; never opened here.
    value.setdefault('checkpoint', str(ROOT / '.local' / 'unused-original-BEST.pt'))
    value['config_path'] = str(path)
    return value

def load_experiment(path=None):
    path = repo_path(path or 'configs/research.json')
    value = read(path)
    if value.get('schema') != 'research-experiment/v1':
        raise ValueError('Expected research-experiment/v1.')
    if value.get('scene_profile') != 'attempt005' or value.get('environment_seed') != 42:
        raise ValueError('Research scenes extend the attempt005 base with environment seed 42.')
    if value.get('scene_sha256') != SCENE_SHA256:
        raise ValueError('Frozen scene identity changed; implement a separate scene backend first.')
    limits = dict(value.get('limits', {}))
    if (type(limits.get('per_simulator_seconds')) is not int or not 1 <= limits['per_simulator_seconds'] <= 1200
            or limits.get('cleanup_seconds') != 5):
        raise ValueError('Use at most 1200 seconds per simulator and 5 seconds cleanup.')
    # Training is update-count limited, never elapsed-time limited. Normalize
    # older generated configs in memory without rewriting the user's files.
    value['legacy_training_time_limit_ignored'] = limits.get('training_seconds') is not None
    limits['training_seconds'] = None
    value['limits'] = limits
    value['config_path'] = str(path)
    return value

def load_positions(experiment):
    from pipeline.scene import validate_scene_position
    rows, identifiers, locations = [], set(), []
    for filename in experiment['positions']:
        manifest = read(repo_path(filename))
        if manifest.get('schema') != 'research-positions/v1' or manifest.get('frame') != 'WORLD':
            raise ValueError('Position manifests require research-positions/v1, WORLD metres.')
        for source in manifest['positions']:
            row = dict(source)
            ident = row.get('id')
            if not isinstance(ident, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', ident) or ident in identifiers:
                raise ValueError('Duplicate or invalid position ID: ' + str(ident))
            identifiers.add(ident)
            xyz = row.get('authored_xyz_m', [])
            if len(xyz) != 3 or not all(type(x) in (int, float) and math.isfinite(x) for x in xyz):
                raise ValueError('Position must contain three finite world metres: ' + ident)
            row.setdefault('authored_quaternion_wxyz', [1, 0, 0, 0])
            validate_scene_position(experiment, row)
            if row.get('split') not in ('train', 'validation', 'test'):
                raise ValueError('Position split must be train, validation or test.')
            seeds = row.get('evaluation_seeds', [15345])
            if not seeds or any(type(x) is not int or x < 0 for x in seeds) or len(set(seeds)) != len(seeds):
                raise ValueError('Evaluation seeds must be unique nonnegative integers.')
            row['evaluation_seeds'] = seeds
            for previous in locations:
                distance = math.dist(xyz, previous['authored_xyz_m'])
                same_orientation = abs(sum(a*b for a, b in zip(row['authored_quaternion_wxyz'], previous['authored_quaternion_wxyz']))) >= 1 - 1e-9
                if (distance == 0 and same_orientation) or (row['split'] != previous['split'] and distance < 0.01):
                    raise ValueError('Duplicate position or cross-split distance below 10 mm: ' + ident + '/' + previous['id'])
            locations.append(row)
            rows.append(row)
    return rows

def scene_identity(experiment):
    """Editing a scene/camera/position/script invalidates prior review approval."""
    from pipeline.scene import scene_setup_identity
    paths = [experiment['script'], *experiment['positions'], 'camera_presets/HAND_DEMO_ONLY.json',
             'camera_presets/HD_TWINS_DEMO_ONLY.json', 'camera_presets/apply_cameras.py']
    return {'scene_profile': experiment['scene_profile'], 'scene_sha256': experiment['scene_sha256'],
            'environment_seed': experiment['environment_seed'], 'script_options': experiment.get('script_options', {}),
            'scene_setup_identity': scene_setup_identity(experiment),
            'inputs': {p: (None if p == experiment['script'] and not repo_path(p).is_file()
                           else digest(repo_path(p))) for p in paths}}

def code_identity():
    paths = list((ROOT / 'pipeline').rglob('*.py'))
    paths += [ROOT / x for x in ('load_scene.py', 'scene_setup.py', 'scripted_run.py', 'recorder.py', 'train.py', 'live_run.py')]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths) if p.is_file()}
