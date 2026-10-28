"""Explicit plan/execute commands for the editable research pipeline.

Importing this module starts no processes and loads no checkpoint.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time

from pipeline.settings import ROOT, SCENE_SHA256, add_frozen_source, code_identity, digest, json_digest, load_experiment, load_host, load_positions, read, repo_path, scene_identity, write
from pipeline.scene import scene_setup_identity

def compiled_script(experiment, position):
    import numpy as np
    path = repo_path(experiment['script'])
    spec = importlib.util.spec_from_file_location('_research_demo', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actions = np.asarray(module.build_demo(position, experiment.get('script_options', {})))
    if actions.dtype != np.float32 or actions.shape != (200, 5) or not np.isfinite(actions).all():
        raise ValueError('Script must produce finite float32[200,5]; other durations need a new capture protocol.')
    if not np.isin(actions[:, 3:], [0, 1]).all() or not (actions[:50, 3:] == 1).all():
        raise ValueError('Jaw values must be binary; the first 50 setup commands must be open.')
    # The shared fixed-wrist action boundary is reused without simulator imports.
    add_frozen_source()
    from chicgrasp_policy.interface import validate_policy_action
    validate_policy_action(actions)
    return actions

def _position_selection(experiment, mode, position_id=None, limit=None, selection_seed=None):
    if selection_seed is not None and mode != 'record':
        raise ValueError('--selection-seed applies only to recorder.py.')
    seed = 42 if mode == 'record' and selection_seed is None else selection_seed
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError('--selection-seed must be a nonnegative integer.')
    rows = load_positions(experiment)
    if position_id:
        rows = [row for row in rows if row['id'] == position_id]
        if not rows: raise ValueError('Unknown position: ' + position_id)
    elif mode == 'record':
        rows = [row for row in rows if row['split'] in ('train', 'validation')]
    elif mode == 'live':
        rows = [row for row in rows if row['split'] == 'test']
    else:
        rows = rows[:1]
    if mode == 'record' and any(row['split'] == 'test' for row in rows):
        raise ValueError('Do not record test positions into the training dataset.')
    if not rows: raise ValueError('No selected positions.')
    pool = rows
    algorithm = 'explicit-position' if position_id else 'all-in-source-order'
    seed_applied = False
    if limit is not None:
        if mode == 'record':
            if type(limit) is not int or limit < 1:
                raise ValueError('--limit must be a positive integer.')
            if limit > len(pool):
                raise ValueError(f'--limit {limit} exceeds {len(pool)} available recording positions.')
            if limit < len(pool):
                rows = random.Random(seed).sample(pool, limit)
                algorithm, seed_applied = 'python-random-sample/v1', True
        else:
            if limit < 1: raise ValueError('--limit must be positive.')
            rows = rows[:limit]
            if not position_id: algorithm = 'prefix-in-source-order'
    selection = {'schema':'research-position-selection/v1', 'mode':mode, 'algorithm':algorithm,
                 'selection_seed':seed, 'seed_applied':seed_applied, 'requested_limit':limit,
                 'available_count':len(pool), 'requested_count':len(pool) if limit is None else limit,
                 'selected_count':len(rows), 'selected_ids':[p['id'] for p in rows],
                 'eligible_ids':[p['id'] for p in pool], 'eligible_pool_sha256':json_digest(pool),
                 'source_files':{name:{'sha256':digest(repo_path(name)), 'bytes':repo_path(name).stat().st_size}
                                 for name in experiment['positions']}}
    return rows, selection


def select_positions(experiment, mode, position_id=None, limit=None, selection_seed=None):
    return _position_selection(experiment, mode, position_id, limit, selection_seed)[0]

def make_plan(experiment, mode, *, position_id=None, limit=None, evaluation_seed=None, selection_seed=None):
    if mode == 'train':
        if selection_seed is not None: raise ValueError('--selection-seed applies only to recorder.py.')
        rows, selection = [], None
    else:
        rows, selection = _position_selection(experiment, mode, position_id, limit, selection_seed)
    if mode == 'preview' and (len(rows) != 1 or limit is not None):
        raise ValueError('A scripted preview runs one position; use --position, not --limit.')
    if evaluation_seed is not None:
        if mode != 'live' or any(evaluation_seed not in p['evaluation_seeds'] for p in rows):
            raise ValueError('--seed must be a configured evaluation seed and used with live_run.py.')
        rows = [dict(p, evaluation_seeds=[evaluation_seed]) for p in rows]
    if mode != 'scene':
        for row in rows: compiled_script(experiment, row)
    starts = 0 if mode in ('script', 'train') else sum(len(p['evaluation_seeds']) if mode == 'live' else 1 for p in rows)
    return {'schema': 'research-plan/v1', 'label': 'RESEARCH_NOT_ORIGINAL_18_OF_30',
            'mode': mode, 'experiment_id': experiment['experiment_id'], 'positions': rows,
            'position_selection':selection,
            'planned_kit_starts': starts, 'executed': False, 'model_loaded': False,
            'scene_identity': scene_identity(experiment), 'code_identity': code_identity(),
            'scene_setup_identity': scene_setup_identity(experiment),
            'script': None if mode == 'scene' else experiment['script'],
            'script_options': {} if mode == 'scene' else experiment.get('script_options', {}),
            'action_count': 0 if mode == 'scene' else 200, 'policy_hz': 10,
            'setup_actions': 0 if mode == 'scene' else 50,
            'training': experiment['training'],
            'video': {'enabled':False} if mode == 'preview' else experiment.get('video', {}),
            'limits': experiment['limits']}

def require_approval(experiment):
    receipt = read(repo_path(experiment['scene_approval']))
    if receipt.get('OWNER_APPROVED') is not True or receipt.get('scene_identity') != scene_identity(experiment):
        raise ValueError('Scene approval is missing or stale. Review and approve the current configuration first.')
    return receipt

@contextmanager
def _scene_file_lock(path):
    """Short metadata lock; the OS releases it even if this process exits."""
    with Path(path).open('a+b') as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b'\0'); stream.flush()
        if os.name == 'nt':
            import msvcrt
            def acquire():
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            def release():
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            def acquire(): fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            def release(): fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        deadline = time.monotonic() + 5
        while True:
            try:
                acquire(); break
            except OSError:
                if time.monotonic() >= deadline:
                    raise ValueError('Another scene metadata operation is active; retry this command.')
                time.sleep(.025)
        try:
            yield
        finally:
            release()


def _atomic_scene_json(path, value):
    """Publish a complete JSON file under the caller's metadata lock."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                prefix='.' + path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()


def _scene_series(requested):
    match = re.fullmatch(r'(.*?)([0-9]+)', requested.name)
    if match:
        return match[1], len(match[2]), int(match[2])
    return requested.name + '-', 3, 1


def _latest_scene_review(requested):
    pointer = requested.with_name(requested.name + '.latest.json')
    if not pointer.exists(): return requested
    value = read(pointer)
    if (not isinstance(value, dict) or value.get('schema') != 'research-latest-scene-review/v1'
            or value.get('requested_output') != str(requested)
            or not isinstance(value.get('output_directory'), str)):
        raise ValueError('Invalid latest scene review pointer: ' + str(pointer))
    raw_target = Path(value['output_directory'])
    target = raw_target.resolve()
    prefix, width, first = _scene_series(requested)
    suffix = target.name[len(prefix):] if target.name.startswith(prefix) else ''
    in_series = (suffix.isascii() and suffix.isdigit() and int(suffix) > first
                 and target.name == prefix + str(int(suffix)).zfill(width))
    if (not raw_target.is_absolute() or target.parent != requested.parent
            or not target.is_dir() or (target != requested and not in_series)):
        raise ValueError('Latest scene review target is not an existing sibling in this review series.')
    return target


def _allocate_scene_output(requested):
    requested = Path(requested).resolve()
    requested.parent.mkdir(parents=True, exist_ok=True)
    pointer = requested.with_name(requested.name + '.latest.json')
    with _scene_file_lock(pointer.with_suffix('.lock')):
        _latest_scene_review(requested)  # Do not overwrite an unrelated/malformed sidecar.
        prefix, width, number = _scene_series(requested)
        output = requested
        while True:
            try:
                output.mkdir(exist_ok=False)
                break
            except FileExistsError:
                number += 1
                output = requested.with_name(prefix + str(number).zfill(width))
        _atomic_scene_json(pointer, {'schema':'research-latest-scene-review/v1',
            'requested_output':str(requested), 'output_directory':str(output),
            'allocated_at_utc':datetime.now(timezone.utc).isoformat()})
    return output


def _allocate_preview_output(requested):
    """Reserve a new preview folder without changing inspection/approval metadata."""
    requested = Path(requested).resolve()
    requested.parent.mkdir(parents=True, exist_ok=True)
    prefix, width, number = _scene_series(requested)
    output = requested
    while True:
        try:
            output.mkdir(exist_ok=False)
            return output
        except FileExistsError:
            number += 1
            output = requested.with_name(prefix + str(number).zfill(width))


def _archive_scene_approval(review_dir, raw):
    number = 1
    while True:
        path = review_dir / f'previous-approval{number:03d}.json'
        try:
            with path.open('xb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            return path
        except FileExistsError:
            if path.is_file() and path.read_bytes() == raw: return path
            number += 1


def approve_review(experiment, review_dir, output=None):
    requested = Path(review_dir).resolve()
    target = repo_path(output or experiment['scene_approval'])
    with _scene_file_lock(requested.with_name(requested.name + '.latest.lock')):
        actual = _latest_scene_review(requested)
        summary_path = actual / 'summary.json'
        try:
            review = read(summary_path)
        except FileNotFoundError as exc:
            raise ValueError(
                f'No saved completion summary for the latest inspection: {summary_path}. '
                'It may still be running or may have been interrupted. '
                'If it is still open, use Finish inspection and wait for COMPLETED before approving. '
                'If the interrupted session is already closed, rerun the same load_scene.py --execute '
                'command, finish it cleanly, then approve again.') from exc
        if review.get('mode') != 'scene' or review.get('completion') != 'COMPLETED':
            raise ValueError('A completed scene review receipt is required; an exit code alone is insufficient.')
        identity = scene_identity(experiment)
        if review.get('scene_identity') != identity:
            raise ValueError('Configuration changed since that review.')
        summary_hash = digest(summary_path)
        value = {'schema': 'research-scene-approval/v1', 'OWNER_APPROVED': True,
                 'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'scene_identity': identity,
                 'review_summary_sha256': summary_hash, 'review_directory':str(actual)}
        target.parent.mkdir(parents=True, exist_ok=True)
        with _scene_file_lock(target.with_name(target.name + '.lock')):
            archive = None
            if target.exists():
                raw = target.read_bytes()
                old = json.loads(raw.decode('utf-8-sig'))
                if not isinstance(old, dict) or old.get('schema') != value['schema'] or old.get('OWNER_APPROVED') is not True:
                    raise ValueError('Refuse to replace an unrelated file at the scene approval path.')
                if old.get('scene_identity') == identity and old.get('review_summary_sha256') == summary_hash:
                    return {'approval':str(target), 'sha256':digest(target), 'review_directory':str(actual), 'unchanged':True}
                archive = _archive_scene_approval(actual, raw)
            _atomic_scene_json(target, value)
        return {'approval':str(target), 'sha256':digest(target), 'review_directory':str(actual),
                'unchanged':False, 'previous_approval':str(archive) if archive else None}

def checkpoint_record(experiment, artifact_path=None):
    artifact_path = repo_path(artifact_path or experiment['model_artifact'])
    value = read(artifact_path)
    if value.get('schema') != 'research-model-artifact/v1' or value.get('training_window_start') != 50:
        raise ValueError('Live backend requires a research artifact with setup/window start 50.')
    if SCENE_SHA256 not in value.get('scene_sha256s', []):
        raise ValueError('Model scene does not match this scene.')
    from pipeline.models import validate_artifact_scene_identity
    if validate_artifact_scene_identity(value) != scene_setup_identity(experiment):
        raise ValueError('Model editable scene identity differs from the approved experiment.')
    if digest(repo_path(experiment['script'])) not in value.get('script_sha256s', []):
        raise ValueError('The current demonstration differs from the model dataset provenance.')
    if value.get('script_options', {}) != experiment.get('script_options', {}):
        raise ValueError('Script options differ from the model dataset provenance.')
    checkpoint = (artifact_path.parent / value['checkpoint_path']).resolve()
    normalizer = (artifact_path.parent / value['normalizer_path']).resolve()
    for key, path in (('checkpoint', checkpoint), ('normalizer', normalizer)):
        if not path.is_relative_to(artifact_path.parent) or digest(path) != value[key]['sha256'] or path.stat().st_size != value[key]['bytes']:
            raise ValueError('Model artifact identity mismatch: ' + key)
    return {'path': str(checkpoint), 'sha256': value['checkpoint']['sha256'],
            'artifact_path': str(artifact_path), 'artifact_sha256': digest(artifact_path),
            'model_type': value['model_type']}

def preflight(host, *, video=False):
    from pipeline.runtime.host import preflight as original_preflight
    result = original_preflight(host, include_checkpoint=False)
    if result['status'] != 'PASS':
        raise ValueError('Host/input preflight failed: ' + '; '.join(result['errors']))
    if video and not Path(host.get('ffmpeg', '')).is_file():
        raise ValueError('Set the FFmpeg executable in the local host file before video execution.')
    if video:
        ffmpeg = Path(host['ffmpeg'])
        if not ffmpeg.with_name('ffprobe' + ffmpeg.suffix).is_file():
            raise ValueError('Put ffprobe beside FFmpeg before video execution.')
    return result

def run_native(experiment, host, mode, position, output, *, evaluation_seed=None,
               checkpoint=None, runner=None):
    from pipeline.runtime.host import environment
    if runner is None:
        from pipeline.runtime.process import run_owned
        runner = run_owned
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    request = {'schema': 'research-native-request/v1', 'mode': mode,
               'environment_seed': 42, 'scene_profile': 'attempt005', 'scene_sha256': SCENE_SHA256,
               'scene_setup_identity': scene_setup_identity(experiment),
               'position': position, 'evaluation_seed': evaluation_seed,
               'checkpoint': checkpoint,
               'video': {**experiment.get('video', {}), 'enabled': mode == 'live' and experiment.get('video', {}).get('enabled', False),
                         'ffmpeg_path': host.get('ffmpeg')},
               'deadline_unix': time.time() + experiment['limits']['per_simulator_seconds']}
    if mode != 'scene':
        import numpy as np
        commands = output / 'commands.npy'
        np.save(commands, compiled_script(experiment, position), allow_pickle=False)
        request.update(script={'path':str(commands.resolve()), 'sha256':digest(commands), 'bytes':commands.stat().st_size},
                       script_source_sha256=digest(repo_path(experiment['script'])),
                       script_options=experiment.get('script_options', {}))
    write(output / 'request.json', request)
    write(output / 'provenance.json', {'scene_identity': scene_identity(experiment), 'code_identity': code_identity()})
    command = [str(Path(host['isaac_sim']) / 'kit/python/python.exe'), '-B', '-s', '-u',
               '-m', 'pipeline.runtime.native', '--config', host['config_path'],
               '--request', str((output / 'request.json').resolve()), '--attempt', str((output / 'native').resolve())]
    receipt = runner(command, cwd=ROOT, env=environment(host), directory=output / 'process',
                     timeout_seconds=experiment['limits']['per_simulator_seconds'], cleanup_seconds=experiment['limits']['cleanup_seconds'])
    result_path = output / 'native/research_result.json'
    result = read(result_path) if result_path.is_file() else {'completion': 'FAILED', 'failure': 'Missing research_result.json'}
    scene_matches = result.get('scene_setup_identity') == request['scene_setup_identity']
    return {'mode': mode, 'position_id': position['id'], 'evaluation_seed': evaluation_seed,
            'completion': 'COMPLETED' if receipt.get('success') and result.get('completion') == 'COMPLETED' and scene_matches else 'FAILED',
            'parent_errors': [] if scene_matches else ['Missing or mismatched native scene_setup_identity'],
            'process_receipt': str((output / 'process/receipt.json').resolve()), 'native': result,
            'result_path': str(result_path.resolve())}

def execute_simulation(experiment, mode, output, *, position_id=None, limit=None, evaluation_seed=None,
                       selection_seed=None, artifact_path=None, runner=None):
    plan = make_plan(experiment, mode, position_id=position_id, limit=limit, evaluation_seed=evaluation_seed,
                     selection_seed=selection_seed)
    host = load_host(repo_path(experiment['host']))
    checked = preflight(host, video=mode == 'live' and experiment.get('video', {}).get('enabled', False))
    if mode != 'scene': require_approval(experiment)
    if scene_setup_identity(experiment) != plan['scene_setup_identity']:
        raise ValueError('Scene identity changed during planning/approval.')
    checkpoint = checkpoint_record(experiment, artifact_path) if mode == 'live' else None
    requested_output = Path(output).resolve()
    if mode == 'scene':
        output = _allocate_scene_output(requested_output)
    elif mode == 'preview':
        output = _allocate_preview_output(requested_output)
    else:
        output = requested_output
        output.mkdir(parents=True, exist_ok=False)
    write(output / 'plan.json', plan)
    write(output / 'preflight.json', checked)
    runs, episodes = [], []
    failed = False
    for position in plan['positions']:
        for seed in position['evaluation_seeds'] if mode == 'live' else [None]:
            name = position['id'] + (f'_seed{seed}' if seed is not None else '')
            record = run_native(experiment, host, 'scripted' if mode == 'record' else mode,
                                position, output / name, evaluation_seed=seed, checkpoint=checkpoint, runner=runner)
            native = record['native']
            if native.get('scene_setup_identity') != plan['scene_setup_identity']:
                record['completion'] = 'FAILED'
                record.setdefault('parent_errors', []).append('Native scene does not match the planned scene')
            if mode == 'record' and record['completion'] == 'COMPLETED' and native.get('training_eligible') is True:
                try:
                    from pipeline.training import verify_episode_binding
                    episode_path = Path(native['episode']['path'])
                    if not episode_path.is_absolute(): episode_path = output / name / 'native' / episode_path
                    if not episode_path.resolve().is_relative_to(output):
                        raise ValueError('Native episode path escaped the new run directory.')
                    binding_record = dict(native['episode'].get('binding') or {})
                    if binding_record.get('path') and not Path(binding_record['path']).is_absolute():
                        binding_record['path'] = str(output / name / 'native' / binding_record['path'])
                    binding = verify_episode_binding(episode_path, binding_record, plan['scene_setup_identity'], base_dir=output)
                    episodes.append({'id': name, 'path': episode_path.resolve().relative_to(output).as_posix(),
                                     'split': position['split'], 'position_id': position['id'],
                                     'script_sha256': digest(repo_path(experiment['script'])),
                                     'scene_sha256': SCENE_SHA256, 'complete': True,
                                     'scene_setup_identity': plan['scene_setup_identity'],
                                     'episode_binding': {'path':Path(binding['path']).relative_to(output).as_posix(),
                                                         'bytes':binding['bytes'], 'sha256':binding['sha256']},
                                     'commands_sha256': digest(output / name / 'commands.npy'),
                                     'script_options': experiment.get('script_options', {})})
                except (ValueError, OSError, KeyError, TypeError) as error:
                    record['completion'] = 'FAILED'
                    record.setdefault('parent_errors', []).append('Episode provenance: ' + str(error))
            runs.append(record)
            write(output / f'{name}.json', record)
            if record['completion'] != 'COMPLETED':
                failed = True
                break
        if failed: break
    if mode == 'record':
        write(output / 'dataset.json', {'schema': 'research-dataset/v1', 'label': 'NEW_RESEARCH_DATASET',
                                      'experiment_id': experiment['experiment_id'], 'episodes': episodes,
                                      'source_plan_sha256': digest(output / 'plan.json'),
                                      'position_selection':plan['position_selection'],
                                      'run_results': [Path(r['result_path']).relative_to(output).as_posix() for r in runs]})
    summary = {'schema': 'research-summary/v1', 'mode': mode, 'completion': 'FAILED' if failed else 'COMPLETED',
               'requested_output':str(requested_output), 'output_directory':str(output),
               'scene_identity': plan['scene_identity'], 'runs_completed': len(runs),
               'scene_setup_identity': plan['scene_setup_identity'],
               'position_selection':plan['position_selection'],
               'training_episodes': len(episodes), 'planned_kit_starts': plan['planned_kit_starts'],
               'results': [r['result_path'] for r in runs], 'original_18_of_30_reproduced': False}
    write(output / 'summary.json', summary)
    return summary

def prepare_training(experiment, *, model_type=None, updates=None, resume=None):
    """Resolve count and resume metadata without starting or loading a model."""
    training = dict(experiment['training'])
    receipt = None
    if resume is not None:
        from pipeline.training import inspect_resume
        path = resume['checkpoint_path'] if isinstance(resume, dict) else resume
        receipt = inspect_resume(repo_path(path))
        if isinstance(resume, dict) and receipt != resume:
            raise ValueError('Resume inputs changed after planning.')
        saved = receipt['plan']
        if updates is not None and updates != saved['updates']:
            raise ValueError('Resume keeps the original total update target; omit --updates or use that same total.')
        if model_type is not None and model_type != saved['model_type']:
            raise ValueError('Resume must use the checkpoint model type.')
        if saved['scene_setup_identity'] != scene_setup_identity(experiment):
            raise ValueError('Resume scene differs from the current experiment.')
        training = dict(saved['config'], model_type=saved['model_type'],
                        updates=saved['updates'], seed=saved['seed'], device=saved['device'])
    else:
        if model_type: training['model_type'] = model_type
        if updates is not None: training['updates'] = updates
    if type(training.get('updates')) is not int or training['updates'] < 1:
        raise ValueError('Updates must be a positive integer.')
    return training, receipt


def execute_training(experiment, output, *, dataset=None, model_type=None, updates=None, resume=None, runner=None):
    from pipeline.runtime.process import run_training_owned
    from pipeline.training import consistent_scene_identity, read_manifest
    training, resume_receipt = prepare_training(experiment, model_type=model_type, updates=updates, resume=resume)
    host = load_host(repo_path(experiment['host']))
    python = Path(host.get('training_python', ''))
    if not python.is_file(): raise ValueError('Set training_python to the prepared training environment executable.')
    require_approval(experiment)
    dataset_path = repo_path(dataset or experiment['dataset'])
    _, records = read_manifest(dataset_path)
    scene = scene_setup_identity(experiment)
    if consistent_scene_identity(records) != scene:
        raise ValueError('Dataset scene does not match the approved experiment scene.')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    request = {'dataset': str(dataset_path), 'output': str(output / 'model'), 'training': training,
               'scene_setup_identity': scene, 'resume': resume_receipt}
    write(output / 'training_request.json', request)
    restart = bool(resume_receipt and resume_receipt.get('restart_from_zero'))
    if restart:
        print('Note: ' + resume_receipt['notice'], flush=True)
    if python.resolve() == (Path(host['isaac_sim']) / 'kit/python/python.exe').resolve():
        from pipeline.runtime.host import environment
        env = environment(host)
    else:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        env.pop('PYTHONHOME', None)
        # A separate training environment supplies its own Torch stack.
        env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT / 'frozen_runtime/src')])
    receipt = (runner or run_training_owned)([str(python), '-B', '-m', 'pipeline.training', str(output / 'training_request.json')],
                                   cwd=ROOT, env=env, directory=output / 'process',
                                   timeout_seconds=None, cleanup_seconds=5)
    artifact = output / 'model/artifact.json'
    errors = []
    completed_updates = None
    try:
        from pipeline.models import validate_artifact_scene_identity
        artifact_value = read(artifact)
        if validate_artifact_scene_identity(artifact_value) != scene:
            raise ValueError('Training artifact scene differs from the approved experiment.')
        completed_updates = artifact_value.get('completed_updates')
        if type(completed_updates) is not int or completed_updates != training['updates']:
            raise ValueError('Training artifact did not complete the requested total update count.')
    except (ValueError, OSError, KeyError, TypeError) as error:
        errors.append(str(error))
    result = {'completion': 'COMPLETED' if receipt.get('success') and not errors else 'FAILED',
              'artifact': str(artifact) if artifact.is_file() else None, 'process_receipt': str(output / 'process/receipt.json'),
              'scene_setup_identity':scene, 'errors':errors, 'requested_updates':training['updates'],
              'completed_updates':completed_updates, 'training_time_limit_seconds':None,
              'resume_checkpoint':resume_receipt['checkpoint_path'] if resume_receipt and not restart else None,
              'restarted_from_zero':restart, 'restart_notice':resume_receipt['notice'] if restart else None}
    write(output / 'summary.json', result)
    return result

def main(mode):
    parser = argparse.ArgumentParser(description='Editable chicgrasp_isaacsim research workflow. Default: plan only; no Kit/model starts.')
    parser.add_argument('--config', default='configs/research.json')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--preview', action='store_true', help='scripted_run.py only: watch one full demonstration in Isaac Sim; no training dataset or video.')
    parser.add_argument('--output')
    parser.add_argument('--position')
    parser.add_argument('--limit', type=int, help='Recorder: unique random subset; omitted means all 50 supplied positions.')
    parser.add_argument('--selection-seed', type=int, help='Recorder position-selection seed (default: 42), independent of policy/environment seeds.')
    parser.add_argument('--artifact')
    parser.add_argument('--dataset')
    parser.add_argument('--model', help='Model adapter registered in pipeline/models.py')
    parser.add_argument('--updates', type=int, help='Total optimizer updates; overrides the configured default.')
    parser.add_argument('--resume', metavar='LAST.pt', help='Training only: restore saved state and continue its original total target into a new output directory.')
    parser.add_argument('--seed', type=int, help='Select one configured live inference seed')
    parser.add_argument('--approve', metavar='SCENE_REVIEW_DIRECTORY')
    parser.add_argument('--approval-file')
    args = parser.parse_args()
    try:
        experiment = load_experiment(args.config)
        if args.preview and mode != 'script':
            raise ValueError('--preview applies only to scripted_run.py.')
        preview = mode == 'script' and (args.preview or args.execute)
        if preview and any(value is not None for value in (
                args.limit, args.selection_seed, args.seed, args.artifact, args.dataset,
                args.resume, args.model, args.updates, args.approve, args.approval_file)):
            raise ValueError('Scripted preview accepts --config, --position and optional --output; training, collection, approval and policy flags do not apply.')
        if preview:
            args.execute = True
        if args.resume is not None and mode != 'train':
            raise ValueError('--resume applies only to train.py.')
        if args.selection_seed is not None and mode != 'record':
            raise ValueError('--selection-seed applies only to recorder.py.')
        if args.model or args.updates is not None:
            if mode != 'train': raise ValueError('--model and --updates apply to train.py only.')
            experiment['training'] = dict(experiment['training'])
            if args.model: experiment['training']['model_type'] = args.model
            if args.updates is not None: experiment['training']['updates'] = args.updates
        if args.approve:
            if mode != 'scene' or args.execute: raise ValueError('Approval is a separate load_scene.py operation.')
            result = approve_review(experiment, args.approve, args.approval_file)
        elif not args.execute:
            resume_receipt = None
            if mode == 'train':
                training, resume_receipt = prepare_training(experiment, model_type=args.model, updates=args.updates, resume=args.resume)
                experiment = dict(experiment, training=training)
            result = make_plan(experiment, mode, position_id=args.position, limit=args.limit,
                               evaluation_seed=args.seed, selection_seed=args.selection_seed)
            if mode == 'train': result['training_time_limit_seconds'] = None
            if resume_receipt:
                result['resume'] = {'checkpoint_path':resume_receipt['checkpoint_path'],
                                    'checkpoint':resume_receipt['checkpoint'],
                                    'original_total_updates':training['updates'], 'state_loaded':False}
                if resume_receipt.get('restart_from_zero'):
                    result['resume'].update(restart_from_zero=True, logged_updates=resume_receipt['logged_updates'],
                                            notice=resume_receipt['notice'])
            if args.output: write(Path(args.output), result)
        else:
            if preview and not args.output:
                args.output = str(repo_path(experiment['scene_approval']).parent / 'preview001')
            if not args.output: raise ValueError('--execute requires a new --output directory.')
            if mode == 'train':
                result = execute_training(experiment, args.output, dataset=args.dataset, model_type=args.model, updates=args.updates, resume=args.resume)
            else:
                result = execute_simulation(experiment, 'preview' if preview else mode, args.output, position_id=args.position,
                                            limit=args.limit, evaluation_seed=args.seed,
                                            selection_seed=args.selection_seed, artifact_path=args.artifact)
        print(json.dumps(result, indent=2))
        if result.get('completion') == 'FAILED': raise SystemExit(1)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, str(exc) + '\n')
