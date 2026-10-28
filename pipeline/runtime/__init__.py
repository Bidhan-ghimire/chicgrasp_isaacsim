"""Use the configured SDK Python for workflow commands; preserve launch intent."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from pipeline.settings import ROOT, load_host, read, repo_path

ENTRY = {'scene':'load_scene.py', 'script':'scripted_run.py', 'record':'recorder.py',
         'train':'train.py', 'live':'live_run.py'}
MARKER = 'CHICGRASP_SDK_HANDOFF'

def handoff(mode, argv, *, runner=subprocess.run, executable=None, env=None):
    if mode not in ENTRY: raise ValueError('Unknown workflow entry point.')
    if '-h' in argv or '--help' in argv: return None
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', default='configs/research.json')
    args, _ = parser.parse_known_args(argv)
    experiment = read(repo_path(args.config))
    host_path = repo_path(experiment['host'])
    if not host_path.is_file():
        raise ValueError('Local setup is missing. Run python setup_project.py from the repository root first.')
    host = load_host(host_path)
    python = Path(host['isaac_sim']) / 'kit/python/python.exe'
    if not python.is_file(): raise ValueError('Configured SDK Python is missing. Run python setup_project.py --check.')
    current = Path(executable or sys.executable).resolve()
    source_env = os.environ if env is None else env
    marker = str(host_path.resolve())
    if source_env.get(MARKER) == marker:
        if current != python.resolve(): raise ValueError('SDK handoff interpreter mismatch; refusing a relaunch loop.')
        return None
    from pipeline.runtime.host import environment
    child_env = environment(host)
    child_env[MARKER] = marker
    result = runner([str(python), '-B', '-s', str(ROOT / ENTRY[mode]), *argv],
                    cwd=ROOT, env=child_env)
    return result.returncode

def main(mode):
    try:
        code = handoff(mode, sys.argv[1:])
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
    if code is not None: raise SystemExit(code)
    from pipeline.runtime.workflow import main as workflow_main
    workflow_main(mode)
