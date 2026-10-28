"""Bounded Windows standalone-SDK discovery. Read-only, standard library only.

No Python candidate, package installer, simulator, encoder or GPU is executed.
Git is queried with optional locks disabled to authenticate a Lab checkout.
"""
from importlib import metadata
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LAB_COMMIT = "37ddf626871758333d6ed89cf64ad702aef127d0"
REQUIRED = ("isaac_sim", "isaac_lab", "policy_support")
KEYS = (*REQUIRED, "ffmpeg", "training_python")
MAX_ROOTS, MAX_CHILDREN, MAX_PATH_ENTRIES = 32, 64, 96
SIM_LAYOUT = ("kit/python/python.exe", "site/sitecustomize.py", "python.bat", "kit/kernel/plugins")
LAB_LAYOUT = ("source/isaaclab/isaaclab/app", "source/isaaclab_tasks", "source/isaaclab_assets")
ENV_KEYS = {
    "isaac_sim": ("ISAAC_SIM_PATH", "ISAACSIM_PATH", "ISAAC_PATH"),
    "isaac_lab": ("ISAACLAB_PATH", "ISAAC_LAB_PATH"),
    "policy_support": ("CHICGRASP_POLICY_SUPPORT",),
    "ffmpeg": ("CHICGRASP_FFMPEG", "FFMPEG_PATH", "FFMPEG_BINARY"),
    "training_python": ("CHICGRASP_TRAINING_PYTHON",),
}


def _canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _sha256(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("file changed during hashing: " + str(path))
    return digest.hexdigest()


def _resource_rows():
    value = json.loads((ROOT / "pipeline/runtime/data/SDK_RESOURCE_IDENTITIES.json").read_text(encoding="utf-8"))
    if value.get("schema") != "hc-sdk-resources/v1":
        raise ValueError("Unsupported pinned SDK resource manifest")
    return value["resources"]


def _verify_resource_subset(path, key):
    errors, checked = [], []
    rows = [row for row in _resource_rows() if row["config_key"] == key]
    if not rows:
        return ["No pinned SDK resources for " + key], []
    for row in rows:
        target = path / row["relative_path"]
        try:
            if target.stat().st_size != row["bytes"] or _sha256(target) != row["sha256"]:
                errors.append("Pinned SDK bytes differ: " + row["name"])
            else:
                checked.append(row["name"])
        except (OSError, ValueError) as exc:
            errors.append("Pinned SDK resource unavailable: " + row["name"] + ": " + str(exc))
    return errors, checked


def _git_identity(path):
    """Read-only Git queries; no index refresh write, fsmonitor hook or network."""
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    base = ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(path)]
    result = {}
    for key, args in (("head", ["rev-parse", "HEAD"]),
                      ("tracked_changes", ["status", "--porcelain=v1", "--untracked-files=no"])):
        try:
            process = subprocess.run(base + args, capture_output=True, text=True, timeout=10,
                env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"error": str(exc)}
        if process.returncode != 0:
            return {"error": process.stderr.strip() or "Git query failed"}
        result[key] = process.stdout.strip()
    return result


def load_support_pins(requirements_path=None):
    path = Path(requirements_path or ROOT / "environments/policy-support-win311.txt")
    pins = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.partition("#")[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if not match:
            raise ValueError("Support requirements must contain only exact name==version pins")
        name = _canonical(match[1])
        if name in pins:
            raise ValueError("Duplicate support pin: " + name)
        pins[name] = match[2]
    if len(pins) != 36:
        raise ValueError("Expected all 36 pinned support packages")
    return pins


def validate_support(path, *, requirements_path=None, distributions_reader=None):
    """Check every pinned distribution version from metadata without importing it."""
    directory = Path(path).expanduser().resolve()
    errors, warnings, found = [], [], {}
    if not directory.is_dir():
        return dict(status="FAIL", errors=["Support directory is missing: " + str(directory)], warnings=[], checks={})
    try:
        pins = load_support_pins(requirements_path)
        reader = distributions_reader or metadata.distributions
        for distribution in reader(path=[str(directory)]):
            name = distribution.metadata.get("Name")
            if not isinstance(name, str) or not name:
                errors.append("Distribution metadata has no Name")
                continue
            found.setdefault(_canonical(name), []).append(distribution.version)
    except Exception as exc:
        return dict(status="FAIL", errors=["Support metadata could not be read: " + str(exc)], warnings=[], checks={})
    for name in ("torch", "torchvision", "numpy"):
        if name in found or (directory/name).exists() or (directory/(name+".py")).exists():
            errors.append("Support must not override SDK core package: " + name)
    for name, versions in found.items():
        if len(versions) != 1:
            errors.append("Duplicate distribution metadata: " + name)
    for name, wanted in pins.items():
        versions = found.get(name, [])
        if versions != [wanted]:
            errors.append(f"Support {name}: expected {wanted}, found {versions or 'missing'}")
    extras = sorted(set(found)-set(pins)-{"torch", "torchvision", "numpy"})
    if extras:
        warnings.append("Additional unpinned support metadata: " + ", ".join(extras))
    return dict(status="FAIL" if errors else "PASS", errors=errors, warnings=warnings,
        checks=dict(path=str(directory), required_pin_count=len(pins),
            matched_pin_count=sum(found.get(name)==[version] for name, version in pins.items()),
            installed_versions=found, imported_packages=False))


def _validate_location(key, value, *, git_reader=None, distributions_reader=None, requirements_path=None):
    path = Path(value).resolve()
    errors, warnings, checks = [], [], {"path": str(path)}
    if key == "policy_support":
        return validate_support(path, requirements_path=requirements_path, distributions_reader=distributions_reader)
    if key in ("isaac_sim", "isaac_lab"):
        layout = SIM_LAYOUT if key == "isaac_sim" else LAB_LAYOUT
        missing = [name for name in layout
                   if not ((path/name).is_file() if Path(name).suffix else (path/name).is_dir())]
        if missing:
            errors.append("Missing " + key + " layout: " + ", ".join(missing))
        if not errors:
            resource_errors, checked = _verify_resource_subset(path, key)
            errors.extend(resource_errors)
            checks["authenticated_resources"] = checked
        if key == "isaac_lab" and not errors:
            identity = (git_reader or _git_identity)(path)
            checks["git"] = identity
            if identity.get("error"):
                errors.append("Isaac Lab source cannot be authenticated: " + identity["error"])
            else:
                if identity.get("head") != LAB_COMMIT:
                    errors.append("Isaac Lab HEAD must be " + LAB_COMMIT)
                if identity.get("tracked_changes") not in ("", []):
                    errors.append("Isaac Lab has modified tracked files")
            checks["archive_source_supported"] = False
    elif key == "ffmpeg":
        if not path.is_file() or path.name.casefold() != "ffmpeg.exe":
            errors.append("Expected a Windows ffmpeg.exe file")
        probe = path.with_name("ffprobe.exe")
        if not probe.is_file():
            errors.append("Matching sibling ffprobe.exe is missing")
        checks.update(ffmpeg_path=str(path), ffprobe_path=str(probe), binaries_executed=False)
    elif key == "training_python":
        if not path.is_file():
            errors.append("Configured training Python is missing")
        checks["interpreter_executed"] = False
        warnings.append("Training interpreter version and imports were not executed or verified")
    return dict(status="FAIL" if errors else "PASS", errors=errors, warnings=warnings, checks=checks)


def validate_host(config, *, resource_checker=None, git_reader=None, distributions_reader=None, requirements_path=None):
    """Read-only validation. FFmpeg is optional when not explicitly configured."""
    errors, warnings, checks = [], [], {}
    for key in KEYS:
        value = config.get(key)
        if not value:
            if key in REQUIRED:
                errors.append("Missing host setting: " + key)
            continue
        try:
            check = _validate_location(key, value, git_reader=git_reader,
                distributions_reader=distributions_reader, requirements_path=requirements_path)
        except Exception as exc:
            check = dict(status="FAIL", errors=[str(exc)], warnings=[], checks={})
        checks[key] = check
        errors.extend(key + ": " + error for error in check["errors"])
        warnings.extend(check["warnings"])
    if all(checks.get(key, {}).get("status") == "PASS" for key in ("isaac_sim", "isaac_lab")):
        try:
            if resource_checker is None:
                from pipeline.runtime.host import verify_sdk_resources
                resource_checker = verify_sdk_resources
            resource_checker(config)
            checks["sdk_resource_pin"] = dict(status="PASS", method="pipeline.runtime.host.verify_sdk_resources")
        except Exception as exc:
            errors.append("SDK resource identity: " + str(exc))
            checks["sdk_resource_pin"] = dict(status="FAIL", reason=str(exc))
    video_status = "READY" if checks.get("ffmpeg", {}).get("status") == "PASS" else "video_unconfigured"
    if not config.get("ffmpeg"):
        warnings.append("video_unconfigured: scene review and recording do not require FFmpeg")
    return dict(status="FAIL" if errors else "PASS", errors=errors, warnings=warnings,
        checks=checks, video_status=video_status, simulator_started=False, packages_imported=False,
        supported_platform="Windows standalone Isaac Sim 5.1 + Isaac Lab v2.3.2")


def _expand_path(value, env, base):
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError("Expected a nonempty filesystem path")
    raw = str(value).strip().strip('"')
    raw = re.sub(r"%([^%]+)%", lambda m: env.get(m[1].upper(), m[0]), raw)
    raw = re.sub(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)",
                 lambda m: env.get((m[1] or m[2]).upper(), m[0]), raw)
    if raw.startswith("~"):
        home = env.get("USERPROFILE") or env.get("HOME")
        if home:
            raw = str(home)+raw[1:]
    path = Path(raw)
    return (Path(base)/path if not path.is_absolute() else path).resolve()


def discover(*, overrides=None, env=None, existing_host=None, search_roots=None, repo_root=None,
             resource_checker=None, git_reader=None, distributions_reader=None,
             requirements_path=None, include_common=True):
    """CLI > explicit environment > stored host > bounded discovery.

    A bad selected-priority value never falls through to a lower-priority path.
    Multiple valid paths at that priority require an explicit choice.
    """
    env = {key.upper():value for key, value in (os.environ if env is None else env).items()}
    repo = Path(repo_root or ROOT).resolve()
    overrides = dict(overrides or {})
    unknown = set(overrides)-set(KEYS)
    if unknown:
        raise ValueError("Unknown path override: " + ", ".join(sorted(unknown)))
    issues, warnings = [], []
    existing_base = repo
    if isinstance(existing_host, (str, os.PathLike)):
        existing_path = Path(existing_host).resolve()
        existing_base = existing_path.parent
        try:
            existing = json.loads(existing_path.read_text(encoding="utf-8-sig"))
            if not isinstance(existing, dict):
                raise ValueError("Host configuration must be a JSON object")
        except Exception as exc:
            return dict(selected={}, candidates={}, issues=["Existing host file cannot be read: " + str(exc)],
                missing_keys=list(REQUIRED), warnings=[], video_status="video_unconfigured")
    else:
        existing = dict(existing_host or {})
        if existing.get("config_path"):
            existing_base = Path(existing["config_path"]).resolve().parent
    candidates = {key:{} for key in KEYS}
    invalid_priority = {}
    invalid_inputs = {key:[] for key in KEYS}

    def add(key, value, source, priority=3, base=repo):
        try:
            path = _expand_path(value, env, base)
            if key == "ffmpeg" and path.is_dir():
                possibilities = [p for p in (path/"ffmpeg.exe", path/"bin/ffmpeg.exe") if p.is_file()]
                if not possibilities:
                    possibilities = [path/"ffmpeg.exe"]
            else:
                possibilities = [path]
            for target in possibilities:
                label = str(target).casefold()
                if label not in candidates[key]:
                    candidates[key][label] = dict(path=str(target), sources=[], priority=priority)
                row = candidates[key][label]
                if source not in row["sources"]:
                    row["sources"].append(source)
                row["priority"] = min(row["priority"], priority)
        except (OSError, ValueError) as exc:
            if priority < 3:
                invalid_priority[key] = min(priority, invalid_priority.get(key, 99))
                invalid_inputs[key].append(dict(source=source, priority=priority, reason=str(exc)))
            else:
                warnings.append(source + ": " + str(exc))

    for key in KEYS:
        if key in overrides:
            add(key, overrides[key], "CLI override", 0)
        for name in ENV_KEYS[key]:
            if env.get(name):
                add(key, env[name], "environment " + name, 1)
        if key in existing and existing[key] is not None:
            add(key, existing[key], "existing host", 2, existing_base)
    add("policy_support", repo/".local/policy_support", "repository local support")
    scan_roots = list(search_roots or [])
    if len(scan_roots) > MAX_ROOTS:
        raise ValueError("At most 32 search roots are allowed")
    scan_roots.append(repo.parent)
    if include_common:
        drive = env.get("SYSTEMDRIVE", "C:") + "/"
        for path in (Path(drive)/"isaacsim", Path(drive)/"IsaacSim"):
            add("isaac_sim", path, "common standalone SDK location")
        for path in (Path(drive)/"robotics/IsaacLab", Path(drive)/"IsaacLab"):
            add("isaac_lab", path, "common Lab checkout location")
        for variable, suffix in (("LOCALAPPDATA", "ov/pkg"), ("PROGRAMFILES", "NVIDIA Corporation")):
            if env.get(variable):
                scan_roots.append(Path(env[variable])/suffix)
        for variable in ("USERPROFILE", "HOME"):
            if env.get(variable):
                add("isaac_sim", Path(env[variable])/"isaacsim", "home installation location")
                add("isaac_lab", Path(env[variable])/"IsaacLab", "home checkout location")
    seen_roots = set()
    for raw in scan_roots:
        try:
            root = _expand_path(raw, env, repo)
            if str(root).casefold() in seen_roots:
                continue
            seen_roots.add(str(root).casefold())
            for key in ("isaac_sim", "isaac_lab"):
                add(key, root, "search root")
            for key, names in (("isaac_sim", ("isaacsim", "IsaacSim", "isaac-sim")),
                ("isaac_lab", ("IsaacLab", "isaaclab")),
                ("policy_support", ("chicgrasp-policy-support", "policy_support")),
                ("ffmpeg", ("ffmpeg.exe", "bin/ffmpeg.exe", "ffmpeg/bin/ffmpeg.exe"))):
                for name in names:
                    add(key, root/name, "bounded known child path")
            if root.is_dir() and root != Path(root.anchor):
                with os.scandir(root) as entries:
                    children = [Path(entry.path) for entry in itertools.islice(entries, MAX_CHILDREN+1)]
                if len(children) > MAX_CHILDREN:
                    warnings.append("Immediate-child scan truncated at 64: " + str(root))
                for child in children[:MAX_CHILDREN]:
                    name = child.name.casefold()
                    if name.startswith(("isaacsim", "isaac-sim")):
                        add("isaac_sim", child, "immediate installer-named child")
                    elif name.startswith("isaaclab"):
                        add("isaac_lab", child, "immediate installer-named child")
                    elif name.startswith("ffmpeg"):
                        add("ffmpeg", child, "immediate FFmpeg-named child")
        except (OSError, ValueError) as exc:
            warnings.append("Search root unavailable: " + str(raw) + ": " + str(exc))
    path_entries = env.get("PATH", "").split(os.pathsep)
    if len(path_entries) > MAX_PATH_ENTRIES:
        warnings.append("PATH discovery truncated at 96 entries")
    for raw in path_entries[:MAX_PATH_ENTRIES]:
        if not raw:
            continue
        try:
            folder = _expand_path(raw, env, repo)
            if (folder/"ffmpeg.exe").is_file():
                add("ffmpeg", folder/"ffmpeg.exe", "PATH")
            if (folder/"python.exe").is_file() or (folder/"python.bat").is_file():
                for ancestor in (folder, folder.parent, folder.parent.parent):
                    add("isaac_sim", ancestor, "PATH Python bounded ancestor")
        except (OSError, ValueError) as exc:
            warnings.append("PATH entry unavailable: " + str(exc))
    # Only the exact conventional link in already-enumerated Lab roots is read.
    for row in list(candidates["isaac_lab"].values()):
        link = Path(row["path"])/"_isaac_sim"
        if link.exists():
            add("isaac_sim", link, "Isaac Lab _isaac_sim link")
    selected, rendered, selection_failures = {}, {}, set()
    for key in KEYS:
        rows = list(candidates[key].values())
        for row in rows:
            try:
                check = _validate_location(key, row["path"], git_reader=git_reader,
                    distributions_reader=distributions_reader, requirements_path=requirements_path)
            except Exception as exc:
                check = dict(status="FAIL", errors=[str(exc)], warnings=[], checks={})
            row.update(valid=check["status"] == "PASS", reasons=check["errors"],
                       warnings=check["warnings"], checks=check["checks"])
        rendered[key] = sorted(rows, key=lambda row:(row["priority"], row["path"].casefold()))
        priority = min([row["priority"] for row in rows] + [invalid_priority.get(key, 99)])
        pool = [row for row in rows if row["priority"] == priority and row["valid"]]
        if invalid_priority.get(key) == priority:
            selection_failures.add(key)
            issues.extend(row["source"] + ": " + row["reason"] for row in invalid_inputs[key]
                          if row["priority"] == priority)
            continue
        if len(pool) == 1:
            selected[key] = pool[0]["path"]
        elif len(pool) > 1:
            issues.append("Multiple valid " + key + " locations; choose an explicit override")
            selection_failures.add(key)
        elif key in REQUIRED or priority < 3:
            issues.append("No valid " + key + " at selected priority; no automatic fallback")
            selection_failures.add(key)
    sdk_validated = False
    if all(key in selected for key in ("isaac_sim", "isaac_lab")):
        try:
            if resource_checker is None:
                from pipeline.runtime.host import verify_sdk_resources
                resource_checker = verify_sdk_resources
            resource_checker(selected)
            sdk_validated = True
        except Exception as exc:
            issues.append("SDK resource identity: " + str(exc))
            for key in ("isaac_sim", "isaac_lab"):
                selected.pop(key, None)
                selection_failures.add(key)
    if "training_python" not in selected and not candidates["training_python"] and "training_python" not in invalid_priority and sdk_validated:
        selected["training_python"] = str(Path(selected["isaac_sim"])/"kit/python/python.exe")
        rendered["training_python"].append(dict(path=selected["training_python"],
            sources=["authenticated selected standalone SDK"], priority=3, valid=True, reasons=[],
            warnings=["Interpreter not executed; version/import runtime validation remains pending"], checks={}))
    return dict(selected=selected, candidates=rendered, issues=issues, warnings=warnings, invalid_inputs=invalid_inputs,
        missing_keys=[key for key in KEYS if key not in selected and (key in REQUIRED or key in selection_failures)], sdk_validated=sdk_validated,
        video_status="READY" if "ffmpeg" in selected else "video_unconfigured",
        bounds=dict(search_roots=len(seen_roots), maximum_children_per_root=MAX_CHILDREN,
                    maximum_path_entries=MAX_PATH_ENTRIES), writes_performed=False)


"""Standard-library setup. No simulator/model imports or implicit installation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import uuid

from pipeline.settings import ROOT, digest, read, repo_path

REQUIREMENTS = ROOT / 'environments/policy-support-win311.txt'
REQUIRED = ('isaac_sim', 'isaac_lab', 'policy_support')

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def save_host(path, proposed, *, update=False):
    """Preserve bytes on no-op; explicit updates keep an exact prior copy."""
    path = Path(path)
    if path.exists():
        previous = path.read_bytes()
        if json.loads(previous.decode('utf-8-sig')) == proposed:
            return {'action': 'UNCHANGED', 'path': str(path)}
        if not update:
            raise ValueError('Existing host settings differ. Use --update-config to retain a backup and update them.')
        backup = path.with_name(path.name + '.backup-' + uuid.uuid4().hex + '.json')
        with backup.open('xb') as stream: stream.write(previous)
        atomic_json(path, proposed)
        return {'action': 'UPDATED', 'path': str(path), 'backup': str(backup)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(proposed, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return {'action': 'CREATED', 'path': str(path)}

def candidate_config(selected, existing, repo_root):
    result = dict(existing or {})
    result.update(schema='hc-reproduction-host/v1')
    for key in (*REQUIRED, 'ffmpeg', 'training_python'):
        if selected.get(key): result[key] = str(Path(selected[key]).resolve())
    if selected.get('isaac_sim') and not result.get('training_python'):
        result['training_python'] = str(Path(selected['isaac_sim']) / 'kit/python/python.exe')
    result.setdefault('checkpoint', str(Path(repo_root) / '.local/models/BEST-120000.pt'))
    return result

def validate_install_target(target, config, repo_root):
    target = Path(target).resolve()
    local = (Path(repo_root) / '.local').resolve()
    if not target.is_relative_to(local) or target == local:
        raise ValueError('Extra packages may be installed only into a NEW folder below this repository .local directory.')
    for key in ('isaac_sim', 'isaac_lab'):
        if config.get(key) and target.is_relative_to(Path(config[key]).resolve()):
            raise ValueError('Refusing to install packages into an SDK/source installation.')
    if target.exists():
        raise ValueError('Support target already exists; it will not be changed. Select a new .local folder with --policy-support.')
    return target

def install_extras(config, target, *, repo_root=ROOT, requirements=REQUIREMENTS,
                   runner=None, support_checker=None):
    from pipeline.runtime.host import environment
    from pipeline.runtime.process import run_owned
    target = validate_install_target(target, config, repo_root)
    requirements = Path(requirements)
    # Do not execute nested requirements, index overrides or arbitrary URL installs.
    pins = [s.strip() for s in requirements.read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
    if not pins or any(not re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+', s) for s in pins):
        raise ValueError('Support requirements must contain explicit package==version pins only.')
    source_sha = digest(requirements)
    local = (Path(repo_root) / '.local').resolve()
    local.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='policy-support-staging-', dir=local))
    logs = local / ('setup-install-' + uuid.uuid4().hex)
    staged_config = dict(config, policy_support=str(staging))
    command = [str(Path(config['isaac_sim']) / 'kit/python/python.exe'), '-B', '-s', '-m', 'pip', 'install',
               '--disable-pip-version-check', '--no-deps', '--target', str(staging), '-r', str(requirements)]
    env = environment(staged_config)
    env['PIP_CACHE_DIR'] = str(local / 'pip-cache')
    receipt = (runner or run_owned)(command, cwd=repo_root, env=env, directory=logs,
                                    timeout_seconds=600, cleanup_seconds=5)
    result = {'target': str(target), 'staging': str(staging), 'process_receipt': str(logs / 'receipt.json'),
              'requirements_sha256': source_sha, 'status': 'INSTALL_FAILED'}
    if not receipt.get('success'):
        result['reason'] = 'Installer failed or did not finish cleanly; staging and logs retained.'
        return result
    if digest(requirements) != source_sha:
        result['reason'] = 'Requirements changed during installation; staging retained.'
        return result
    checked = (support_checker or validate_support)(staging)
    result['validation'] = checked
    if checked.get('status') != 'PASS':
        result['reason'] = 'Installed metadata did not match all pinned extras; staging retained.'
        return result
    target.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(target)  # Refuses to replace an existing target on supported Windows.
    result.update(status='INSTALLED', staging=None)
    return result

def prepare(args, *, repo_root=ROOT, system=None, discover_fn=None, validate_fn=None,
            preflight_fn=None, installer=None, input_fn=input, interactive=None):
    from pipeline.runtime.host import preflight
    repo_root = Path(repo_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute(): config_path = repo_root / config_path
    config_path = config_path.resolve()
    report = {'schema': 'chicgrasp-automatic-setup/v1', 'status': 'SETUP_INCOMPLETE',
              'config_path': str(config_path), 'errors': [], 'warnings': [],
              'simulator_started': False, 'model_loaded': False, 'software_installed': False,
              'runtime_validated': False}
    if (system or platform.system()) != 'Windows':
        report.update(status='UNSUPPORTED_PLATFORM', errors=['The current adapter supports Windows standalone Isaac Sim only. Linux, container and pip-only layouts are not implemented.'])
        return report
    if sys.version_info[:2] != (3, 11):
        report['errors'].append('Run setup with Python 3.11, matching the pinned SDK.')
        return report
    existing = read(config_path) if config_path.exists() else None
    if existing and existing.get('schema') not in ('hc-reproduction-host/v1', 'research-host/v1'):
        raise ValueError('Existing host file has an unsupported schema; it was not changed.')
    # Resolve saved relative paths against the host file, not the current terminal.
    resolved_existing = dict(existing or {})
    for key in (*REQUIRED, 'training_python', 'ffmpeg', 'checkpoint'):
        if resolved_existing.get(key):
            p = Path(os.path.expandvars(resolved_existing[key])).expanduser()
            resolved_existing[key] = str(p.resolve() if p.is_absolute() else (config_path.parent / p).resolve())
    if args.check:
        if not existing:
            report['errors'].append('Local settings are missing. Run python setup_project.py first.')
            return report
        checked = (validate_fn or validate_host)(resolved_existing)
        report['validation'] = checked
        report['errors'].extend(checked['errors'])
        report['warnings'].extend(checked.get('warnings', []))
        report['video_status'] = 'READY' if checked.get('video_status') == 'READY' else 'NOT_CONFIGURED'
        if not report['errors']:
            pf = (preflight_fn or preflight)(resolved_existing, include_checkpoint=False)
            report['preflight'] = pf
            report['errors'].extend(pf['errors'])
        if args.require_video and report['video_status'] != 'READY':
            report['errors'].append('Video requires both FFmpeg and FFprobe; supply --ffmpeg during configuration.')
        report['status'] = 'READY' if not report['errors'] else 'SETUP_INCOMPLETE'
        return report
    overrides = {k: getattr(args, k) for k in (*REQUIRED, 'ffmpeg', 'training_python') if getattr(args, k, None) is not None}
    discovery = (discover_fn or discover)(overrides=overrides, existing_host=resolved_existing or None,
                                          search_roots=args.search_root, repo_root=repo_root)
    ask = (sys.stdin.isatty() and not args.non_interactive) if interactive is None else interactive
    if ask and not args.dry_run:
        for key in ('isaac_sim', 'isaac_lab'):
            if key in discovery['missing_keys'] and key not in overrides:
                response = input_fn(f'Could not choose {key}. Enter its installed folder (blank to stop): ').strip().strip('"')
                if response: overrides[key] = response
        discovery = (discover_fn or discover)(overrides=overrides, existing_host=resolved_existing or None,
                                              search_roots=args.search_root, repo_root=repo_root)
    report['discovery'] = discovery
    selected = dict(discovery['selected'])
    if args.checkpoint:
        p = Path(args.checkpoint)
        resolved_existing['checkpoint'] = str(p.resolve() if p.is_absolute() else (repo_root / p).resolve())
    config = candidate_config(selected, resolved_existing, repo_root)
    missing = set(discovery['missing_keys'])
    # Never install when core SDK selection is missing/ambiguous or explicitly wrong.
    if missing <= {'policy_support'} and 'policy_support' in missing and args.install_extras and not args.dry_run:
        target = overrides.get('policy_support') or resolved_existing.get('policy_support') or str(repo_root / '.local/policy_support')
        proposed = dict(config, policy_support=str(Path(target).resolve()))
        if existing and proposed != existing and not args.update_config:
            report['errors'].append('Existing settings would change. Use --update-config before installing for those new settings.')
        else:
            installed = (installer or install_extras)(config, target, repo_root=repo_root)
            report['installation'] = installed
            if installed['status'] == 'INSTALLED':
                report['software_installed'] = True
                overrides['policy_support'] = installed['target']
                discovery = (discover_fn or discover)(overrides=overrides, existing_host=resolved_existing or None,
                                                      search_roots=args.search_root, repo_root=repo_root)
                report['discovery'] = discovery
                selected = dict(discovery['selected'])
                config = candidate_config(selected, resolved_existing, repo_root)
                missing = set(discovery['missing_keys'])
            else: report['errors'].append(installed['reason'])
    for issue in discovery.get('issues', []):
        if issue not in report['errors']: report['errors'].append(issue)
    if missing:
        report['errors'].append('Missing or ambiguous settings: ' + ', '.join(sorted(missing)))
        if 'policy_support' in missing:
            report['next_command'] = 'python setup_project.py --install-extras'
    report['video_status'] = 'READY' if discovery.get('video_status') == 'READY' else 'NOT_CONFIGURED'
    report['proposed_config'] = config
    if not report['errors']:
        checked = (validate_fn or validate_host)(config)
        report['validation'] = checked
        report['errors'].extend(checked['errors'])
        report['warnings'].extend(checked.get('warnings', []))
        report['video_status'] = 'READY' if checked.get('video_status') == 'READY' else 'NOT_CONFIGURED'
        if not report['errors']:
            pf = (preflight_fn or preflight)(config, include_checkpoint=False)
            report['preflight'] = pf
            report['errors'].extend(pf['errors'])
    if args.require_video and report['video_status'] != 'READY':
        report['errors'].append('Both FFmpeg and FFprobe are required for video; use --ffmpeg PATH.')
    if not report['errors']:
        if args.dry_run:
            report['configuration'] = {'action': 'DRY_RUN_NO_WRITE', 'path': str(config_path)}
        else:
            report['configuration'] = save_host(config_path, config, update=args.update_config)
        report['status'] = 'READY'
        report['next_command'] = 'python load_scene.py'
    if not args.dry_run:
        report['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(repo_root / '.local/setup-report.json', report)
    return report

def parser():
    p = argparse.ArgumentParser(description='Configure the installed Windows SDKs; no Kit/model start. Extra-package installation is explicit.')
    p.add_argument('--config', default='.local/host.json')
    for name in (*REQUIRED, 'ffmpeg', 'training_python', 'checkpoint'):
        p.add_argument('--' + name.replace('_', '-'))
    p.add_argument('--search-root', action='append', default=[])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Validate saved configuration without writing anything')
    mode.add_argument('--dry-run', action='store_true', help='Discover and report without writing or installing')
    p.add_argument('--install-extras', action='store_true', help='Install pinned extras into a NEW project-local folder; never modify the SDK')
    p.add_argument('--update-config', action='store_true', help='Permit a configuration update, preserving an exact backup')
    p.add_argument('--require-video', action='store_true')
    p.add_argument('--non-interactive', action='store_true')
    p.add_argument('--json', action='store_true')
    return p

def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.check and (args.install_extras or args.update_config):
        p.error('--check never installs or updates; use a separate configure command.')
    try:
        report = prepare(args)
    except (OSError, ValueError, KeyError) as exc:
        report = {'status':'SETUP_INCOMPLETE','errors':[str(exc)],'warnings':[]}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print('Setup: ' + report['status'])
        if report.get('configuration'): print('Settings: ' + report['configuration']['path'])
        for error in report.get('errors', []): print('  ' + error)
        for warning in report.get('warnings', []): print('  Note: ' + warning)
        if report.get('video_status'): print('Video tools: ' + report['video_status'])
        if report.get('next_command'): print('Next: ' + report['next_command'])
        print('No simulator or model was started. READY means offline setup checks passed, not runtime validation.')
    return 0 if report['status'] == 'READY' else 2


"""Create new experiment settings from the unchanged research template."""
import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10))}


def validate_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("Use 1-64 ASCII letters, digits, underscores or hyphens; start with a letter or digit.")
    if name.upper() in RESERVED:
        raise ValueError("That experiment name is reserved by Windows: " + name)
    return name


def _inside(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError("Path escapes the repository: " + str(relative))
    return path


def create_experiment(name, *, repo_root=ROOT):
    """Create only configs/experiments/NAME.json. Never create run directories."""
    name = validate_name(name)
    root = Path(repo_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Repository root must be an existing directory.")
    config_relative = Path("configs") / "experiments" / (name + ".json")
    output_relative = Path("research_outputs") / name
    target = _inside(root, config_relative)
    output_root = _inside(root, output_relative)
    for path in (target, output_root):
        if os.path.lexists(path):
            raise FileExistsError("Already exists; choose a new experiment name: " + str(path))
    template = _inside(root, Path("configs/research.json"))
    config = json.loads(template.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or config.get("schema") != "research-experiment/v1":
        raise ValueError("The template must be a research-experiment/v1 JSON object.")
    outputs = {key: (output_relative / (directory + "001")).as_posix()
               for key, directory in (("scene", "scene"), ("collection", "collection"),
                                      ("training", "training"), ("live", "live"))}
    receipt = {
        "status": "CREATED", "name": name, "config": config_relative.as_posix(),
        "root": output_relative.as_posix(), "outputs": outputs,
        "scene_approval": (output_relative / "scene-approval001.json").as_posix(),
        "dataset": outputs["collection"] + "/dataset.json",
        "artifact": outputs["training"] + "/model/artifact.json",
    }
    config.update(experiment_id=name, scene_approval=receipt["scene_approval"],
                  dataset=receipt["dataset"], model_artifact=receipt["artifact"])
    contents = json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Recheck the resolved boundary after creating the parent. Exclusive creation
    # also refuses a config created by another process since the earlier check.
    _inside(root, config_relative)
    if os.path.lexists(output_root):
        raise FileExistsError("Experiment output already exists: " + str(output_root))
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(contents)
    return receipt


def create_experiment_main(argv=None, *, repo_root=ROOT):
    parser = argparse.ArgumentParser(description="Create a new experiment config and report its output paths. No simulator starts.")
    parser.add_argument("name", metavar="NAME")
    parser.add_argument("--json", action="store_true", help="Print only the JSON result on success")
    args = parser.parse_args(argv)
    try:
        result = create_experiment(args.name, repo_root=repo_root)
    except (OSError, ValueError) as exc:
        print("Could not create experiment: " + str(exc), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("Created experiment: " + result["name"])
        print("Config: " + result["config"])
        for key, value in result["outputs"].items():
            print(key.capitalize() + ": " + value)
        print("Scene approval: " + result["scene_approval"])
        print("Dataset: " + result["dataset"])
        print("Model artifact: " + result["artifact"])
        print("No run directories were created and no simulator was started.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
