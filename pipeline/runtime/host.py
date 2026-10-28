"""Relocatable input/configuration handling. No simulator, NumPy or model imports."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / 'frozen_runtime'
SOURCE_COMMIT = '3d98c85228821d955479730292d2b026fd1d63ed'
MODEL_SHA = '1e91f1db31fb9b2856845abc2266a9078d9c97c1b291bc1e1faaa3c29a3855fe'
ALLOCATION_ORDER = tuple(f'HC{n:02d}_R{r}' for r in range(3) for n in range(1, 11))
TIMEOUT_SECONDS = 1200
CLEANUP_SECONDS = 5
CHECKER_TIMEOUT_SECONDS = 180
BINDINGS_SHA = '61846dfa29898cf0fbbcfddd4a807c7c0c827eaacd3d930f7bde1fa62bd7e845'

def adapter_snapshot():
    paths = sorted(set((ROOT/'pipeline/runtime').rglob('*.py')) | set((ROOT/'pipeline/runtime/data').glob('*.json')) | {ROOT/'pipeline/original_evaluation.py', ROOT/'reproduce.py'})
    return json_hash({p.relative_to(ROOT).as_posix():sha256(p) for p in paths})

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha256(path):
    path=Path(path); before=path.stat(); digest=hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8*1024*1024): digest.update(block)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError(f'Input changed during hashing: {path}')
    return digest.hexdigest()

def json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def write_json(path, value, *, exclusive=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def load_config(path):
    path=Path(path).resolve(strict=True); value=load_json(path)
    if value.get('schema')!='hc-reproduction-host/v1': raise ValueError('Unsupported host configuration schema')
    for key in ('isaac_sim','isaac_lab','policy_support','checkpoint'):
        raw=value.get(key)
        if not isinstance(raw,str) or not raw: raise ValueError(f'Missing host setting: {key}')
        target=Path(os.path.expandvars(raw)).expanduser()
        value[key]=str((path.parent/target).resolve() if not target.is_absolute() else target.resolve())
    value['config_path']=str(path)
    return value

def make_binding(config, point, *, checkpoint_verified=False):
    """Relocate operative paths only; checkpoint metadata identities stay intact."""
    if point not in ALLOCATION_ORDER: raise ValueError(f'Unknown frozen allocation: {point}')
    bindings=ROOT/'pipeline/runtime/data/HC_BINDINGS.json'
    if sha256(bindings)!=BINDINGS_SHA: raise ValueError('Frozen scientific bindings changed')
    value=deepcopy(load_json(bindings)['allocations'][point])
    for key in ('checkpoint','checkpoint_identity'):
        value[key]['path']=config['checkpoint']
    normalizer=FROZEN/'outputs/random_xy_50_collection/attempt001/package/normalization.json'
    value['normalizer']['path']=str(normalizer)
    value['scene_profile']['path']=str(FROZEN/'docs/plans/CHICGRASP_SCALED_TABLE_AND_COVERAGE/attempt005/SCENE_PROFILE.json')
    value['policy_support_path']=config['policy_support']
    value['checkpoint_verified']=bool(checkpoint_verified)
    return value

def environment(config):
    sim=Path(config['isaac_sim']); lab=Path(config['isaac_lab'])
    result=dict(os.environ)
    result.pop('PYTHONHOME',None)
    result.update(PYTHONPATH=os.pathsep.join(map(str,[ROOT,FROZEN/'src',FROZEN/'source/chicgrasp_isaac',lab/'source/isaaclab',lab/'source/isaaclab_tasks',lab/'source/isaaclab_assets',Path(config['policy_support']),sim/'site'])),
        PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',
        CARB_APP_PATH=str(sim/'kit'),ISAAC_PATH=str(sim),EXP_PATH=str(sim/'apps'),CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return result

def preflight(config, *, include_checkpoint=True):
    errors=[];checked=[]
    if sha256(ROOT/'pipeline/runtime/data/HC_BINDINGS.json')!=BINDINGS_SHA:
        errors.append('Frozen scientific binding snapshots changed.')
    for row in load_json(ROOT/'manifests/CORE_INPUTS.json')['files']:
        if row['group']=='model' and not include_checkpoint: continue
        path=Path(config['checkpoint']) if row['group']=='model' else FROZEN/row['path']
        if not path.is_file():
            errors.append(f'Missing {row["path"]}; restore the model parts or run git lfs pull for assets.');continue
        if path.stat().st_size != row['bytes']:
            errors.append(f'Wrong size or unpulled LFS pointer: {row["path"]}');continue
        if sha256(path)!=row['sha256']:
            errors.append(f'Frozen input SHA256 mismatch: {row["path"]}');continue
        checked.append(row['path'])
    expected=[('isaac_sim','kit/python/python.exe'),('isaac_sim','kit/kernel/plugins'),
              ('isaac_lab','source/isaaclab/isaaclab/app'),('isaac_lab','source/isaaclab_tasks'),
              ('isaac_lab','source/isaaclab_assets')]
    for key,child in expected:
        if not (Path(config[key])/child).exists():errors.append(f'Missing {key}/{child}; check the pinned installation path.')
    support=Path(config['policy_support'])
    for name in ('diffusers','einops','huggingface_hub','zarr'):
        if not (support/name).is_dir():errors.append(f'Missing policy support package: {name}; follow docs/TUTORIAL.md.')
    for name in ('torch','torchvision','numpy'):
        if (support/name).exists():errors.append(f'Policy support must not override Isaac {name}.')
    # Metadata inspection does not import packages or initialize CUDA.
    from importlib.metadata import distributions
    expected_versions={'diffusers':'0.11.1','einops':'0.4.1','huggingface-hub':'0.25.2',
                       'zarr':'2.18.7','numcodecs':'0.15.1'}
    found={d.metadata['Name'].lower().replace('_','-'):d.version for d in distributions(path=[str(support)])}
    for name,version in expected_versions.items():
        if found.get(name)!=version:errors.append(f'Policy support {name}: expected {version}, found {found.get(name)}.')
    try:
        verify_sdk_resources(config)
    except (ValueError,KeyError,OSError) as exc:errors.append(f'Pinned SDK identity: {exc}')
    return {'status':'PASS' if not errors else 'FAIL','errors':errors,'verified_input_count':len(checked),
            'checkpoint_verified':include_checkpoint and not any('BEST.pt' in x for x in errors),
            'simulator_started':False,'model_loaded':False}

def scientific_projection(binding):
    """Fields that may not change during path relocation or new administration."""
    names=('point','heldout_allocation','normalization','recipe_identity','source_identity','schedule_identity',
           'model_configuration','authored_world_xyz_m','authored_quaternion_wxyz','original_authored_world_xyz_m',
           'setup_actions','setup_action_identity','setup_source_indices','production_configuration',
           'evaluation_seed','seed','device','weights_kind','optimizer_step','envelope_v2','envelope_protocol','definition_id')
    return {key:binding[key] for key in names}


"""Host-path adapters for the immutable original HC runtime.

Importing this module is standard-library-only. install() imports the frozen
CPU-side modules but neither starts Kit nor deserializes a checkpoint. Only the
original recorder's explicit execute path may cross those runtime boundaries.
"""
from copy import deepcopy
import ast
import functools
import importlib.util
import inspect
from pathlib import Path
import sys
import textwrap


ORIGINAL_GROUND_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Grid/default_environment.usd"


def normalized_path(value):
    return str(value).replace("\\", "/").casefold()


def rewrite_literal_paths(source, replacements):
    """Rewrite only explicitly enumerated string literals, each exactly once."""
    tree = ast.parse(textwrap.dedent(source))
    counts = {name: 0 for name in replacements}

    class Rewrite(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in replacements:
                counts[node.value] += 1
                return ast.copy_location(ast.Constant(replacements[node.value]), node)
            return node

    tree = Rewrite().visit(tree)
    if any(count != 1 for count in counts.values()):
        raise ValueError("host_literal_allowlist_mismatch:" + repr(counts))
    ast.fix_missing_locations(tree)
    return tree


def relocate_function(function, replacements):
    tree = rewrite_literal_paths(inspect.getsource(function), replacements)
    scope = dict(function.__globals__)
    exec(compile(tree, inspect.getsourcefile(function), "exec"), scope)
    changed = scope[function.__name__]
    functools.update_wrapper(changed, function)
    return changed


def relocate_renderer_record(record, resource):
    """Normalize only the identity of the authenticated rendering experience."""
    result = deepcopy(record)
    experience = result["experience"]
    if normalized_path(experience["path"]) not in {
        normalized_path(resource["original_path"]), normalized_path(resource["path"])
    }:
        raise ValueError("unapproved_renderer_experience_path")
    if any(experience[key] != resource[key] for key in ("bytes", "sha256")):
        raise ValueError("renderer_experience_bytes_changed")
    experience["path"] = resource["path"]
    return result


def relocate_shader_report(report, resource, shader_property):
    """Keep saved readback truthful; normalize a copy for the frozen checker."""
    result = deepcopy(report)
    rows = [row for row in result.get("dependencies", [])
            if row.get("property") == shader_property]
    if len(rows) != 1:
        raise ValueError("ground_shader_dependency_count")
    row = rows[0]
    for key in ("authored", "resolved"):
        if normalized_path(row.get(key)) != normalized_path(resource["path"]):
            raise ValueError("ground_shader_unapproved_resolution:" + key)
        row[key] = resource["original_path"]
    return result


def verify_sdk_resources(config):
    manifest = load_json(ROOT / "pipeline/runtime/data/SDK_RESOURCE_IDENTITIES.json")
    if manifest.get("schema") != "hc-sdk-resources/v1":
        raise ValueError("sdk_resource_manifest_schema")
    resources = {}
    for row in manifest["resources"]:
        path = (Path(config[row["config_key"]]) / row["relative_path"]).resolve(strict=True)
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError("sdk_resource_changed:" + row["name"])
        resources[row["name"]] = dict(row, path=path.as_posix())
    if set(resources) != {"experience", "ground_shader", "physx_binding", "sim_version", "lab_version"}:
        raise ValueError("sdk_resource_set")
    return resources


def verify_bundled_ground():
    manifest = load_json(ROOT / "pipeline/runtime/data/BUNDLED_GROUND.json")
    if (manifest.get("schema") != "hc-bundled-sdk-ground/v1"
            or manifest.get("original_url") != ORIGINAL_GROUND_URL):
        raise ValueError("bundled_ground_manifest_schema_or_url")
    root = (FROZEN / "assets/portable_sdk_ground").resolve(strict=True)
    paths = []
    for row in manifest["files"]:
        path = (FROZEN / row["path"]).resolve(strict=True)
        if (not path.is_relative_to(root) or path.stat().st_size != row["bytes"]
                or sha256(path) != row["sha256"] or row["sha256"] != row["original_cache_sha256"]):
            raise ValueError("bundled_ground_identity:" + row["path"])
        paths.append(path.relative_to(root).as_posix())
    expected = {"default_environment.usd", "Materials/Textures/Wireframe_blue.png",
                "Materials/Textures/WireframeBlur_basecolor.png", "Materials/Textures/WireframeBlur_blue.png"}
    if len(paths) != 4 or set(paths) != expected:
        raise ValueError("bundled_ground_exact_file_set")
    return dict(original_url=ORIGINAL_GROUND_URL, path=str(root / "default_environment.usd"),
                files=deepcopy(manifest["files"]))


def configure_with_bundled_ground(cfg, session, original, ground):
    """Relocate the unchanged ground asset, then call original start setup once."""
    if cfg.scene.ground.spawn.usd_path != ORIGINAL_GROUND_URL:
        raise ValueError("unexpected_original_ground_url")
    cfg.scene.ground.spawn.usd_path = ground["path"]
    return original(cfg, session)


def execution_identity(invocation, attempt, output):
    attempt = Path(attempt).resolve()
    if Path(output).resolve() != attempt / "capture/experiment":
        raise ValueError("portable_execution_output_mismatch")
    return dict(attempt_id=attempt.name, executed_commit=invocation["executed_commit"],
        source_commit=SOURCE_COMMIT, runtime_snapshot_sha256=invocation["runtime_snapshot_sha256"],
        adapter_snapshot_sha256=invocation["adapter_snapshot_sha256"],
        provenance_kind="portable_reproduction", commit_source="current public repository HEAD, or null for source archive; authenticated frozen snapshot recorded separately",
        errors=[])


def install(config, point, attempt, invocation, preflight):
    """Install narrow host adaptations after complete frozen-input verification."""
    if preflight.get("status") != "PASS" or preflight.get("checkpoint_verified") is not True:
        raise ValueError("complete_frozen_and_checkpoint_preflight_required")
    expected = make_binding(config, point, checkpoint_verified=True)
    if invocation.get("r4_live_binding") != expected:
        raise ValueError("portable_invocation_binding_changed")
    if (invocation.get("source_commit") != SOURCE_COMMIT
            or invocation.get("provenance_kind") != "portable_reproduction"
            or invocation.get("runtime_snapshot_sha256") != sha256(ROOT / "manifests/CORE_INPUTS.json")):
        raise ValueError("portable_snapshot_provenance_changed")
    if invocation.get("adapter_snapshot_sha256") != adapter_snapshot():
        raise ValueError("portable_adapter_snapshot_changed")
    resources = verify_sdk_resources(config)
    ground = verify_bundled_ground()
    if str(FROZEN / "src") not in sys.path:
        sys.path.insert(0, str(FROZEN / "src"))

    from chicgrasp_policy import render_profile, grey_studio_check, dome_recovery
    from chicgrasp_policy import attachment_capture_runtime, attachment_owner, r4_live_binding

    def verify_binding(requested, *, verify_checkpoint=True, envelope_v2=False, envelope_v3c=False):
        if requested != point or envelope_v2 or not envelope_v3c:
            raise ValueError("portable_exact_hc_v3c_binding_required")
        return make_binding(config, point, checkpoint_verified=bool(verify_checkpoint))

    r4_live_binding.verify_binding = verify_binding
    original_configure_start = r4_live_binding.configure_start

    def configure_start(cfg, session):
        return configure_with_bundled_ground(cfg, session, original_configure_start, ground)

    r4_live_binding.configure_start = configure_start
    experience = resources["experience"]
    original_renderer = render_profile.validate_renderer
    render_profile.SHARED_EXPERIENCE = experience["path"]

    def validate_renderer(record, baseline=None):
        actual = relocate_renderer_record(record, experience)
        expected_renderer = None if baseline is None else relocate_renderer_record(baseline, experience)
        return original_renderer(actual, expected_renderer)

    render_profile.validate_renderer = validate_renderer
    attachment_capture_runtime.SHARED_EXPERIENCE = experience["path"]
    attachment_capture_runtime.APP_CONFIGURATION["experience"] = experience["path"]
    sim = Path(config["isaac_sim"])
    dlls = {"C:/isaacsim": sim.as_posix(), "C:/isaacsim/kit": (sim / "kit").as_posix(),
            "C:/isaacsim/kit/kernel/plugins": (sim / "kit/kernel/plugins").as_posix()}
    attachment_capture_runtime._NativeDependencies.launch_app = relocate_function(
        attachment_capture_runtime._NativeDependencies.launch_app, dlls)
    versions = {"C:/isaacsim/VERSION": resources["sim_version"]["path"],
                "C:/robotics/IsaacLab/VERSION": resources["lab_version"]["path"]}
    attachment_owner._installed_identity = relocate_function(attachment_owner._installed_identity, versions)
    attachment_owner.PINNED_RUNTIME["binding_path"] = str(Path(resources["physx_binding"]["path"]))

    shader = resources["ground_shader"]
    shader_property = grey_studio_check.SHADER + ".info:mdl:sourceAsset"
    original_grey = grey_studio_check.validate_report

    def validate_report(report, calibration):
        return original_grey(relocate_shader_report(report, shader, shader_property), calibration)

    grey_studio_check.validate_report = validate_report
    original_dome = dome_recovery.apply_overrides

    def apply_overrides(stage, *, edit_context=None):
        original_dome(stage, edit_context=edit_context)
        from pxr import Sdf, Usd
        prim = stage.GetPrimAtPath(grey_studio_check.SHADER)
        attr = prim.GetAttribute("info:mdl:sourceAsset")
        if not attr.IsValid() or normalized_path(attr.Get().path) != normalized_path(shader["original_path"]):
            raise ValueError("ground_shader_original_asset_path_changed")
        # Only resolution changes, in a task-owned memory layer. Frozen files,
        # shader bytes, material parameters, lighting and physics stay untouched.
        if normalized_path(shader["path"]) != normalized_path(shader["original_path"]):
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                if not attr.Set(Sdf.AssetPath(shader["path"])):
                    raise ValueError("ground_shader_host_alias_write_failed")

    dome_recovery.apply_overrides = apply_overrides
    source = FROZEN / "scripts/policy/attachment_assisted_lift_record.py"
    spec = importlib.util.spec_from_file_location("portable_original_hc_recorder", source)
    recorder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = recorder
    spec.loader.exec_module(recorder)
    recorder.execution_identity = lambda output: execution_identity(invocation, attempt, output)
    receipt = dict(schema="hc-host-adapter/v1", source_commit=SOURCE_COMMIT,
        frozen_files_modified=False, original_scientific_callbacks_preserved=True,
        resources=resources, bundled_ground=ground,
        literal_replacements={"native_launch": dlls, "owner_identity": versions},
        hooks=["r4_live_binding.verify_binding", "recorder.execution_identity",
               "r4_live_binding.configure_start: ground path prelude, original called once",
               "render_profile experience path and validator alias", "grey_studio shader path alias",
               "attachment_owner runtime identity host paths"],
        restrictions="No model, observation, seed, physics, scoring or rendering-parameter changes.")
    return recorder, receipt
