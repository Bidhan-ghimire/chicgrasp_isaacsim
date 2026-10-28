"""Bounded, explicit launcher for the original HC allocations; no automatic retry."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pipeline.runtime.host import ROOT, FROZEN, SOURCE_COMMIT, MODEL_SHA, ALLOCATION_ORDER, TIMEOUT_SECONDS, CLEANUP_SECONDS, CHECKER_TIMEOUT_SECONDS, adapter_snapshot, load_json, load_config, write_json, sha256, make_binding, environment, preflight, scientific_projection, json_hash

BATCH_LIMIT_SECONDS=12*3600
MIN_FREE_BYTES=5*1024**3

def now():
    return datetime.now(timezone.utc).isoformat()

def selection(allocation):
    if allocation=='all': return list(ALLOCATION_ORDER)
    if allocation not in ALLOCATION_ORDER: raise ValueError('Unknown frozen allocation: '+allocation)
    return [allocation]

def make_plan(config, allocation='all'):
    points=selection(allocation)
    bindings=[make_binding(config,point) for point in points]
    return dict(schema='hc-portable-reproduction-plan/v1',created_utc=now(),
        scope='ORIGINAL_UNCAPPED_HC_V3C',source_commit=SOURCE_COMMIT,
        historical_result='18/30; historical observation, not a guaranteed new outcome',
        scientific_binding_sha256=json_hash([scientific_projection(v) for v in bindings]),
        policy_observation='LEFT and RIGHT RGB 320x240, 6 arm joints, 2 jaw bits; no HAND images',
        optional_demo_cameras_enabled=False,model_sha256=MODEL_SHA,
        allocations=[dict(point=v['point'],evaluation_seed=v['evaluation_seed'],environment_seed=v['seed'],
            authored_world_xyz_m=v['authored_world_xyz_m']) for v in bindings],
        per_trial_limit_seconds=TIMEOUT_SECONDS,cleanup_seconds=CLEANUP_SECONDS,
        checker_limit_seconds=CHECKER_TIMEOUT_SECONDS,batch_limit_seconds=BATCH_LIMIT_SECONDS,
        automatic_retries=0,simulator_started=False,model_loaded=False)

def current_commit():
    try:result=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,timeout=10)
    except FileNotFoundError:return None
    value=result.stdout.strip()
    return value if result.returncode==0 and re.fullmatch('[0-9a-f]{40}',value) else None

def file_identity(path):
    path=Path(path)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha256(path))

def native_command(config,point,attempt,mode):
    return [str(Path(config['isaac_sim'])/'kit/python/python.exe'),'-B','-s','-u',
        '-m','pipeline.original_evaluation','--native','--config',config['config_path'],
        '--point',point,'--attempt',str(attempt),'--mode',mode]

def process_record(receipt, invocation, attempt):
    """Translate only actual observations; retain the complete supervisor receipt."""
    result=dict(receipt)
    for key in ('run_type','geometric_mode','stage3_learned','stage3_witness','r4_live_point',
                'checkpoint_sha256','normalizer_sha256','r4_live_binding','envelope_v2',
                'envelope_protocol','witness_rule_sha256','executed_commit'):
        result[key]=invocation[key]
    result['kind']='attachment_capture'
    result['invocation_sha256']=sha256(attempt/'invocation.json')
    result['child_start_local']=datetime.fromtimestamp(receipt['started_unix'],timezone.utc).isoformat()
    result['kit_start_observation']=None
    # A completed startup line establishes a start; its observation interval ends
    # at this read, not at an invented live-poll time.
    pattern=re.compile(r'\[ext: [^\]\r\n]+\] startup\s*$')
    for name in ('stdout','stderr'):
        path=Path(receipt[name]['path'])
        if not path.is_file(): continue
        with path.open('r',encoding='utf-8',errors='replace') as stream:
            for line in stream:
                if pattern.search(line):
                    result['kit_start_observation']=dict(source='complete Kit extension startup line',
                        line=line.rstrip(),observed_local=now(),read_phase='after_owned_tree_exit')
                    break
        if result['kit_start_observation']:break
    result['actual_kit_starts']=1 if result['kit_start_observation'] else None
    result['kit_start_scan_complete']=True
    journal=attempt/'capture/structured/events.jsonl'
    if journal.is_file(): result['completed_journal']=file_identity(journal)
    result['final_journal_read_completed_ms']=math.ceil((time.monotonic()-receipt['started_monotonic'])*1000)
    result['duration_seconds']=(time.monotonic()-receipt['started_monotonic'])
    result['pre_kit_failure_verified']=False
    return result

def capture_clean(process):
    return (process.get('exit_code') in (0,1) and process.get('owned_tree_gone_verified') is True
        and process.get('capture_complete') is True and process.get('actual_kit_starts')==1
        and not any(process.get(k) for k in ('errors','timed_out','forced_termination','stop_requested','cleanup_budget_exceeded')))

def trial(config, point, output, preflight_result, runner):
    attempt=output/point/'native001';attempt.mkdir(parents=True,exist_ok=False)
    binding=make_binding(config,point,checkpoint_verified=True)
    invocation=dict(schema='hc-portable-invocation/v1',kind='attachment_capture',
        created_utc=now(),provenance_kind='portable_reproduction',source_commit=SOURCE_COMMIT,
        executed_commit=current_commit(),runtime_snapshot_sha256=sha256(ROOT/'manifests/CORE_INPUTS.json'),
        adapter_snapshot_sha256=adapter_snapshot(),run_type='HARNESS',evidence_authority=False,
        execution_approved=False,geometric_mode=True,stage3_learned=True,stage3_witness=True,
        envelope_v2=True,envelope_protocol='v3c',r4_live_point=point,r4_live_binding=binding,
        checkpoint_sha256=MODEL_SHA,normalizer_sha256=binding['normalizer']['sha256'],
        witness_rule_sha256='770faeab856c154bf6f9ab48512fce3932bfe1fee494e451f7ec56da8eb00782',
        timeout_ms=TIMEOUT_SECONDS*1000,output_directory=str(attempt/'capture'),
        parent_preflight=preflight_result,
        historical_grant_is_provenance_only=True,owner_issued_runtime_authorization=False,
        operator_explicit_execute=True,automatic_retries=0)
    write_json(attempt/'invocation.json',invocation)
    receipt=runner(native_command(config,point,attempt,'execute'),cwd=ROOT,env=environment(config),
        directory=attempt/'capture',timeout_seconds=TIMEOUT_SECONDS,cleanup_seconds=CLEANUP_SECONDS,
        stop_file=output/'STOP')
    process=process_record(receipt,invocation,attempt)
    write_json(attempt/'capture/process_result.json',process)
    if not capture_clean(process):
        return dict(point=point,status='TECHNICAL_STOP',reason='Capture process or start accounting not qualified',actual_kit_starts=process['actual_kit_starts'])
    checked=runner(native_command(config,point,attempt,'adjudicate'),cwd=ROOT,env=environment(config),
        directory=attempt/'checker',timeout_seconds=CHECKER_TIMEOUT_SECONDS,cleanup_seconds=CLEANUP_SECONDS,
        stop_file=output/'STOP')
    result_path=attempt/'attempt_result.json'
    if not checked['success'] or not result_path.is_file():
        return dict(point=point,status='TECHNICAL_STOP',reason='Independent checker process failed or result missing',actual_kit_starts=1)
    answer=load_json(result_path)
    if answer.get('technical_clean') is not True or answer.get('status')!='HARNESS_COMPLETE':
        return dict(point=point,status='TECHNICAL_STOP',reason='Independent original checker did not qualify the trial',result=answer,actual_kit_starts=1)
    historical=next(row['V3C_VERDICT'] for row in load_json(ROOT/'results/HC_18_OF_30.json')['rows'] if row['allocation']==point)
    verdict=answer.get('end_state',{}).get('status')
    return dict(point=point,status='HARNESS_COMPLETE',actual_kit_starts=1,
        verdict=verdict,historical_verdict=historical,comparison='MATCH' if verdict==historical else 'DIFFERENT',result=answer)

def run_batch(config, output, allocation='all', *, runner=None):
    if os.name!='nt':raise ValueError('This containment adapter supports Windows only.')
    output=Path(output).resolve()
    for protected in (FROZEN,Path(config['isaac_sim']),Path(config['isaac_lab']),Path(config['policy_support'])):
        if output==protected or protected in output.parents:raise ValueError('Output must be outside frozen inputs and SDK/package directories.')
    if output.exists():raise FileExistsError('Output directory already exists; nothing overwritten.')
    nearest=output.parent
    while not nearest.exists():nearest=nearest.parent
    if shutil.disk_usage(nearest).free < MIN_FREE_BYTES:raise ValueError('At least 5 GiB free space is required for this output.')
    checked=preflight(config)
    if checked['status']!='PASS':raise ValueError('Preflight failed: '+'; '.join(checked['errors']))
    lock=ROOT/'.local/ACTIVE_RUN.lock'
    lock.parent.mkdir(exist_ok=True)
    with lock.open('x',encoding='utf-8') as stream:json.dump(dict(pid=os.getpid(),created_utc=now(),output=str(output)),stream)
    if runner is None:
        from pipeline.runtime.process import run_owned
        runner=run_owned
    rows=[];begin=time.monotonic();status='STOPPED';created=False
    try:
        output.mkdir(parents=True,exist_ok=False)
        created=True
        write_json(output/'PLAN.json',make_plan(config,allocation))
        for point in selection(allocation):
            if (output/'STOP').exists():break
            if time.monotonic()-begin+TIMEOUT_SECONDS+CLEANUP_SECONDS+CHECKER_TIMEOUT_SECONDS+CLEANUP_SECONDS>BATCH_LIMIT_SECONDS:
                break
            try:row=trial(config,point,output,checked,runner)
            except (Exception,KeyboardInterrupt) as exc:
                row=dict(point=point,status='TECHNICAL_STOP',reason=f'{type(exc).__name__}: {exc}',actual_kit_starts=None)
            rows.append(row)
            write_json(output/f'{point}/SUMMARY.json',row)
            print(point+': '+row['status'],flush=True)
            if row['status']!='HARNESS_COMPLETE':break
        if len(rows)==len(selection(allocation)) and all(r['status']=='HARNESS_COMPLETE' for r in rows):status='COMPLETE'
    finally:
        if created:write_json(output/'RUN_RESULT.json',dict(status=status,finished_utc=now(),
            historical_success_count=18,historical_trial_count=30,attempted_allocations=len(rows),
            completed_trials=sum(r['status']=='HARNESS_COMPLETE' for r in rows),
            observed_success_count=sum(r.get('verdict')=='PASS' for r in rows),
            actual_kit_starts_known=sum(r.get('actual_kit_starts') or 0 for r in rows),
            start_accounting_unknown=any(r.get('actual_kit_starts') is None for r in rows),
            rows=rows,automatic_retries=0,acceptance_evidence=False))
        lock.unlink()
    return 0 if status=='COMPLETE' else 1

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    configure=commands.add_parser('configure',help='Write new machine-local paths; install/launch nothing')
    for key in ('isaac-sim','isaac-lab','policy-support','checkpoint'):configure.add_argument('--'+key,required=True)
    configure.add_argument('--config',default='.local/host.json')
    check=commands.add_parser('check',help='Verify bytes and installed-package metadata without loading a model')
    check.add_argument('--without-checkpoint',action='store_true')
    plan=commands.add_parser('plan',help='Write the fixed allocation plan; no simulator or checkpoint access')
    run=commands.add_parser('run',help='Explicitly execute the bounded scientific reproduction')
    run.add_argument('--execute',action='store_true',required=True)
    for command in (check,plan,run):command.add_argument('--config',default='.local/host.json')
    for command in (plan,run):
        command.add_argument('--output',required=True)
        command.add_argument('--allocation',default='all',choices=['all',*ALLOCATION_ORDER])
    args=parser.parse_args(argv)
    try:
        if args.command=='configure':
            write_json(args.config,dict(schema='hc-reproduction-host/v1',**{key:str(Path(getattr(args,key)).expanduser().resolve()) for key in ('isaac_sim','isaac_lab','policy_support','checkpoint')}))
            print('Configuration saved. No packages installed; no simulator started.');return 0
        config=load_config(args.config)
        if args.command=='plan':
            write_json(args.output,make_plan(config,args.allocation));print('Plan saved; no simulator started.');return 0
        if args.command=='check':
            result=preflight(config,include_checkpoint=not args.without_checkpoint)
            print(json.dumps(result,indent=2));return 0 if result['status']=='PASS' else 1
        return run_batch(config,args.output,args.allocation)
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
        print(f'STOP: {exc}',file=sys.stderr);return 1


"""Explicit portable execute/check entry; --help never imports native modules."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.runtime.host import ALLOCATION_ORDER, load_config, load_json, preflight, write_json
from pipeline.runtime.host import install


def native_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--point", choices=ALLOCATION_ORDER, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--mode", choices=("execute", "adjudicate"), required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    attempt = args.attempt.resolve(strict=True)
    invocation = load_json(attempt / "invocation.json")
    if (invocation.get("r4_live_point") != args.point
            or Path(invocation["output_directory"]).resolve() != attempt / "capture"
            or invocation.get("timeout_ms") != 1200000 or invocation.get("run_type") != "HARNESS"
            or invocation.get("execution_approved") or invocation.get("evidence_authority")
            or invocation.get("stage3_learned") is not True or invocation.get("stage3_witness") is not True
            or invocation.get("envelope_v2") is not True or invocation.get("envelope_protocol") != "v3c"):
        raise ValueError("portable_invocation_scope")
    prefix = "native" if args.mode == "execute" else "checker"
    # Full byte verification, including BEST, occurs before any model load.
    checked = preflight(config, include_checkpoint=True)
    write_json(attempt / (prefix + "_preflight.json"), checked)
    recorder, adaptation = install(config, args.point, attempt, invocation, checked)
    write_json(attempt / (prefix + "_host_adapter.json"), adaptation)
    if args.mode == "adjudicate":
        result = recorder.adjudicate(attempt)
        write_json(attempt / "attempt_result.json", result)
        return 0 if result.get("technical_clean") is True else 1
    return recorder.main(["--execute", "--output-dir", str(attempt / "capture/experiment"),
        "--journal", str(attempt / "capture/structured/events.jsonl"), "--seed", "42",
        "--geometric-mode", "--run-type", "HARNESS", "--stage3-learned",
        "--stage3-witness", "--r4-live-point", args.point, "--envelope-v3c"])

if __name__ == "__main__":
    if sys.argv[1:2] == ["--native"]:
        raise SystemExit(native_main(sys.argv[2:]))
    raise SystemExit(main())
