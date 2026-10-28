"""Sequential owner-authorized dispatch through existing bounded launcher only."""
from pathlib import Path
import subprocess,datetime,time,json,sys
from readout import ROOT,DOC,OUT,load,save,summary
PW='C:/Program Files/PowerShell/7/pwsh.exe'
clock=load(DOC/'CLOCK.json');release=load(DOC/'RELEASE.json')
frozen={r['allocation']:r for r in load(DOC/'POSITIONS.json')['attempts']}
positions=[frozen[p] for p in release['allocation_order']]
if __name__=='__main__':
 if sys.argv[1:]!=['--execute-released-batch']:raise SystemExit('Explicit batch dispatch flag required')
 preparation=load(OUT/'PREPARATION.json');expected=preparation['executed_commit']
 assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==expected
 freeze=load(OUT/'FREEZE_PUBLICATION.json')
 assert freeze['commit']==subprocess.check_output(['git','-C',str(ROOT),'rev-parse','chicgrasp-scaled50-hc-freeze^{commit}'],text=True).strip()
 assert freeze['positions_sha256']=='6b72af4c8f822d4db52d20c89516c0dfb7a0ee36d0512c6bcd6e0132c8b50cee'
 for row in positions:
  point=row['allocation'];attempt=OUT/point/preparation.get('attempt_numbers',{}).get(point,'native001')
  if attempt.exists():
   # Receipt-based orchestration recovery never repeats a consumed native start.
   native=attempt/'capture/experiment/result.json';pr=attempt/'capture/process_result.json'
   if not (native.exists() and pr.exists()):raise SystemExit('Existing incomplete attempt: reconcile owned process before dispatch')
   if not load(pr).get('owned_tree_gone_verified'):raise SystemExit('Existing owned tree not proven drained')
   if not (attempt/'attempt_result.json').exists():
    from chicgrasp_policy.r4_live_check import adjudicate
    save(attempt/'attempt_result.json',adjudicate(attempt))
   value=summary();current=next(x for x in value['rows'] if x['allocation']==point)
   if not current.get('technical_clean') or not current.get('PRIMARY_VERDICT_ESTABLISHED'):
    raise SystemExit('Existing unresolved allocation; no automatic native replay')
   print('REUSED '+point,flush=True);continue
  if time.time()+1385>clock['absolute_deadline_unix']:raise SystemExit('Fixed batch deadline cannot fit native/checker budget')
  suffix='' if not (OUT/(point+'_launcher.log')).exists() else '_admin'+str(load(OUT/'PREPARATION.json')['admin_cycles_used'])
  log=OUT/(point+suffix+'_launcher.log')
  command=[PW,'-NoLogo','-NoProfile','-NonInteractive','-File',str(ROOT/'scripts/policy/launch_attachment_capture.ps1'),'-AttemptDirectory',str(attempt),'-GeometricMode','-Stage3Learned','-Stage3Witness','-EnvelopeV3C','-R4LivePoint',point,'-RunType','HARNESS','-Purpose','CHICGRASP_SCALED50_HC_V3C_BATCH']
  save(OUT/(point+suffix+'_dispatch.json'),dict(command=command,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),executed_commit=expected))
  print('DISPATCH '+point,flush=True)
  with log.open('xb') as handle:
   process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
   while process.poll() is None:
    save(OUT/'RUNNING.json',dict(point=point,launcher_pid=process.pid,status='RUNNING',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    time.sleep(10)
  value=summary();current=next(x for x in value['rows'] if x['allocation']==point)
  print(json.dumps(dict(point=point,launcher_exit=process.returncode,verdict=current.get('V3C_VERDICT'),close=current.get('scheduled_anchor'),N=(current.get('event_schedule') or {}).get('total_actions'),technical_clean=current.get('technical_clean'),successes=value['V3C_SUCCESSES'],established=value['PRIMARY_VERDICTS_ESTABLISHED'])),flush=True)
  if not current.get('technical_clean') or not current.get('owned_tree_gone_verified') or not current.get('PRIMARY_VERDICT_ESTABLISHED'):
   save(OUT/'RUNNING.json',dict(point=point,status='STOPPED_FOR_REVIEW',launcher_exit=process.returncode,readout_error=current.get('readout_error')))
   raise SystemExit('Unresolved technical/evidence/process condition; further launches stopped')
  if value['POLICY_PREDICTIONS']>680 or value['ACTUAL_KIT_STARTS']>33 or value['DAILY_ACTUAL_STARTS']>64:raise SystemExit('Count cap exceeded')
 save(OUT/'RUNNING.json',dict(status='NATIVE_BATCH_COMPLETE_PENDING_PUBLICATION',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
