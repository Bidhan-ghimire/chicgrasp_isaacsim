"""Saved-evidence HB v3c readout, adapted from published v2 continuation. No launch, model load or prediction."""
from pathlib import Path
from collections import Counter
import json,sys,hashlib,csv,os,tempfile
ROOT=Path(__file__).resolve().parents[3];DOC=Path(__file__).resolve().parent
OUT=ROOT/'outputs/scaled50_hb_v3c_batch/attempt001'
sys.path.insert(0,str(ROOT/'src'))
def established(r):return r.get('PRIMARY_VERDICT_ESTABLISHED',r.get('SCORE_EVIDENCE_STATUS')=='ESTABLISHED')
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []
def save(p,v):
 p=Path(p);data=(json.dumps(v,indent=2,allow_nan=False)+'\n').encode('utf-8')
 # Publish complete bytes without truncating the currently readable receipt.
 # A failed replacement preserves both the old receipt and the pending bytes.
 with tempfile.NamedTemporaryFile(prefix=p.name+'.pending.',dir=p.parent,delete=False) as f:
  pending=Path(f.name);f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(pending,p)
def ref(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def classify(rows):
 count_established=sum(established(r) for r in rows)
 successes=sum(established(r) and r.get('V3C_VERDICT')=='PASS' for r in rows)
 full=len(rows)==30 and count_established==30
 return dict(BATCH_STATUS='COMPLETE' if full else 'INCOMPLETE',CAMPAIGN_RESULT=('WORKED' if successes>=15 else 'PARTIAL' if successes>=12 else 'FAILED') if full else 'NOT_CLASSIFIED',V3C_SUCCESSES=successes,PLANNED_TRIALS=30,PRIMARY_VERDICTS_ESTABLISHED=count_established,MILESTONE_50=('MET' if successes>=15 else 'NOT_MET') if full else 'PENDING',MILESTONE_60=('MET' if successes>=18 else 'NOT_MET') if full else 'PENDING')
def assess(frozen,attempt,*,save_readout=True,adjudication_path=None):
 cap=attempt/'capture/experiment';a=load(adjudication_path or attempt/'attempt_result.json');native=load(cap/'result.json');score=a.get('end_state') or native.get('experiment_result') or {};process=load(attempt/'capture/process_result.json')
 inv=load(attempt/'invocation.json');b=inv['r4_live_binding']
 assert inv['r4_live_point']==frozen['allocation'] and b['authored_world_xyz_m']==frozen['authored_world_xyz_m'] and b['evaluation_seed']==frozen['evaluation_seed']
 events=lines(cap/'stage3_command_events.jsonl');actions=lines(cap/'stage3_actions.jsonl');accepted={e['global_action'] for e in events if e['event']=='ACCEPTANCE_OBSERVED'}
 closes={s:None for s in ('LEFT','RIGHT','BOTH')}
 for action in actions:
  g=action['global_action'];bits=action['action'][3:]
  if g not in accepted:continue
  for side,yes in [('LEFT',bits[0]==0),('RIGHT',bits[1]==0),('BOTH',bits==[0,0])]:
   if yes and closes[side] is None:closes[side]=dict(action=g,chunk=None if g<50 else (g-50)//12,full_slot=None if g<50 else (g-50)%12+1)
 close=a.get('first_accepted_close')
 if close is None and score.get('event_schedule'):close=score['event_schedule']['close_action']
 cohort=str(close) if close in (108,120,132,144) else 'NO_ANCHOR' if close is None else 'OTHER'
 gate=a.get('admission') or score.get('admission') or {};margins=[]
 for side in gate.get('sides',[]):
  margins.append(dict(side=side['side'],signed_reference_deviation=side['signed_reference_deviation'],envelope_headroom_mm=1000*(.003-side['maximum_absolute_deviation_m']),worst_samples=side.get('witnesses'),jaws=[dict(jaw_id=j['jaw_id'],progress_headroom_mm=None if j['progress'] is None else 1000*(j['progress']['min_m']-.001),shortfall_headroom_mm=1000*(j['shortfall']['min_m']-.001)) for j in side['jaws']]))
 movement=score.get('movement') or {};world=a.get('motion',{}).get('WORLD') or {};rise=world.get('rise') or movement.get('world_rise') or {};room=world.get('WORLD_ROOM') or movement.get('world_room') or {}
 measure=load(cap/'measurements.json');rows=measure['rows'];schedule=score.get('event_schedule') or a.get('event_schedule');owner=load(cap/'owner_before_cleanup.json')
 from chicgrasp_policy.r4_live_check import v1_shadow,event_motion_metrics
 import numpy as np
 shadow=a.get('v1_shadow') or v1_shadow(rows,lines(cap/'witness_transitions.jsonl'),dynamics_protocol='v3c')
 motion=a.get('motion') or event_motion_metrics(rows,np.asarray([x['action'] for x in actions],dtype=np.float32),sum(x['event']=='DISPATCH_REQUESTED' for x in events),close,protocol='v3c')
 counts={}
 if schedule:
  for name,start,end in [('admission',schedule['admission_start'],schedule['decision']),('rise',schedule['confirmation'],schedule['rise_end']),('room',schedule['room_start'],schedule['score_end'])]:
   segment=rows[start:min(len(rows),end+1)];counts[name]=dict(required=end-start+1,available=len(segment),valid=sum(not r.get('issues') and r.get('pose',{}).get('valid') is True for r in segment))
 else:counts={name:dict(required=n,available=0,valid=0,status='NOT_REACHED') for name,n in [('admission',61),('rise',550),('room',240)]}
 checkpoint=load(cap/'stage3_checkpoint.json') if (cap/'stage3_checkpoint.json').exists() else {}
 predictions=lines(cap/'stage3_inferences.jsonl');technical=a.get('technical_clean') is True and a.get('CAPTURE_INTEGRITY')=='PASS' and a.get('PROCESS_COMPLETION')=='PASS'
 result=dict(allocation=frozen['allocation'],position=frozen['position'],round=frozen['round'],base_diffusion_seed=frozen['evaluation_seed'],environment_seed=42,attempt=str(attempt),executed_commit=inv['executed_commit'],identities={k:b.get(k) for k in ('checkpoint_identity','normalizer','scene_profile','approved_protocol','hb_manifest','grant')},first_accepted_close=closes,scheduled_anchor=close,cohort=cohort,event_schedule=schedule,attachment=owner.get('attachment'),attachment_history=owner.get('event_evidence',{}).get('records'),technical_clean=technical,technical_status=a.get('status'),category=a.get('category'),CAPTURE_INTEGRITY=a.get('CAPTURE_INTEGRITY'),PROCESS_QUALIFICATION=a.get('PROCESS_COMPLETION'),issues=a.get('issues'),process_issues=a.get('process_issues'),primary_errors=native.get('primary_errors'),admission=gate,admission_margins=margins,root_tcp_headroom_mm=None if gate.get('distance_m') is None else 1000*(.15-gate['distance_m']),rise=rise,rise_at_h_m=movement.get('rise_at_lift_end_m'),world_room=room,window_rows=counts,signed_margins=movement.get('signed_margins'),separate_motion=motion,v1_shadow=shadow,policy_predictions=len(predictions),constructor_shape_forwards=len(checkpoint.get('constructor_receipts',[])),native_process_seconds=process.get('duration_seconds'),process_exit_code=process.get('exit_code'),owned_tree_gone_verified=process.get('owned_tree_gone_verified'),cleanup_elapsed_ms=process.get('cleanup_elapsed_ms'),raw_result=ref(cap/'result.json'),independent_result=ref(adjudication_path or attempt/'attempt_result.json'),process_receipt=ref(attempt/'capture/process_result.json'))
 result['native_scored_metrics']=native.get('experiment_result')
 result['capture_errors']={k:native.get(k) for k in ('primary_errors','recording_errors','cleanup_errors')}
 result['artifact_storage_bytes']=sum(p.stat().st_size for p in attempt.rglob('*') if p.is_file() and p.name!='HB_READOUT.json')
 result['score_agreement']=a.get('score_agreement')
 result['native_verdict']=native.get('experiment_result',{}).get('V3C_VERDICT')
 result['extension_receipts']=[{k:x.get(k) for k in ('chunk_index','global_start','execute_count','seed','authorized_bound_at_prediction')} for x in predictions]
 result['observed_substeps']=len(rows)
 for key in ('V3C_VERDICT','FIRST_V3C_FAILURE_GATE','SCORE_EVIDENCE_STATUS','TAIL_RELEASE','TAIL_TECHNICAL','PROCESS_COMPLETION','CLEANUP_STATUS'):result[key]=score.get(key,'UNKNOWN')
 result['PRIMARY_VERDICT_ESTABLISHED']=bool(a.get('end_state') and a.get('end_state',{}).get('SCORE_EVIDENCE_STATUS')=='ESTABLISHED' and a.get('CAPTURE_INTEGRITY')=='PASS')
 result['score_origin']='INDEPENDENT_CHECKER' if a.get('end_state') else 'NATIVE_ONLY_NOT_ACCEPTED_BY_REQUIRED_CHECKER'
 if not score:result['SCORE_EVIDENCE_STATUS']='NOT_ESTABLISHED'
 if save_readout:save(attempt/'HB_READOUT.json',result)
 return result

def prediction_reading(predicted, result):
 if predicted=='ADMISSION_MAY_CHANGE_COMPOSITE_UNRESOLVED' or not established(result):return 'UNRESOLVED'
 if predicted in ('FAIL_RISE_TO_PASS','UNCHANGED_PASS'):
  return 'HIT' if result.get('V3C_VERDICT')=='PASS' else 'MISS'
 if predicted=='UNCHANGED_ADMISSION_FAILURE':
  return 'HIT' if result.get('FIRST_V3C_FAILURE_GATE')=='ADMISSION' and result.get('V3C_VERDICT')=='FAIL' else 'MISS'
 raise ValueError('Unknown frozen prediction: '+predicted)

def summary():
 release=load(DOC/'RELEASE.json');predref=release['predictions']
 assert ref(predref['path'])==predref, 'Frozen predictions changed'
 assert ref(release['v2_result']['path'])==release['v2_result'], 'Published native v2 result changed'
 with Path(predref['path']).open(newline='',encoding='utf-8-sig') as f:predictions={r['allocation']:r for r in csv.DictReader(f)}
 v2={r['allocation']:r for r in load(release['v2_result']['path'])['rows']}
 rows=[];paired=[];forecast=[]
 for frozen in load(DOC/'POSITIONS.json')['attempts']:
  point=frozen['allocation'];attempts=sorted((OUT/point).glob('native*/attempt_result.json'))
  if attempts:
   try:
    attempt=attempts[-1].parent;cached=attempt/'HB_READOUT.json'
    if cached.exists():
     r=load(cached)
     assert ref(attempt/'attempt_result.json')==r['independent_result']
     assert ref(attempt/'capture/experiment/result.json')==r['raw_result']
    else:r=assess(frozen,attempt)
   except Exception as exc:r=dict(allocation=point,round=frozen['round'],position=frozen['position'],SCORE_EVIDENCE_STATUS='NOT_ESTABLISHED',V3C_VERDICT='UNKNOWN',readout_error=repr(exc))
  else:r=dict(allocation=point,round=frozen['round'],position=frozen['position'],SCORE_EVIDENCE_STATUS='NOT_RUN',V3C_VERDICT='NOT_RUN')
  rows.append(r)
  old=v2[point];known=established(r) and old['PRIMARY_VERDICT_ESTABLISHED']
  pair=old['V2_VERDICT']+'_TO_'+r['V3C_VERDICT'] if known else 'UNADJUDICATED'
  paired.append(dict(allocation=point,round=frozen['round'],v2_verdict=old['V2_VERDICT'],v3c_verdict=r['V3C_VERDICT'],pair=pair,v2_first_gate=old.get('FIRST_V2_FAILURE_GATE'),v3c_first_gate=r.get('FIRST_V3C_FAILURE_GATE'),first_gate_changed=known and old.get('FIRST_V2_FAILURE_GATE')!=r.get('FIRST_V3C_FAILURE_GATE'),v2_attempt=old['attempt'],v3c_attempt=r.get('attempt'),v2_native_result_sha256=old['raw_result']['sha256'],v2_publication=release['v2_publication']))
  pred=predictions[point];reading=prediction_reading(pred['prediction'],r)
  forecast.append(dict(allocation=point,prediction=pred['prediction'],native_v3c_verdict=r['V3C_VERDICT'],primary_established=established(r),HIT_MISS_UNRESOLVED=reading,prediction_source_sha256=predref['sha256'],original_composite_prediction='UNRESOLVED' if point=='HB06_R0' else pred['prediction'],v2_admission=old.get('admission',{}).get('status'),v3c_admission=r.get('admission',{}).get('status'),admission_changed=known and old.get('admission',{}).get('status')!=r.get('admission',{}).get('status')))
 value=classify(rows);value['rows']=rows;value['paired_rows']=paired;value['prediction_results']=forecast
 notes=[load(p) for p in (ROOT/'outputs/attachment_harness_notes').glob('*.json')];notes=[n for n in notes if n.get('task')=='CHICGRASP_SCALED50_HB_V3C_BATCH']
 for outkey,key in [('ACTUAL_KIT_STARTS','actual_kit_starts'),('EXEMPT_PRE_ACTION_KIT_STARTS','exempt_pre_action_kit_starts'),('CONSUMED_KIT_STARTS','consumed_kit_starts')]:value[outkey]=None if any(n.get(key) is None for n in notes) else sum(n[key] for n in notes)
 value.update(DEFINITION_ID=release['definition_id'],COMPARISON_TYPE='MATCHED_DEVELOPMENT_COMPARISON',V2_PUBLISHED_SUCCESSES='11/30',V2_PUBLISHED_CAMPAIGN_RESULT='FAILED',
  FIRST_FAILURE_GATE_COUNTS=dict(Counter(r.get('FIRST_V3C_FAILURE_GATE') for r in rows if established(r) and r.get('V3C_VERDICT')=='FAIL')),
  TAIL_RELEASE_COUNT=sum(r.get('TAIL_RELEASE') is True for r in rows),TAIL_TECHNICAL_COUNT=sum(r.get('TAIL_TECHNICAL') is True for r in rows),
  POLICY_PREDICTIONS=sum(len(lines(p)) for p in OUT.glob('*/native*/capture/experiment/stage3_inferences.jsonl')),
  CONSTRUCTOR_SHAPE_FORWARDS=sum(len(load(p).get('constructor_receipts',[])) for p in OUT.glob('*/native*/capture/experiment/stage3_checkpoint.json')),
  V1_SHADOW_MODE='NONINTERVENING_V1_SHADOW_ON_V3C_DYNAMICS',V1_SHADOW_EVALUABLE_COUNT=sum(r.get('v1_shadow',{}).get('status') in ('PASS','FAIL') for r in rows),
  V1_SHADOW_NOT_OBSERVED_COUNT=sum(r.get('v1_shadow',{}).get('status')=='NOT_OBSERVED' for r in rows),V1_SHADOW_NOT_COMPUTED_COUNT=sum(r.get('v1_shadow',{}).get('status')=='NOT_COMPUTED_OPTIONAL' for r in rows),
  V1_SHADOW_SUCCESSES=sum(r.get('v1_shadow',{}).get('status')=='PASS' for r in rows),UNRUN_OR_UNADJUDICATED_ALLOCATIONS=[r['allocation'] for r in rows if not established(r)],
  OPTIMIZER_UPDATES=0,NEW_TRAINING_RECORDINGS=0,EXPERT_PILOT_STARTS=0,EXTRA_DIAGNOSTIC_STARTS=0,DEMO_ONLY_STARTS_UNDER_THIS_GRANT=0,HC_EVALUATIONS=0,
  SMOKE_EXCLUDED_FROM_HB='YES',SMOKE_ACCOUNTING=dict(consumed_starts=1,policy_predictions=14),ADMIN_REPAIR_CYCLES=load(OUT/'PREPARATION.json')['admin_cycles_used'],
  PREDICTIONS_SOURCE_UNCHANGED='YES',HB06_R0_ORIGINAL_PREDICTION='UNRESOLVED',UNIQUE_SCORED_ALLOCATIONS=value['PRIMARY_VERDICTS_ESTABLISHED'])
 for name in ('PASS_TO_PASS','FAIL_TO_PASS','PASS_TO_FAIL','FAIL_TO_FAIL','UNADJUDICATED'):value['PAIRED_'+name]=sum(r['pair']==name for r in paired)
 for name in ('HIT','MISS','UNRESOLVED'):value['PREDICTION_'+{'HIT':'HITS','MISS':'MISSES','UNRESOLVED':'UNRESOLVED'}[name]]=sum(r['HIT_MISS_UNRESOLVED']==name for r in forecast)
 value['prediction_misses']=[r for r in forecast if r['HIT_MISS_UNRESOLVED']=='MISS']
 value['per_position']={p:dict(successes=sum(established(r) and r['V3C_VERDICT']=='PASS' for r in rows if r['position']['position_id']==p),planned=3,established=sum(established(r) for r in rows if r['position']['position_id']==p)) for p in [f'HB{i:02d}' for i in range(1,11)]}
 value['cohorts']={c:dict(count=sum(r.get('cohort')==c for r in rows),successes=sum(r.get('cohort')==c and r['V3C_VERDICT']=='PASS' and established(r) for r in rows)) for c in ['108','120','132','144','OTHER','NO_ANCHOR']}
 for c in ('108','120','132','144','OTHER'):value['CLOSE'+('_OTHER' if c=='OTHER' else c)+'_COUNT']=value['cohorts'][c]['count']
 value['NO_ANCHOR_COUNT']=value['cohorts']['NO_ANCHOR']['count']
 value['NATIVE_CHECKER_VERDICT_MISMATCHES']=sum(len((r.get('score_agreement') or {}).get('verdict_mismatches',[])) for r in rows)
 value['BOUNDARY_ROW_COUNT']=sum((r.get('score_agreement') or {}).get('BOUNDARY_ROW_COUNT',0) for r in rows)
 value['native_process_seconds']=sum(load(p).get('duration_seconds',0) for p in OUT.glob('*/native*/capture/process_result.json'))
 value['artifact_storage_bytes']=sum(r.get('artifact_storage_bytes',0) for r in rows)
 save(DOC/'PROGRESS.json',value)
 return value

if __name__=='__main__':
 v=summary();print(json.dumps({k:v[k] for k in ('BATCH_STATUS','V3C_SUCCESSES','PRIMARY_VERDICTS_ESTABLISHED','POLICY_PREDICTIONS','ACTUAL_KIT_STARTS')}))
