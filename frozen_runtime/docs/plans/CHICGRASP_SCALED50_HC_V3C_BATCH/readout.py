"""HC saved receipts through the published HB v3c assessor; no forecasts or launches."""
from pathlib import Path
from collections import Counter
import importlib.util,json
ROOT=Path(__file__).resolve().parents[3];DOC=Path(__file__).resolve().parent
OUT=ROOT/'outputs/scaled50_hc_v3c_batch/attempt001'
spec=importlib.util.spec_from_file_location('published_hb_v3c_readout',ROOT/'docs/plans/CHICGRASP_SCALED50_HB_V3C_BATCH/readout.py')
hb=importlib.util.module_from_spec(spec);spec.loader.exec_module(hb)
load,save,ref,lines,established=hb.load,hb.save,hb.ref,hb.lines,hb.established

def summary():
 rows=[]
 for frozen in load(DOC/'POSITIONS.json')['attempts']:
  point=frozen['allocation'];attempts=sorted((OUT/point).glob('native*/attempt_result.json'));valid=[]
  for ap in attempts:
   attempt=ap.parent;cached=attempt/'HC_READOUT.json'
   try:
    if cached.exists():
     r=load(cached)
     assert ref(ap)==r['independent_result'] and ref(attempt/'capture/experiment/result.json')==r['raw_result'] and ref(attempt/'capture/process_result.json')==r['process_receipt']
    else:
     r=hb.assess(frozen,attempt,save_readout=False)
     r.update(local_round=frozen['local_round'],continued_round=frozen['continued_round'])
     b=load(attempt/'invocation.json')['r4_live_binding'];r['identities']['hc_manifest']=b['hc_manifest']
     history=r.get('attachment_history');source=attempt/'capture/experiment/owner_before_cleanup.json'
     r['attachment_history']=dict(representation='HASHED_SAVED_SOURCE_REFERENCE',source=ref(source),json_pointer='/event_evidence/records',record_count=len(history) if history is not None else None)
     save(cached,r)
    if established(r):valid.append(r)
   except Exception as exc:r=dict(allocation=point,round=frozen['round'],position=frozen['position'],SCORE_EVIDENCE_STATUS='NOT_ESTABLISHED',V3C_VERDICT='UNKNOWN',readout_error=repr(exc))
  if len(valid)>1:raise RuntimeError('Multiple scored outcomes for one HC allocation; no selection allowed')
  if valid:r=valid[0]
  elif not attempts:r=dict(allocation=point,round=frozen['round'],local_round=frozen['local_round'],continued_round=frozen['continued_round'],position=frozen['position'],SCORE_EVIDENCE_STATUS='NOT_RUN',V3C_VERDICT='NOT_RUN')
  r['all_native_attempt_ids']=[str(p) for p in sorted((OUT/point).glob('native*')) if p.is_dir()]
  rows.append(r)
 v=hb.classify(rows);v.update(rows=rows,HC_SUCCESSES=v['V3C_SUCCESSES'],HEADLINE='ten previously untested positions, 30 trials, temporary-attachment protocol',DEFINITION_ID='V1_WINDOWS_WITH_EARLIEST_DEMO_ATTACHMENT',COMPARISON_TYPE='PRESPECIFIED_NEW_POSITION_EVALUATION')
 notes=[load(p) for p in (ROOT/'outputs/attachment_harness_notes').glob('*.json')];own=[n for n in notes if n.get('task')=='CHICGRASP_SCALED50_HC_V3C_BATCH']
 for key,source in [('ACTUAL_KIT_STARTS','actual_kit_starts'),('EXEMPT_PRE_ACTION_KIT_STARTS','exempt_pre_action_kit_starts'),('CONSUMED_KIT_STARTS','consumed_kit_starts')]:v[key]=None if any(n.get(source) is None for n in own) else sum(n[source] for n in own)
 daily=[n for n in notes if str(n.get('invocation_local_time','')).startswith('2026-10-25')]
 v['DAILY_ACTUAL_STARTS']=None if any(n.get('actual_kit_starts') is None for n in daily) else sum(n['actual_kit_starts'] for n in daily)
 v.update(ACTUAL_HC_STARTS=v['ACTUAL_KIT_STARTS'],EXEMPT_PRE_ACTION_STARTS=v['EXEMPT_PRE_ACTION_KIT_STARTS'],CONSUMED_HC_ATTEMPTS=v['CONSUMED_KIT_STARTS'],
  INFRASTRUCTURE_REPLACEMENTS=sum(max(0,sum(n.get('consumed_kit_starts',0) for n in own if n.get('r4_live_point')==p)-1) for p in [r['allocation'] for r in rows]),
  UNIQUE_SCORED_ALLOCATIONS=v['PRIMARY_VERDICTS_ESTABLISHED'],FIRST_FAILURE_GATE_COUNTS=dict(Counter(r.get('FIRST_V3C_FAILURE_GATE') for r in rows if established(r) and r.get('V3C_VERDICT')=='FAIL')),
  TAIL_RELEASE_COUNT=sum(r.get('TAIL_RELEASE') is True for r in rows),TAIL_TECHNICAL_COUNT=sum(r.get('TAIL_TECHNICAL') is True for r in rows),
  POLICY_PREDICTIONS=sum(len(lines(p)) for p in OUT.glob('*/native*/capture/experiment/stage3_inferences.jsonl')),
  CONSTRUCTOR_SHAPE_FORWARDS=sum(len(load(p).get('constructor_receipts',[])) for p in OUT.glob('*/native*/capture/experiment/stage3_checkpoint.json')),
  V1_SHADOW_MODE='NONINTERVENING_V1_SHADOW_ON_V3C_DYNAMICS',V1_SHADOW_EVALUABLE_COUNT=sum(r.get('v1_shadow',{}).get('status') in ('PASS','FAIL') for r in rows),V1_SHADOW_NOT_OBSERVED_COUNT=sum(r.get('v1_shadow',{}).get('status')=='NOT_OBSERVED' for r in rows),V1_SHADOW_NOT_COMPUTED_COUNT=sum(r.get('v1_shadow',{}).get('status')=='NOT_COMPUTED_OPTIONAL' for r in rows),
  UNRUN_OR_UNADJUDICATED_ALLOCATIONS=[r['allocation'] for r in rows if not established(r)],ADMIN_REPAIR_CYCLES=load(OUT/'PREPARATION.json')['admin_cycles_used'],
  OPTIMIZER_UPDATES=0,NEW_TRAINING_RECORDINGS=0,EXPERT_PILOT_STARTS=0,DEMO_ONLY_STARTS_UNDER_THIS_GRANT=0)
 v['per_position']={p:dict(successes=sum(established(r) and r['V3C_VERDICT']=='PASS' for r in rows if r['position']['position_id']==p),planned=3,established=sum(established(r) for r in rows if r['position']['position_id']==p)) for p in [f'HC{i:02d}' for i in range(1,11)]}
 v['HC_3_OF_3_POSITIONS']=[p for p,x in v['per_position'].items() if x['successes']==x['established']==3]
 v['cohorts']={c:dict(count=sum(r.get('cohort')==c for r in rows),successes=sum(r.get('cohort')==c and r['V3C_VERDICT']=='PASS' and established(r) for r in rows)) for c in ['108','120','132','144','OTHER','NO_ANCHOR']}
 for c in ('108','120','132','144','OTHER'):v['CLOSE'+('_OTHER' if c=='OTHER' else c)+'_COUNT']=v['cohorts'][c]['count']
 v['NO_ANCHOR_COUNT']=v['cohorts']['NO_ANCHOR']['count']
 v['NATIVE_CHECKER_VERDICT_MISMATCHES']=sum(len((r.get('score_agreement') or {}).get('verdict_mismatches',[])) for r in rows)
 v['BOUNDARY_ROW_COUNT']=sum((r.get('score_agreement') or {}).get('BOUNDARY_ROW_COUNT',0) for r in rows)
 v['native_process_seconds']=sum(load(p).get('duration_seconds',0) for p in OUT.glob('*/native*/capture/process_result.json'))
 v['REMAINING_FILMING_START_CAP']=None if v['DAILY_ACTUAL_STARTS'] is None else min(12,72-v['DAILY_ACTUAL_STARTS'])
 save(DOC/'PROGRESS.json',v);return v

if __name__=='__main__':
 v=summary();print(json.dumps({k:v[k] for k in ('BATCH_STATUS','HC_SUCCESSES','PRIMARY_VERDICTS_ESTABLISHED','POLICY_PREDICTIONS','ACTUAL_HC_STARTS')}))
