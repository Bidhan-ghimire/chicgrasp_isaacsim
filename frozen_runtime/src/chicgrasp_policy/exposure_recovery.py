"""Restore the existing recovery format without changing the training recipe."""
from __future__ import annotations

import random

import numpy as np

from . import overfit_policy as policy_api
from . import stage2_evaluation as evidence

require = evidence.data.require
IMMUTABLE = ('step','normalization','recipe_identity','source_identity','schedule_identity','seed')


def _metadata(saved,expected=None):
    metadata=saved['metadata'];step=saved['optimizer_step']
    require(isinstance(metadata,dict), 'Recovery metadata must be an object')
    horizon=metadata.get('max_optimizer_steps',10000)
    require(type(horizon) is int and horizon>0 and type(step) is int and 0<step<=horizon,
            'Recovery requires completed updates within the configured horizon')
    if expected is not None:
        require(horizon==expected.get('max_optimizer_steps',10000),
                'Recovery binding differs: max_optimizer_steps')
    for key in IMMUTABLE:
        require(key in metadata,'Recovery metadata missing '+key)
        if expected is not None:
            require(key in expected and metadata[key]==expected[key],'Recovery binding differs: '+key)
    require(metadata['step'] in (1,2) and metadata['seed']==42,'Recovery phase or seed changed')
    require(metadata.get('run_kind')=='counted','Smoke weights cannot initialize counted training or replay')
    for key in ('completed_updates','sampler_position','credited_updates'):
        require(type(metadata.get(key)) is int and metadata[key]==step,'Recovery counter mismatch: '+key)
    rates=metadata.get('learning_rate_positive')
    require(isinstance(rates,list) and len(rates)==step and all(type(value) is bool for value in rates)
            and rates==[False]+[True]*(step-1),'Recovery positive-learning-rate prefix mismatch')
    return metadata,step,rates


def _rng(saved):
    """Validate CPU bookkeeping without advancing any live random generator."""
    import torch
    state=saved['rng'];numpy=state['numpy']
    random.Random(0).setstate(state['python'])
    value=(numpy['name'],np.asarray(numpy['keys'],dtype=np.uint32),numpy['position'],
           numpy['has_gauss'],numpy['cached_gaussian'])
    np.random.RandomState(0).set_state(value)
    torch.Generator(device='cpu').set_state(state['torch_cpu'])
    cuda=state['torch_cuda']
    require(isinstance(cuda,list) and all(isinstance(item,torch.Tensor) and item.dtype==torch.uint8
            and item.ndim==1 and item.numel()>0 for item in cuda),'Recovery CUDA RNG payload invalid')
    require(len(cuda)==(torch.cuda.device_count() if torch.cuda.is_available() else 0),
            'Recovery CUDA generator count changed')
    return state,value


def restore_training(path,*,policy,ema_policy,optimizer,scheduler,ema,expected_metadata):
    """Restore LAST; caller resumes the frozen sampler at completed_updates.

    Model construction precedes this call. RNG restoration is deliberately last,
    after state loading, so reinitialization cannot perturb the retained stream.
    Exceptions propagate to the caller; partial restore is never a saved state.
    """
    import torch
    saved=evidence.load_recovery(path)
    metadata,step,rates=_metadata(saved,expected_metadata)
    require(policy.gate3_config==saved['model_config'] and ema_policy.gate3_config==saved['model_config'],
            'Recovery target architecture differs')
    require(ema.averaged_model is ema_policy,'EMA target object differs')
    control=saved['ema_control'];current={key:value for key,value in vars(ema).items() if key!='averaged_model'}
    require(set(control)==set(current),'EMA control fields changed')
    for key in current:
        if key not in ('optimization_step','decay'):
            require(control[key]==current[key],'EMA recipe differs: '+key)
    require(control['optimization_step']==step,'EMA completed-update mismatch')
    schedule=saved['scheduler_state'];groups=saved['optimizer_state']['param_groups']
    require(schedule.get('last_epoch')==step and schedule.get('_step_count')==step+1,
            'Scheduler completed-update mismatch')
    require(schedule.get('_last_lr')==[group['lr'] for group in groups],
            'Optimizer and scheduler learning rates differ')
    require(len(groups)==len(optimizer.param_groups),'Optimizer group count changed')
    for stored,active in zip(groups,optimizer.param_groups):
        for key in ('betas','eps','weight_decay','foreach','initial_lr'):
            require(stored.get(key)==active.get(key),'Optimizer recipe differs: '+key)
    for state in saved['optimizer_state']['state'].values():
        require(float(state.get('step',-1))==step,'Optimizer completed-update mismatch')
    # The normalizer is a registered part of each model state. Its frozen values
    # must agree with the freshly loaded phase normalization before any mutation.
    norm={key:value for key,value in policy.state_dict().items() if key.startswith('normalizer.')}
    require(bool(norm),'Recovery normalizer state missing')
    for name,value in norm.items():
        for model_state in (saved['model_state'],saved['ema_state']):
            require(name in model_state and torch.equal(value.detach().cpu(),model_state[name]),
                    'Recovery normalizer differs: '+name)
    rng,numpy_rng=_rng(saved)
    device=next(policy.parameters()).device
    require(next(ema_policy.parameters()).device==device,'Recovery model devices differ')
    policy.load_state_dict(saved['model_state'],strict=True)
    ema_policy.load_state_dict(saved['ema_state'],strict=True)
    # The official normalizer rebuilds ParameterDict from the CPU checkpoint.
    # Place all restored model tensors without a dtype conversion, before RNG.
    policy.to(device=device);ema_policy.to(device=device)
    require(all(value.device==device for model in (policy,ema_policy)
                for value in (*model.parameters(),*model.buffers())),
            'Restored model tensors must share the original device')
    optimizer.load_state_dict(saved['optimizer_state'])
    scheduler.load_state_dict(schedule)
    for key,value in control.items():setattr(ema,key,value)
    policy.train();ema_policy.eval()
    random.setstate(rng['python']);np.random.set_state(numpy_rng)
    torch.set_rng_state(rng['torch_cpu'])
    if rng['torch_cuda']:torch.cuda.set_rng_state_all(rng['torch_cuda'])
    return dict(completed_updates=step,learning_rate_positive=list(rates))


def loss_reference(path,restored_step):
    """Read persisted loss rows only; no model construction or inference."""
    import json
    from collections import deque
    tail=deque(maxlen=100);repeats=[]
    with open(path,encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if 'loss' not in row:continue
            tail.append(row)
            if row['step']>restored_step:repeats.append(row)
    require(len(tail)==100,'Last 100 persisted losses unavailable')
    require(all(row['step']==restored_step+i+1 for i,row in enumerate(repeats)),
            'Repeated-loss source is not a contiguous saved sequence')
    return dict(baseline=list(tail),repeats=repeats,restored_step=restored_step,
                precrash_last_step=tail[-1]['step'])


def compare_resume_losses(reference,rows):
    """Owner loss checks; same-step repeat check and first100 new-update backstop."""
    import statistics
    import math
    repeated=[];failures=[]
    for i,row in enumerate(rows[:len(reference['repeats'])]):
        old=reference['repeats'][i]
        difference=abs(row['loss']-old['loss']);bound=.05*abs(old['loss'])
        same_sequence=(row['step']==old['step'] and
            row['sampled_episode_chunks']==old['sampled_episode_chunks'])
        passed=math.isfinite(row['loss']) and difference<=bound and same_sequence
        repeated.append(dict(step=row['step'],previous_loss=old['loss'],resumed_loss=row['loss'],
            absolute_difference=difference,relative_difference=difference/abs(old['loss']) if old['loss'] else None,
            within_5_percent=passed,same_sampler_sequence=same_sequence))
        if not passed:failures.append(f'repeated update {row["step"]} exceeds 5% or sequence differs')
    baseline=[r['loss'] for r in reference['baseline']]
    result=dict(passed=not failures,failures=failures,repeated_updates=repeated,
        repeated_expected=len(reference['repeats']),new_updates_observed=max(0,len(rows)-len(reference['repeats'])),
        baseline_mean=statistics.mean(baseline),baseline_median=statistics.median(baseline),
        first100_new_result='NOT_REACHED')
    if len(rows)>=100:
        result['first100_resumed_beside_precrash']=[dict(previous_step=a['step'],previous_loss=a['loss'],
            resumed_step=b['step'],resumed_loss=b['loss']) for a,b in zip(reference['baseline'],rows[:100])]
    new=rows[len(reference['repeats']):len(reference['repeats'])+100]
    if len(new)==100:
        values=[r['loss'] for r in new]
        mean=statistics.mean(values);median=statistics.median(values)
        for name,value in (('mean',mean),('median',median)):
            base=result['baseline_'+name]
            if not math.isfinite(value) or abs(value-base)>.2*abs(base):
                failures.append(f'first100 new-update {name} differs by more than 20%')
        result.update(first100_new_mean=mean,first100_new_median=median,
            first100_new_range=[new[0]['step'],new[-1]['step']],
            first100_new_result='PASS' if not failures else 'FAIL')
    result['passed']=not failures
    return result


def load_ema(path,device='cuda:0'):
    """Load BEST EMA directly; caller supplies the existing cuda_constructor scope."""
    require(device=='cuda:0','Deployment recovery loader requires cuda:0')
    saved=evidence.load_recovery(path)
    metadata,step,_=_metadata(saved)
    model=policy_api.build_policy(saved['model_config'],device=device)
    model.load_state_dict(saved['ema_state'],strict=True)
    model.to(device);model.eval()
    return model,dict(metadata,optimizer_step=step,weights_kind='EMA')
