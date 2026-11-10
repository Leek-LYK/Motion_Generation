"""Concrete M5 matrix configurations and conservative numerical routing gates."""
import json
from pathlib import Path
from training.m3 import ROOT

TINY={'latent_dim':256,'ff_size':1024,'num_layers':4,'num_heads':4}
SMALL={'latent_dim':512,'ff_size':2048,'num_layers':8,'num_heads':8}


def configuration(stage,model=TINY,seed=3101,start=0,end=40000,loss='norm7',microbatch=16,fk_weight=1.,anchor=False):
    parent=json.loads((ROOT/'runs/m4_research_20261103/R3_D_high_noise/config.json').read_text())
    cfg=dict(parent,stage=stage,model=model,seed=seed,stage_start_step=start,max_steps=end,minimum_steps=end,
      batch_size=64,microbatch=microbatch,learning_rate=2e-5,minimum_learning_rate=2e-6,
      scheduler='cosine',warmup_steps=1000 if start==0 else 500,precision='bfloat16',
      loss_objective=loss,fk_weight=fk_weight,loss_mode='smooth_l1',legacy_exact_update=False,
      sampling='stratified',cache_events=2048,high_noise_after=5000,high_noise_fraction=.5,
      high_noise_start=800,eval_every=1000,generation_every=2000,full_generation_every=10000,
      checkpoint_every=5000,log_every=50,eval_batch_size=16,ddim_steps=100,full_seeds=[0,42,1234],
      objective='M5 '+loss+'; see frozen m5_losses.py',time_sampling='uniform until global update 5000, then 50% high-noise replacement',
      postprocess_training=False)
    if anchor:
        cfg.update(model=TINY,seed=parent['seed'],batch_size=16,microbatch=16,precision='float32',
          learning_rate=parent['learning_rate'],scheduler='constant',sampling='legacy',cache_events=1024,
          high_noise_after=0,legacy_exact_update=loss=='norm6',time_sampling=parent['time_sampling'])
    return cfg


def numerical_ok(report):return report['all_finite'] and report['all_valid_qpos']


def core_passed(report):
    return numerical_ok(report) and all(v['core_amplitude_passed'] for v in report['conditions'].values())


def no_core_regression(candidate,control,stratum=None,tolerance=.05):
    reasons=[]
    for condition,values in candidate['conditions'].items():
        other=control['conditions'][condition]
        if stratum:
            values=values['length_strata'].get(stratum);other=other['length_strata'].get(stratum)
            if not values or not other:return False,['Missing length stratum']
        for group in ('walk','jog','jump','raise_arms'):
            a=values['groups'].get(group,{}).get('noncollapsed_fraction');b=other['groups'].get(group,{}).get('noncollapsed_fraction')
            if a is None or b is None or a+1e-12<b-tolerance:reasons.append(f'{condition}/{group}: {a} vs {b}')
    return not reasons,reasons


def fk_mean(report):
    return sum(g['mean_fk_inconsistency_m'] for g in report['groups'].values())/len(report['groups'])


def final_geometry_passed(report):
    return core_passed(report) and all(g['p95_clip_max_penetration_m']<=.05 and g['max_penetration_m']<=.10
      and g['mean_penetration_over_5cm_fraction']<=.01 and g['mean_projection_node_p95_m']<=.03
      and g['max_projection_node_error_m']<=.10 and g['mean_rotation_threshold_exceed_fraction']<=.001
      for g in report['groups'].values())
