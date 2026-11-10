#!/usr/bin/env python3
"""Matched-budget optimizer/time ablations; no intermediate recipe selection."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from training.m3 import atomic_json,sha
from training.m5_data import GROUPS
from training.m5_review_statistics import source_clusters
from scripts.analyze_m5_physical_milestone import bootstrap_difference


def read(path):return json.loads(path.read_text())


def trace(run,start,end):
    rows={}
    with (run/'sample_trace.jsonl').open() as f:
        for line in f:
            row=json.loads(line);step=row['step']
            if step>end:break
            if start<step:
                if step in rows:raise ValueError('Duplicate sampling update')
                rows[step]=row['indices']
    if set(rows)!=set(range(start+1,end+1)):raise ValueError('Incomplete sampling prefix')
    return rows


def learning_rates(run,start,end):
    rows={}
    with (run/'metrics.jsonl').open() as f:
        for line in f:
            row=json.loads(line)
            if start<row.get('step',-1)<=end and 'learning_rate' in row:rows[row['step']]=row['learning_rate']
    if not rows:raise ValueError('Missing learning-rate evidence')
    return rows


def load(run,step):
    cfg=read(run/'config.json');registered=read(run/'run_config.json');folder=run/'evaluations'/f'generation_full_{step:06d}'
    report=read(folder/'report.json');rows=read(folder/'per_sample.json');checkpoint=run/'checkpoints'/f'step_{step:06d}.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['step']!=step or state['config']!=cfg or report.get('identity')!=state['identity'] or report['identity']!=registered['identity']:
        raise ValueError('Checkpoint/evaluation/configuration lineage differs')
    if report['step']!=step or report['weights']!='ema' or set(report['groups'])!=set(GROUPS) or set(report['conditions'])!={'native','heading0'}:
        raise ValueError('Unexpected evaluation scope')
    if report['seeds']!=[0,42,1234] or report['events']!=300 or report['generated_clips']!=1800 or len(rows)!=1800 or report['ddim_steps']!=100 or report['guidance']!=2.:
        raise ValueError('Incomplete or unmatched generation battery')
    if registered.get('engineering_fixture') or cfg.get('engineering_fixture'):raise ValueError('Engineering fixtures cannot rank formal models')
    groups=list(report['groups'].values());denoising=read(run/'evaluations'/f'denoising_{step:06d}.json')['ema']
    metrics={'mean_fk_self_m':float(np.mean([g['mean_fk_inconsistency_m'] for g in groups])),
      'pooled_projection_p95_m':report['pooled_projection_node_p95_m'],
      'macro_mean_clip_projection_p95_m':float(np.mean([g['mean_projection_node_p95_m'] for g in groups])),
      'max_penetration_m':max(g['max_penetration_m'] for g in groups),'max_projection_node_error_m':max(g['max_projection_node_error_m'] for g in groups),
      'mean_deep_frame_fraction':float(np.mean([g['mean_penetration_over_5cm_fraction'] for g in groups])),
      'mean_rotation_exceedance_fraction':float(np.mean([g['mean_rotation_threshold_exceed_fraction'] for g in groups])),
      'core_passed_both_headings':all(v['core_amplitude_passed'] for v in report['conditions'].values()),
      'per_heading_length_core':{h:{length:v['core_amplitude_passed'] for length,v in value['length_strata'].items()} for h,value in report['conditions'].items()},
      'common_norm6_validation':denoising['macro_loss'],'supervised_fk_validation':denoising['parts']['fk'],
      'checkpoint_sha256':sha(checkpoint),'all_finite':report['all_finite'],'all_valid_qpos':report['all_valid_qpos']}
    result={'config':cfg,'report':report,'rows':rows,'registered':registered,'rngs':state['rngs'],
      'optimizer_group':state['optimizer']['param_groups'][0],'metrics':metrics,'optimizer_kind':state.get('optimizer_kind','adamw-fp32-v1')}
    del state;return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--kind',choices=['optimizer','uniform_time'],required=True)
    p.add_argument('--step',type=int,required=True);a=p.parse_args();root=a.experiment.resolve()
    kind={'optimizer':('adam_atan2','O1_small_raw7_atan2',0),'uniform_time':('uniform_time','U1_small_raw7_uniform',5000)}[a.kind]
    directory,name,start=kind;protocol_path=root/'auxiliary_protocols'/directory/'protocol.json';protocol=read(protocol_path)
    if protocol['run']!=name or protocol['control']!='C_small_raw7':raise ValueError('Registered comparison changed')
    if a.kind=='uniform_time' and a.step!=20000:raise ValueError('Uniform-time study has a fixed 20k endpoint')
    if a.kind=='optimizer' and a.step not in [10000,20000,30000,40000]:raise ValueError('Use registered full-generation milestones')
    candidate_run=root/name;control_run=root/'C_small_raw7';candidate=load(candidate_run,a.step);control=load(control_run,a.step)
    cfg=candidate['config'];base=control['config'];differences={k:[base.get(k),cfg.get(k)] for k in set(cfg)|set(base) if base.get(k)!=cfg.get(k)}
    common={'stage','launch_code_sha256'}
    allowed=common|({'optimizer_a','optimizer_b','optimizer_betas','optimizer_name','optimizer_weight_decay'} if a.kind=='optimizer' else
      {'engineering_fixture','high_noise_fraction','legality_weight','lr_schedule_origin_step','minimum_steps','stage_start_step','training_stop_step'})
    if set(differences)-allowed:raise ValueError('Additional factors differ: '+str(set(differences)-allowed))
    for key in ['m2_fingerprint','physical_input_fingerprint','training_view_fingerprint','text_config_fingerprint','subset_sha256','stats_sha256']:
        if candidate['report']['identity'][key]!=control['report']['identity'][key]:raise ValueError('Input contract differs: '+key)
    if sha(candidate_run/'config.json')!=protocol['configuration_sha256']:raise ValueError('Candidate configuration differs from protocol')
    for key,value in base['launch_code_sha256'].items():
        if cfg['launch_code_sha256'][key]!=value:raise ValueError('Shared training code changed: '+key)
    lineage=read(candidate_run/'lineage.json')
    if a.kind=='optimizer':
        if candidate['registered']['initial_model_sha256']!=control['registered']['initial_model_sha256']:raise ValueError('Initial weights differ')
        if lineage['parent_sha256']!=protocol['initial_checkpoint_sha256'] or sha(Path(protocol['initial_weights']))!=lineage['parent_sha256']:raise ValueError('Initial checkpoint changed')
        if cfg['optimizer_name']!=protocol['optimizer'] or candidate['optimizer_kind']!=protocol['optimizer']:raise ValueError('Optimizer implementation differs')
        for key in ['betas','weight_decay']:
            if candidate['optimizer_group'][key]!=control['optimizer_group'][key]:raise ValueError('Optimizer hyperparameters differ: '+key)
        if cfg['optimizer_a']!=1. or cfg['optimizer_b']!=1.:raise ValueError('Unexpected angular scaling')
    else:
        done=read(candidate_run/'COMPLETED.json')
        if done['step']!=20000 or sha(candidate_run/'checkpoints/latest.pt')!=done['terminal_checkpoint_sha256']:raise ValueError('Uniform endpoint not finalized')
        if lineage['parent_sha256']!=protocol['parent_sha256'] or sha(Path(protocol['parent_checkpoint']))!=lineage['parent_sha256'] or not lineage['optimizer_rng_preserved']:
            raise ValueError('Uniform-time fork does not preserve the registered parent state')
        if cfg['high_noise_fraction']!=0. or base['high_noise_fraction']!=.5 or cfg['lr_schedule_origin_step']!=0 or cfg['max_steps']!=40000:
            raise ValueError('Time policy or learning-rate schedule differs')
        if any(cfg.get(k,0.) for k in ['consistency_weight','axis_weight','ground_weight','legality_weight']):raise ValueError('Uniform-time study cannot enable extra physical losses')
    draws=trace(candidate_run,start,a.step)
    if draws!=trace(control_run,start,a.step):raise ValueError('Matched sample draws diverged')
    lr=learning_rates(candidate_run,start,a.step)
    if lr!=learning_rates(control_run,start,a.step):raise ValueError('Matched learning-rate records diverged')
    matching={k:torch.equal(candidate['rngs'][k],control['rngs'][k]) for k in ['batch_rng','noise_rng','text_rng','heading_rng','high_noise_rng']}
    if not all(v for k,v in matching.items() if k!='high_noise_rng'):raise ValueError('Data/noise/dropout/yaw streams diverged')
    if matching['high_noise_rng']!=(a.kind=='optimizer'):raise ValueError('Unexpected high-noise replacement RNG history')
    clusters=source_clusters(read(control_run/'subset.json')['val'])
    result={'scope':'Matched-budget development evidence from one training seed. Fixed terminal budgets remain unchanged; no interim recipe selection or independent-training replication claim.',
      'kind':a.kind,'step':a.step,'candidate':name,'control':'C_small_raw7','configuration_differences':differences,
      'metrics':{'candidate':candidate['metrics'],'control':control['metrics']},'rng_states_equal':matching,'sample_streams_equal':True,
      'sampling_prefix_sha256':hashlib.sha256(json.dumps(draws,sort_keys=True).encode()).hexdigest(),'matching_logged_learning_rates':len(lr),
      'paired_candidate_minus_control':{key:bootstrap_difference(candidate['rows'],control['rows'],clusters,key) for key in ['fk_inconsistency_m','projection_node_p95_m']},
      'relative_candidate_improvement':{key:1-candidate['metrics'][key]/control['metrics'][key] for key in ['mean_fk_self_m','pooled_projection_p95_m']},
      'script_sha256':sha(__file__),'protocol_sha256':sha(protocol_path),'M5_complete':False,'final_audit_model_scores_read':False}
    atomic_json(root/'diagnostics'/f'{a.kind}_{a.step:06d}.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
