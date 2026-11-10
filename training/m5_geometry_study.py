"""Predeclared M5 geometry-study construction and comparisons; no M5 exit claim."""
import json
from pathlib import Path
import numpy as np
from training.m3 import atomic_json,sha,ROOT
from training.m5_protocol import core_passed,no_core_regression,final_geometry_passed

FIELDS=('consistency_weight','axis_weight','ground_weight','legality_weight')
RUNS=('G0_control','G1_consistency','G2_legality','G3_ground')
COMBINATIONS={'G4_consistency_legality':('consistency_weight','legality_weight'),
  'G4_consistency_ground':('consistency_weight','ground_weight'),'G4_legality_ground':('legality_weight','ground_weight'),
  'G4_all':('consistency_weight','legality_weight','ground_weight')}


def build_study(source,output,parent,calibration,code,fixture=False,subset=None):
    source=Path(source).resolve();output=Path(output).resolve();parent=Path(parent).resolve()
    if (output/'activation.json').exists():raise FileExistsError('Study activation already frozen')
    if not calibration.get('passed') or not fixture and not calibration.get('formal_coefficients_usable'):raise ValueError('Formal calibration is not usable')
    if not fixture and (calibration['loss_code_sha256']!=code['training/m5_constraint_losses.py'] or calibration['legality_code_sha256']!=code['training/g1_legality_loss.py']):raise ValueError('Calibrated operator code changed')
    parent_sha=sha(parent)
    if calibration['parent_sha256']!=parent_sha:raise ValueError('Calibration parent differs')
    cfg0=json.loads((source/'config.json').read_text());parent_step=calibration['step']
    if not fixture and (parent_step!=40000 or any(cfg0.get(k,0.) for k in FIELDS)):raise ValueError('Formal study requires an unregularized fixed 40k parent')
    subset=Path(subset or source/'subset.json').resolve();manifest=json.loads(subset.read_text());view=Path(cfg0['training_view']);ready=json.loads(view.read_text())
    keys={'consistency_weight':'consistency','legality_weight':'legality','ground_weight':'ground'}
    low={key:calibration['coefficients'][name]['low'] for key,name in keys.items()}
    if any(v is None or not np.isfinite(v) or v<=0 for v in low.values()):raise ValueError('Invalid calibrated coefficients')
    coefficients={name:{key:0. for key in FIELDS} for name in (*RUNS,*COMBINATIONS)}
    coefficients['G1_consistency']['consistency_weight']=low['consistency_weight'];coefficients['G2_legality']['legality_weight']=low['legality_weight'];coefficients['G3_ground']['ground_weight']=low['ground_weight']
    for name,fields in COMBINATIONS.items():coefficients[name].update({key:low[key] for key in fields})
    target=parent_step+(2 if fixture else 10000);schedule_end=parent_step+(20 if fixture else 20000)
    activation={'schema':'m5-constraint-activation-v1','passed':True,'engineering_fixture':fixture,
      'physical_input_fingerprint':ready['input_fingerprint'],'training_view_fingerprint':ready['fingerprint'],
      'runs':{name:{'parent_sha256':parent_sha,'coefficients':values} for name,values in coefficients.items()}}
    atomic_json(output/'activation.json',activation);hashes={}
    for name,values in coefficients.items():
        cfg=dict(cfg0,stage=name,stage_start_step=parent_step,max_steps=schedule_end,minimum_steps=target,training_stop_step=target,
          lr_schedule_origin_step=parent_step,lr_schedule_end_step=schedule_end,learning_rate=2e-5,minimum_learning_rate=2e-6,
          scheduler='cosine',warmup_steps=2 if fixture else 500,engineering_fixture=fixture,launch_code_sha256=code,
          subset_sha256=sha(subset),constraint_activation_protocol=str(output/'activation.json'),**values)
        if fixture:cfg.update(eval_every=10000,generation_every=10000,full_generation_every=10000,checkpoint_every=10000,
          log_every=1,full_seeds=[42],ddim_steps=10,eval_batch_size=8,cache_events=6)
        path=output/'configs'/(name+'.json');atomic_json(path,cfg);hashes[name]=sha(path)
    plan={'schema':'m5-geometry-study-v1','source_run':str(source),'parent_checkpoint':str(parent),'parent_sha256':parent_sha,
      'parent_step':parent_step,'target_step':target,'lr_schedule_end_step':schedule_end,'subset':str(subset),'subset_sha256':sha(subset),
      'first_wave':list(RUNS),'conditional_combinations':COMBINATIONS,'max_parallel':1 if fixture else 4,
      'new_updates_first_wave':4*(target-parent_step),'maximum_new_updates':5*(target-parent_step),
      'config_sha256':hashes,'code_sha256':code,'engineering_fixture':fixture,'M5_complete':False,
      'scope':'Bounded development research. All endpoints and failed cases retained; no audit release, no automatic D/E or M6 entry.',
      'combination_rule':'After all singles, run at most one combination: exactly the union of >=2 directions that pass their predeclared tests. Restart from the original parent; coefficients unchanged; no trained-weight merging.'}
    atomic_json(output/'protocol.json',plan);return plan


def evidence(run,target):
    run=Path(run);done=json.loads((run/'COMPLETED.json').read_text());cfg=json.loads((run/'config.json').read_text())
    if done['step']!=target or cfg.get('training_stop_step',cfg['max_steps'])!=target:raise ValueError('Incomplete trial budget')
    if sha(run/'checkpoints/latest.pt')!=done['terminal_checkpoint_sha256']:raise ValueError('Terminal checkpoint changed')
    folder=run/'evaluations'/f'generation_full_{target:06d}';r=json.loads((folder/'report.json').read_text());d=json.loads((run/'evaluations'/f'denoising_{target:06d}.json').read_text())['ema']
    if r.get('identity')!=done['identity'] or r['step']!=target:raise ValueError('Evaluation/checkpoint lineage differs')
    groups=list(r['groups'].values());m={key:float(np.mean([g[key] for g in groups])) for key in ['mean_fk_inconsistency_m','mean_projection_node_p95_m','mean_penetration_over_5cm_fraction','mean_rotation_threshold_exceed_fraction']}
    m.update(pooled_projection_node_p95_m=r['pooled_projection_node_p95_m'],max_penetration_m=max(g['max_penetration_m'] for g in groups),
      max_projection_node_error_m=max(g['max_projection_node_error_m'] for g in groups),geometry_score_m=r['geometry_score_m'],
      norm6_validation=d['macro_loss'],supervised_fk_validation=d['parts']['fk'])
    return {'run':str(run),'report':r,'denoising':d,'metrics':m,'checkpoint_sha256':done['terminal_checkpoint_sha256']}


def compare(candidate,control):
    a=candidate['metrics'];b=control['metrics'];r=candidate['report'];base=control['report'];reasons=[]
    good,reasons=no_core_regression(r,base);short_good,short_reasons=no_core_regression(r,base,stratum='short')
    core_short=all(v.get('length_strata',{}).get('short',{}).get('core_amplitude_passed',False) for v in r['conditions'].values())
    new_class_regression=[]
    for heading,value in r['conditions'].items():
        for group in ['turn','squat']:
            x=value['groups'].get(group,{}).get('noncollapsed_fraction');y=base['conditions'][heading]['groups'].get(group,{}).get('noncollapsed_fraction')
            if x is None or y is None or x+.05+1e-12<y:new_class_regression.append(heading+'/'+group)
    common={'all_finite':r['all_finite'],'valid_qpos':r['all_valid_qpos'],'short_core_generation':core_short,
      'short_core_fraction_no_regression':short_good,'new_class_fraction_no_regression':not new_class_regression,
      'common_norm6_validation_guard':a['norm6_validation']<=1.05*b['norm6_validation'],
      'supervised_fk_validation_guard':a['supervised_fk_validation']<=1.1*b['supervised_fk_validation'],
      'supervised_fk_any_t_guard':all(candidate['denoising']['geometry_by_timestep'][str(t)]['fk']<=1.25*control['denoising']['geometry_by_timestep'][str(t)]['fk'] for t in (0,249,499,749,999))}
    directional={'consistency':all(common.values()) and a['mean_fk_inconsistency_m']<=.95*b['mean_fk_inconsistency_m'] and a['pooled_projection_node_p95_m']<=1.2*b['pooled_projection_node_p95_m'],
      'legality':all(common.values()) and a['pooled_projection_node_p95_m']<=.9*b['pooled_projection_node_p95_m'] and a['mean_fk_inconsistency_m']<=1.1*b['mean_fk_inconsistency_m'],
      'ground':all(common.values()) and ((b['mean_penetration_over_5cm_fraction']>1e-4 and a['mean_penetration_over_5cm_fraction']<=.8*b['mean_penetration_over_5cm_fraction']) or (b['max_penetration_m']>.01 and a['max_penetration_m']<=.8*b['max_penetration_m']))
        and a['mean_fk_inconsistency_m']<=1.1*b['mean_fk_inconsistency_m'] and a['pooled_projection_node_p95_m']<=1.2*b['pooled_projection_node_p95_m']}
    joint={**common,'full_core_generation':core_passed(r),'full_core_fraction_no_regression':good,
      'all_six_amplitude_both_headings':all(len(v['groups'])==6 and all(g['amplitude_passed'] for g in v['groups'].values()) for v in r['conditions'].values()),
      'self_improvement_ge10pct':a['mean_fk_inconsistency_m']<=.9*b['mean_fk_inconsistency_m'],
      'projection_no_regression':a['pooled_projection_node_p95_m']<=1.05*b['pooled_projection_node_p95_m'],
      'deep_penetration_no_regression':a['mean_penetration_over_5cm_fraction']<=1.1*b['mean_penetration_over_5cm_fraction']+1e-4,
      'rotation_no_regression':a['mean_rotation_threshold_exceed_fraction']<=1.1*b['mean_rotation_threshold_exceed_fraction']+1e-5,
      'max_penetration_le10cm':a['max_penetration_m']<=.1,'max_projection_le10cm':a['max_projection_node_error_m']<=.1}
    return {'run':candidate['run'],'checkpoint_sha256':candidate['checkpoint_sha256'],'metrics':a,'common_guards':common,'directional_evidence':directional,
      'joint_development_passed':all(joint.values()),'joint_checks':joint,'failed_joint_checks':[k for k,v in joint.items() if not v],
      'full_final_geometry_passed':bool(final_geometry_passed(r) and r['pooled_projection_node_p95_m']<=.03),
      'regression_reasons':reasons,'short_regression_reasons':short_reasons,'new_class_regression':new_class_regression,
      'M5_complete':False,'scope':'Development recipe evidence, not semantic review, full curriculum, independent training seed or final audit.'}


def combination_for(assessments):
    selected=[]
    for run,key in [('G1_consistency','consistency_weight'),('G2_legality','legality_weight'),('G3_ground','ground_weight')]:
        direction={'consistency_weight':'consistency','legality_weight':'legality','ground_weight':'ground'}[key]
        if assessments[run]['directional_evidence'][direction]:selected.append(key)
    if len(selected)<2:return None
    return next(name for name,keys in COMBINATIONS.items() if set(keys)==set(selected))
