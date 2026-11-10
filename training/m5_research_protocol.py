"""Predeclared M5-A research comparisons; old failed admission records stay intact."""
import json
from pathlib import Path
import numpy as np
from training.m3 import ROOT,sha
from training.m5_protocol import core_passed,no_core_regression


def configuration(name,consistency=0.,axis=0.,seed=20261102,norm6_control=False):
    cfg=json.loads((ROOT/'runs/m5_20261103/A1_tiny_fk/config.json').read_text())
    cfg.update(stage=name,seed=seed,stage_start_step=20000,max_steps=30000,minimum_steps=30000,
      consistency_weight=consistency,axis_weight=axis,consistency_smoothing_m=.001,
      objective='norm6 + supervised normalized FK + metre-space self-consistency + G1 hinge-axis defect',
      full_generation_every=5000,checkpoint_every=2500,loss_objective='norm7',fk_weight=1.,legacy_exact_update=False)
    if norm6_control:cfg.update(loss_objective='norm6',fk_weight=0.,legacy_exact_update=True,objective='original normalized six-term control')
    return cfg


def evidence(run,diagnostic=None):
    run=Path(run);generation=json.loads((run/'evaluations/generation_full_030000/report.json').read_text())
    diagnostic=Path(diagnostic) if diagnostic else run/'diagnostics_terminal/report.json'
    detail=json.loads(diagnostic.read_text());groups=generation['groups']
    checksum=sha(run/'checkpoints/latest.pt')
    if detail['checkpoint_sha256']!=checksum or generation['step']!=30000:raise ValueError('Evidence/checkpoint mismatch')
    complete=json.loads((run/'COMPLETED.json').read_text())
    if complete['terminal_checkpoint_sha256']!=checksum:raise ValueError('Completed checkpoint changed')
    means={key:float(np.mean([g[key] for g in groups.values()])) for key in
      ('mean_fk_inconsistency_m','mean_projection_node_p95_m','mean_penetration_over_5cm_fraction','mean_rotation_threshold_exceed_fraction')}
    means['max_penetration_m']=max(g['max_penetration_m'] for g in groups.values())
    means['axis_defect']=float(np.mean([r['axis_defect'] for r in detail['saved_generation']['records']]))
    means['geometry_score_m']=generation['geometry_score_m']
    return {'run':str(run),'generation':generation,'denoising':detail['denoising_summary'],'metrics':means}


def compare(candidate,control6,control_fk):
    metrics=candidate['metrics'];base=control6['metrics'];fk=control_fk['metrics']
    amplitude,amplitude_reasons=no_core_regression(candidate['generation'],control6['generation'])
    ratios=[];means=[];norm_ratios=[]
    for heading in ('None','0.0'):
        values=candidate['denoising'][heading];reference=control_fk['denoising'][heading]
        ratios.extend(values[str(t)]['fk_gt_norm']/max(reference[str(t)]['fk_gt_norm'],1e-12) for t in (0,249,499,749,999))
        means.append(float(np.mean([v['fk_gt_norm'] for v in values.values()])/max(np.mean([v['fk_gt_norm'] for v in reference.values()]),1e-12)))
        norm_ratios.append(float(np.mean([v['norm6'] for v in values.values()])/max(np.mean([v['norm6'] for v in reference.values()]),1e-12)))
    guards={'core_generation_both_headings':core_passed(candidate['generation']),'no_core_fraction_regression_over_5pp':amplitude,
      'supervised_fk_mean_not_worse_over_10pct':max(means)<=1.10,'supervised_fk_any_t_not_worse_over_25pct':max(ratios)<=1.25,
      'norm6_validation_not_worse_over_5pct':max(norm_ratios)<=1.05}
    joint={**guards,'historical_self_improvement_at_least_10pct':metrics['mean_fk_inconsistency_m']<=.9*base['mean_fk_inconsistency_m'],
      'projection_not_worse_than_control6_over_5pct':metrics['mean_projection_node_p95_m']<=1.05*base['mean_projection_node_p95_m'],
      'deep_penetration_not_worse_than_control6_over_10pct':metrics['mean_penetration_over_5cm_fraction']<=1.1*base['mean_penetration_over_5cm_fraction']+1e-4,
      'maximum_penetration_under_10cm':metrics['max_penetration_m']<=.10,
      'rotation_exceedance_not_worse_over_10pct':metrics['mean_rotation_threshold_exceed_fraction']<=1.1*base['mean_rotation_threshold_exceed_fraction']+1e-5}
    directions={'consistency':all(guards.values()) and metrics['mean_fk_inconsistency_m']<=.95*fk['mean_fk_inconsistency_m']
      and metrics['mean_projection_node_p95_m']<=1.2*fk['mean_projection_node_p95_m'],
      'axis':all(guards.values()) and metrics['axis_defect']<=.8*fk['axis_defect']
      and metrics['mean_projection_node_p95_m']<=.9*fk['mean_projection_node_p95_m']}
    return {'run':candidate['run'],'metrics':metrics,'joint_gate_passed':all(joint.values()),'checks':joint,
      'directional_evidence':directions,'failed_checks':[k for k,v in joint.items() if not v],
      'core_regression_reasons':amplitude_reasons,'self_improvement_vs_control6':1-metrics['mean_fk_inconsistency_m']/base['mean_fk_inconsistency_m'],
      'supervised_fk_mean_ratios':list(map(float,means)),'largest_supervised_fk_t_ratio':float(max(ratios)),
      'scope':'Exploratory development evidence, not independent final audit or M5 completion'}
