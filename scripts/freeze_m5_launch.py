#!/usr/bin/env python3
"""Freeze the verified M5 launch protocol, adjustments and full main-matrix registry."""
import argparse,json,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.m3 import atomic_json,sha,validate_inputs
from training.m5_protocol import configuration,SMALL


def read(p):return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);args=p.parse_args();root=args.experiment.resolve()
    if (root/'protocol/launch.json').exists():raise FileExistsError('Launch already frozen; do not overwrite')
    ready=validate_inputs();data=read(root/'data/READY.json');verify=read(root/'verification/report.json')
    extra=read(root/'verification/additional_checks.json');benchmark=read(root/'benchmark/report.json')
    reference=read(root/'reference_review/agent_review.json');calibration=read(root/'reference_review/metric_calibration.json')
    smoke=read(root/'verification/runner_smoke_final/COMPLETED.json');standalone=read(root/'verification/standalone_evaluation/report.json')
    probe=read(root/'verification/text_probe/report.json')
    for record in (data,verify,extra,benchmark,reference):assert record['passed']
    for name,checksum in verify['code_sha256'].items():assert sha(ROOT/name)==checksum,name
    for name,checksum in benchmark['code_sha256'].items():assert sha(ROOT/name)==checksum,name
    for name,checksum in smoke['identity']['code_sha256'].items():assert sha(ROOT/name)==checksum,name
    assert standalone['all_finite'] and standalone['all_valid_qpos'] and standalone['metrics_sha256']==sha(ROOT/'training/m5_evaluation.py')
    assert standalone['evaluator_sha256']==sha(ROOT/'scripts/evaluate_m5.py')
    assert probe['script_sha256']==sha(ROOT/'scripts/probe_m5_text.py')
    assert reference['events_reviewed']==120 and reference['candidate_model_scores_used_for_selection'] is False
    assert calibration['candidate_outputs_used'] is False
    for name,h in data['manifest_sha256'].items():assert sha(root/'data'/name)==h,name
    m4=read(ROOT/'runs/m4_research_20261103/R3_D_high_noise/COMPLETED.json')['identity']
    assert sha(ROOT/'training/m3.py')==m4['training_code_sha256']
    for name,h in m4['m4_code_sha256'].items():assert sha(ROOT/name)==h,name
    files=sorted(set(list((ROOT/'training').glob('m5*.py'))+list((ROOT/'scripts').glob('*m5*.py'))+
       [ROOT/f for f in ['training/m3.py','training/m4.py','training/m4_research.py','training/g1_qpos.py','training/video.py','data_pipeline/model_input_dataset.py']]))
    hashes={str(f.relative_to(ROOT)):sha(f) for f in files}
    snapshot=root/'source_at_launch'
    for source in files:
        target=snapshot/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    plan=ROOT/'docs/科研进度/20261103_M5_完整训练与实验计划.md';shutil.copyfile(plan,snapshot/'M5_plan_v1.md')
    for name in ['pyproject.toml','uv.lock']:shutil.copyfile(ROOT/name,snapshot/name)
    stages=[('A0','R3 20k full state','anchor',10000),('A1','same R3 parent','anchor',10000),
       ('B0','random Tiny seed3101','short',40000),('B1','random Small seed3101','short',40000),
       ('C','B1 identical initial weights, raw seven-term loss','short',40000),
       ('D0','selected Small 40k','short',20000),('D1','same Small 40k','mixed_small',20000),
       ('E0','D1 60k','mixed_small',20000),('E1','same D1 60k','mixed_large',20000),
       ('F0','matched Tiny full curriculum; reuse B0 only if exact match','matched_curriculum',40000),
       ('F1','independent Small seed3102 full 40k+20k+20k','matched_curriculum',80000),
       ('G','development baselines/review then frozen audit','locked_audit',0)]
    registry=[{'stage':name,'initialization':init,'data_role':role,'new_updates_nominal':updates,'status':'queued',
               'depends_on_previous_stage_gate':True} for name,init,role,updates in stages]
    protocol={'schema':'m5-launch-protocol-v1','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
      'user_authorization':'Formally start the complete M5 main experiment matrix in section 6 sequentially.',
      'm2_fingerprint':ready['fingerprint'],'text_config_fingerprint':ready['text_config_fingerprint'],
      'R3_parent_sha256':sha(ROOT/'runs/m4_research_20261103/R3_D_high_noise/checkpoints/latest.pt'),
      'code_sha256':hashes,'data_manifest_sha256':data['manifest_sha256'],'plan_sha256':sha(plan),
      'actual_data_counts':data['actual_counts'],'data_adjustment':data['adjustment_reason'],
      'microbatch':benchmark['recommended_microbatch'],'effective_batch':64,'small_parameters':56755617,
      'rotation_thresholds_rad':calibration['rotation_thresholds_rad'],'rotation_threshold_formula':calibration['formula'],
      'formal_update_budget_nominal':340000,'extra_matched_tiny_updates_if_raw7_selected':40000,
      'allowed_A1_retry_updates':10000,'other_optional_training_branches_enabled':False,
      'development_sampling_seeds':[0,42,1234],'audit_sampling_seeds':[23017,23073,232026],
      'headings':[None,0.0],'guidance':2.,'ddim_steps':100,
      'automatic_routing':{'A':'FK mean <= 0.9 * control, core generation passes both headings, no core fraction regression > .05; at most one lambda .25 retry',
        'C':'Both candidates trained. Core generation and no >.05 core fraction regression vs Tiny, then minimum declared geometry score. Provisional numerical selection, not semantic acceptance.',
        'D':'Extended-length core gate passes and short core fraction does not regress > .05 vs continuation control',
        'E':'Core gate + no > .05 core regression + lower geometry score than matched-data continuation; otherwise stop',
        'G':'Two independent human reviewers and absolute/relative geometry gates before audit release. Audit reviewed separately. Never auto-start M6.'},
      'geometry_score_definition':'Unweighted class macro mean of FK position inconsistency + p95 per-clip maximum foot penetration + mean per-clip p95 projection node distance; all in metres.',
      'final_geometry_thresholds':{'p95_clip_max_penetration_m':.05,'max_penetration_m':.10,'mean_penetration_over_5cm_fraction':.01,
        'mean_projection_node_p95_m':.03,'max_projection_node_error_m':.10,'rotation_threshold_exceed_fraction':.001},
      'relative_improvement_named_metrics':['mean_fk_inconsistency_m','p95_clip_max_penetration_m'],'relative_improvement_required':.30,
      'semantic_acceptance':{'route':'blind paired preference lower clustered 95% CI > .5','main_action_macro_accuracy_min':.8,
         'complete_jump_rating_fraction_min':.8,'independent_raters':2,'events_per_class_per_small_model':20},
      'implementation_limits':['Numerical routing lets the authorized matrix run unattended; semantic selection remains provisional until G. Semantic failure stops acceptance and may require further research.',
        'No G1-compatible TMR evaluator has been validated for this launch; no claimed TMR/FID scores.',
        'Static crouch and small hops remain represented; amplitude and flight diagnostics are not semantic success.',
        'Final data expansion is 3500 to 4500 actual events, not the originally proposed 10000 to 30000.'],
      'evidence_sha256':{name:sha(root/name) for name in ['verification/report.json','verification/additional_checks.json',
        'verification/runner_smoke_final/COMPLETED.json','verification/standalone_evaluation/report.json','verification/text_probe/report.json',
        'benchmark/report.json','reference_review/agent_review.json','reference_review/metric_calibration.json','data/READY.json']}}
    atomic_json(root/'experiment_registry.json',registry);atomic_json(root/'protocol/launch.json',protocol)
    atomic_json(root/'queue_status.json',{'phase':'preflight_complete_ready_to_launch','M5_complete':False,'protocol_sha256':sha(root/'protocol/launch.json')})
    print(json.dumps({'ready':True,'actual_data_counts':data['actual_counts'],'microbatch':protocol['microbatch'],'matrix_stages':[r['stage'] for r in registry],
                      'protocol_sha256':sha(root/'protocol/launch.json')},indent=2),flush=True)


if __name__=='__main__':main()
