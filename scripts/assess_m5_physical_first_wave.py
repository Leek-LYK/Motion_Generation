#!/usr/bin/env python3
"""Fixed-budget physical-v2 comparisons; final semantic/audit acceptance stays separate."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.m3 import atomic_json,sha
from training.m5_protocol import core_passed,no_core_regression,final_geometry_passed,fk_mean


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();root=a.experiment.resolve()
    launch=json.loads((root/'protocol/launch.json').read_text());reports={};initial={};traces={};details={}
    for name in launch['jobs']:
        run=root/name;done=json.loads((run/'COMPLETED.json').read_text());config=json.loads((run/'config.json').read_text());step=config['max_steps']
        if config!=json.loads((root/'configs'/(name+'.json')).read_text()):raise ValueError('Run configuration drift: '+name)
        if done['step']!=step or sha(run/'checkpoints/latest.pt')!=done['terminal_checkpoint_sha256']:raise ValueError('Terminal state mismatch: '+name)
        if sha(root/'configs'/(name+'.json'))!=launch['config_sha256'][name] or done['identity']['subset_sha256']!=launch['subset_sha256']:
            raise ValueError('Frozen inputs changed: '+name)
        r=json.loads((run/'evaluations'/f'generation_full_{step:06d}/report.json').read_text());reports[name]=r
        initial[name]=json.loads((run/'run_config.json').read_text())['initial_model_sha256'];traces[name]=sha(run/'sample_trace.jsonl')
        details[name]={'checkpoint_sha256':done['terminal_checkpoint_sha256'],'mean_fk_inconsistency_m':fk_mean(r),
          'pooled_projection_node_p95_m':r['pooled_projection_node_p95_m'],
          'mean_projection_node_p95_m':float(np.mean([g['mean_projection_node_p95_m'] for g in r['groups'].values()])),
          'max_penetration_m':max(g['max_penetration_m'] for g in r['groups'].values()),
          'mean_deep_penetration_fraction':float(np.mean([g['mean_penetration_over_5cm_fraction'] for g in r['groups'].values()])),
          'max_projection_node_error_m':max(g['max_projection_node_error_m'] for g in r['groups'].values()),
          'core_passed':core_passed(r),'all_six_class_amplitude_passed':r['all_six_class_amplitude_passed'],
          'final_geometry_passed':bool(final_geometry_passed(r) and r['pooled_projection_node_p95_m']<=.03),
          'geometry_score_m':r['geometry_score_m']}
    if initial['P0_tiny_norm6']!=initial['B0_tiny_norm7'] or initial['B1_small_norm7']!=initial['C_small_raw7']:raise ValueError('Matched initialization mismatch')
    if len(set(traces.values()))!=1:raise ValueError('Data draw streams diverged')
    baseline=reports['P0_tiny_norm6'];tiny=reports['B0_tiny_norm7'];amplitude,reasons=no_core_regression(tiny,baseline)
    improvement=1-fk_mean(tiny)/fk_mean(baseline)
    repair={'self_improvement_vs_norm6':improvement,'core_passed':core_passed(tiny),'no_core_regression':amplitude,
      'self_improvement_at_least_10pct':improvement>=.10,'passed':bool(core_passed(tiny) and amplitude and improvement>=.10),
      'regression_reasons':reasons,'scope':'New from-scratch diagnostic. The failed legacy R3 continuation results are preserved and not relabelled.'}
    candidates=[];candidate_checks={}
    for name in ['B1_small_norm7','C_small_raw7']:
        report=reports[name];good,reasons=no_core_regression(report,tiny)
        candidate_checks[name]={'core_passed':core_passed(report),'no_core_regression_vs_tiny_norm7':good,'regression_reasons':reasons,
          'matched_tiny_same_objective_available':name=='B1_small_norm7'}
        if core_passed(report) and good:candidates.append((report['geometry_score_m'],name))
    chosen=min(candidates)[1] if candidates else None
    result={'phase':'first_wave_assessed','protocol_sha256':sha(root/'protocol/launch.json'),'metrics':details,'repair_fk_diagnostic':repair,
      'small_candidate_checks':candidate_checks,'provisional_small_candidate':chosen,'paired_initializations_equal':True,'sample_streams_equal':True,
      'main_curriculum_auto_started':False,'semantic_acceptance':'pending independent human review','final_audit_model_scores_read':False,'M5_complete':False,
      'next_step':'Inspect all evidence before dependent D/E. A failed repair diagnostic or failed core generation requires further M5 research; no threshold changes.'}
    atomic_json(root/'first_wave_assessment.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
