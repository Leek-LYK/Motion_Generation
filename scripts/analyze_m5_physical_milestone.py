#!/usr/bin/env python3
"""Matched-budget exploratory diagnostics; preserve the frozen 40k selection endpoint."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from training.m3 import atomic_json,sha
from training.m5_data import GROUPS
from training.m5_review_statistics import source_clusters


def bootstrap_difference(candidate,control,clusters,metric):
    key=lambda r:(r['sample_id'],r['seed'],r['condition'])
    other={key(r):r for r in control};assert set(other)=={key(r) for r in candidate}
    labels=sorted(set(clusters.values()));by_label={s:i for i,s in enumerate(labels)};sums=np.zeros((len(labels),6));counts=np.zeros_like(sums)
    for row in candidate:
        i=by_label[clusters[row['sample_id']]];j=GROUPS.index(row['group']);sums[i,j]+=row[metric]-other[key(row)][metric];counts[i,j]+=1
    rng=np.random.default_rng(7379);values=[];rejected=0
    while sum(len(x) for x in values)<2000:
        n=min(500,2000-sum(len(x) for x in values));weights=rng.multinomial(len(labels),np.full(len(labels),1/len(labels)),size=n)
        denom=weights@counts;ok=(denom>0).all(1);rejected+=int((~ok).sum())
        if ok.any():values.append(((weights[ok]@sums)/denom[ok]).mean(1))
    return {'candidate_minus_control_m':float((sums.sum(0)/counts.sum(0)).mean()),'cluster_bootstrap_95_ci_m':np.quantile(np.concatenate(values),[.025,.975]).tolist(),
      'source_family_clusters':len(labels),'draws':2000,'seed':7379,'empty_class_draws_rejected':rejected,
      'scope':'Exploratory intermediate comparison; repeated seeds, headings and correlated source families stay clustered.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--step',type=int,required=True);a=p.parse_args();root=a.experiment.resolve()
    launch=json.loads((root/'protocol/launch.json').read_text());manifest=json.loads(Path(launch['subset']).read_text());clusters=source_clusters(manifest['val'])
    reports={};rows={};metrics={};traces={};initial={}
    for name in launch['jobs']:
        run=root/name;folder=run/'evaluations'/f'generation_full_{a.step:06d}';report=json.loads((folder/'report.json').read_text());reports[name]=report
        if report['step']!=a.step or report.get('identity',{}).get('subset_sha256')!=launch['subset_sha256']:raise ValueError('Milestone report not finalized or identity differs: '+name)
        rows[name]=json.loads((folder/'per_sample.json').read_text());denoise=json.loads((run/'evaluations'/f'denoising_{a.step:06d}.json').read_text())['ema']
        groups=list(report['groups'].values());checkpoint=run/'checkpoints'/f'step_{a.step:06d}.pt'
        metrics[name]={'step':a.step,'checkpoint_sha256':sha(checkpoint),'all_finite':report['all_finite'],'all_valid_qpos':report['all_valid_qpos'],
          'core_amplitude_passed':report['core_amplitude_passed'],'core_passed_both_headings':all(v['core_amplitude_passed'] for v in report['conditions'].values()),
          'mean_fk_self_m':float(np.mean([v['mean_fk_inconsistency_m'] for v in groups])),
          'macro_mean_clip_projection_p95_m':float(np.mean([v['mean_projection_node_p95_m'] for v in groups])),
          'pooled_projection_p95_m':report['pooled_projection_node_p95_m'],'max_penetration_m':max(v['max_penetration_m'] for v in groups),
          'mean_deep_penetration_fraction':float(np.mean([v['mean_penetration_over_5cm_fraction'] for v in groups])),
          'mean_rotation_exceedance_fraction':float(np.mean([v['mean_rotation_threshold_exceed_fraction'] for v in groups])),
          'common_norm6_denoising_macro':denoise['macro_loss'],'denoising_fk_gt_norm':denoise['parts']['fk'],
          'denoising_native_axis_defect':denoise['parts']['axis_defect']}
        with (run/'sample_trace.jsonl').open() as f:prefix=[next(f) for _ in range(a.step)]
        traces[name]=hashlib.sha256(''.join(prefix).encode()).hexdigest();initial[name]=json.loads((run/'run_config.json').read_text())['initial_model_sha256']
    assert len(set(traces.values()))==1 and initial['B1_small_norm7']==initial['C_small_raw7'] and initial['P0_tiny_norm6']==initial['B0_tiny_norm7']
    paired={key:bootstrap_difference(rows['C_small_raw7'],rows['B1_small_norm7'],clusters,key) for key in ['fk_inconsistency_m','projection_node_p95_m']}
    result={'scope':'Intermediate diagnostics only. Continue all frozen 40k budgets; no model/recipe selection from this snapshot.','step':a.step,
      'events_per_model':len(manifest['val']),'clips_per_model':len(next(iter(rows.values()))),'source_family_clusters':len(set(clusters.values())),
      'metrics':metrics,'C_raw7_minus_B1_norm7':paired,'sample_stream_sha256':traces,'matched_initializations_verified':True,
      'script_sha256':sha(__file__),'protocol_sha256':sha(root/'protocol/launch.json'),'M5_complete':False}
    path=root/'diagnostics'/f'milestone_{a.step:06d}.json';atomic_json(path,result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
