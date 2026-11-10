#!/usr/bin/env python3
"""Same-recipe Tiny/Small raw-seven capacity comparison at one fixed milestone."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from training.m3 import atomic_json,sha
from training.m5_review_statistics import source_clusters
from scripts.analyze_m5_physical_milestone import bootstrap_difference


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--step',type=int,required=True)
    a=p.parse_args();root=a.experiment.resolve();names={'small':'C_small_raw7','tiny':'F0_tiny_raw7_short'};data={}
    for role,name in names.items():
        run=root/name;folder=run/'evaluations'/f'generation_full_{a.step:06d}';config=json.loads((run/'config.json').read_text());report=json.loads((folder/'report.json').read_text())
        if report.get('identity',{}).get('subset_sha256')!=config['subset_sha256'] or report['step']!=a.step:raise ValueError('Unfinalized report or changed subset')
        with (run/'sample_trace.jsonl').open() as f:prefix=''.join(next(f) for _ in range(a.step))
        data[role]={'name':name,'config':config,'report':report,'rows':json.loads((folder/'per_sample.json').read_text()),'trace':prefix,
          'parameters':json.loads((run/'run_config.json').read_text())['parameters'],'checkpoint_sha256':sha(run/'checkpoints'/f'step_{a.step:06d}.pt')}
    keys=set(data['small']['config'])|set(data['tiny']['config']);differences={k:[data['small']['config'].get(k),data['tiny']['config'].get(k)] for k in keys if data['small']['config'].get(k)!=data['tiny']['config'].get(k)}
    if set(differences)-{'stage','model'}:raise ValueError('Capacity comparison recipe differs: '+str(set(differences)))
    if data['small']['trace']!=data['tiny']['trace']:raise ValueError('Matched training draws diverged')
    manifest=json.loads((root/names['small']/'subset.json').read_text());clusters=source_clusters(manifest['val']);metrics={}
    for role,entry in data.items():
        report=entry['report'];groups=list(report['groups'].values())
        metrics[role]={'run':entry['name'],'parameters':entry['parameters'],'checkpoint_sha256':entry['checkpoint_sha256'],
          'mean_fk_self_m':float(np.mean([g['mean_fk_inconsistency_m'] for g in groups])),
          'pooled_projection_p95_m':report['pooled_projection_node_p95_m'],'macro_mean_clip_projection_p95_m':float(np.mean([g['mean_projection_node_p95_m'] for g in groups])),
          'max_penetration_m':max(g['max_penetration_m'] for g in groups),'mean_deep_frame_fraction':float(np.mean([g['mean_penetration_over_5cm_fraction'] for g in groups])),
          'core_passed_both_headings':all(v['core_amplitude_passed'] for v in report['conditions'].values()),
          'per_heading_length_core':{h:{l:v['core_amplitude_passed'] for l,v in value['length_strata'].items()} for h,value in report['conditions'].items()}}
    result={'scope':'One matched seed and intermediate budget; no final model selection or independent-training replication claim.',
      'step':a.step,'metrics':metrics,'configuration_differences':differences,'data_streams_match':True,
      'source_family_clusters':len(set(clusters.values())),
      'paired_small_minus_tiny':{key:bootstrap_difference(data['small']['rows'],data['tiny']['rows'],clusters,key) for key in ['fk_inconsistency_m','projection_node_p95_m']},
      'relative_small_improvement':{key:1-metrics['small'][key]/metrics['tiny'][key] for key in ['mean_fk_self_m','pooled_projection_p95_m']},
      'script_sha256':sha(__file__),'M5_complete':False}
    atomic_json(root/'diagnostics'/f'raw_capacity_{a.step:06d}.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
