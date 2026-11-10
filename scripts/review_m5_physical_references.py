#!/usr/bin/env python3
"""Calibrate physical-v2 continuity on full development references, not candidates."""
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from kimodo.motion_rep import KimodoMotionRep
from kimodo.skeleton.registry import build_skeleton
from training.m3 import atomic_json,sha
from training.m5_physical_data import PhysicalSubsetDataset
from training.m5_physical_losses import geometry
from training.m5_physical_evaluation import PhysicalGeometryMetrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();root=a.experiment.resolve();torch.set_num_threads(4)
    ready=json.loads((root/'data/READY.json').read_text());manifest=json.loads((root/'data/D_short_3500.json').read_text())
    samples=PhysicalSubsetDataset(manifest['val'],ready['input_dir'],root/'stats/motion');rep=KimodoMotionRep(build_skeleton(34),30,str(root/'stats/motion'))
    metric=PhysicalGeometryMetrics();rotations=defaultdict(list);records=[]
    with torch.inference_mode():
        for i in range(len(samples)):
            sample=samples[i];_,_,motion,_=geometry(rep,sample['x0'][None]);motion={k:v[0] for k,v in motion.items()}
            R=motion['global_rot_mats'].numpy();relative=np.swapaxes(R[:-1],-1,-2)@R[1:]
            angles=np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1));rotations[sample['group']].append(angles.reshape(-1))
            measured,_,_=metric.measure(motion,sample['group']);records.append({'sample_id':sample['sample_id'],'group':sample['group'],**measured})
            if (i+1)%50==0:print('physical reference review',i+1,'/',len(samples),flush=True)
    thresholds={g:max(np.pi/4,float(np.quantile(np.concatenate(v),.999))) for g,v in rotations.items()}
    report={'schema':'m5-physical-reference-review-v2','passed':True,'events':len(samples),'training_view_fingerprint':ready['fingerprint'],
      'metric_thresholds':thresholds,'rotation_policy':'max(45 degrees/frame, per-class physical-reference P99.9), fixed before candidates',
      'max_reference_bounded_penetration_m':max(r['max_penetration_m'] for r in records),
      'max_reference_raw_fk_penetration_m':max(r['raw_fk_max_penetration_m'] for r in records),
      'max_reference_projection_node_error_m':max(r['projection_node_max_m'] for r in records),
      'mean_reference_projection_node_p95_m':float(np.mean([r['projection_node_p95_m'] for r in records])),
      'reference_limit_violating_events':sum(r['legacy_limit_violation_rad']>1e-4 for r in records),
      'records':records,'model_scores_used':False,'script_sha256':sha(__file__)}
    atomic_json(root/'reference_review.json',report);print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':main()
