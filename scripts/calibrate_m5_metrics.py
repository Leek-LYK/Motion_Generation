#!/usr/bin/env python3
"""Freeze per-class rotation thresholds from development references only."""
import argparse,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from training.m3 import make_model,atomic_json,sha
from training.m5_data import SubsetDataset,GROUPS


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4);data=json.loads((args.experiment/'data/READY.json').read_text());path=args.experiment/'data'/data['manifests']['mixed_large']
    manifest=json.loads(path.read_text());dataset=SubsetDataset(manifest['val'])
    rep=make_model({'latent_dim':64,'ff_size':128,'num_layers':1,'num_heads':4}).motion_rep;angles={g:[] for g in GROUPS}
    with torch.inference_mode():
        for i in range(len(dataset)):
            event=dataset[i];r=rep.inverse(event['x0'][None],is_normalized=True)['global_rot_mats'][0].numpy()
            rel=np.swapaxes(r[:-1],-1,-2)@r[1:];theta=np.arccos(np.clip((np.trace(rel,axis1=-2,axis2=-1)-1)/2,-1,1))
            angles[event['group']].append(theta.ravel())
    quantiles={g:float(np.quantile(np.concatenate(v),.999)) for g,v in angles.items()}
    thresholds={g:max(math.pi/4,q) for g,q in quantiles.items()}
    result={'reference_only':True,'candidate_outputs_used':False,'reference_q999_rotation_angle_rad':quantiles,
      'rotation_thresholds_rad':thresholds,'formula':'max(pi/4, per-class reference q99.9 SO(3) adjacent-frame angle)',
      'manifest_sha256':sha(path),'script_sha256':sha(__file__)}
    atomic_json(args.experiment/'reference_review/metric_calibration.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
