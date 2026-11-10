#!/usr/bin/env python3
"""Audit canonical native hinge limits of every selected training source frame."""
import argparse,json,sys
from collections import OrderedDict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from training.g1_physical import physical_constants
from training.m3 import atomic_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();root=a.experiment.resolve()
    view=json.loads((root/'data/READY.json').read_text());ids=json.loads((root/'data/stats_train_ids.json').read_text())
    by_id={r['sample_id']:r for r in pd.read_parquet(Path(view['input_dir'])/'samples.parquet').to_dict('records')}
    bounds=physical_constants()['limits'];cache=OrderedDict();bad=[];maximum=0.;frames=violation_frames=0
    for i,sid in enumerate(ids):
        row=by_id[sid];assert row['split']=='train';path=ROOT/row['npz_path']
        if path not in cache:
            with np.load(path,allow_pickle=False) as z:cache[path]=z['qpos_mujoco'][:,7:].copy()
            if len(cache)>8:cache.popitem(last=False)
        q=cache[path][row['start_frame']:row['end_frame_exclusive']];q=np.arctan2(np.sin(q),np.cos(q))
        excess=np.maximum(0,np.maximum(bounds[:,0]-q,q-bounds[:,1]));worst=float(excess.max());n=int(np.any(excess>1e-4,axis=-1).sum())
        frames+=len(q);violation_frames+=n;maximum=max(maximum,worst)
        if n:bad.append({'sample_id':sid,'max_canonical_limit_violation_rad':worst,'violating_frames':n,'frames':len(q)})
        if (i+1)%1500==0:print('source target limit audit',i+1,'/',len(ids),flush=True)
    result={'passed':not bad,'events':len(ids),'frames':frames,'violating_events':len(bad),'violating_frames':violation_frames,
      'max_canonical_limit_violation_rad':maximum,'records':bad,'training_view_fingerprint':view['fingerprint'],'script_sha256':sha(__file__),
      'note':'Source target eligibility audit; no model scores or validation/test filtering.'}
    atomic_json(root/'training_target_limits_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':main()
