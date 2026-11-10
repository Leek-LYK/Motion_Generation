#!/usr/bin/env python3
"""Freeze a qualified training view without changing legacy data or evaluation sets."""
import argparse,hashlib,json,re,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from kimodo.motion_rep import KimodoMotionRep
from kimodo.skeleton.registry import build_skeleton
from data_pipeline.motion_features import moments,merge_moments
from training.m3 import atomic_json,sha

OBJECT_PATTERN=r'\b(?:crutch(?:es)?|canes?|umbrella|wheelbarrow|edges?)\b'


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);out=a.output.resolve();source=a.input_dir.resolve()
    if (out/'data/READY.json').exists():raise FileExistsError('Qualified view is already frozen')
    proof_path=out/'cache_audit.json';proof=json.loads(proof_path.read_text());ready=json.loads((source/'READY').read_text())
    if not proof['passed'] or proof['cache_fingerprint']!=ready['fingerprint']:raise ValueError('Cache audit mismatch')
    quality={r['sample_id']:r for r in proof['records']};data=out/'data';data.mkdir(parents=True,exist_ok=True)
    original=ROOT/'runs/m5_20261103/data';names=['D_short_3500','D_mixed_small_3500','D_mixed_large_4500']
    kept={};excluded=[];manifest_hashes={};counts={};eval_flags={}
    for name in names:
        path=original/(name+'.json');m=json.loads(path.read_text());train=[]
        for r in m['train']:
            q=quality[r['sample_id']];reasons=[]
            if q['max_penetration_m']>.10:reasons.append('reference_foot_penetration_over_10cm')
            if q['deep_frame_fraction']>.10:reasons.append('reference_over_5cm_for_over_10pct_frames')
            if re.search(OBJECT_PATTERN,r['text'],re.I):reasons.append('text_requires_unsupported_prop_or_edge')
            if reasons:excluded.append({'manifest':name,'sample_id':r['sample_id'],'text':r['text'],'group':r['group'],'reasons':reasons,'reference':q})
            else:train.append(r);kept[r['sample_id']]=r
        # Keep all development/monitor events. Flags are strata, never silent drops.
        for r in m['val']:
            q=quality[r['sample_id']];eval_flags[r['sample_id']]={'unsupported_context':bool(re.search(OBJECT_PATTERN,r['text'],re.I)),
                'reference_quality_flag':bool(q['max_penetration_m']>.10 or q['deep_frame_fraction']>.10)}
        m.update(schema='kimodo-m5-physical-subset-v2',parent_manifest=str(path),parent_manifest_sha256=sha(path),
                 physical_cache_fingerprint=ready['fingerprint'],normalization='qualified unique main-training union, physical-v2',
                 train=train,actual_train_events=len(train),human_semantic_review='pending; deterministic eligibility is not a human semantic review')
        target=data/(name+'.json');atomic_json(target,m);manifest_hashes[name]=sha(target)
        counts[name]={'train':len(train),'excluded':len(json.loads(path.read_text())['train'])-len(train),'val':len(m['val']),
                     'monitor':len(m['monitor']),'groups':dict(Counter(r['group'] for r in train))}
    small={r['sample_id'] for r in json.loads((data/'D_mixed_small_3500.json').read_text())['train']}
    large={r['sample_id'] for r in json.loads((data/'D_mixed_large_4500.json').read_text())['train']};assert small<=large
    rows=pd.read_parquet(source/'samples.parquet');by_id={r['sample_id']:r for r in rows.to_dict('records')}
    x=np.load(source/'features.npy',mmap_mode='r');rep=KimodoMotionRep(build_skeleton(34),30,stats_path=None)
    stats={g:(0,np.zeros(w),np.zeros(w)) for g,w in [('global_root',5),('body',412),('local_root',4)]}
    with torch.inference_mode():
        for sid in sorted(kept):
            r=by_id[sid];assert r['split']=='train';n=int(r['num_frames']);start=int(r['frame_offset']);raw=np.asarray(x[start:start+n])
            local=rep.global_root_to_local_root(torch.from_numpy(raw[None,:,:5].copy()),normalized=False,lengths=torch.tensor([n]))[0].numpy()
            for g,values in [('global_root',raw[:,:5]),('body',raw[:,5:]),('local_root',local)]:stats[g]=merge_moments(stats[g],moments(values))
    stats_hashes={}
    for g,(n,mean,m2) in stats.items():
        folder=out/'stats/motion'/g;folder.mkdir(parents=True,exist_ok=True)
        for name,values in [('mean',mean),('std',np.sqrt(m2/n))]:
            f=folder/(name+'.npy');np.save(f,values.astype(np.float32));stats_hashes[str(f.relative_to(out))]=sha(f)
    policy={'schema':'m5-physical-training-view-v2','input_dir':str(source),'input_fingerprint':ready['fingerprint'],
      'cache_audit_sha256':sha(proof_path),'preparation_script_sha256':sha(__file__),'unsupported_context_pattern':OBJECT_PATTERN,
      'ground_filter':'Training only: reference max foot penetration >0.10m OR >0.05m for >10% of frames. No root shifts.',
      'statistics_policy':'Unique qualified main-training union only; legacy anchor/val/test excluded; population standard deviation.',
      'unique_stats_events':len(kept),'statistics_frames':stats['body'][0],'stats_sha256':stats_hashes,'manifests_sha256':manifest_hashes,
      'counts':counts,'development_events_removed':0,'locked_audit_model_scores_read':False,
      'locked_audit_note':'Existing v2 cache includes deterministic source-quality audit, not generated model scores. Final semantic audit remains unreleased.'}
    policy['fingerprint']=hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()
    atomic_json(data/'training_exclusions.json',excluded);atomic_json(data/'development_flags.json',eval_flags)
    atomic_json(data/'stats_train_ids.json',sorted(kept));atomic_json(data/'READY.json',dict(policy,passed=True))
    print(json.dumps({k:v for k,v in policy.items() if k!='stats_sha256'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
