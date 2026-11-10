#!/usr/bin/env python3
"""Build a versioned physical-FK M5 cache from preserved source qpos.

Reuse only the M2 smooth-root/heading block, which is invariant to this hinge
rotation-order repair. Recompute positions, rotations, velocities and contacts.
Statistics use the union of selected training events, never development/test.
"""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import argparse,fcntl,hashlib,json,multiprocessing as mp,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from collections import OrderedDict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from training.m3 import atomic_json,sha,INPUTS,validate_inputs
from data_pipeline.motion_features import moments,merge_moments

_worker=None


def initialize():
    global _worker
    import torch
    from training.g1_physical import PhysicalG1Converter
    from kimodo.motion_rep import KimodoMotionRep
    torch.set_num_threads(1)
    converter=PhysicalG1Converter();rep=KimodoMotionRep(converter.skeleton,30,stats_path=None)
    _worker=(converter,rep,OrderedDict())


def convert(row):
    import torch
    from kimodo.geometry import matrix_to_cont6d
    from kimodo.motion_rep.feature_utils import compute_vel_xyz,compute_heading_angle
    from kimodo.motion_rep.feet import foot_detect_from_pos_and_vel
    converter,rep,mapped=_worker;node=(ROOT/row['npz_path']).resolve();node.relative_to(ROOT/'data/processed/g1_30hz_nodes_v1')
    with np.load(node,allow_pickle=False) as archive:
        if str(archive['source_sha256'])!=row['source_sha256']:raise ValueError('Node/source identity mismatch')
        q=archive['qpos_mujoco'][row['start_frame']:row['end_frame_exclusive']].astype(np.float64)
    n=int(row['num_frames']);assert q.shape==(n,36)
    source=row['feature_path']
    if source not in mapped:
        mapped[source]=np.load(INPUTS/source,mmap_mode='r',allow_pickle=False)
        if len(mapped)>8:mapped.popitem(last=False)
    prefix=np.asarray(mapped[source][row['frame_offset']:row['frame_offset']+n,:5]).copy()
    with torch.inference_mode():
        motion=converter.qpos_to_motion_dict(q);root=motion['root_positions'].clone();origin=root[0,[0,2]].clone()
        root[:,[0,2]]-=origin
        rotations=motion['local_rot_mats'];G,P,_=converter.skeleton.fk(rotations,root)
        heading=compute_heading_angle(P[None],converter.skeleton)[0];h=torch.stack((torch.cos(heading),torch.sin(heading)),-1)
        heading_error=float(abs(h-torch.from_numpy(prefix[:,3:5])).max());height_error=float(abs(root[:,1]-torch.from_numpy(prefix[:,1])).max())
        if heading_error>1e-5 or height_error>1e-5:raise ValueError(f'Root block is not invariant: {heading_error}/{height_error}')
        smooth=torch.from_numpy(prefix[:,:3]).double();offset=smooth.clone();offset[:,1]=0
        positions=P-offset[:,None];lengths=torch.tensor([n]);velocity=compute_vel_xyz(P[None],30,lengths=lengths)[0]
        contact=foot_detect_from_pos_and_vel(P[None],velocity[None],converter.skeleton,.15,.10)[0]
        raw=torch.cat((torch.from_numpy(prefix).double(),positions.reshape(n,102),matrix_to_cont6d(G).reshape(n,204),velocity.reshape(n,102),contact.double()),-1).float().numpy()
        local_root=rep.global_root_to_local_root(torch.from_numpy(prefix)[None],normalized=False,lengths=lengths)[0].numpy()
        foot=converter.foot_bottom({'posed_joints':P,'global_rot_mats':G}).numpy()
    if raw.shape!=(n,417) or not np.isfinite(raw).all():raise ValueError('Malformed physical feature')
    return {'sample_id':row['sample_id'],'raw':raw,'local_root':local_root,'first_heading_angle':float(np.arctan2(prefix[0,4],prefix[0,3])),
      'origin_x':float(origin[0]),'origin_z':float(origin[1]),'root_block_error':max(heading_error,height_error),
      'reference_max_penetration_m':float(max(0,-foot.min()))}


def selection():
    directory=ROOT/'runs/m5_20261103/data';ready=json.loads((directory/'READY.json').read_text());manifests=[json.loads((directory/name).read_text()) for name in ready['manifests'].values()]
    manifests.append(json.loads((directory/'D_anchor_1024.json').read_text()))
    train={r['sample_id'] for m in manifests for r in m['train']};val={r['sample_id'] for m in manifests for r in m['val']}
    audit=json.loads((directory/'locked_audit.json').read_text())['events'];test={r['sample_id'] for r in audit}
    assert not train&val and not (train|val)&test
    wanted=train|val|test;frames=[]
    for split in ('train','val','test_content','test_repetition'):
        f=pd.read_parquet(INPUTS/f'{split}.parquet');frames.append(f[f.sample_id.isin(wanted)].copy())
    all_rows=pd.concat(frames,ignore_index=True).sort_values(['split','sample_id']).reset_index(drop=True)
    if set(all_rows.sample_id)!=wanted or all_rows.sample_id.nunique()!=len(wanted):raise ValueError('Missing or duplicate source events')
    all_rows['physical_stats_train']=all_rows.sample_id.isin(train)
    all_rows['physical_frame_offset']=all_rows.num_frames.cumsum().shift(1,fill_value=0).astype('int64')
    return all_rows,{'unique_train_events':len(train),'unique_val_events':len(val),'unique_audit_events':len(test),'total_events':len(wanted)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8);args=p.parse_args();out=args.output.resolve()
    if (out/'READY').exists():raise FileExistsError('Physical cache already complete')
    out.mkdir(parents=True,exist_ok=True);lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ready=validate_inputs();audit=ROOT/'runs/m5_physical_20261103/physical_contract_audit.json';proof=json.loads(audit.read_text());assert proof['passed']
    if proof['physical_code_sha256']!=sha(ROOT/'training/g1_physical.py'):raise ValueError('Physical contract code changed')
    frame,counts=selection();rows=frame.to_dict('records');total_frames=int(frame.num_frames.sum())
    config={'schema':'m5-physical-inputs-v2','m2_parent_fingerprint':ready['fingerprint'],'feature_dim':417,'fps':30,
      'physical_version':proof['version'],'physical_code_sha256':sha(ROOT/'training/g1_physical.py'),'builder_sha256':sha(__file__),
      'selection_sha256':hashlib.sha256(json.dumps(sorted(frame.sample_id.tolist())).encode()).hexdigest(),
      'statistics_policy':'Unique selected training-event union; valid frames only. val/test never fit statistics.',
      'root_policy':'Reuse invariant M2 root/heading block; recompute all body features from physical source qpos.',
      'text_cache':str(INPUTS/'text_encoder/embeddings.npy'),'text_config_fingerprint':ready['text_config_fingerprint'],
      'normalization_epsilon':1e-5,**counts,'frames':total_frames,'old_cache_modified':False}
    config['fingerprint']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();atomic_json(out/'config.json',config)
    packed=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype=np.float32,shape=(total_frames,417))
    stats={name:(0,np.zeros(width),np.zeros(width)) for name,width in [('global_root',5),('body',412),('local_root',4)]}
    updates={};start=time.monotonic();done=0;max_root=0.;max_reference_penetration=0.
    try:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
            iterator=iter(rows);pending={}
            def fill():
                while len(pending)<args.workers*2:
                    try:row=next(iterator)
                    except StopIteration:break
                    pending[pool.submit(convert,row)]=row
            fill()
            while pending:
                completed,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in completed:
                    row=pending.pop(future);result=future.result();raw=result.pop('raw');local=result.pop('local_root')
                    a=row['physical_frame_offset'];n=row['num_frames'];packed[a:a+n]=raw
                    if row['physical_stats_train']:
                        stats['global_root']=merge_moments(stats['global_root'],moments(raw[:,:5]));stats['body']=merge_moments(stats['body'],moments(raw[:,5:]));stats['local_root']=merge_moments(stats['local_root'],moments(local))
                    updates[row['sample_id']]={k:result[k] for k in ('first_heading_angle','origin_x','origin_z')}
                    max_root=max(max_root,result['root_block_error']);max_reference_penetration=max(max_reference_penetration,result['reference_max_penetration_m']);done+=1
                    if done%100==0 or done==len(rows):
                        packed.flush();atomic_json(out/'status.json',{'phase':'building','completed_events':done,'total_events':len(rows),'elapsed_s':time.monotonic()-start})
                        print(done,'/',len(rows),'events',round(time.monotonic()-start,1),'s',flush=True)
                fill()
        packed.flush()
        for name,(n,mean,m2) in stats.items():
            directory=out/'stats/motion'/name;directory.mkdir(parents=True,exist_ok=True)
            np.save(directory/'mean.npy',mean.astype(np.float32));np.save(directory/'std.npy',np.sqrt(m2/n).astype(np.float32))
        for key in ('first_heading_angle','origin_x','origin_z'):frame[key]=frame.sample_id.map(lambda sid:updates[sid][key])
        frame['frame_offset']=frame['physical_frame_offset'];frame['feature_path']='features.npy';frame.to_parquet(out/'samples.parquet',index=False)
        for split in ('train','val','test_content','test_repetition'):frame[frame.split==split].to_parquet(out/f'{split}.parquet',index=False)
        stat_hash={str(f.relative_to(out)):sha(f) for f in (out/'stats').rglob('*.npy')}
        verification={'passed':True,'fingerprint':config['fingerprint'],**counts,'frames':total_frames,'statistics_frames':stats['body'][0],
          'root_block_max_error':max_root,'reference_max_penetration_m':max_reference_penetration,'physical_contract_audit_sha256':sha(audit),
          'stats_sha256':stat_hash,'samples_sha256':sha(out/'samples.parquet'),'feature_sha256':sha(out/'features.npy'),
          'audit_model_scores_read':False,'completed_at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
        atomic_json(out/'READY',verification);atomic_json(out/'status.json',{'phase':'complete','completed_events':done,'total_events':len(rows),'elapsed_s':time.monotonic()-start})
        print(json.dumps(verification,indent=2),flush=True)
    except Exception as error:atomic_json(out/'status.json',{'phase':'failed','completed_events':done,'error':repr(error)});raise


if __name__=='__main__':main()
