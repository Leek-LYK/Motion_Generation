#!/usr/bin/env python3
"""Audit physical-v2 packed features, real-source roundtrips and reference quality."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from kimodo.geometry import cont6d_to_matrix
from kimodo.motion_rep import KimodoMotionRep
from training.g1_physical import PhysicalG1Converter
from training.m3 import atomic_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();torch.set_num_threads(4)
    directory=args.input_dir;ready=json.loads((directory/'READY').read_text());config=json.loads((directory/'config.json').read_text())
    assert ready['passed'] and ready['fingerprint']==config['fingerprint']
    for name,h in ready['stats_sha256'].items():assert sha(directory/name)==h
    assert sha(directory/'samples.parquet')==ready['samples_sha256']
    frame=pd.read_parquet(directory/'samples.parquet');packed=np.load(directory/'features.npy',mmap_mode='r');converter=PhysicalG1Converter();rep=KimodoMotionRep(converter.skeleton,30,stats_path=None)
    records=[];max_fk_error=0.;max_q_error=0.;max_root_error=0.;max_compiled_error=0.
    import mujoco
    mj=mujoco.MjModel.from_xml_path(str(converter.xml_path));data=mujoco.MjData(mj);c=converter.constants
    with torch.inference_mode():
        for i,row in enumerate(frame.to_dict('records')):
            a=row['frame_offset'];n=row['num_frames'];raw=np.asarray(packed[a:a+n]);assert raw.shape==(n,417) and np.isfinite(raw).all()
            P=raw[:,5:107].reshape(n,34,3).copy();P[:,:,0]+=raw[:,None,0];P[:,:,2]+=raw[:,None,2]
            G=cont6d_to_matrix(torch.from_numpy(raw[:,107:311].copy()).reshape(n,34,6))
            bottom=converter.foot_bottom({'posed_joints':torch.from_numpy(P),'global_rot_mats':G}).numpy()
            depth=float(max(0,-bottom.min()));fraction=float(np.mean(bottom.min(1)<-.05))
            records.append({'sample_id':row['sample_id'],'split':row['split'],'text':row['text'],'source_path':row['npz_path'],
              'physical_stats_train':bool(row['physical_stats_train']),'max_penetration_m':depth,'deep_frame_fraction':fraction})
            if i%50==0:
                times=np.array([0,n//2,n-1]);motion=rep.inverse(torch.from_numpy(raw[times].copy())[None],is_normalized=False)
                error=float(torch.linalg.vector_norm(motion['posed_joints'][0]-torch.from_numpy(P[times]),dim=-1).max());max_fk_error=max(max_fk_error,error)
                recovered=converter.dict_to_qpos({k:v[0] for k,v in motion.items()})
                with np.load(ROOT/row['npz_path']) as z:q=z['qpos_mujoco'][row['start_frame']:row['end_frame_exclusive']][times].astype(float)
                q[:,:2]-=q[0:1,:2] if row['start_frame']==0 else 0
                # Use the true segment origin, not the first selected audit frame.
                q[:,0]+=0 # kept explicit below for a single centering convention
                with np.load(ROOT/row['npz_path']) as z:segment=z['qpos_mujoco'][row['start_frame']:row['end_frame_exclusive']].astype(float)
                q=segment[times].copy();q[:,0]-=segment[0,0];q[:,1]-=segment[0,1]
                dq=np.arctan2(np.sin(recovered[:,7:]-q[:,7:]),np.cos(recovered[:,7:]-q[:,7:]))
                max_q_error=max(max_q_error,float(abs(dq).max()));max_root_error=max(max_root_error,float(abs(recovered[:,:3]-q[:,:3]).max()))
                for t,qpos in enumerate(q):
                    data.qpos[:]=qpos;mujoco.mj_forward(mj,data)
                    expected=np.stack([c['C'].T@data.xanchor[j] for j in c['joint_ids']])
                    max_compiled_error=max(max_compiled_error,float(np.linalg.norm(P[times[t],c['indices']]-expected,axis=-1).max()))
            if (i+1)%1000==0:print('audited',i+1,'/',len(frame),flush=True)
    assert max_fk_error<1e-5 and max_q_error<1e-4 and max_root_error<1e-5 and max_compiled_error<1e-5,(max_fk_error,max_q_error,max_root_error,max_compiled_error)
    counts={split:{'events':len(part),'gt_over_10cm':sum(r['max_penetration_m']>.10 for r in part),
       'gt_over_5cm_10pct_frames':sum(r['deep_frame_fraction']>.10 for r in part),'max_depth_m':max(r['max_penetration_m'] for r in part)}
       for split in ('train','val','test_content','test_repetition') for part in [[r for r in records if r['split']==split]] if part}
    report={'passed':True,'cache_fingerprint':config['fingerprint'],'events':len(frame),'max_feature_fk_error_m':max_fk_error,
      'max_real_source_joint_angle_roundtrip_error_rad':max_q_error,'max_root_roundtrip_error_m':max_root_error,
      'max_compiled_joint_position_error_m':max_compiled_error,'reference_quality_by_split':counts,'records':records,
      'script_sha256':sha(__file__),'note':'Numerical/physical contract audit passed. Reference-quality flags are separate; do not use bad targets as automatic training acceptance.'}
    atomic_json(args.output,report);print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)
    for r in sorted(records,key=lambda r:-r['max_penetration_m'])[:8]:print('REFERENCE_TAIL',r,flush=True)


if __name__=='__main__':main()
