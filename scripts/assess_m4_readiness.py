#!/usr/bin/env python3
"""Measure actual pure-noise motion amplitude; never equate numeric success with M5 readiness."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import numpy as np
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.skeleton.registry import build_skeleton
from training.m3 import atomic_json,batch_to,configure,make_model,sha,validate_inputs
from training.m4 import load_data,sample,rotate_batch
from training.m4_readiness import movement,summarize
from training.g1_qpos import G1BoundedDecoder,DECODER_VERSION


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',default='best_val_ema.pt')
    p.add_argument('--protocol',type=Path,default=ROOT/'runs/m4_research_20261103/M5_entry_protocol.json')
    p.add_argument('--guidance',type=float,default=1.0)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--split',choices=['val','test_repetition'],default='val')
    p.add_argument('--heading',type=float,help='Apply a common first heading to condition and reference before evaluation')
    p.add_argument('--seeds',type=int,nargs='+',help='Explicit sampling seeds, frozen before final audit')
    p.add_argument('--decoder',choices=['legacy','bounded-v1'],default='legacy')
    p.add_argument('--save-samples',action='store_true',help='Retain raw decoded motion and both qpos variants')
    p.add_argument('--quick',action='store_true');args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.quick and args.seeds:raise ValueError('quick and explicit seeds are mutually exclusive')
    seeds=args.seeds or ([42] if args.quick else [0,42,1234])
    if len(set(seeds))!=len(seeds):raise ValueError('Duplicate seeds')
    ready=validate_inputs()
    payload=(args.run_dir/'checkpoints'/args.checkpoint).read_bytes()
    checkpoint_hash=hashlib.sha256(payload).hexdigest()
    state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
    del payload
    if state['identity']['m2_fingerprint']!=ready['fingerprint']:raise ValueError('Data mismatch')
    manifest_path=args.manifest or args.run_dir/'subset.json'
    if args.split!='val':
        release_path=args.protocol.parent/'audit_release.json'
        if not release_path.exists():raise ValueError('Locked audit has not been released after development checks')
        release=json.loads(release_path.read_text())
        if checkpoint_hash not in release['allowed_checkpoint_sha256'] or args.guidance!=release['guidance'] or sha(manifest_path)!=release['audit_manifest_sha256'] or sha(args.protocol)!=release['protocol_sha256'] or args.quick:
            raise ValueError('Audit configuration differs from the frozen release')
        if (seeds!=release['sampling_seeds'] or args.heading not in release['conditioning_headings'] or
            args.decoder!=release['decoder'] or not args.save_samples or
            sha(__file__)!=release['assessor_sha256'] or
            sha(ROOT/'training/g1_qpos.py')!=release['decoder_sha256'] or
            sha(ROOT/'training/m4_readiness.py')!=release['metric_sha256']):
            raise ValueError('Audit seeds, heading, decoder, retention or code differ from frozen release')
    args.output.mkdir(parents=True)
    configure(state['config']['seed']);model=make_model(state['config']['model'],args.run_dir/'stats/motion').cuda().eval()
    model.load_state_dict(state['ema']);manifest=json.loads(manifest_path.read_text())
    validation=load_data(manifest,args.split);rows=[]
    converter=MujocoQposConverter(model.motion_rep.skeleton)
    bounded=G1BoundedDecoder(build_skeleton(34)) if args.decoder=='bounded-v1' else None
    protocol=json.loads(args.protocol.read_text())
    with torch.inference_mode():
        for seed in seeds:
            for start in range(0,len(validation),16):
                events=validation[start:start+16];batch=batch_to(events,'cuda:0')
                if args.heading is not None:
                    batch=rotate_batch(model,batch,args.heading-batch['first_heading_angle'])
                generated=sample(model,batch,seed+start,100,args.guidance,state['config']['text_dropout']>0)
                for i,event in enumerate(events):
                    n=event['length'];prediction=model.motion_rep.inverse(generated[i:i+1,:n],is_normalized=True)
                    reference=model.motion_rep.inverse(batch['x0'][i:i+1,:n],is_normalized=True)
                    pred={k:v[0] for k,v in prediction.items()};ref={k:v[0] for k,v in reference.items()}
                    a=movement(pred,model.motion_rep.skeleton,event['group']);b=movement(ref,model.motion_rep.skeleton,event['group'])
                    legacy_qpos=converter.dict_to_qpos(prediction)[0]
                    qpos,diagnostics=bounded.decode(pred) if bounded else (legacy_qpos,None)
                    quaternion_error=float(np.max(np.abs(np.linalg.norm(qpos[:,3:7],axis=1)-1)))
                    finite=all(bool(torch.isfinite(x).all()) for x in pred.values()) and bool(np.isfinite(qpos).all()) and quaternion_error<1e-4
                    valid_qpos=finite and (not bounded or diagnostics['bounded']['max_limit_violation_rad']<=1e-6)
                    row={'sample_id':event['sample_id'],'group':event['group'],'cohort':event['cohort'],'text':event['text'],
                                 'seed':seed,'guidance':args.guidance,'generated_amplitude':a,'reference_amplitude':b,'amplitude_ratio':a/max(b,1e-8),
                                 'qpos_shape':list(qpos.shape),'quaternion_norm_max_error':quaternion_error,
                                 'finite':finite,'valid_qpos':valid_qpos,'decoder_diagnostics':diagnostics,
                                 'world_mpjpe_m':float(torch.linalg.vector_norm(pred['posed_joints']-ref['posed_joints'],dim=-1).mean())}
                    if bounded:
                        rebuilt=bounded.converter.qpos_to_motion_dict(qpos.astype(np.float32),source_fps=30)
                        row['bounded_amplitude']=movement(rebuilt,model.motion_rep.skeleton,event['group'])
                    if args.save_samples:
                        folder=args.output/f'seed_{seed}';folder.mkdir(exist_ok=True)
                        name=event['sample_id'];stem=folder/name
                        np.savez_compressed(stem.with_suffix('.npz'),**{k:v.detach().cpu().numpy() for k,v in pred.items()})
                        np.savetxt(stem.with_suffix('.csv'),qpos,delimiter=',')
                        if bounded:np.savetxt(folder/f'{name}_legacy.csv',legacy_qpos,delimiter=',')
                        row['artifact_prefix']=str(stem.relative_to(args.output))
                    rows.append(row)
            print('amplitude assessed seed',seed,flush=True)
    result=summarize(rows,protocol)
    result.update(checkpoint_step=state['step'],checkpoint_sha256=checkpoint_hash,
                  protocol_sha256=sha(args.protocol),events=len(validation),seeds=seeds,guidance=args.guidance,
                  source_split=args.split,source_manifest_sha256=sha(manifest_path),
                  conditioning_heading=args.heading,
                  decoder=DECODER_VERSION if bounded else 'upstream-euler',postprocess=bool(bounded),
                  decoder_sha256=sha(ROOT/'training/g1_qpos.py') if bounded else None,
                  all_exported_qpos_valid=all(r['valid_qpos'] for r in rows),
                  max_legacy_limit_violation_rad=max((r['decoder_diagnostics']['legacy']['max_limit_violation_rad'] for r in rows),default=0) if bounded else None,
                  max_exported_joint_speed_rad_s=max((r['decoder_diagnostics']['bounded']['max_joint_speed_rad_s'] for r in rows),default=0) if bounded else None,
                  amplitude_uses_raw_unprojected_motion=True,
                  script_sha256=sha(__file__),metric_code_sha256=sha(ROOT/'training/m4_readiness.py'))
    atomic_json(args.output/'per_sample.json',rows);atomic_json(args.output/'report.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
