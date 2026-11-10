#!/usr/bin/env python3
"""Matched denoising FK diagnostics and saved-generation projection decomposition."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,io,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from kimodo.skeleton.registry import build_skeleton
from kimodo.model.diffusion import Diffusion
from training.m3 import make_model,configure,batch_to,forward,sha,atomic_json,validate_inputs,INPUTS
from training.m4 import rotate_batch,per_sample_losses
from training.m5_data import SubsetDataset
from training.m5_research_losses import HingeGeometry,per_sample_terms
from training.g1_qpos import G1BoundedDecoder,nearest_angles


def denoising_details(model,samples,headings=(None,0.),batch_size=16):
    model.eval();device=next(model.parameters()).device;hinge=HingeGeometry(model.motion_rep.skeleton,device)
    diffusion=Diffusion(1000).to(device);records=[]
    with torch.inference_mode():
        for heading in headings:
            for start in range(0,len(samples),batch_size):
                events=[samples[i] for i in range(start,min(start+batch_size,len(samples)))];batch=batch_to(events,device)
                if heading is not None:batch=rotate_batch(model,batch,heading-batch['first_heading_angle'])
                noise=torch.zeros_like(batch['x0'])
                for i,event in enumerate(events):
                    seed=int(hashlib.sha256((event['sample_id']+':M4val').encode()).hexdigest()[:12],16)
                    rng=torch.Generator(device=device).manual_seed(seed)
                    noise[i,:event['length']]=torch.randn((event['length'],417),device=device,generator=rng)
                for timestamp in (0,249,499,749,999):
                    t=torch.full((len(events),),timestamp,device=device,dtype=torch.long)
                    prediction=forward(model,diffusion.q_sample(batch['x0'],t,noise),batch,t)
                    values,_=per_sample_terms(model.motion_rep,prediction,batch['x0'],batch['x_pad_mask'],hinge)
                    common,_=per_sample_losses(prediction,batch['x0'],batch['x_pad_mask']);values['norm6']=common
                    for i,event in enumerate(events):
                        records.append({'sample_id':event['sample_id'],'group':event['group'],'heading':heading,'t':timestamp,
                                        **{key:float(v[i]) for key,v in values.items()}})
            print('denoising heading',heading,'complete',flush=True)
    return records


def saved_generation_details(run):
    root=run/'evaluations/generation_full_030000';path=root/'per_sample.json'
    if not path.exists():return None
    rows=json.loads(path.read_text());skeleton=build_skeleton(34);decoder=G1BoundedDecoder(skeleton);hinge=HingeGeometry(skeleton)
    mean=np.r_[np.load(INPUTS/'stats/motion/global_root/mean.npy'),np.load(INPUTS/'stats/motion/body/mean.npy')]
    std=np.r_[np.load(INPUTS/'stats/motion/global_root/std.npy'),np.load(INPUTS/'stats/motion/body/std.npy')];scale=np.sqrt(std**2+1e-5)
    sums={k:np.zeros(34) for k in ('position_to_fk','axis_only_projection','limits_additional_projection','bounded_projection')}
    axis_sums=np.zeros(29);sample_count=0;records=[]
    with torch.inference_mode():
        for row in rows:
            stem=root/row['artifact_prefix']
            with np.load(stem.with_suffix('.npz')) as z:motion={k:torch.from_numpy(z[k].copy()) for k in z.files}
            features=np.load(stem.with_name(stem.name+'_features.npy'))*scale+mean
            direct=features[:,5:107].reshape(-1,34,3).copy();direct[:,:,0]+=features[:,None,0];direct[:,:,2]+=features[:,None,2]
            local=motion['local_rot_mats'];r=decoder.offsets[None]@local.numpy()[:,decoder.indices]
            q=np.loadtxt(stem.with_suffix('.csv'),delimiter=',');unbounded=q.copy()
            unbounded[:,7:]=nearest_angles(r,decoder.axes,decoder.bounds,np.zeros(29,dtype=bool))
            axis_motion=decoder.converter.qpos_to_motion_dict(unbounded.astype(np.float32),source_fps=30)
            bounded=decoder.converter.qpos_to_motion_dict(q.astype(np.float32),source_fps=30)
            raw=motion['posed_joints'].numpy();axis_pos=axis_motion['posed_joints'].numpy();bounded_pos=bounded['posed_joints'].numpy()
            values={'position_to_fk':np.linalg.norm(direct-raw,axis=-1),'axis_only_projection':np.linalg.norm(raw-axis_pos,axis=-1),
                    'limits_additional_projection':np.linalg.norm(axis_pos-bounded_pos,axis=-1),'bounded_projection':np.linalg.norm(raw-bounded_pos,axis=-1)}
            energy,angles=hinge.defects(local)
            for name,value in values.items():sums[name]+=value.mean(0)
            axis_sums+=angles.numpy().mean(0);sample_count+=1
            records.append({k:row[k] for k in ('sample_id','group','seed','condition','text')}|{
                **{name+'_mean_m':float(value.mean()) for name,value in values.items()},
                'axis_defect':float(energy.mean()),'mean_axis_angle_rad':float(angles.mean())})
    return {'clips':sample_count,'records':records,'per_node_mean_m':{name:dict(zip(skeleton.bone_order_names,(v/sample_count).tolist())) for name,v in sums.items()},
            'per_hinge_mean_axis_angle_rad':dict(zip(hinge.names,(axis_sums/sample_count).tolist())),
            'note':'Axis-only and subsequent limit effects are sequential distances, not an additive decomposition of p95 or Euclidean norms.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--saved-only',action='store_true');args=p.parse_args()
    if (args.output/'report.json').exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=True);validate_inputs();run=args.run_dir.resolve()
    payload=(run/'checkpoints/latest.pt').read_bytes();checksum=hashlib.sha256(payload).hexdigest()
    state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False);del payload
    configure(state['config']['seed']);records=[]
    if not args.saved_only:
        model=make_model(state['config']['model']).cuda().eval();model.load_state_dict(state['ema'])
        manifest=json.loads((ROOT/'runs/m5_20261103/data/D_anchor_1024.json').read_text());samples=SubsetDataset(manifest['val'],cache_size=64)
        records=denoising_details(model,samples)
        del model;torch.cuda.empty_cache()
    saved=saved_generation_details(run);summary={}
    for heading in (None,0.):
        summary[str(heading)]={str(t):{key:float(np.mean([r[key] for r in records if r['t']==t and r['heading']==heading]))
          for key in ('norm6','fk_gt_norm','fk_gt_m','fk_self_m','axis_defect','axis_angle_rad')} for t in (0,249,499,749,999)} if records else {}
    result={'run':str(run),'checkpoint_step':state['step'],'checkpoint_sha256':checksum,'denoising_summary':summary,'denoising_records':records,
            'saved_generation':saved,'source_script_sha256':sha(__file__),'loss_code_sha256':sha(ROOT/'training/m5_research_losses.py'),
            'audit_used':False,'optimizer_updates':0}
    atomic_json(args.output/'report.json',result)
    print(json.dumps({'run':run.name,'step':state['step'],'denoising_native':summary['None'],
       'saved_clips':saved['clips'] if saved else 0},indent=2),flush=True)


if __name__=='__main__':main()
