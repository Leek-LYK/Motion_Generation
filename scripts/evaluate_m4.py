#!/usr/bin/env python3
"""Evaluate the selected best-val EMA; controlled text/yaw probes and all-validation generation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import ImageDraw
from kimodo.model.diffusion import Diffusion
from kimodo.exports.mujoco import MujocoQposConverter
from training.m3 import atomic_json,batch_to,configure,forward,make_model,motion_image,sha,validate_inputs
from training.m4 import GROUPS,evaluate_full,export_preview,load_data,per_sample_losses,sample


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();run=args.run_dir.resolve();output=run/'final_evaluation';output.mkdir(exist_ok=True)
    completed=json.loads((run/'COMPLETED.json').read_text())
    checkpoint=run/'checkpoints/best_val_ema.pt'
    checksum=sha(checkpoint)
    if checksum!=completed['best_val_ema_checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
    if (output/'COMPLETE.json').exists():
        if json.loads((output/'COMPLETE.json').read_text())['checkpoint_sha256']!=checksum:raise ValueError('Evaluation checkpoint changed')
        print('Evaluation already complete');return
    ready=validate_inputs();state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['identity']['m2_fingerprint']!=ready['fingerprint']:raise ValueError('Data changed')
    for name,digest in state['identity']['m4_code_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Implementation changed: '+name)
    config=state['config'];configure(config['seed'])
    model=make_model(config['model'],run/'stats/motion').cuda().eval();model.load_state_dict(state['ema'])
    manifest=json.loads((run/'subset.json').read_text());validation=load_data(manifest,'val')
    native=evaluate_full(model,validation,'cuda:0')
    rotated=evaluate_full(model,validation,'cuda:0',heading_offset=math.pi/2)
    dropped=config['text_dropout']>0
    weights=[1.0,1.5,2.0] if dropped else [1.0]
    seeds=[42] if args.smoke else [0,42,1234]
    ddim_steps=4 if args.smoke else 100
    converter=MujocoQposConverter(model.motion_rep.skeleton)
    records=[]
    with torch.inference_mode():
        for weight in weights:
            for seed in seeds:
                for start in range(0,len(validation),16):
                    chunk=validation[start:start+16];batch=batch_to(chunk,'cuda:0')
                    generated=sample(model,batch,seed+start,ddim_steps,weight,dropped)
                    for i,event in enumerate(chunk):
                        n=event['length'];motion=model.motion_rep.inverse(generated[i:i+1,:n],is_normalized=True)
                        reference=model.motion_rep.inverse(batch['x0'][i:i+1,:n],is_normalized=True)
                        a,b=motion['posed_joints'][0],reference['posed_joints'][0]
                        qpos=converter.dict_to_qpos(motion)[0]
                        if not np.isfinite(qpos).all():raise FloatingPointError('Invalid generated qpos')
                        feet=a[:,model.motion_rep.skeleton.foot_joint_idx];contacts=motion['foot_contacts'][0]
                        active=contacts[:-1]&contacts[1:]
                        speed=torch.linalg.vector_norm(torch.diff(feet,dim=0),dim=-1)*30
                        records.append({'sample_id':event['sample_id'],'group':event['group'],'cohort':event['cohort'],
                            'seed':seed,'batch_seed':seed+start,'guidance':weight,'frames':n,
                            'world_mpjpe_m':float(torch.linalg.vector_norm(a-b,dim=-1).mean()),
                            'root_relative_mpjpe_m':float(torch.linalg.vector_norm((a-a[:,:1])-(b-b[:,:1]),dim=-1).mean()),
                            'max_foot_penetration_m':float(torch.clamp(-feet[...,1].min(),min=0)),
                            'predicted_contact_fraction':float(contacts.float().mean()),
                            'predicted_contact_foot_speed_mps':float(speed[active].mean()) if active.any() else None,
                            'finite_qpos':True})
                print(f"generated cfg={weight} seed={seed} events={len(validation)}",flush=True)
        # All classes and cohorts, identical xt; cross-class shuffle avoids same-caption swaps.
        batch=batch_to(validation,'cuda:0');t=torch.full((len(validation),),999,device='cuda:0',dtype=torch.long)
        generator=torch.Generator(device='cuda:0').manual_seed(20261102)
        noisy=Diffusion(1000).cuda().q_sample(batch['x0'],t,torch.randn(batch['x0'].shape,device='cuda:0',generator=generator))
        ablation={}
        for label,text in [('correct',batch['text_feat']),('cross_class_shuffle',batch['text_feat'].roll(16,0)),('zero',torch.zeros_like(batch['text_feat']))]:
            prediction=forward(model,noisy,dict(batch,text_feat=text),t)
            loss,_=per_sample_losses(prediction,batch['x0'],batch['x_pad_mask']);ablation[label]=float(loss.mean())
    representatives=[next(s for s in validation if s['group']==g and s['cohort']==cohort)
                     for g in GROUPS for cohort in ('seen_text_new_motion','unseen_text_same_class')]
    for weight in weights:
        export_preview(model,representatives,output/f'previews_cfg{weight:g}',42,ddim_steps,weight,dropped)
    # Identical duration, global heading and initial noise for every text.
    controlled=output/'prompt_swap';controlled.mkdir(exist_ok=True);control_records=[]
    with torch.inference_mode():
        for event in representatives:
            modified=dict(event,length=75,x0=torch.zeros(75,417),first_heading_angle=torch.tensor(0.0))
            predicted=sample(model,batch_to([modified],'cuda:0'),42,ddim_steps,1.0,dropped)
            motion=model.motion_rep.inverse(predicted,is_normalized=True);name=event['group']+'_'+event['cohort']
            qpos=converter.dict_to_qpos(motion)[0]
            np.savetxt(controlled/f'{name}.csv',qpos,delimiter=',')
            np.savez(controlled/f'{name}.npz',**{k:v[0].cpu().numpy() for k,v in motion.items()})
            xyz=motion['posed_joints'][0].cpu().numpy()
            picture=motion_image(xyz,xyz,model.motion_rep.skeleton.joint_parents.cpu().tolist(),event['text'])
            draw=ImageDraw.Draw(picture);draw.rectangle((0,27,picture.width,55),fill='white')
            draw.text((12,30),'Generated only | common noise seed 42 / length 75 / heading 0 / CFG 1',fill='black')
            picture.save(controlled/f'{name}.png')
            control_records.append({'sample_id':event['sample_id'],'text':event['text'],'group':event['group'],
                                    'cohort':event['cohort'],'seed':42,'heading':0,'frames':75})
    summaries={}
    for weight in weights:
        rows=[r for r in records if r['guidance']==weight]
        per_group={g:float(np.mean([r['world_mpjpe_m'] for r in rows if r['group']==g])) for g in GROUPS}
        summaries[str(weight)]={'generated_sequences':len(rows),'world_mpjpe_macro_m':float(np.mean(list(per_group.values()))),
             'world_mpjpe_by_class_m':per_group,
             'world_mpjpe_by_cohort_m':{c:float(np.mean([r['world_mpjpe_m'] for r in rows if r['cohort']==c])) for c in ('seen_text_new_motion','unseen_text_same_class')},
             'world_mpjpe_p95_m':float(np.quantile([r['world_mpjpe_m'] for r in rows],.95)),
             'root_relative_mpjpe_mean_m':float(np.mean([r['root_relative_mpjpe_m'] for r in rows])),
             'all_qpos_finite':all(r['finite_qpos'] for r in rows)}
    report={'passed':True,'stage':config['stage'],'checkpoint':str(checkpoint),'checkpoint_sha256':checksum,
            'checkpoint_step':state['step'],'final_training_step':completed['step'],'weights':'ema',
            'validation_events':len(validation),'native_validation':native,'rotated90_validation':rotated,
            'generation_summary':summaries,'text_ablation_t999':ablation,'zero_text_trained':dropped,
            'seeds':seeds,'ddim_steps':ddim_steps,'smoke':args.smoke,'postprocess':False,
            'limitations':['Small rule-selected validation set; semantic human review pending',
                'Paired generation errors are diagnostics, not a unique correct-motion criterion',
                'Foot speed uses predicted contact; report contact coverage alongside it',
                'No physics controller or official test evaluation']}
    atomic_json(output/'report.json',report);atomic_json(output/'per_generation.json',records)
    atomic_json(controlled/'manifest.json',control_records)
    atomic_json(output/'COMPLETE.json',{'passed':True,'checkpoint_sha256':checksum,'report_sha256':sha(output/'report.json'),
                'validation_events':len(validation),'generated_sequences':len(records),'smoke':args.smoke})
    print(json.dumps({'stage':config['stage'],'checkpoint_step':state['step'],'validation_macro':native['macro_loss'],
                       'generation':summaries},indent=2),flush=True)


if __name__=='__main__':main()
