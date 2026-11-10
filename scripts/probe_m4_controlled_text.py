#!/usr/bin/env python3
"""Repeat the same eight controlled prompts under three shared-noise seeds."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,sys,hashlib,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image,ImageDraw
from kimodo.exports.mujoco import MujocoQposConverter
from training.m3 import atomic_json,batch_to,configure,make_model,motion_image,sha
from training.m4 import GROUPS,load_data,sample
from training.m4_readiness import movement


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--checkpoint',default='best_val_ema.pt');p.add_argument('--output',type=Path)
    p.add_argument('--guidance',type=float,default=1.0)
    p.add_argument('--group',choices=list(GROUPS));p.add_argument('--cohort',choices=['seen_text_new_motion','unseen_text_same_class']);args=p.parse_args()
    run=args.run_dir.resolve();out=args.output or run/'controlled_three_seeds';out.mkdir(parents=True,exist_ok=False)
    checkpoint=run/'checkpoints'/args.checkpoint;payload=checkpoint.read_bytes();checkpoint_hash=hashlib.sha256(payload).hexdigest()
    state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False);del payload
    configure(state['config']['seed']);model=make_model(state['config']['model'],run/'stats/motion').cuda().eval();model.load_state_dict(state['ema'])
    events=load_data(json.loads((run/'subset.json').read_text()),'val')
    if args.group or args.cohort:
        chosen=[s for s in events if (not args.group or s['group']==args.group) and (not args.cohort or s['cohort']==args.cohort)]
    else:
        chosen=[next(s for s in events if s['group']==g and s['cohort']==c) for g in GROUPS for c in ('seen_text_new_motion','unseen_text_same_class')]
    controlled=[dict(s,length=75,x0=torch.zeros(75,417),first_heading_angle=torch.tensor(0.0)) for s in chosen]
    converter=MujocoQposConverter(model.motion_rep.skeleton);records=[]
    with torch.inference_mode():
        for seed in (0,42,1234):
            batch=batch_to(controlled,'cuda:0');features=sample(model,batch,seed,100,args.guidance,state['config']['text_dropout']>0,shared_noise=True)
            folder=out/f'seed_{seed}';folder.mkdir();sheet=Image.new('RGB',(1600,372*((len(chosen)+1)//2)),'white')
            for i,event in enumerate(chosen):
                decoded=model.motion_rep.inverse(features[i:i+1],is_normalized=True)
                motion={k:v[0] for k,v in decoded.items()};qpos=converter.dict_to_qpos(decoded)[0]
                ref=model.motion_rep.inverse(event['x0'][None].cuda(),is_normalized=True)
                reference={k:v[0] for k,v in ref.items()}
                amp=movement(motion,model.motion_rep.skeleton,event['group']);refamp=movement(reference,model.motion_rep.skeleton,event['group'])
                xyz=motion['posed_joints'].cpu().numpy();name=event['group']+'_'+event['cohort']
                if args.group or args.cohort:name=f'{i:02d}_'+name
                picture=motion_image(xyz,xyz,model.motion_rep.skeleton.joint_parents.cpu().tolist(),event['text'])
                draw=ImageDraw.Draw(picture);draw.rectangle((0,27,picture.width,55),fill='white')
                draw.text((12,30),f'Generated only | common noise seed {seed} / 75 frames / heading 0 / CFG{args.guidance:g} | amplitude {amp:.3f}',fill='black')
                picture.save(folder/f'{name}.png');sheet.paste(picture.resize((800,372)),((i%2)*800,(i//2)*372))
                np.savetxt(folder/f'{name}.csv',qpos,delimiter=',');np.savez(folder/f'{name}.npz',**{k:v.cpu().numpy() for k,v in motion.items()})
                records.append({'seed':seed,'group':event['group'],'cohort':event['cohort'],'text':event['text'],
                    'sample_id':event['sample_id'],'generated_amplitude':amp,'reference_event_amplitude':refamp,
                    'reference_event_frames':event['length'],'generated_frames':75,'heading':0,'guidance':args.guidance,
                    'finite_qpos':bool(np.isfinite(qpos).all())})
            sheet.save(out/f'seed_{seed}_sheet.png');print('Controlled seed',seed,'complete',flush=True)
    atomic_json(out/'report.json',{'checkpoint_sha256':checkpoint_hash,'checkpoint_step':state['step'],'guidance':args.guidance,
        'cases':records,'semantic_review':'pending','note':'Reference durations differ; these are controlled qualitative probes, not paired trajectory scores.'})


if __name__=='__main__':main()
