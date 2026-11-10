#!/usr/bin/env python3
"""Same-noise/length/heading text controls, with zero and shuffled text ablations."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,io,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import numpy as np
from PIL import Image,ImageDraw
from training.m3 import configure,make_model,batch_to,atomic_json,motion_image,sha
from training.m5_data import SubsetDataset,GROUPS
from training.m5_evaluation import sample_keyed,GeometryMetrics,amplitude


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--checkpoint',default='latest.pt')
    p.add_argument('--output',type=Path);args=p.parse_args();run=args.run_dir.resolve();out=args.output or run/'controlled_terminal'
    if (out/'report.json').exists():raise FileExistsError(out)
    payload=(run/'checkpoints'/args.checkpoint).read_bytes();checksum=hashlib.sha256(payload).hexdigest()
    state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False);del payload
    configure(state['config']['seed']);model=make_model(state['config']['model']).cuda().eval();model.load_state_dict(state['ema'])
    manifest=json.loads((run/'subset.json').read_text());rows=[]
    for g in GROUPS:
        options=[r for r in manifest['val'] if r['group']==g]
        for cohort in sorted({r.get('cohort','unknown') for r in options}):
            rows.append(next(r for r in options if r.get('cohort','unknown')==cohort))
    dataset=SubsetDataset(rows);events=[]
    for i in range(len(dataset)):
        event=dict(dataset[i]);event.update(length=75,x0=torch.zeros(75,417),first_heading_angle=torch.tensor(0.),sample_id='M5-shared-noise-control')
        events.append(event)
    batch=batch_to(events,'cuda:0');metrics=GeometryMetrics(state['config'].get('metric_thresholds'));records=[];sensitivity=[]
    out.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for seed in (0,42,1234):
            outputs={mode:sample_keyed(model,batch,events,seed,100,2.,text_mode=mode) for mode in ('real','zero','shuffled')}
            zero_difference=float((outputs['zero']-outputs['zero'][:1]).abs().max())
            shuffle_error=float((outputs['shuffled']-outputs['real'].roll(1,0)).abs().max())
            if zero_difference>1e-4 or shuffle_error>1e-4:raise AssertionError('Matched-noise text-control permutation failed')
            sensitivity.append({'seed':seed,'zero_text_across_rows_max_difference':zero_difference,'shuffled_vs_real_permutation_max_difference':shuffle_error,
              'real_vs_zero_normalized_feature_rms':float((outputs['real']-outputs['zero']).square().mean().sqrt())})
            for mode,features in outputs.items():
                folder=out/f'seed_{seed}'/mode;folder.mkdir(parents=True,exist_ok=True)
                sheet=Image.new('RGB',(1600,372*((len(events)+1)//2)),'white')
                for i,event in enumerate(events):
                    raw=model.motion_rep.inverse(features[i:i+1],is_normalized=True);motion={k:v[0] for k,v in raw.items()}
                    info,q,legacy=metrics.measure(motion,rows[i]['group']);name=f'{i:02d}_{rows[i]["group"]}'
                    np.savez_compressed(folder/f'{name}.npz',**{k:v.cpu().numpy() for k,v in motion.items()})
                    np.save(folder/f'{name}_features.npy',features[i].cpu().numpy());np.savetxt(folder/f'{name}.csv',q,delimiter=',');np.savetxt(folder/f'{name}_legacy.csv',legacy,delimiter=',')
                    xyz=motion['posed_joints'].cpu().numpy();picture=motion_image(xyz,xyz,model.motion_rep.skeleton.joint_parents.cpu().tolist(),rows[i]['text'])
                    draw=ImageDraw.Draw(picture);draw.rectangle((0,27,picture.width,55),fill='white');draw.text((8,30),f'{mode} text | common noise {seed}, 75 frames, heading 0, CFG2',fill='black')
                    sheet.paste(picture.resize((800,372)),((i%2)*800,(i//2)*372))
                    records.append({'sample_id':rows[i]['sample_id'],'group':rows[i]['group'],'cohort':rows[i].get('cohort'),
                      'text':rows[i]['text'],'text_mode':mode,'seed':seed,'artifact_prefix':str((folder/name).relative_to(out)),
                      'generated_amplitude':amplitude(motion,model.motion_rep.skeleton,rows[i]['group']),**info})
                sheet.save(folder/'sheet.png')
            print('Text controls seed',seed,'complete',flush=True)
    atomic_json(out/'report.json',{'checkpoint_sha256':checksum,'step':state['step'],'seeds':[0,42,1234],
       'frames':75,'heading':0,'guidance':2,'shared_noise_across_texts':True,'text_sensitivity':sensitivity,'cases':records,
       'semantic_correctness':'pending visual review; sensitivity does not prove text correctness','script_sha256':sha(__file__)})


if __name__=='__main__':main()
