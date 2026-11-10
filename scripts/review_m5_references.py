#!/usr/bin/env python3
"""Data-only reference QC and 120-event review sheets before candidate selection."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image,ImageDraw
from training.m3 import make_model,atomic_json,motion_image,sha
from training.m5_data import SubsetDataset,GROUPS,rank
from training.m5_evaluation import GeometryMetrics,amplitude


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4);out=args.experiment/'reference_review';out.mkdir(parents=True,exist_ok=True)
    ready=json.loads((args.experiment/'data/READY.json').read_text());manifest=json.loads((args.experiment/'data'/ready['manifests']['mixed_large']).read_text())
    model=make_model({'latent_dim':64,'ff_size':128,'num_layers':1,'num_heads':4});rep=model.motion_rep;metrics=GeometryMetrics()
    validation=SubsetDataset(manifest['val'],cache_size=len(manifest['val']));records=[]
    with torch.inference_mode():
        for i in range(len(validation)):
            event=validation[i];raw=rep.inverse(event['x0'][None],is_normalized=True);motion={k:v[0] for k,v in raw.items()}
            quality,q,legacy=metrics.measure(motion)
            records.append({'sample_id':event['sample_id'],'group':event['group'],'reference_amplitude':amplitude(motion,rep.skeleton,event['group']),**quality})
    choices=[];pool=manifest['val']+manifest['train'];used=set()
    for g in GROUPS:
        for length in ('short','extended'):
            rows=[r for r in pool if r['group']==g and (r['num_frames']<=90)==(length=='short')]
            rows=sorted(rows,key=lambda r:(r['split']!='val',rank(r['sample_id'],'reference-review')))
            cell=[]
            for r in rows:
                if r['sample_id'] not in used:cell.append(r);used.add(r['sample_id'])
                if len(cell)==10:break
            if len(cell)!=10:raise ValueError('Insufficient reference review cell')
            dataset=SubsetDataset(cell,cache_size=10);sheet=Image.new('RGB',(1600,372*5),'white')
            with torch.inference_mode():
                for i in range(10):
                    event=dataset[i];motion=rep.inverse(event['x0'][None],is_normalized=True)
                    xyz=motion['posed_joints'][0].numpy();picture=motion_image(xyz,xyz,rep.skeleton.joint_parents.tolist(),event['text'])
                    draw=ImageDraw.Draw(picture);draw.rectangle((0,27,picture.width,54),fill='white')
                    draw.text((8,30),f'REFERENCE ONLY | {g}/{length} | {event["sample_id"]} | {event["length"]} frames',fill='black')
                    sheet.paste(picture.resize((800,372)),((i%2)*800,(i//2)*372))
                    choices.append(dict(cell[i],sheet=f'{g}_{length}.png',sheet_index=i))
            sheet.save(out/f'{g}_{length}.png');print('reference sheet',g,length,flush=True)
    atomic_json(out/'selected_120.json',choices)
    result={'reference_only':True,'trained_model_outputs_used':False,'validation_events':len(records),'review_sheet_events':len(choices),
      'all_valid_qpos':all(r['valid_qpos'] for r in records),'max_reference_penetration_m':max(r['max_penetration_m'] for r in records),
      'reference_metrics':records,'agent_visual_review':'pending','independent_human_review':'pending',
      'manifest_sha256':sha(args.experiment/'data'/ready['manifests']['mixed_large']),'script_sha256':sha(__file__)}
    atomic_json(out/'report.json',result);print({k:v for k,v in result.items() if k!='reference_metrics'},flush=True)


if __name__=='__main__':main()
