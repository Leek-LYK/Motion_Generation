#!/usr/bin/env python3
"""Produce complete validation contact sheets without labeling human review as complete."""
import argparse
import json
from pathlib import Path
import sys
import textwrap

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from training.m3 import configure,make_model,atomic_json
from training.m4 import GROUPS,load_data


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);args=p.parse_args()
    manifest=json.loads((args.experiment/'data/BCD.json').read_text())
    configure(20261102);model=make_model({'latent_dim':64,'ff_size':128,'num_layers':1,'num_heads':4})
    samples=load_data(manifest,'val');output=args.experiment/'data/reference_review';output.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    parents=model.motion_rep.skeleton.joint_parents.tolist();metrics=[]
    with torch.inference_mode():
        for group in GROUPS:
            sheet=Image.new('RGB',(1600,1400),'white');draw=ImageDraw.Draw(sheet)
            selected=[s for s in samples if s['group']==group]
            for row,sample in enumerate(selected):
                motion=model.motion_rep.inverse(sample['x0'][None],is_normalized=True)
                xyz=motion['posed_joints'][0].numpy();root=motion['root_positions'][0].numpy()
                metric={'sample_id':sample['sample_id'],'group':group,'cohort':sample['cohort'],'text':sample['text'],
                        'frames':len(xyz),'root_travel_m':float(np.linalg.norm(root[-1,[0,2]]-root[0,[0,2]])),
                        'root_height_range_m':[float(root[:,1].min()),float(root[:,1].max())],
                        'finite':bool(np.isfinite(xyz).all())};metrics.append(metric)
                x0,y0=(row%4)*400,(row//4)*350
                draw.rectangle((x0,y0,x0+399,y0+349),outline=(180,180,180))
                draw.text((x0+6,y0+4),f'{row:02d} {group} | {sample["cohort"]}',font=font,fill='black')
                for line,text in enumerate(textwrap.wrap(sample['text'],53)[:5]):
                    draw.text((x0+6,y0+23+line*15),text,font=font,fill='black')
                draw.text((x0+6,y0+106),f'travel {metric["root_travel_m"]:.2f}m; root Y {metric["root_height_range_m"][0]:.2f}..{metric["root_height_range_m"][1]:.2f}m',font=font,fill='black')
                for col,t in enumerate(np.linspace(0,len(xyz)-1,4).astype(int)):
                    relative=xyz[t].copy();relative[:,[0,2]]-=root[t,[0,2]]
                    for view,axis in enumerate((0,2)):
                        px=relative[:,axis]*48+x0+50+100*col
                        py=-relative[:,1]*48+y0+220+view*110
                        for j,parent in enumerate(parents):
                            if parent>=0:draw.line((px[j],py[j],px[parent],py[parent]),fill=(20,90,170),width=2)
                        draw.text((x0+8+col*100,y0+126+view*110),f'{"XY" if view==0 else "ZY"} f{t}',font=font,fill='black')
            sheet.save(output/f'{group}.png')
    atomic_json(output/'numeric_reference_audit.json',{'samples':len(metrics),'all_finite':all(r['finite'] for r in metrics),
                'human_review':'pending','samples_metadata':metrics})
    print(output,flush=True)


if __name__=='__main__':main()
