#!/usr/bin/env python3
"""Compare true and decoded motion against the separately predicted velocity channels."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from training.m3 import make_model,batch_to,configure,atomic_json,sha
from training.m4 import load_data,sample,GROUPS
from training.m4_readiness import movement


def main():
    root=ROOT/'runs/m4_research_20261103';run=root/'R1_D_continue'
    state=torch.load(run/'checkpoints/best_val_ema.pt',map_location='cpu',weights_only=False)
    configure(state['config']['seed']);model=make_model(state['config']['model'],run/'stats/motion').cuda().eval()
    model.load_state_dict(state['ema']);manifest=json.loads((run/'subset.json').read_text())
    train=load_data(manifest,'train');val=load_data(manifest,'val');rows=[]
    with torch.inference_mode():
        for split,events in [('train',train),('val',val)]:
            chosen=[s for g in GROUPS for s in [x for x in events if x['group']==g][:8]]
            for start in range(0,len(chosen),8):
                chunk=chosen[start:start+8];batch=batch_to(chunk,'cuda:0');features=sample(model,batch,42+start,100,1,True)
                for i,s in enumerate(chunk):
                    n=s['length'];pred=model.motion_rep.inverse(features[i:i+1,:n],is_normalized=True)
                    ref=model.motion_rep.inverse(batch['x0'][i:i+1,:n],is_normalized=True)
                    pa={k:v[0] for k,v in pred.items()};ra={k:v[0] for k,v in ref.items()}
                    pf=model.motion_rep.unnormalize(features[i,:n]);rf=model.motion_rep.unnormalize(batch['x0'][i,:n])
                    rows.append({'split':split,'sample_id':s['sample_id'],'group':s['group'],'text':s['text'],
                        'reference_amplitude':movement(ra,model.motion_rep.skeleton,s['group']),
                        'generated_amplitude':movement(pa,model.motion_rep.skeleton,s['group']),
                        'reference_velocity_feature_rms':float(rf[:,311:413].square().mean().sqrt()),
                        'predicted_velocity_feature_rms':float(pf[:,311:413].square().mean().sqrt()),
                        'decoded_velocity_rms':float((torch.diff(pa['posed_joints'],dim=0)*30).square().mean().sqrt())})
                print(split,chunk[0]['group'],'done',flush=True)
    summary={}
    for split in ['train','val']:
        for group in GROUPS:
            rs=[r for r in rows if r['split']==split and r['group']==group]
            summary[split+'/'+group]={k:sum(r[k] for r in rs)/len(rs) for k in
                ['reference_amplitude','generated_amplitude','reference_velocity_feature_rms','predicted_velocity_feature_rms','decoded_velocity_rms']}
    atomic_json(root/'diagnostics/R1_train_vs_val_velocity.json',{'checkpoint_step':state['step'],'summary':summary,'rows':rows,'script_sha256':sha(__file__)})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
