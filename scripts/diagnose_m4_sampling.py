#!/usr/bin/env python3
"""Independent DDIM arithmetic and partial-denoising probes for the M4 collapse."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from kimodo.model.diffusion import Diffusion,DDIMSampler
from training.m3 import atomic_json,batch_to,configure,forward,make_model
from training.m4 import GROUPS,load_data,sample
from training.m4_readiness import movement


@torch.inference_mode()
def independent(model,batch,noise,start_t=999,oracle=False):
    diffusion=Diffusion(1000).cuda();schedule=torch.linspace(0,start_t,100,device='cuda:0').round().long()
    alpha=diffusion.alphas_cumprod_base
    x=noise.clone() if start_t==999 and not oracle else diffusion.q_sample(batch['x0'],torch.full((len(noise),),start_t,device='cuda:0',dtype=torch.long),noise)
    for i in range(len(schedule)-1,-1,-1):
        t=schedule[i].expand(len(x));clean=batch['x0'] if oracle else forward(model,x,batch,t)
        a=alpha[schedule[i]];previous=alpha[schedule[i-1]] if i else torch.ones((),device='cuda:0')
        epsilon=(x-a.sqrt()*clean)/(1-a).sqrt()
        x=previous.sqrt()*clean+(1-previous).sqrt()*epsilon
        x=x.masked_fill(~batch['x_pad_mask'][...,None],0)
    return x


def main():
    run=ROOT/'runs/m4_20261102/D_1024_heading';out=ROOT/'runs/m4_research_20261103/diagnostics'
    out.mkdir(parents=True,exist_ok=True)
    state=torch.load(run/'checkpoints/best_val_ema.pt',map_location='cpu',weights_only=False)
    configure(state['config']['seed']);model=make_model(state['config']['model'],run/'stats/motion').cuda().eval();model.load_state_dict(state['ema'])
    manifest=json.loads((run/'subset.json').read_text());val=load_data(manifest,'val')
    events=[next(s for s in val if s['group']==g and s['cohort']=='seen_text_new_motion') for g in GROUPS]
    batch=batch_to(events,'cuda:0');generator=torch.Generator(device='cuda:0').manual_seed(42)
    noise=torch.randn(batch['x0'].shape,device='cuda:0',generator=generator)
    with torch.inference_mode():
        official=sample(model,batch,42,100,1,True)
        separate=independent(model,batch,noise)
        error=float((official-separate).abs().max());assert error<.001,error
        oracle=independent(model,batch,noise,oracle=True)
        oracle_error=float((oracle-batch['x0']).abs().max());assert oracle_error<1e-5,oracle_error
        probes={}
        for t in (249,499,749,999):
            pred=independent(model,batch,noise,t)
            rows=[]
            for i,event in enumerate(events):
                n=event['length'];a=model.motion_rep.inverse(pred[i:i+1,:n],is_normalized=True)
                b=model.motion_rep.inverse(batch['x0'][i:i+1,:n],is_normalized=True)
                pm={k:v[0] for k,v in a.items()};rm={k:v[0] for k,v in b.items()}
                rows.append({'group':event['group'],'world_mpjpe_m':float(torch.linalg.vector_norm(a['posed_joints']-b['posed_joints'],dim=-1).mean()),
                    'generated_amplitude':movement(pm,model.motion_rep.skeleton,event['group']),
                    'reference_amplitude':movement(rm,model.motion_rep.skeleton,event['group'])})
            probes[str(t)]=rows
        # A frozen pose must have zero motion under every amplitude definition.
        motion=model.motion_rep.inverse(batch['x0'][:1,:events[0]['length']],is_normalized=True)
        static={k:v[0,:1].expand_as(v[0]) for k,v in motion.items()}
        static_amplitudes={g:movement(static,model.motion_rep.skeleton,g) for g in GROUPS}
        assert max(static_amplitudes.values())<1e-6
    report={'passed':True,'independent_vs_existing_ddim_max_abs':error,'oracle_ddim_reconstruction_max_abs':oracle_error,
        'static_pose_amplitudes':static_amplitudes,'partial_denoising_probes':probes,
        'interpretation_boundary':'Four fixed held-out examples; lower starting t retains reference information and is not pure text generation.'}
    atomic_json(out/'sampling_audit.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
