#!/usr/bin/env python3
"""Choose extra-loss coefficients from frozen-parent training gradients only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from training.m3 import atomic_json,configure,make_model,batch_to,forward,sha,validate_inputs
from training.m4 import rotate_batch
from training.m5_data import SubsetDataset
from training.m5_research_losses import HingeGeometry,research_objective
from kimodo.model.diffusion import Diffusion


def vector_norm(values):return torch.sqrt(sum(v.square().sum() for v in values if v is not None))
def dot(a,b):return sum((x*y).sum() for x,y in zip(a,b) if x is not None and y is not None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);args=p.parse_args();root=args.experiment
    verified=json.loads((root/'verification/report.json').read_text());assert verified['passed']
    for name,h in verified['code_sha256'].items():assert sha(ROOT/name)==h
    if (root/'calibration/report.json').exists():raise FileExistsError('Calibration already frozen')
    ready=validate_inputs();parent=ROOT/'runs/m4_research_20261103/R3_D_high_noise/checkpoints/latest.pt'
    state=torch.load(parent,map_location='cpu',weights_only=False);configure(6101)
    model=make_model(state['config']['model']).cuda().train();model.load_state_dict(state['model'])
    hinge=HingeGeometry(model.motion_rep.skeleton,'cuda:0');manifest=json.loads((ROOT/'runs/m5_20261103/data/D_anchor_1024.json').read_text())
    dataset=SubsetDataset(manifest['train'],cache_size=128);diffusion=Diffusion(1000).cuda()
    cpu=torch.Generator().manual_seed(6101);rng=torch.Generator(device='cuda').manual_seed(6102)
    parameters=[p for p in model.body_model.parameters() if p.requires_grad];records=[]
    for batch_index in range(8):
        indices=torch.randint(len(dataset),(16,),generator=cpu).tolist();events=dataset.batch(indices);batch=batch_to(events,'cuda:0')
        angles=(torch.rand(16,device='cuda',generator=rng)*2-1)*torch.pi;batch=rotate_batch(model,batch,angles)
        with torch.no_grad():
            truth_motion=model.motion_rep.inverse(batch['x0'],is_normalized=True)
            truth_axis,_=hinge.defects(truth_motion['local_rot_mats'])
            truth_axis_mean=float((truth_axis*batch['x_pad_mask'][...,None]).sum()/(batch['x_pad_mask'].sum()*29))
            if truth_axis_mean>1e-5:raise ValueError('Real training reference not compatible with hinge-axis contract')
        dropped=torch.rand(16,device='cuda',generator=rng)<.1;batch['text_feat']=batch['text_feat'].masked_fill(dropped[:,None,None],0)
        t=torch.randint(1000,(16,),device='cuda',generator=rng);replace=torch.rand(16,device='cuda',generator=rng)<.5
        high=torch.randint(800,1000,(16,),device='cuda',generator=rng);t=torch.where(replace,high,t)
        noise=torch.randn(batch['x0'].shape,device='cuda',generator=rng)
        pred=forward(model,diffusion.q_sample(batch['x0'],t,noise),batch,t)
        base,parts=research_objective(model.motion_rep,pred,batch['x0'],batch['x_pad_mask'],{'fk_weight':1.},hinge)
        gradients={k:torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
          for k,value in [('base',base),('consistency',parts['consistency_loss_m']),('axis',parts['axis_defect'])]}
        norms={k:float(vector_norm(g)) for k,g in gradients.items()}
        if not all(np.isfinite(v) and v>1e-10 for v in norms.values()):raise ValueError('Unusable gradient scale')
        records.append({'batch':batch_index,'sample_ids':[e['sample_id'] for e in events],'t':t.cpu().tolist(),'ground_truth_axis_energy':truth_axis_mean,
           'body_gradient_norms':norms,'losses':{k:float(parts[k].detach()) for k in ('norm6_comparison','fk_gt_norm','consistency_loss_m','axis_defect')},
           'cosine_with_base':{k:float(dot(gradients['base'],gradients[k])/(norms['base']*norms[k])) for k in ('consistency','axis')}})
        del gradients,parts,base,pred
    ratios={k:float(np.median([r['body_gradient_norms']['base']/r['body_gradient_norms'][k] for r in records])) for k in ('consistency','axis')}
    coefficients={k:{label:ratios[k]*fraction for label,fraction in [('low',.1),('medium',.3)]} for k in ratios}
    if not all(1e-6<=v<=20 for table in coefficients.values() for v in table.values()):raise ValueError('Unexpected coefficient scale; inspect before training')
    result={'passed':True,'parent_sha256':sha(parent),'m2_fingerprint':ready['fingerprint'],'training_events_only':True,
      'validation_scores_used_to_choose_weights':False,'optimizer_updates':0,'body_gradient_ratio_medians':ratios,'coefficients':coefficients,
      'target_initial_gradient_fractions':{'low':.1,'medium':.3},'records':records,'script_sha256':sha(__file__),
      'loss_code_sha256':sha(ROOT/'training/m5_research_losses.py'),
      'note':'Weights are fixed after training-only calibration; gradient fractions may change as learning progresses.'}
    atomic_json(root/'calibration/report.json',result);print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':main()
