#!/usr/bin/env python3
"""CUDA regression checks for M4's new conditioning, full validation and continuation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import copy
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from training.m3 import atomic_json,batch_to,configure,forward,sha,validate_inputs
from training.m4 import M4Trainer,cfg_prediction,evaluate_full,identity_for,load_data,rotate_batch,sample
from verify_m3 import max_difference


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);args=p.parse_args()
    experiment=args.experiment.resolve();output=experiment/'verification'
    output.mkdir(parents=True,exist_ok=False)
    ready=validate_inputs();path=experiment/'data/BCD.json';manifest=json.loads(path.read_text())
    train=load_data(manifest,'train');val=load_data(manifest,'val')
    samples=[train[i] for i in [0,1,256,257,512,513,768,769]]
    config=json.loads((ROOT/'configs/m4_D.json').read_text());config['ema_every']=2
    configure(config['seed']);identity=identity_for(path,ready);trainer=M4Trainer(config,samples,identity)
    initial={k:v.cpu().clone() for k,v in trainer.model.state_dict().items()}
    batch=batch_to(samples[:4],'cuda:0');angles=torch.tensor([0.0,.4,-1.2,math.pi/2],device='cuda:0')
    with torch.inference_mode():
        rotated=rotate_batch(trainer.model,batch,angles)
        raw=trainer.model.motion_rep.unnormalize(batch['x0'])
        singleton=torch.cat([trainer.model.motion_rep.rotate(raw[i:i+1],angles[i:i+1]) for i in range(len(raw))])
        expected=trainer.model.motion_rep.normalize(singleton).masked_fill(~batch['x_pad_mask'][...,None],0)
        single_error=float((expected-rotated['x0']).abs().max());assert single_error<1e-4
        inverse=rotate_batch(trainer.model,rotated,-angles)
        inverse_error=float((inverse['x0']-batch['x0']).abs().max());assert inverse_error<5e-4,inverse_error
        assert torch.count_nonzero(rotated['x0'][~batch['x_pad_mask']])==0
        decoded=trainer.model.motion_rep.inverse(batch['x0'],is_normalized=True)
        transformed=trainer.model.motion_rep.inverse(rotated['x0'],is_normalized=True)
        c,s=torch.cos(angles),torch.sin(angles)
        positions=decoded['posed_joints'];target=positions.clone()
        target[...,0]=c[:,None,None]*positions[...,0]+s[:,None,None]*positions[...,2]
        target[...,2]=-s[:,None,None]*positions[...,0]+c[:,None,None]*positions[...,2]
        fk_error=float((transformed['posed_joints']-target)[batch['x_pad_mask']].abs().max());assert fk_error<1e-4,fk_error
    original_p=trainer.config['text_dropout'];trainer.config['text_dropout']=1.0
    prepared,dropped,_=trainer.prepare_training_batch(batch)
    assert dropped.all() and torch.count_nonzero(prepared['text_feat'])==0
    trainer.config['text_dropout']=0.0;unchanged,dropped,_=trainer.prepare_training_batch(batch)
    assert not dropped.any() and torch.equal(unchanged['text_feat'],batch['text_feat'])
    trainer.config['text_dropout']=original_p
    for _ in range(3):trainer.update()
    trainer.record_validation(1.0,1.0);trainer.save(output/'step3.pt')
    for _ in range(3):trainer.update()
    expected_model={k:v.cpu().clone() for k,v in trainer.model.state_dict().items()}
    expected_ema={k:v.cpu().clone() for k,v in trainer.ema.state_dict().items()}
    expected_opt=copy.deepcopy(trainer.optimizer.state_dict())
    expected_rng={n:getattr(trainer,n).get_state() for n in ('batch_rng','noise_rng','text_rng','heading_rng')}
    resumed=M4Trainer(config,samples,identity);resumed.restore(output/'step3.pt')
    for _ in range(3):resumed.update()
    differences={'model':max_difference(expected_model,resumed.model.state_dict()),'ema':max_difference(expected_ema,resumed.ema.state_dict()),
                 'optimizer':max_difference(expected_opt,resumed.optimizer.state_dict()),
                 **{n:max_difference(v,getattr(resumed,n).get_state()) for n,v in expected_rng.items()}}
    assert max(differences.values())==0.0,differences
    changed={name:any(not torch.equal(initial[k],v.cpu()) for k,v in resumed.model.state_dict().items() if k.startswith(name+'.'))
             for name in ('root_model','body_model')};assert all(changed.values())
    resumed.model.eval();t=torch.tensor([0,249,749,999],device='cuda:0');x=batch['x0']
    with torch.inference_mode():
        conditional=forward(resumed.model,x,batch,t);unconditional=forward(resumed.model,x,dict(batch,text_feat=torch.zeros_like(batch['text_feat'])),t)
        assert torch.equal(cfg_prediction(resumed.model,x,batch,t,1.0),conditional)
        cfg_error=float((cfg_prediction(resumed.model,x,batch,t,2.0)-(unconditional+2*(conditional-unconditional))).abs().max())
        assert cfg_error==0
    schedules=resumed.diffusion.alphas_cumprod.clone();rngs={n:getattr(resumed,n).get_state() for n in expected_rng}
    first=sample(resumed.model,batch,42,4,2.0,True);second=sample(resumed.model,batch,42,4,2.0,True)
    assert torch.equal(first,second)
    assert torch.equal(schedules,resumed.diffusion.alphas_cumprod)
    assert all(torch.equal(v,getattr(resumed,n).get_state()) for n,v in rngs.items())
    try:sample(resumed.model,batch,42,4,2.0,False)
    except ValueError:pass
    else:raise AssertionError('Untrained CFG should be rejected')
    evaluation=evaluate_full(resumed.ema,val,'cuda:0')
    assert evaluation['events']==64 and len(evaluation['strata'])==8
    assert {r['sample_id'] for r in evaluation['per_sample']}=={r['sample_id'] for r in val}
    report={'passed':True,'both_stages_updated':changed,'continuous_vs_resume_max_abs':differences,
            'yaw_batch_vs_singleton_max_abs':single_error,'yaw_inverse_max_abs':inverse_error,'yaw_fk_equivariance_error_m':fk_error,
            'text_dropout_zero_and_one_checked':True,'cfg_weight1_equals_conditional':True,'cfg_weight2_formula_max_abs':cfg_error,
            'same_seed_sampling_exact':True,'sampling_preserves_training_rngs_and_schedule':True,
            'full_validation_events':evaluation['events'],'full_validation_strata':list(evaluation['strata']),
            'm2_fingerprint':ready['fingerprint'],'m4_code_sha256':identity['m4_code_sha256']}
    atomic_json(output/'report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
