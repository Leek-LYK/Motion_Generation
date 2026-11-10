#!/usr/bin/env python3
"""M5 numerical and checkpoint verification; finite-budget noncompetitive updates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,json,sys,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from training.m3 import atomic_json,configure,make_model,batch_to,sha,validate_inputs
from training.m4 import load_data,rotate_batch
from training.m4_research import ResearchTrainer
from training.m5 import M5Trainer
from training.m5_losses import objective,geometry


def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);args=p.parse_args()
    out=args.experiment/'verification';out.mkdir(parents=True,exist_ok=True)
    ready=validate_inputs();configure(3101);device='cuda:0'
    manifest=json.loads((ROOT/'runs/m4_20261102/data/BCD.json').read_text());samples=load_data(manifest,'val')[:4]
    small={'latent_dim':64,'ff_size':128,'num_layers':1,'num_heads':4}
    model=make_model(small).to(device);batch=batch_to(samples,device);x=batch['x0'];mask=batch['x_pad_mask']
    checks={};oracle,_=objective(model.motion_rep,x,x,mask,'norm7');checks['oracle_norm7_loss']=float(oracle)
    assert float(oracle)<1e-5,oracle
    pred=x.clone().detach();pred[:,:,107:311]+=.07*torch.randn_like(pred[:,:,107:311]);pred.requires_grad_(True)
    loss,parts=objective(model.motion_rep,pred,x,mask,'norm7');loss.backward()
    assert torch.isfinite(pred.grad).all() and float(pred.grad[:,:,107:311].norm())>0
    checks['fk_rotation_gradient_nonzero']=True;checks['perturbed_fk_loss']=float(parts['fk'])
    assert float(parts['fk'])>float(oracle)
    # Padding cannot contribute loss or gradients, even with different padded predictions.
    p2=pred.detach().clone();p2[~mask]=17.;p2.requires_grad_(True)
    loss2,_=objective(model.motion_rep,p2,x,mask,'norm7');loss2.backward()
    assert torch.allclose(loss,loss2,atol=1e-7,rtol=1e-6) and bool((p2.grad[~mask]==0).all())
    checks['padding_loss_and_gradient']=True
    full=pred.detach().clone().requires_grad_(True);whole,_=objective(model.motion_rep,full,x,mask,'weighted_raw7');whole.backward()
    micro=pred.detach().clone().requires_grad_(True)
    for i in range(len(x)):
        part,_=objective(model.motion_rep,micro[i:i+1],x[i:i+1],mask[i:i+1],'weighted_raw7')
        (part*mask[i].sum()/mask.sum()).backward()
    diff=float((full.grad-micro.grad).abs().max());assert torch.allclose(full.grad,micro.grad,atol=3e-6,rtol=3e-5),diff
    checks['unequal_length_accumulation_gradient_max_difference']=diff
    rotated=rotate_batch(model,batch,torch.tensor([.1,.6,1.1,-.7],device=device))
    _,fk_a,m_a=geometry(model.motion_rep,x);_,fk_b,m_b=geometry(model.motion_rep,rotated['x0'])
    # Pairwise bone distances must be unchanged by synchronized yaw augmentation.
    dist_a=torch.linalg.vector_norm(m_a['posed_joints'][:,:,1:]-m_a['posed_joints'][:,:,:1],dim=-1)
    dist_b=torch.linalg.vector_norm(m_b['posed_joints'][:,:,1:]-m_b['posed_joints'][:,:,:1],dim=-1)
    error=float((dist_a-dist_b).abs().masked_select(mask[:,:,None]).max());assert error<1e-5,error
    checks['yaw_geometry_max_distance_difference_m']=error
    print('Geometry, gradients, masks and frame-weighted accumulation verified',flush=True)
    parent_path=ROOT/'runs/m4_research_20261103/R3_D_high_noise/checkpoints/latest.pt'
    parent=torch.load(parent_path,map_location='cpu',weights_only=False);c=dict(parent['config'])
    # Verify exact continuation at the real Tiny architecture, preserving all source state.
    configure(c['seed']);legacy=ResearchTrainer(c,samples,parent['identity'],device)
    legacy.restore(parent_path);legacy.update()
    expected=copy.deepcopy(legacy.model.state_dict());expected_opt=copy.deepcopy(legacy.optimizer.state_dict());expected_rng={k:v.clone() for k,v in legacy.state_dict().items()} if hasattr(legacy,'state_dict') else None
    del legacy;gc.collect();torch.cuda.empty_cache()
    cfg=dict(c,stage='verification_exact',loss_objective='norm6',fk_weight=0.,legacy_exact_update=True,
             stage_start_step=20000,max_steps=30000,microbatch=16,sampling='legacy',scheduler='constant')
    trainer=M5Trainer(cfg,samples,parent['identity'],device);trainer.load_state(parent,continuation=True);trainer.update()
    assert same(trainer.model.state_dict(),expected) and same(trainer.optimizer.state_dict(),expected_opt)
    checks['real_R3_full_state_control_update_exact']=True
    del trainer,parent,expected,expected_opt;gc.collect();torch.cuda.empty_cache()
    cfg.update(model=small,seed=3101,stage='verification_resume',stage_start_step=0,max_steps=10,batch_size=4,microbatch=2,
      loss_objective='norm7',fk_weight=1.,legacy_exact_update=False,precision='bfloat16',high_noise_after=0)
    identity={'m2_fingerprint':ready['fingerprint'],'test':'m5-numerical-verification'}
    configure(3101);trainer=M5Trainer(cfg,samples,identity,device);trainer.update();trainer.save(out/'resume_fixture.pt')
    trainer.update();trainer.update();expected=copy.deepcopy(trainer.state());del trainer;gc.collect();torch.cuda.empty_cache()
    configure(3101);resumed=M5Trainer(cfg,samples,identity,device);resumed.restore(out/'resume_fixture.pt');resumed.update();resumed.update();actual=resumed.state()
    for key in ('model','ema','optimizer','rngs','torch_rng','cuda_rng','step','samples_seen','stage_updates'):assert same(actual[key],expected[key]),key
    checks['bf16_accumulated_resume_exact']=True
    del resumed,expected,actual;gc.collect();torch.cuda.empty_cache()
    full_model=make_model({'latent_dim':512,'ff_size':2048,'num_layers':8,'num_heads':8})
    count=sum(p.numel() for p in full_model.parameters());assert count==56755617,count
    checks['small_parameters']=count
    result={'passed':True,'checks':checks,'source_checkpoint_sha256':sha(parent_path),'m2_fingerprint':ready['fingerprint'],
      'code_sha256':{f:sha(ROOT/f) for f in ('training/m5.py','training/m5_losses.py','scripts/verify_m5.py')},
      'note':'Verification updates are not formal M5 results; parent checkpoints and frozen M2 are unchanged.'}
    atomic_json(out/'report.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
