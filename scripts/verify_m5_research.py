#!/usr/bin/env python3
"""Verify hinge geometry, gradient paths, zero-addition equivalence and exact resume."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,gc,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from kimodo.geometry import axis_angle_to_matrix
from training.m3 import configure,atomic_json,sha,validate_inputs,make_model,batch_to
from training.m4 import load_data,rotate_batch
from training.m5 import M5Trainer
from training.m5_protocol import configuration
from training.m5_research import ResearchTrainer
from training.m5_research_losses import HingeGeometry,research_objective,per_sample_terms


def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);args=p.parse_args()
    out=args.experiment/'verification';out.mkdir(parents=True,exist_ok=True);ready=validate_inputs();configure(6101)
    manifest=json.loads((ROOT/'runs/m5_20261103/data/D_anchor_1024.json').read_text());events=load_data(manifest,'val')[:4]
    model=make_model({'latent_dim':64,'ff_size':128,'num_layers':1,'num_heads':4}).cuda();hinge=HingeGeometry(model.motion_rep.skeleton,'cuda:0')
    batch=batch_to(events,'cuda:0');x=batch['x0'];mask=batch['x_pad_mask'];terms,_=per_sample_terms(model.motion_rep,x,x,mask,hinge)
    checks={'ground_truth_fk_gt_norm':float(terms['fk_gt_norm'].mean()),'ground_truth_self_loss':float(terms['consistency_norm'].mean()),
            'ground_truth_axis_energy':float(terms['axis_defect'].mean())}
    assert checks['ground_truth_fk_gt_norm']<1e-5 and checks['ground_truth_self_loss']<1e-5
    assert checks['ground_truth_axis_energy']<1e-5,checks
    local=torch.eye(3,device='cuda').expand(1,2,34,3,3).clone()
    theta=torch.linspace(-2.0,2.0,29,device='cuda')
    around=axis_angle_to_matrix(hinge.axes*theta[:,None])
    local=local.index_copy(2,hinge.indices,(hinge.offsets.transpose(-1,-2)@around)[None,None].expand(1,2,29,3,3))
    energy,_=hinge.defects(local);checks['legal_hinge_axis_energy_max']=float(energy.max());assert energy.max()<1e-10
    axis=hinge.axes[0];basis=torch.tensor([1.,0.,0.],device='cuda')
    if abs(float(axis@basis))>.9:basis=torch.tensor([0.,1.,0.],device='cuda')
    perpendicular=torch.linalg.cross(axis,basis);perpendicular=perpendicular/perpendicular.norm()
    angle=torch.tensor(.15,device='cuda',requires_grad=True)
    inserted=(hinge.offsets[0].T@axis_angle_to_matrix(perpendicular*angle))[None,None,None].expand(1,2,1,3,3)
    perturbed=local.index_copy(2,hinge.indices[:1],inserted)
    energy,_=hinge.defects(perturbed);value=energy[0,0,0];value.backward()
    assert torch.isfinite(angle.grad) and abs(float(angle.grad)-float(torch.sin(angle.detach())))<1e-5
    checks['off_axis_finite_correct_gradient']=float(angle.grad)
    pred=x.clone();pred[:,:,8:107]+=.02;pred.requires_grad_(True)
    terms,_=per_sample_terms(model.motion_rep,pred,x,mask,hinge);terms['consistency_smooth_m'].sum().backward()
    assert torch.isfinite(pred.grad).all() and pred.grad[:,:,5:107].norm()>0 and pred.grad[:,:,107:311].norm()>0
    checks['self_consistency_reaches_both_position_and_rotation']=True
    rotated=rotate_batch(model,dict(batch,x0=pred.detach()),torch.tensor([.2,.6,-.9,1.2],device='cuda'))
    original,_=per_sample_terms(model.motion_rep,pred.detach(),x,mask,hinge)
    after,_=per_sample_terms(model.motion_rep,rotated['x0'],rotated['x0'],mask,hinge)
    delta=float((original['axis_defect']-after['axis_defect']).abs().max());assert delta<2e-6,delta
    checks['axis_yaw_invariance_max_error']=delta
    print('Geometry and gradient paths verified',flush=True)
    parent=ROOT/'runs/m4_research_20261103/R3_D_high_noise/checkpoints/latest.pt';state=torch.load(parent,map_location='cpu',weights_only=False)
    cfg=configuration('zero_extra_equivalence',start=20000,end=30000,anchor=True);cfg.update(consistency_weight=0.,axis_weight=0.)
    configure(cfg['seed']);old=M5Trainer(cfg,events,state['identity']);old.load_state(copy.deepcopy(state),continuation=True);old.update()
    expected=copy.deepcopy(old.state());del old;gc.collect();torch.cuda.empty_cache()
    configure(cfg['seed']);new=ResearchTrainer(cfg,events,state['identity']);new.load_state(copy.deepcopy(state),continuation=True);new.update()
    current=new.state()
    for key in ('model','ema','optimizer','rngs','torch_rng','cuda_rng','samples_seen','step','stage_updates'):assert same(expected[key],current[key]),key
    checks['zero_extra_update_exact_vs_A1_trainer']=True
    # The legacy no-FK control must increment stage_updates exactly once.
    control_cfg=dict(cfg,legacy_exact_update=True,loss_objective='norm6',fk_weight=0.)
    control=ResearchTrainer(control_cfg,events,state['identity']);control.load_state(copy.deepcopy(state),continuation=True);control.update()
    assert control.stage_updates==1 and control.step==20001
    checks['legacy_control_counter_increment_once']=True
    del control
    del state
    del new,expected,current;gc.collect();torch.cuda.empty_cache()
    cfg.update(model={'latent_dim':64,'ff_size':128,'num_layers':1,'num_heads':4},stage='research_resume',seed=6101,
      stage_start_step=0,max_steps=10,batch_size=4,microbatch=2,consistency_weight=.5,axis_weight=1.,high_noise_after=0)
    identity={'m2_fingerprint':ready['fingerprint'],'purpose':'verification only'}
    configure(6101);trainer=ResearchTrainer(cfg,events,identity);trainer.update();trainer.save(out/'resume_fixture.pt');trainer.update();trainer.update()
    expected=copy.deepcopy(trainer.state());del trainer;gc.collect();torch.cuda.empty_cache()
    configure(6101);trainer=ResearchTrainer(cfg,events,identity);trainer.restore(out/'resume_fixture.pt');trainer.update();trainer.update();current=trainer.state()
    for key in ('model','ema','optimizer','rngs','torch_rng','cuda_rng','samples_seen','step','stage_updates'):assert same(expected[key],current[key]),key
    checks['combined_loss_resume_exact']=True
    report={'passed':True,'checks':checks,'parent_sha256':sha(parent),'code_sha256':{f:sha(ROOT/f) for f in
      ('training/m5_research_losses.py','training/m5_research.py','scripts/verify_m5_research.py')},'optimizer_updates':8}
    atomic_json(out/'report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
