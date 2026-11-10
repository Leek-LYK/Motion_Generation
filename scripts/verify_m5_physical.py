#!/usr/bin/env python3
"""Physical targets, independent gradients/masks/yaw, BF16 updates and exact resume."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,gc,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from kimodo.geometry import cont6d_to_matrix
from training.m3 import configure,make_model,batch_to,atomic_json,sha
from training.m4 import rotate_batch
from training.m5_physical import PhysicalTrainer
from training.m5_physical_data import PhysicalSubsetDataset
from training.m5_physical_losses import geometry,objective,NativeHingeGeometry,stable_cont6d
from training.m5_physical_evaluation import evaluate_generation,evaluate_denoising


def exact(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();root=a.experiment.resolve();out=root/'verification';out.mkdir(parents=True,exist_ok=True)
    view=json.loads((root/'data/READY.json').read_text());m=json.loads((root/'data/D_short_3500.json').read_text());configure(3101);device='cuda:0'
    # Six classes and unequal durations, using real held-out input events.
    entries=[next(r for r in m['val'] if r['group']==g) for g in ('walk','jog','jump','raise_arms','turn','squat')]
    ds=PhysicalSubsetDataset(entries,view['input_dir'],root/'stats/motion');batch=batch_to(ds.batch(range(len(ds))),device)
    small={'latent_dim':64,'ff_size':128,'num_layers':1,'num_heads':4};model=make_model(small,root/'stats/motion').to(device);rep=model.motion_rep
    hinge=NativeHingeGeometry(rep.skeleton,device);x=batch['x0'];mask=batch['x_pad_mask'];checks={}
    oracle,parts=objective(rep,x,x,mask,{'loss_objective':'norm7'},hinge)
    checks.update(oracle_norm7_loss=float(oracle),oracle_fk_error_m=float(parts['fk_position_inconsistency_m']),oracle_native_axis_defect=float(parts['axis_defect']))
    assert float(oracle)<1e-8 and float(parts['fk_position_inconsistency_m'])<2e-6 and float(parts['axis_defect'])<1e-10,checks
    values=torch.randn(128,6,device=device);actual,fallback=stable_cont6d(values);difference=float(abs(actual-cont6d_to_matrix(values)).max())
    assert difference<2e-5;checks['regular_6d_matches_existing_max_error']=difference
    degenerate=torch.tensor([[0.,0,0,0,0,0],[1.,0,0,1,0,0]],device=device,requires_grad=True);rotation,flags=stable_cont6d(degenerate)
    rotation.sum().backward();assert torch.isfinite(rotation).all() and torch.isfinite(degenerate.grad).all() and flags.all()
    assert torch.allclose(rotation.transpose(-1,-2)@rotation,torch.eye(3,device=device).expand(2,3,3),atol=1e-6)
    checks['degenerate_6d_finite_so3_and_gradients']=True
    pred=x.clone().detach();pred[:,:,107:311]+=.05*torch.randn_like(pred[:,:,107:311]);pred.requires_grad_(True)
    cfg={'loss_objective':'norm7','consistency_weight':.2,'axis_weight':1.,'ground_weight':.5}
    loss,perturbed=objective(rep,pred,x,mask,cfg,hinge);loss.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad[:,:,107:311].norm()>0 and pred.grad[:,:,5:107].norm()>0
    assert perturbed['fk']>parts['fk'] and perturbed['axis_defect']>parts['axis_defect'];checks['perturbation_and_two_branch_gradients']=True
    other=pred.detach().clone();other[~mask]=17.;other.requires_grad_(True)
    value,_=objective(rep,other,x,mask,cfg,hinge);value.backward()
    assert torch.allclose(value,loss,atol=1e-7,rtol=1e-6) and (other.grad[~mask]==0).all();checks['padding_has_zero_loss_and_gradient']=True
    for mode in ('norm7','weighted_raw7'):
        config=dict(cfg,loss_objective=mode);full=pred.detach().clone().requires_grad_(True);whole,_=objective(rep,full,x,mask,config,hinge);whole.backward()
        split=pred.detach().clone().requires_grad_(True)
        for i in range(len(x)):
            value,_=objective(rep,split[i:i+1],x[i:i+1],mask[i:i+1],config,hinge);(value*mask[i].sum()/mask.sum()).backward()
        error=float(abs(full.grad-split.grad).max());assert torch.allclose(full.grad,split.grad,atol=3e-6,rtol=3e-5),error
        checks[mode+'_accumulation_gradient_max_error']=error
    angles=torch.tensor([.1,.6,1.1,-.7,2.0,-2.0],device=device);rotated=rotate_batch(model,batch,angles)
    _,_,before,_=geometry(rep,x);_,_,after,_=geometry(rep,rotated['x0'])
    dist=lambda motion:torch.linalg.vector_norm(motion['posed_joints'][:,:,1:]-motion['posed_joints'][:,:,:1],dim=-1)
    error=float(abs(dist(before)-dist(after)).masked_select(mask[:,:,None]).max());assert error<1e-5;checks['yaw_bone_distance_max_error_m']=error
    foot_error=float(abs(hinge.converter.foot_bottom(before)-hinge.converter.foot_bottom(after)).masked_select(mask[:,:,None]).max());assert foot_error<1e-5
    checks['yaw_collider_height_max_error_m']=foot_error
    print('Targets, native axes, finite rotations, padding, accumulation and yaw verified',flush=True)
    config={'model':small,'seed':3101,'batch_size':6,'microbatch':2,'learning_rate':2e-5,'minimum_learning_rate':2e-6,'scheduler':'warmup_cosine',
      'warmup_steps':2,'stage_start_step':0,'max_steps':10,'minimum_steps':10,'early_stop_relative_delta':.001,'grad_clip':1.,
      'ema_every':1,'ema_decay':.995,'precision':'bfloat16','text_dropout':.1,'heading_augmentation':True,'sampling':'stratified',
      'high_noise_fraction':.5,'high_noise_start':800,'high_noise_after':0,'loss_objective':'norm7'}
    identity={'physical_input_fingerprint':view['input_fingerprint'],'training_view_fingerprint':view['fingerprint'],'text_config_fingerprint':'fixture','verification':True}
    configure(3101);trainer=PhysicalTrainer(config,ds,identity,root/'stats/motion');trainer.update();trainer.save(out/'resume_fixture.pt');trainer.update();trainer.update();expected=copy.deepcopy(trainer.state())
    del trainer;gc.collect();torch.cuda.empty_cache();configure(3101)
    resumed=PhysicalTrainer(config,ds,identity,root/'stats/motion');resumed.restore(out/'resume_fixture.pt');resumed.update();resumed.update();actual=resumed.state()
    for key in ('model','ema','optimizer','rngs','torch_rng','cuda_rng','numpy_rng','python_rng','step','samples_seen','stage_updates'):assert exact(actual[key],expected[key]),key
    checks['bf16_update_resume_exact']=True
    print('BF16 full update and exact checkpoint resume verified',flush=True)
    # Exercise the actual physical decoder/evaluator, including persisted artifacts.
    denoise=evaluate_denoising(resumed.ema,ds,resumed.device,batch_size=3);assert np.isfinite(denoise['macro_loss']) and len(denoise['groups'])==6
    report=evaluate_generation(resumed.ema,ds,out/'generation_smoke',3,seeds=(42,),headings=(None,0.),batch_size=3,steps=10,retain=True)
    assert report['all_valid_qpos'] and report['all_finite'] and report['generated_clips']==12
    assert np.isfinite(report['pooled_projection_node_p95_m']) and report['pooled_projection_node_p95_m']>=0
    checks['six_class_physical_generation_and_export_smoke']=True
    # Parameter count is a requirement of M5, not inferred from a run name.
    full=make_model({'latent_dim':512,'ff_size':2048,'num_layers':8,'num_heads':8},root/'stats/motion');count=sum(p.numel() for p in full.parameters());assert count==56755617
    checks['small_parameters']=count
    result={'passed':True,'checks':checks,'training_view_fingerprint':view['fingerprint'],'script_sha256':sha(__file__),
      'code_sha256':{f:sha(ROOT/f) for f in ['training/m5_physical.py','training/m5_physical_losses.py','training/m5_physical_evaluation.py','training/m5_physical_data.py']},
      'note':'Engineering fixtures are excluded from model ranking; no legacy model weights loaded.'}
    atomic_json(out/'report.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
