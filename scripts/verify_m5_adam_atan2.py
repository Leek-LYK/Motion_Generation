#!/usr/bin/env python3
"""Independent scalar reference, gradient-scale tests and real BF16 state restoration."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,gc,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from training.adam_atan2 import AdamAtan2,VERSION
from training.m5_atan2 import Atan2PhysicalTrainer
from training.m5_physical import PhysicalTrainer
from training.m5_physical_data import PhysicalSubsetDataset
from training.m3 import configure,atomic_json,sha
from scripts.verify_m5_physical import exact


def main():
    root=ROOT/'runs/m5_physical_20261103';out=root/'verification/adam_atan2';out.mkdir(parents=True,exist_ok=True);checks={}
    rng=np.random.default_rng(7411);gradients=rng.normal(0,.3,(31,7));gradients[:,0]=0;gradients[:,1]*=1e-10
    beta1,beta2=.9,.999;lr=2e-5;start=np.linspace(-1.,1.,7);expected=start.copy()
    for t in range(1,len(gradients)+1):
        weights1=(1-beta1)*beta1**np.arange(t-1,-1,-1);weights2=(1-beta2)*beta2**np.arange(t-1,-1,-1)
        m=(gradients[:t]*weights1[:,None]).sum(0)/(1-beta1**t);v=(gradients[:t]**2*weights2[:,None]).sum(0)/(1-beta2**t)
        expected-=lr*np.arctan2(m,np.sqrt(v))
    p=torch.nn.Parameter(torch.tensor(start,dtype=torch.float64));opt=AdamAtan2([p],lr=lr,betas=(beta1,beta2))
    for g in gradients:p.grad=torch.tensor(g,dtype=torch.float64);opt.step()
    error=float(abs(p.detach().numpy()-expected).max());assert error<1e-12;checks['closed_form_numpy_reference_max_error']=error
    outcomes=[]
    for scale in [1e-6,1.,1e6]:
        p=torch.nn.Parameter(torch.tensor(start,dtype=torch.float64));opt=AdamAtan2([p],lr=lr)
        for g in gradients:p.grad=torch.tensor(g*scale,dtype=torch.float64);opt.step()
        outcomes.append(p.detach().clone())
    assert all(torch.allclose(v,outcomes[1],atol=1e-12,rtol=1e-12) for v in outcomes);checks['positive_gradient_scale_invariance_within_precision']=True
    p=torch.nn.Parameter(torch.zeros(7));opt=AdamAtan2([p]);p.grad=torch.zeros_like(p);opt.step();assert torch.equal(p,torch.zeros_like(p)) and torch.isfinite(p).all()
    checks['zero_gradient_atan2_0_0_is_finite_and_zero']=True
    p.grad=torch.ones_like(p);before=p.detach().clone();opt.step();assert abs(p-before).max()<=lr*np.pi/2+1e-9;checks['angular_update_bound']=True
    configure(3101);view=json.loads((root/'data/READY.json').read_text());manifest=json.loads((root/'data/D_short_3500.json').read_text())
    entries=[next(r for r in manifest['train'] if r['group']==g) for g in ['walk','jog','jump','raise_arms','turn','squat']]
    ds=PhysicalSubsetDataset(entries,view['input_dir'],root/'stats/motion',cache_size=6)
    cfg={'model':{'latent_dim':64,'ff_size':128,'num_layers':1,'num_heads':4},'seed':3101,'batch_size':6,'microbatch':2,'learning_rate':2e-5,
      'minimum_learning_rate':2e-6,'scheduler':'cosine','warmup_steps':2,'stage_start_step':0,'max_steps':10,'minimum_steps':10,'early_stop_relative_delta':.001,
      'grad_clip':1.,'ema_every':1,'ema_decay':.995,'precision':'bfloat16','text_dropout':.1,'heading_augmentation':True,'sampling':'stratified',
      'high_noise_fraction':.5,'high_noise_start':800,'high_noise_after':0,'loss_objective':'weighted_raw7',
      'axis_weight':.02,'consistency_weight':.01,'ground_weight':.1,'optimizer_name':VERSION,'optimizer_betas':[.9,.999],
      'optimizer_weight_decay':0.,'optimizer_a':1.,'optimizer_b':1.}
    identity={'physical_input_fingerprint':view['input_fingerprint'],'training_view_fingerprint':view['fingerprint'],'text_config_fingerprint':'fixture','fixture':True}
    trainer=Atan2PhysicalTrainer(cfg,ds,identity,root/'stats/motion');trainer.update();trainer.save(out/'resume_fixture.pt');trainer.update();trainer.update();expected=copy.deepcopy(trainer.state())
    del trainer;gc.collect();torch.cuda.empty_cache();configure(3101);resumed=Atan2PhysicalTrainer(cfg,ds,identity,root/'stats/motion');resumed.restore(out/'resume_fixture.pt');resumed.update();resumed.update();actual=resumed.state()
    for key in ['model','ema','optimizer','rngs','torch_rng','cuda_rng','step','samples_seen','stage_updates']:assert exact(actual[key],expected[key]),key
    checks['bf16_accumulation_nonzero_physics_resume_exact']=True
    configure(3101);legacy=PhysicalTrainer(cfg,ds,identity,root/'stats/motion');zero=copy.deepcopy(legacy.state());fresh=Atan2PhysicalTrainer(cfg,ds,identity,root/'stats/motion');fresh.load_state(zero,initialization_only=True)
    assert exact(fresh.model.state_dict(),zero['model']) and not fresh.optimizer.state and fresh.step==0;checks['step_zero_initial_weights_shared_without_momentum']=True
    rejected=False
    try:resumed.load_state(zero,initialization_only=True)
    except ValueError:rejected=True
    assert rejected;checks['initial_weight_reload_into_used_optimizer_rejected']=True
    legacy.update();bad=legacy.state();rejected=False
    try:resumed.load_state(bad,continuation=True)
    except ValueError:rejected=True
    assert rejected;checks['trained_AdamW_continuation_rejected']=True
    report={'passed':True,'checks':checks,'optimizer_version':VERSION,'a':1.,'b':1.,'bias_correction':'inside atan2','betas':[.9,.999],
      'paper_reference':'https://arxiv.org/html/2407.05872v2#A3.SS5','scope':'Numerical verification only; no optimizer quality or M5 completion claim.',
      'code_sha256':{p:sha(ROOT/p) for p in ['training/adam_atan2.py','training/m5_atan2.py','scripts/verify_m5_adam_atan2.py']}}
    atomic_json(out/'report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
