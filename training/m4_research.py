"""Single-variable high-noise curriculum ablation; original M4 implementation is unchanged."""
import os
import random
from pathlib import Path
import numpy as np
import torch
from training.m3 import COMPONENTS,batch_to,forward,losses
from training.m4 import M4Trainer


class ResearchTrainer(M4Trainer):
    def __init__(self,config,samples,identity,device='cuda:0'):
        super().__init__(config,samples,identity,device)
        self.high_noise_rng=torch.Generator(device=self.device).manual_seed(config['seed']+5)

    def update(self):
        self.model.train()
        indices=torch.randint(len(self.samples),(self.config['batch_size'],),generator=self.batch_rng).tolist()
        batch,dropped,angles=self.prepare_training_batch(batch_to([self.samples[i] for i in indices],self.device))
        t=torch.randint(1000,(len(indices),),device=self.device,generator=self.noise_rng)
        fraction=self.config.get('high_noise_fraction',0.0)
        replacement=torch.zeros_like(t,dtype=torch.bool)
        if fraction:
            replacement=torch.rand(len(t),device=self.device,generator=self.high_noise_rng)<fraction
            high=torch.randint(self.config.get('high_noise_start',800),1000,(len(t),),device=self.device,generator=self.high_noise_rng)
            t=torch.where(replacement,high,t)
        noise=torch.randn(batch['x0'].shape,device=self.device,generator=self.noise_rng)
        prediction=forward(self.model,self.diffusion.q_sample(batch['x0'],t,noise),batch,t)
        if self.config.get('loss_mode','smooth_l1')=='mse':
            error=torch.nn.functional.mse_loss(prediction,batch['x0'],reduction='none');mask=batch['x_pad_mask']
            parts={name:(error[...,a:b]*mask[...,None]).sum()/(mask.sum()*(b-a)) for name,(a,b) in COMPONENTS.items()}
            total=torch.stack(list(parts.values())).mean()
            parts['all_feature_mean']=(error*mask[...,None]).sum()/(mask.sum()*417)
        else:
            total,parts=losses(prediction,batch['x0'],batch['x_pad_mask'])
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite loss')
        self.optimizer.zero_grad(set_to_none=True);total.backward();norms={}
        for name,stage in [('root',self.model.root_model),('body',self.model.body_model)]:
            norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in stage.parameters() if p.grad is not None]))
            if not torch.isfinite(norm) or norm<=0:raise FloatingPointError('Invalid gradients')
            norms[name]=float(norm)
        grad=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config['grad_clip'],error_if_nonfinite=True)
        self.optimizer.step();self.step+=1;self.samples_seen+=len(indices);self.text_dropped+=int(dropped.sum())
        if self.step%self.config['ema_every']==0:
            self.ema_updates+=1;decay=min(self.config['ema_decay'],(1+self.ema_updates)/(10+self.ema_updates))
            with torch.no_grad():
                for a,p in zip(self.ema.parameters(),self.model.parameters()):a.lerp_(p,1-decay)
        return {'loss':float(total.detach()),**{k:float(v.detach()) for k,v in parts.items()},'grad_norm':float(grad),
            'root_grad_norm':norms['root'],'body_grad_norm':norms['body'],'text_drop_fraction':float(dropped.float().mean()),
            'text_drop_fraction_cumulative':self.text_dropped/self.samples_seen,'heading_abs_delta_mean':float(angles.abs().mean()),
            'sample_exposures':self.samples_seen,'equivalent_passes':self.samples_seen/len(self.samples),
            'high_noise_replacement_fraction':float(replacement.float().mean()),'t_ge800_fraction':float((t>=800).float().mean())}

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        state={'schema':'kimodo-m4-checkpoint-v1','config':self.config,'identity':self.identity,
            'model':self.model.state_dict(),'ema':self.ema.state_dict(),'optimizer':self.optimizer.state_dict(),
            **{name:getattr(self,name) for name in ['step','ema_updates','best_raw','best_ema','best_raw_step','best_ema_step',
              'early_best','bad_evaluations','samples_seen','text_dropped','last_evaluation_step']},
            'rngs':{n:getattr(self,n).get_state() for n in ['batch_rng','noise_rng','text_rng','heading_rng','high_noise_rng']},
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
            'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
        temp=path.with_suffix('.tmp')
        with temp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
