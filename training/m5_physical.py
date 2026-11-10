"""From-scratch M5 trainer using audited physical-v2 inputs and qualified statistics."""
import copy,random
from pathlib import Path
import numpy as np,torch
from kimodo.model.diffusion import Diffusion
from training.m3 import make_model,batch_to,forward
from training.m5 import M5Trainer,COUNTERS,RNGS
from training.m5_physical_losses import objective,NativeHingeGeometry


class PhysicalTrainer(M5Trainer):
    def __init__(self,config,samples,identity,stats_dir,device='cuda:0'):
        # Deliberately construct once with the new statistics. Legacy constructors
        # would silently load M2 statistics, which are nonpersistent state buffers.
        self.config=copy.deepcopy(config);self.samples=samples;self.identity=identity;self.device=torch.device(device)
        self.model=make_model(config['model'],stats_dir=stats_dir).to(self.device)
        self.ema=copy.deepcopy(self.model).eval().requires_grad_(False)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=config['learning_rate'],weight_decay=0.,foreach=False)
        self.diffusion=Diffusion(1000).to(self.device)
        for i,name in enumerate(RNGS,1):
            setattr(self,name,torch.Generator(device='cpu' if name=='batch_rng' else self.device).manual_seed(config['seed']+i))
        self.step=self.ema_updates=self.stage_updates=self.best_raw_step=self.best_ema_step=0
        self.bad_evaluations=self.samples_seen=self.text_dropped=0;self.last_evaluation_step=-1
        self.best_raw=self.best_ema=self.early_best=float('inf');self.pools={}
        if config.get('sampling')=='stratified':
            for i,row in enumerate(samples.entries):
                bucket=0 if row['num_frames']<=90 else 1 if row['num_frames']<=120 else 2
                self.pools.setdefault(bucket,{}).setdefault(row['group'],[]).append(i)
        self.hinge=NativeHingeGeometry(self.model.motion_rep.skeleton,self.device)

    def update(self):
        self.model.train();indices=self.draw_indices();self.last_indices=indices
        batch,dropped,angles=self.prepare_training_batch(batch_to(self.samples.batch(indices),self.device))
        t=torch.randint(1000,(len(indices),),device=self.device,generator=self.noise_rng)
        fraction=self.config.get('high_noise_fraction',0) if self.step>=self.config.get('high_noise_after',0) else 0
        replacement=torch.zeros_like(t,dtype=torch.bool)
        if fraction:
            replacement=torch.rand(len(t),device=self.device,generator=self.high_noise_rng)<fraction
            high=torch.randint(self.config.get('high_noise_start',800),1000,(len(t),),device=self.device,generator=self.high_noise_rng)
            t=torch.where(replacement,high,t)
        noise=torch.randn(batch['x0'].shape,device=self.device,generator=self.noise_rng);noisy=self.diffusion.q_sample(batch['x0'],t,noise)
        lr=self.current_lr()
        for group in self.optimizer.param_groups:group['lr']=lr
        self.optimizer.zero_grad(set_to_none=True);valid_frames=batch['x_pad_mask'].sum();sums={};value=0.
        for start in range(0,len(indices),self.config['microbatch']):
            end=start+self.config['microbatch']
            chunk={k:v[start:end] if isinstance(v,torch.Tensor) and v.ndim and len(v)==len(indices) else v for k,v in batch.items()}
            with torch.autocast(device_type=self.device.type,dtype=torch.bfloat16,enabled=self.config['precision']=='bfloat16'):
                prediction=forward(self.model,noisy[start:end],chunk,t[start:end])
            with torch.autocast(device_type=self.device.type,enabled=False):
                total,parts=objective(self.model.motion_rep,prediction,chunk['x0'],chunk['x_pad_mask'],self.config,self.hinge)
                weight=chunk['x_pad_mask'].sum()/valid_frames;weighted=total*weight
            if not torch.isfinite(weighted):raise FloatingPointError('Nonfinite physical-v2 loss')
            weighted.backward();value+=float(weighted.detach())
            for k,v in parts.items():sums[k]=sums.get(k,0.)+float(v.detach()*weight)
        norms={}
        for name,stage in [('root',self.model.root_model),('body',self.model.body_model)]:
            norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in stage.parameters() if p.grad is not None]))
            if not torch.isfinite(norm) or norm<=0:raise FloatingPointError('Invalid '+name+' gradients')
            norms[name]=float(norm)
        grad=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config['grad_clip'],error_if_nonfinite=True)
        self.optimizer.step();self.step+=1;self.stage_updates+=1;self.samples_seen+=len(indices);self.text_dropped+=int(dropped.sum())
        if self.step%self.config['ema_every']==0:
            self.ema_updates+=1;decay=min(self.config['ema_decay'],(1+self.ema_updates)/(10+self.ema_updates))
            with torch.no_grad():
                for a,p in zip(self.ema.parameters(),self.model.parameters()):a.lerp_(p,1-decay)
        return {'loss':value,**sums,'grad_norm':float(grad),'root_grad_norm':norms['root'],'body_grad_norm':norms['body'],
          'learning_rate':lr,'valid_frames':int(valid_frames),'padding_fraction':float(1-valid_frames/batch['x_pad_mask'].numel()),
          't_ge800_fraction':float((t>=800).float().mean()),'high_noise_replacement_fraction':float(replacement.float().mean()),
          'text_drop_fraction':float(dropped.float().mean()),'heading_abs_delta_mean':float(angles.abs().mean()),
          'sample_exposures':self.samples_seen,'equivalent_passes_stage':self.stage_updates*len(indices)/len(self.samples)}

    def state(self):
        result=super().state();result['schema']='kimodo-m5-physical-checkpoint-v2';return result

    def load_state(self,state,continuation=False,initialization_only=False):
        if state['schema']!='kimodo-m5-physical-checkpoint-v2':raise ValueError('Legacy representation checkpoints cannot initialize physical-v2')
        for key in ('physical_input_fingerprint','training_view_fingerprint','text_config_fingerprint'):
            if state['identity'][key]!=self.identity[key]:raise ValueError('Physical checkpoint '+key+' mismatch')
        if state['config']['model']!=self.config['model']:raise ValueError('Architecture mismatch')
        if not continuation and not initialization_only and (state['identity']!=self.identity or state['config']!=self.config):raise ValueError('Resume identity/config mismatch')
        self.model.load_state_dict(state['model']);self.ema.load_state_dict(state['ema'])
        if initialization_only:
            if state['step']!=0:raise ValueError('Initialization must be step zero')
            return
        self.optimizer.load_state_dict(state['optimizer'])
        for key in COUNTERS:setattr(self,key,state[key])
        for key,value in state['rngs'].items():getattr(self,key).set_state(value)
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);self.stage_updates=state['stage_updates']
        if continuation:
            if self.step!=self.config['stage_start_step']:raise ValueError('Wrong continuation step')
            self.stage_updates=0;self.best_raw=self.best_ema=self.early_best=float('inf')
            self.best_raw_step=self.best_ema_step=self.bad_evaluations=0;self.last_evaluation_step=-1
