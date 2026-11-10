"""Versioned research trainer; the original M5 implementation is preserved."""
import torch
from training.m3 import batch_to,forward
from training.m5 import M5Trainer
from training.m5_research_losses import HingeGeometry,research_objective


class ResearchTrainer(M5Trainer):
    def __init__(self,config,samples,identity,device='cuda:0'):
        super().__init__(config,samples,identity,device)
        self.hinge=HingeGeometry(self.model.motion_rep.skeleton,self.device)

    def update(self):
        if self.config.get('legacy_exact_update',False):
            return super().update()
        self.model.train();indices=self.draw_indices()
        samples=self.samples.batch(indices) if hasattr(self.samples,'batch') else [self.samples[i] for i in indices]
        batch,dropped,angles=self.prepare_training_batch(batch_to(samples,self.device))
        t=torch.randint(1000,(len(indices),),device=self.device,generator=self.noise_rng)
        fraction=self.config.get('high_noise_fraction',0) if self.step>=self.config.get('high_noise_after',0) else 0
        replacement=torch.zeros_like(t,dtype=torch.bool)
        if fraction:
            replacement=torch.rand(len(t),device=self.device,generator=self.high_noise_rng)<fraction
            high=torch.randint(self.config.get('high_noise_start',800),1000,(len(t),),device=self.device,generator=self.high_noise_rng)
            t=torch.where(replacement,high,t)
        noise=torch.randn(batch['x0'].shape,device=self.device,generator=self.noise_rng)
        noisy=self.diffusion.q_sample(batch['x0'],t,noise)
        lr=self.current_lr()
        for group in self.optimizer.param_groups:group['lr']=lr
        self.optimizer.zero_grad(set_to_none=True)
        valid_frames=batch['x_pad_mask'].sum();parts_sum={};total_value=0.
        micro=self.config.get('microbatch',len(indices))
        for start in range(0,len(indices),micro):
            end=start+micro
            chunk={k:v[start:end] if isinstance(v,torch.Tensor) and v.ndim and len(v)==len(indices) else v for k,v in batch.items()}
            with torch.autocast(device_type=self.device.type,dtype=torch.bfloat16,enabled=self.config['precision']=='bfloat16'):
                prediction=forward(self.model,noisy[start:end],chunk,t[start:end])
            with torch.autocast(device_type=self.device.type,enabled=False):
                total,parts=research_objective(self.model.motion_rep,prediction,chunk['x0'],chunk['x_pad_mask'],self.config,self.hinge)
                weight=chunk['x_pad_mask'].sum()/valid_frames
                weighted=total*weight
            if not torch.isfinite(weighted):raise FloatingPointError('Nonfinite loss')
            weighted.backward();total_value+=float(weighted.detach())
            for name,value in parts.items():parts_sum[name]=parts_sum.get(name,0.)+float((value.detach()*weight))
        norms={}
        for name,stage in [('root',self.model.root_model),('body',self.model.body_model)]:
            norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in stage.parameters() if p.grad is not None]))
            if not torch.isfinite(norm) or norm<=0:raise FloatingPointError(f'Invalid {name} gradients')
            norms[name]=float(norm)
        grad=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config['grad_clip'],error_if_nonfinite=True)
        self.optimizer.step();self.step+=1;self.stage_updates+=1;self.samples_seen+=len(indices);self.text_dropped+=int(dropped.sum())
        if self.step%self.config['ema_every']==0:
            self.ema_updates+=1;decay=min(self.config['ema_decay'],(1+self.ema_updates)/(10+self.ema_updates))
            with torch.no_grad():
                for a,p in zip(self.ema.parameters(),self.model.parameters()):a.lerp_(p,1-decay)
        return {'loss':total_value,**parts_sum,'grad_norm':float(grad),'root_grad_norm':norms['root'],'body_grad_norm':norms['body'],
                'learning_rate':lr,'valid_frames':int(valid_frames),'padding_fraction':float(1-valid_frames/batch['x_pad_mask'].numel()),
                't_ge800_fraction':float((t>=800).float().mean()),'high_noise_replacement_fraction':float(replacement.float().mean()),
                'text_drop_fraction':float(dropped.float().mean()),'heading_abs_delta_mean':float(angles.abs().mean()),
                'sample_exposures':self.samples_seen,'equivalent_passes_stage':self.stage_updates*len(indices)/len(self.samples)}

