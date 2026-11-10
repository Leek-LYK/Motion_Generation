"""Versioned M5 constraint trainer, preserving source optimizer/RNG lineage."""
import math
import torch
from training.m3 import batch_to,forward
from training.m5_physical import PhysicalTrainer
from training.m5_atan2 import SCHEMA as ATAN2_SCHEMA
from training.adam_atan2 import AdamAtan2,VERSION as ATAN2_VERSION
from training.g1_legality_loss import NativeLegalityLoss,VERSION as LEGALITY_VERSION
from training.m5_constraint_losses import constraint_objective

SCHEMA='kimodo-m5-physical-constraints-checkpoint-v1'
ADAMW_KIND='adamw-fp32-v1'


class PhysicalConstraintTrainer(PhysicalTrainer):
    def __init__(self,config,samples,identity,stats_dir,device='cuda:0'):
        super().__init__(config,samples,identity,stats_dir,device)
        name=config.get('optimizer_name')
        if name in (None,'adamw',ADAMW_KIND):
            self.optimizer_kind=ADAMW_KIND
            self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=config['learning_rate'],betas=tuple(config.get('optimizer_betas',[.9,.999])),
              weight_decay=config.get('optimizer_weight_decay',0.),eps=config.get('optimizer_epsilon',1e-8),foreach=False)
        elif name==ATAN2_VERSION:
            self.optimizer_kind=ATAN2_VERSION
            self.optimizer=AdamAtan2(self.model.parameters(),lr=config['learning_rate'],betas=tuple(config['optimizer_betas']),
              weight_decay=config.get('optimizer_weight_decay',0.),a=config.get('optimizer_a',1.),b=config.get('optimizer_b',1.))
        else:raise ValueError('Unknown optimizer implementation')
        for key in ('consistency_weight','axis_weight','ground_weight','legality_weight'):
            if not math.isfinite(config.get(key,0.)) or config.get(key,0.)<0:raise ValueError('Invalid physical coefficient: '+key)
        self.legality=NativeLegalityLoss(self.device)

    def current_lr(self):
        if 'lr_schedule_origin_step' not in self.config:return super().current_lr()
        if self.config.get('scheduler','constant')=='constant':return self.config['learning_rate']
        origin=self.config['lr_schedule_origin_step'];local=self.step-origin;warmup=self.config['warmup_steps']
        if local<warmup:return self.config['learning_rate']*(local+1)/warmup
        end=self.config.get('lr_schedule_end_step',self.config['max_steps']);span=max(1,end-origin-warmup)
        fraction=min(1.,(local-warmup)/span);low=self.config['minimum_learning_rate'];high=self.config['learning_rate']
        return low+(high-low)*.5*(1+math.cos(math.pi*fraction))

    def state(self):
        result=super().state();result.update(schema=SCHEMA,optimizer_kind=self.optimizer_kind,legality_operator=LEGALITY_VERSION);return result

    def load_state(self,state,continuation=False,initialization_only=False):
        source_schema=state.get('schema')
        kinds={'kimodo-m5-physical-checkpoint-v2':ADAMW_KIND,ATAN2_SCHEMA:ATAN2_VERSION,SCHEMA:state.get('optimizer_kind')}
        if source_schema not in kinds:raise ValueError('Unsupported representation/schema')
        if initialization_only:
            if self.step!=0 or self.optimizer.state or state['step']!=0:raise ValueError('Only fresh step-zero initialization is supported')
        else:
            if kinds[source_schema]!=self.optimizer_kind:raise ValueError('Optimizer momentum lineage mismatch')
            if source_schema!=SCHEMA and not continuation:raise ValueError('A base checkpoint requires explicit full-state continuation')
            if source_schema==SCHEMA and state.get('legality_operator')!=LEGALITY_VERSION:raise ValueError('Legality operator version changed')
            for group in state['optimizer']['param_groups']:
                target=self.optimizer.param_groups[0]
                for key in ('betas','weight_decay','eps','a','b'):
                    if group.get(key)!=target.get(key):raise ValueError('Optimizer specification changed at continuation: '+key)
        proxy=dict(state,schema='kimodo-m5-physical-checkpoint-v2')
        return super().load_state(proxy,continuation=continuation,initialization_only=initialization_only)

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
                total,parts=constraint_objective(self.model.motion_rep,prediction,chunk['x0'],chunk['x_pad_mask'],self.config,self.hinge,self.legality)
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
