"""M5 trainer: isolated state lineage, frame-correct accumulation and frozen M2 inputs."""
import copy,math,os,random
from pathlib import Path
import numpy as np
import torch
from training.m3 import batch_to,forward
from training.m4_research import ResearchTrainer
from training.m5_losses import objective

COUNTERS=('step','ema_updates','best_raw','best_ema','best_raw_step','best_ema_step','early_best',
          'bad_evaluations','samples_seen','text_dropped','last_evaluation_step')
RNGS=('batch_rng','noise_rng','text_rng','heading_rng','high_noise_rng')


class M5Trainer(ResearchTrainer):
    def __init__(self,config,samples,identity,device='cuda:0'):
        # Upstream M4 creates the same architecture and initialized optimizer. Its
        # protocol restriction is not modified; M5 adds its own precision policy.
        compatibility=dict(config,precision='float32')
        super().__init__(compatibility,samples,identity,device)
        self.config=copy.deepcopy(config)
        self.pools={}
        if config.get('sampling','legacy')=='stratified':
            for i,row in enumerate(samples.entries):
                bucket=0 if row['num_frames']<=90 else 1 if row['num_frames']<=120 else 2
                self.pools.setdefault(bucket,{}).setdefault(row['group'],[]).append(i)
        self.stage_updates=0

    def draw_indices(self):
        count=self.config['batch_size']
        if not self.pools:return torch.randint(len(self.samples),(count,),generator=self.batch_rng).tolist()
        buckets=sorted(self.pools)
        weights=torch.tensor([sum(len(v) for v in self.pools[b].values()) for b in buckets],dtype=torch.float64)
        bucket=buckets[int(torch.multinomial(weights,1,generator=self.batch_rng))]
        groups=sorted(self.pools[bucket]);selected=torch.randint(len(groups),(count,),generator=self.batch_rng)
        return [self.pools[bucket][groups[int(g)]][int(torch.randint(len(self.pools[bucket][groups[int(g)]]),(1,),generator=self.batch_rng))] for g in selected]

    def current_lr(self):
        if self.config.get('scheduler','constant')=='constant':return self.config['learning_rate']
        local=self.step-self.config['stage_start_step'];warmup=self.config['warmup_steps']
        if local<warmup:return self.config['learning_rate']*(local+1)/warmup
        span=max(1,self.config['max_steps']-self.config['stage_start_step']-warmup)
        fraction=min(1.,(local-warmup)/span)
        low=self.config['minimum_learning_rate'];high=self.config['learning_rate']
        return low+(high-low)*.5*(1+math.cos(math.pi*fraction))

    def update(self):
        if self.config.get('legacy_exact_update',False):
            result=super().update();self.stage_updates+=1
            return dict(result,learning_rate=self.optimizer.param_groups[0]['lr'])
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
                total,parts=objective(self.model.motion_rep,prediction,chunk['x0'],chunk['x_pad_mask'],self.config['loss_objective'],self.config.get('fk_weight',1.))
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

    def state(self):
        return {'schema':'kimodo-m5-checkpoint-v1','config':self.config,'identity':self.identity,
          'model':self.model.state_dict(),'ema':self.ema.state_dict(),'optimizer':self.optimizer.state_dict(),
          **{k:getattr(self,k) for k in COUNTERS},'stage_updates':self.stage_updates,
          'rngs':{n:getattr(self,n).get_state() for n in RNGS},'torch_rng':torch.get_rng_state(),
          'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
        with temp.open('wb') as f:torch.save(self.state(),f);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)

    def load_state(self,state,continuation=False,initialization_only=False):
        if state['identity']['m2_fingerprint']!=self.identity['m2_fingerprint']:raise ValueError('M2 mismatch')
        if state['config']['model']!=self.config['model']:raise ValueError('Architecture mismatch')
        if not continuation and not initialization_only:
            if state['schema']!='kimodo-m5-checkpoint-v1' or state['identity']!=self.identity or state['config']!=self.config:
                raise ValueError('Resume identity/config mismatch')
        self.model.load_state_dict(state['model']);self.ema.load_state_dict(state['ema'])
        if initialization_only:
            if state['step']!=0:raise ValueError('Initialization source must be step zero')
            return
        self.optimizer.load_state_dict(state['optimizer'])
        for k in COUNTERS:setattr(self,k,state[k])
        for name,value in state['rngs'].items():getattr(self,name).set_state(value)
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        self.stage_updates=state.get('stage_updates',0)
        if continuation:
            self.stage_updates=0;self.best_raw=self.best_ema=self.early_best=float('inf')
            self.best_raw_step=self.best_ema_step=0;self.bad_evaluations=0;self.last_evaluation_step=-1
            if self.step!=self.config['stage_start_step']:raise ValueError('Wrong continuation step')

    def restore(self,path):self.load_state(torch.load(path,map_location='cpu',weights_only=False))
