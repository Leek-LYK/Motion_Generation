"""Physical M5 optimizer ablation; random initialization is shared, moments are not."""
from training.m5_physical import PhysicalTrainer
from training.adam_atan2 import AdamAtan2,VERSION

SCHEMA='kimodo-m5-physical-atan2-checkpoint-v1'


class Atan2PhysicalTrainer(PhysicalTrainer):
    def __init__(self,config,samples,identity,stats_dir,device='cuda:0'):
        super().__init__(config,samples,identity,stats_dir,device)
        if config.get('optimizer_name')!=VERSION:raise ValueError('Explicit versioned optimizer required')
        self.optimizer=AdamAtan2(self.model.parameters(),lr=config['learning_rate'],betas=tuple(config['optimizer_betas']),
          weight_decay=config.get('optimizer_weight_decay',0.),a=config.get('optimizer_a',1.),b=config.get('optimizer_b',1.))

    def state(self):
        state=super().state();state.update(schema=SCHEMA,optimizer_kind=VERSION);return state

    def load_state(self,state,continuation=False,initialization_only=False):
        if initialization_only and (self.step!=0 or self.optimizer.state):raise ValueError('Initial weights can only be loaded into a fresh optimizer')
        if initialization_only and state.get('schema')=='kimodo-m5-physical-checkpoint-v2':
            if state['step']!=0:raise ValueError('AdamW checkpoints can supply only unchanged step-zero weights')
            return super().load_state(state,initialization_only=True)
        if state.get('schema')!=SCHEMA or state.get('optimizer_kind')!=VERSION:raise ValueError('Optimizer lineage mismatch; AdamW momentum cannot masquerade as Adam-atan2')
        if continuation:
            for key in ['optimizer_name','optimizer_betas','optimizer_a','optimizer_b','optimizer_weight_decay']:
                if state['config'].get(key)!=self.config.get(key):raise ValueError('Optimizer configuration cannot change during continuation: '+key)
        proxy=dict(state,schema='kimodo-m5-physical-checkpoint-v2')
        return super().load_state(proxy,continuation=continuation,initialization_only=initialization_only)
