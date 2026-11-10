"""Late heading curriculum without text dropout or loss changes."""
import copy
import torch
from training.m4 import M4Trainer,rotate_batch


class OrientationTrainer(M4Trainer):
    def prepare_training_batch(self,batch):
        result=batch
        probability=self.config['heading_augmentation_probability']
        if not 0<=probability<=1:raise ValueError('Invalid heading augmentation probability')
        angles=torch.zeros(len(batch['x0']),device=self.device)
        if self.config['heading_augmentation'] and probability>0:
            angles=(torch.rand(len(batch['x0']),device=self.device,generator=self.heading_rng)*2-1)*torch.pi
            enabled=torch.rand(len(batch['x0']),device=self.device,generator=self.heading_rng)<probability
            angles=angles*enabled
            result=rotate_batch(self.model,result,angles)
        dropped=torch.zeros(len(batch['x0']),device=self.device,dtype=torch.bool)
        if self.config['text_dropout']:
            dropped=torch.rand(len(batch['x0']),device=self.device,generator=self.text_rng)<self.config['text_dropout']
            result=dict(result,text_feat=result['text_feat'].masked_fill(dropped[:,None,None],0))
        return result,dropped,angles


def balanced_validation(native,rotated):
    if native['events']!=rotated['events']:raise ValueError('Unbalanced validation views')
    result=copy.deepcopy(native)
    result['macro_loss']=(native['macro_loss']+rotated['macro_loss'])/2
    result['events']=native['events']+rotated['events']
    result['unique_events']=native['events']
    for family in ('groups','cohorts','strata','timesteps','parts'):
        if native[family].keys()!=rotated[family].keys():raise ValueError('View metric mismatch')
        result[family]={k:(native[family][k]+rotated[family][k])/2 for k in native[family]}
    result['per_sample']=[dict(x,view='native') for x in native['per_sample']]+[dict(x,view='rotated90') for x in rotated['per_sample']]
    result['selection_metric']='equal native/rotated90 class-cohort macro mean'
    return result
