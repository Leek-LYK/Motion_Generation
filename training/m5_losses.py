"""Masked seven-component supervision, including FK against ground truth positions."""
import torch
import torch.nn.functional as F
from training.m3 import COMPONENTS,losses as norm6

WEIGHTS={'root_pos':10.,'heading':2.,'joint_pos':10.,'joint_rot':10.,'joint_vel':3.,'contacts':4.,'fk':5.}


def geometry(rep,normalized):
    raw=rep.unnormalize(normalized.float())
    motion=rep.inverse(normalized.float(),is_normalized=True,posed_joints_from='rotations')
    offset=torch.stack((raw[...,0],torch.zeros_like(raw[...,0]),raw[...,2]),dim=-1)
    fk_features=motion['posed_joints']-offset[...,None,:]
    return raw,fk_features,motion


def objective(rep,prediction,target,mask,mode='norm7',fk_weight=1.):
    prediction=prediction.float();target=target.float()
    base,parts=norm6(prediction,target,mask)
    if mode=='norm6':return base,parts
    if mode not in ('norm7','weighted_raw7'):raise ValueError(mode)
    raw_pred,fk,_=geometry(rep,prediction);raw_target=rep.unnormalize(target)
    target_positions=raw_target[...,5:107].reshape(*target.shape[:2],34,3)
    predicted_positions=raw_pred[...,5:107].reshape_as(target_positions)
    valid=mask[...,None,None];denominator=mask.sum()*102
    if mode=='norm7':
        scale=torch.sqrt(rep.stats.std.to(prediction.device)[5:107]**2+rep.stats.eps).reshape(34,3)
        fk_loss=(F.smooth_l1_loss(fk/scale,target_positions/scale,reduction='none')*valid).sum()/denominator
        total=base+fk_weight*fk_loss
    else:
        parts={name:(F.smooth_l1_loss(raw_pred[...,a:b],raw_target[...,a:b],reduction='none')*mask[...,None]).sum()/(mask.sum()*(b-a))
               for name,(a,b) in COMPONENTS.items()}
        fk_loss=(F.smooth_l1_loss(fk,target_positions,reduction='none')*valid).sum()/denominator
        total=(sum(WEIGHTS[k]*v for k,v in parts.items())+WEIGHTS['fk']*fk_loss)/sum(WEIGHTS.values())
    parts.update(fk=fk_loss,norm6_comparison=base,
                 fk_position_inconsistency_m=(torch.linalg.vector_norm(fk-predicted_positions,dim=-1)*mask[...,None]).sum()/(mask.sum()*34))
    return total,parts
