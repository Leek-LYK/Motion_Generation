"""Isolated M5-A research: supervised FK, self-consistency, and hinge-axis defect.

These are distinct objectives. The hinge term is differentiable and does not
clamp angles, modify trajectories, or introduce a MuJoCo operation into autograd.
"""
import torch
import torch.nn.functional as F
from kimodo.exports.mujoco import MujocoQposConverter
from training.m3 import losses as norm6
from training.m5_losses import geometry


class HingeGeometry:
    def __init__(self,skeleton,device='cpu'):
        converter=MujocoQposConverter(skeleton)
        self.indices=converter._mujoco_indices_to_kimodo_indices.to(device=device,dtype=torch.long)
        self.axes=converter._mujoco_joint_axis_values_f2q_space.to(device=device,dtype=torch.float32)
        self.axes=self.axes/self.axes.norm(dim=-1,keepdim=True)
        self.offsets=converter._rot_offsets_f2q.to(device=device,dtype=torch.float32)[self.indices]
        self.names=[skeleton.bone_order_names[int(i)] for i in self.indices.cpu()]

    def defects(self,local_rotations):
        """For SO(3), 0.5||R*u-u||² vanishes exactly on rotations about u.

        R is first converted into the same f2q rest/axis frame as the frozen
        exporter. Hinge feasibility is measured without joint-limit clamping.
        """
        rotations=self.offsets@local_rotations[...,self.indices,:,:].float()
        transformed=torch.matmul(rotations,self.axes[...,None]).squeeze(-1)
        energy=.5*(transformed-self.axes).square().sum(-1)
        cosine=(transformed*self.axes).sum(-1).clamp(-1,1)
        return energy,torch.acos(cosine)


def per_sample_terms(rep,prediction,target,mask,hinge):
    prediction=prediction.float();target=target.float()
    raw,fk,motion=geometry(rep,prediction);truth=rep.unnormalize(target)
    target_positions=truth[...,5:107].reshape(*target.shape[:2],34,3)
    predicted_positions=raw[...,5:107].reshape_as(target_positions)
    scale=torch.sqrt(rep.stats.std.to(prediction.device)[5:107]**2+rep.stats.eps).reshape(34,3)
    valid=mask[...,None,None];counts=mask.sum(1)
    fk_values=F.smooth_l1_loss(fk/scale,target_positions/scale,reduction='none')
    cons_values=F.smooth_l1_loss(fk/scale,predicted_positions/scale,reduction='none')
    smooth_distance=torch.sqrt((fk-predicted_positions).square().sum(-1)+1e-6)-.001
    axis,angles=hinge.defects(motion['local_rot_mats'])
    return {'fk_gt_norm':(fk_values*valid).sum((1,2,3))/(counts*102),
      'consistency_norm':(cons_values*valid).sum((1,2,3))/(counts*102),
      'consistency_smooth_m':(smooth_distance*mask[...,None]).sum((1,2))/(counts*34),
      'fk_gt_m':(torch.linalg.vector_norm(fk-target_positions,dim=-1)*mask[...,None]).sum((1,2))/(counts*34),
      'fk_self_m':(torch.linalg.vector_norm(fk-predicted_positions,dim=-1)*mask[...,None]).sum((1,2))/(counts*34),
      'axis_defect':(axis*mask[...,None]).sum((1,2))/(counts*len(hinge.indices)),
      'axis_angle_rad':(angles*mask[...,None]).sum((1,2))/(counts*len(hinge.indices))},(raw,fk,motion,axis,angles)


def research_objective(rep,prediction,target,mask,config,hinge):
    prediction=prediction.float();target=target.float()
    base,parts=norm6(prediction,target,mask)
    terms,decoded=per_sample_terms(rep,prediction,target,mask,hinge)
    weights=mask.sum(1)/mask.sum()
    means={k:(v*weights).sum() for k,v in terms.items()}
    # Preserve the original M5 reduction exactly (not mean of per-sample means).
    raw,fk,motion,axis,angles=decoded
    truth=rep.unnormalize(target);positions=truth[...,5:107].reshape(*target.shape[:2],34,3)
    scale=torch.sqrt(rep.stats.std.to(prediction.device)[5:107]**2+rep.stats.eps).reshape(34,3)
    fk_loss=(F.smooth_l1_loss(fk/scale,positions/scale,reduction='none')*mask[...,None,None]).sum()/(mask.sum()*102)
    predicted_positions=raw[...,5:107].reshape_as(positions)
    consistency_norm=(F.smooth_l1_loss(fk/scale,predicted_positions/scale,reduction='none')*mask[...,None,None]).sum()/(mask.sum()*102)
    # Match the metre-space Euclidean consistency criterion, smoothed at 1 mm.
    consistency=((torch.sqrt((fk-predicted_positions).square().sum(-1)+1e-6)-.001)*mask[...,None]).sum()/(mask.sum()*34)
    axis_loss=(axis*mask[...,None]).sum()/(mask.sum()*len(hinge.indices))
    total=base+config.get('fk_weight',1.)*fk_loss
    if config.get('consistency_weight',0):total=total+config['consistency_weight']*consistency
    if config.get('axis_weight',0):total=total+config['axis_weight']*axis_loss
    parts.update(fk=fk_loss,fk_gt_norm=fk_loss,consistency_norm=consistency_norm,consistency_loss_m=consistency,axis_defect=axis_loss,
      norm6_comparison=base,fk_position_inconsistency_m=means['fk_self_m'],fk_gt_position_error_m=means['fk_gt_m'],
      mean_axis_angle_rad=means['axis_angle_rad'])
    return total,parts
