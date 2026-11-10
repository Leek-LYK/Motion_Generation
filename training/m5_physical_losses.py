"""Physical-v2 objectives; fixed MJCF-native hinge frame and explicit diagnostics."""
import torch
import torch.nn.functional as F
from kimodo.skeleton.transforms import global_rots_to_local_rots
from training.m3 import COMPONENTS,losses as norm6
from training.m5_losses import WEIGHTS
from training.g1_physical import physical_constants,PhysicalG1Converter


def stable_cont6d(values,eps=1e-6):
    """Gram-Schmidt with a deterministic finite fallback at degenerate predictions.

    Regular 6D rotations follow the existing column convention exactly. Fallback
    frequency is reported; it is not counted as evidence of feasible rotations.
    """
    a,b=values[...,:3],values[...,3:];na=torch.linalg.vector_norm(a,dim=-1,keepdim=True)
    fallback=torch.zeros_like(a);fallback[...,0]=1
    x=torch.where(na>eps,a/na.clamp_min(eps),fallback)
    residual=b-(x*b).sum(-1,keepdim=True)*x;nb=torch.linalg.vector_norm(residual,dim=-1,keepdim=True)
    # Choose a coordinate direction furthest from x, then orthogonalize it.
    orth=F.one_hot(x.abs().argmin(-1),3).to(values.dtype);orth=orth-(orth*x).sum(-1,keepdim=True)*x
    orth=orth/torch.linalg.vector_norm(orth,dim=-1,keepdim=True).clamp_min(eps)
    y=torch.where(nb>eps,residual/nb.clamp_min(eps),orth);z=torch.cross(x,y,dim=-1)
    return torch.stack((x,y,z),-1),(na[...,0]<=eps)|(nb[...,0]<=eps)


def geometry(rep,normalized):
    raw=rep.unnormalize(normalized.float());B,T=raw.shape[:2]
    global_R,fallback=stable_cont6d(raw[...,107:311].reshape(B,T,34,6))
    local=global_rots_to_local_rots(global_R,rep.skeleton)
    offset=torch.stack((raw[...,0],torch.zeros_like(raw[...,0]),raw[...,2]),-1)
    positions=raw[...,5:107].reshape(B,T,34,3)+offset[...,None,:];root=positions[...,rep.skeleton.root_idx,:]
    _,posed,_=rep.skeleton.fk(local,root)
    motion={'local_rot_mats':local,'global_rot_mats':global_R,'posed_joints':posed,'root_positions':root,
            'smooth_root_pos':raw[...,:3],'global_root_heading':raw[...,3:5],'foot_contacts':raw[...,413:]>0.5}
    return raw,posed-offset[...,None,:],motion,fallback


class NativeHingeGeometry:
    def __init__(self,skeleton,device='cpu'):
        c=physical_constants();self.indices=torch.as_tensor(c['indices'],device=device)
        self.axes=torch.as_tensor(c['axes'],device=device,dtype=torch.float32)
        self.rest=torch.as_tensor(c['rest'],device=device,dtype=torch.float32)
        self.converter=PhysicalG1Converter(skeleton)

    def defects(self,local):
        # Native child-body rotation is rest.T @ R_local, with the native axis.
        rotation=self.rest.transpose(-1,-2)@local[...,self.indices,:,:]
        transformed=(rotation@self.axes[...,None]).squeeze(-1)
        return .5*(transformed-self.axes).square().sum(-1)


def objective(rep,prediction,target,mask,config,hinge):
    prediction=prediction.float();target=target.float();base,parts=norm6(prediction,target,mask)
    raw,fk,motion,fallback=geometry(rep,prediction);truth=rep.unnormalize(target)
    target_P=truth[...,5:107].reshape(*target.shape[:2],34,3);pred_P=raw[...,5:107].reshape_as(target_P)
    scale=torch.sqrt(rep.stats.std.to(prediction.device)[5:107]**2+rep.stats.eps).reshape(34,3)
    valid=mask[...,None,None];denominator=mask.sum()*102
    fk_norm=(F.smooth_l1_loss(fk/scale,target_P/scale,reduction='none')*valid).sum()/denominator
    mode=config['loss_objective']
    if mode=='norm6':total=base
    elif mode=='norm7':total=base+config.get('fk_weight',1.)*fk_norm
    elif mode=='weighted_raw7':
        raw_parts={k:(F.smooth_l1_loss(raw[...,a:b],truth[...,a:b],reduction='none')*mask[...,None]).sum()/(mask.sum()*(b-a)) for k,(a,b) in COMPONENTS.items()}
        raw_fk=(F.smooth_l1_loss(fk,target_P,reduction='none')*valid).sum()/denominator
        total=(sum(WEIGHTS[k]*v for k,v in raw_parts.items())+WEIGHTS['fk']*raw_fk)/sum(WEIGHTS.values())
        parts.update({'raw_'+k:v for k,v in raw_parts.items()},raw_fk=raw_fk)
    else:raise ValueError(mode)
    euclidean=torch.linalg.vector_norm(fk-pred_P,dim=-1)
    consistency=((torch.sqrt((fk-pred_P).square().sum(-1)+1e-6)-.001)*mask[...,None]).sum()/(mask.sum()*34)
    axis=(hinge.defects(motion['local_rot_mats'])*mask[...,None]).sum()/(mask.sum()*29)
    foot=hinge.converter.foot_bottom(motion)
    ground=(F.relu(-foot-.005).square()*mask[...,None]).sum()/(mask.sum()*foot.shape[-1])
    for key,value in [('consistency_weight',consistency),('axis_weight',axis),('ground_weight',ground)]:
        if config.get(key,0):total=total+config[key]*value
    parts.update(norm6_comparison=base,fk=fk_norm,consistency_loss_m=consistency,axis_defect=axis,ground_loss_m2=ground,
      fk_position_inconsistency_m=(euclidean*mask[...,None]).sum()/(mask.sum()*34),
      fk_gt_position_error_m=(torch.linalg.vector_norm(fk-target_P,dim=-1)*mask[...,None]).sum()/(mask.sum()*34),
      rotation_fallback_fraction=(fallback*mask[...,None]).sum()/(mask.sum()*34))
    return total,parts
