"""Versioned physical objectives with an optional bounded-manifold penalty."""
from training.m5_physical_losses import objective as physical_objective,geometry

VERSION='m5-physical-constraints-v1'


def constraint_objective(rep,prediction,target,mask,config,hinge,legality):
    # Accumulate all geometric gradient branches in FP32 before the single cast
    # back to the BF16 network output, rather than rounding each branch separately.
    prediction=prediction.float();target=target.float()
    total,parts=physical_objective(rep,prediction,target,mask,config,hinge)
    # The unchanged base loss is returned directly when the extra coefficient is
    # zero. The new diagnostic has no randomness or gradient side effects.
    _,_,motion,_=geometry(rep,prediction)
    legal=legality.loss(motion['local_rot_mats'],mask)
    if config.get('legality_weight',0.):total=total+config['legality_weight']*legal
    parts['bounded_hinge_legality']=legal
    return total,parts
