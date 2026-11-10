"""Differentiable distance to the MJCF-native bounded-hinge rotation manifold.

The closest legal angle is solved with Torch under no_grad. The residual retains
its gradient w.r.t. predicted rotations; the envelope derivative is exact away
from non-unique projections. No simulator, NumPy exporter, IK, root correction,
or temporal smoothing participates in the autograd path.
"""
import math
import torch,mujoco
from kimodo.assets import skeleton_asset_path
from training.g1_physical import physical_constants,hinge_matrices

VERSION='g1-native-bounded-hinge-envelope-v1'


class NativeLegalityLoss:
    def __init__(self,device='cpu'):
        self.device=torch.device(device);c=physical_constants();model=mujoco.MjModel.from_xml_path(str(skeleton_asset_path('g1skel34','xml','g1.xml')))
        self.indices=torch.as_tensor(c['indices'],device=self.device);self._rest=torch.as_tensor(c['rest'],device=self.device)
        self._axes=torch.as_tensor(c['axes'],device=self.device);self._bounds=torch.as_tensor(c['limits'],device=self.device)
        self.limited=torch.as_tensor(model.jnt_limited[c['joint_ids']].astype(bool),device=self.device)
        lo=float(self._bounds[self.limited,0].min()) if self.limited.any() else -math.pi
        hi=float(self._bounds[self.limited,1].max()) if self.limited.any() else math.pi
        self.turns=tuple(range(math.floor(lo/(2*math.pi))-1,math.ceil(hi/(2*math.pi))+2));self.cache={}

    def constants(self,dtype):
        if dtype not in self.cache:self.cache[dtype]=tuple(t.to(dtype=dtype) for t in [self._rest,self._axes,self._bounds])
        return self.cache[dtype]

    def project_native(self,rotation):
        if rotation.device!=self.device:raise ValueError('Projection device mismatch')
        _,axes,bounds=self.constants(rotation.dtype)
        with torch.no_grad():
            r=rotation.detach();a=torch.diagonal(r,dim1=-2,dim2=-1).sum(-1)-(axes*(r@axes[...,None]).squeeze(-1)).sum(-1)
            skew=torch.stack((r[...,2,1]-r[...,1,2],r[...,0,2]-r[...,2,0],r[...,1,0]-r[...,0,1]),-1);b=(skew*axes).sum(-1)
            phi=torch.atan2(b,a);low,high=bounds.unbind(-1)
            candidates=torch.stack([low.expand_as(phi),high.expand_as(phi),*[phi+2*math.pi*k for k in self.turns]],-1)
            valid=(candidates>=low[...,None])&(candidates<=high[...,None]);scores=a[...,None]*candidates.cos()+b[...,None]*candidates.sin()
            selected=scores.masked_fill(~valid,-torch.inf).argmax(-1)
            bounded=candidates.gather(-1,selected[...,None]).squeeze(-1);angles=torch.where(self.limited,bounded,phi)
            projected=hinge_matrices(axes,angles)
        return projected,angles

    def native(self,local):
        rest,_,_=self.constants(local.dtype);return rest.transpose(-1,-2)@local[...,self.indices,:,:]

    def per_joint(self,local):
        rotation=self.native(local);closest,angles=self.project_native(rotation)
        return .5*(rotation-closest).square().sum((-1,-2)),angles

    def loss(self,local,mask):
        values,_=self.per_joint(local)
        if mask.sum()<=0:raise ValueError('At least one valid frame is required')
        return (values*mask[...,None]).sum()/(mask.sum()*len(self.indices))
