"""G1 physical coordinate contract v2, independently aligned to compiled MJCF.

MJCF hinges rotate in the child body's local frame: R_local = R_rest @ R_axis(q).
The frozen upstream inverse uses the opposite order for noncommuting rest/axis
rotations. This project-local adapter leaves upstream code and legacy data intact.
"""
from functools import lru_cache
import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation
from kimodo.assets import skeleton_asset_path
from kimodo.skeleton.registry import build_skeleton
from kimodo.geometry import quaternion_to_matrix,matrix_to_quaternion
from training.g1_qpos import G1BoundedDecoder

PHYSICAL_VERSION='g1-mjcf-native-hinge-v2'


@lru_cache(maxsize=1)
def physical_constants():
    sk=build_skeleton(34);model=mujoco.MjModel.from_xml_path(str(skeleton_asset_path('g1skel34','xml','g1.xml')))
    data=mujoco.MjData(model);q=model.qpos0.copy();q[:3]=[0,0,.8];q[3:7]=[1,0,0,0];data.qpos[:]=q;mujoco.mj_forward(model,data)
    C=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
    ids=np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_HINGE);indices=[];G=np.tile(np.eye(3),(34,1,1));P=np.zeros((34,3));P[0]=C.T@q[:3]
    for jid in ids:
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,int(jid));idx=sk.bone_index[name.replace('_joint','_skel')]
        indices.append(idx);G[idx]=C.T@data.xmat[model.jnt_bodyid[jid]].reshape(3,3)@C;P[idx]=C.T@data.xanchor[jid]
    rest=np.stack([G[int(sk.joint_parents[i])].T@G[i] for i in indices]);axes=(C.T@model.jnt_axis[ids,:,None]).squeeze(-1)
    axes=axes/np.linalg.norm(axes,axis=-1,keepdims=True)
    foot=[]
    for geom in range(model.ngeom):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,geom) or ''
        if 'foot' not in name or 'collision' not in name:continue
        if model.geom_type[geom]!=mujoco.mjtGeom.mjGEOM_CAPSULE:raise ValueError('Unexpected foot collider')
        body=int(model.geom_bodyid[geom]);anchor=body
        while model.body_jntnum[anchor]==0:anchor=int(model.body_parentid[anchor])
        jid=int(model.body_jntadr[anchor]);jname=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,jid);idx=sk.bone_index[jname.replace('_joint','_skel')]
        R=data.xmat[body].reshape(3,3);O=(C@G[idx]).T@R;shift=R.T@(data.xpos[body]-C@P[idx])
        geom_R=Rotation.from_quat(model.geom_quat[geom],scalar_first=True).as_matrix()
        foot.append({'name':name,'node':idx,'offset':O@(shift+model.geom_pos[geom]),'axis':O@geom_R[:,2],
                     'radius':float(model.geom_size[geom,0]),'half_length':float(model.geom_size[geom,1])})
    return {'C':C,'indices':np.asarray(indices,dtype=np.int64),'rest':rest,'axes':axes,'joint_ids':ids,'foot':foot,
            'limits':model.jnt_range[ids].copy(),'qpos0':model.qpos0.copy()}


def hinge_matrices(axes,angles):
    x,y,z=axes.unbind(-1);zero=torch.zeros_like(x)
    K=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(-1,3,3)
    identity=torch.eye(3,device=angles.device,dtype=angles.dtype)
    return identity+torch.sin(angles)[...,None,None]*K+(1-torch.cos(angles))[...,None,None]*(K@K)


class PhysicalG1Converter:
    def __init__(self,skeleton=None):
        self.skeleton=skeleton or build_skeleton(34);self.constants=physical_constants()
        self.xml_path=skeleton_asset_path('g1skel34','xml','g1.xml')

    def qpos_to_motion_dict(self,qpos,source_fps=30,**kwargs):
        if source_fps!=30:raise ValueError('Physical v2 expects already sampled 30Hz poses')
        q=torch.as_tensor(qpos);single=q.ndim==2
        if single:q=q[None]
        if q.ndim!=3 or q.shape[-1]!=36:raise ValueError('Expected [B,T,36]')
        if not q.is_floating_point():q=q.float()
        device,dtype=q.device,q.dtype;c=self.constants
        C=torch.as_tensor(c['C'],device=device,dtype=dtype);indices=torch.as_tensor(c['indices'],device=device)
        root=(C.T@q[...,:3,None]).squeeze(-1);root_R=C.T@quaternion_to_matrix(q[...,3:7])@C
        rest=torch.as_tensor(c['rest'],device=device,dtype=dtype);axes=torch.as_tensor(c['axes'],device=device,dtype=dtype)
        local=torch.eye(3,device=device,dtype=dtype).expand(*q.shape[:2],34,3,3).clone()
        local=local.index_copy(2,indices,rest@hinge_matrices(axes,q[...,7:]))
        local=local.index_copy(2,torch.tensor([0],device=device),root_R[:,:,None])
        global_R,positions,_=self.skeleton.fk(local,root)
        result={'local_rot_mats':local,'global_rot_mats':global_R,'posed_joints':positions,'root_positions':root}
        return {k:v[0] for k,v in result.items()} if single else result

    def dict_to_qpos(self,motion,device=None,numpy=True,**kwargs):
        local=torch.as_tensor(motion['local_rot_mats'],device=device);root=torch.as_tensor(motion['root_positions'],device=local.device,dtype=local.dtype)
        single=local.ndim==4
        if single:local=local[None];root=root[None]
        c=self.constants;C=torch.as_tensor(c['C'],device=local.device,dtype=local.dtype)
        rest=torch.as_tensor(c['rest'],device=local.device,dtype=local.dtype);axes=torch.as_tensor(c['axes'],device=local.device,dtype=local.dtype)
        indices=torch.as_tensor(c['indices'],device=local.device);r=rest.transpose(-1,-2)@local[...,indices,:,:]
        a=torch.diagonal(r,dim1=-2,dim2=-1).sum(-1)-(axes*(r@axes[...,None]).squeeze(-1)).sum(-1)
        skew=torch.stack((r[...,2,1]-r[...,1,2],r[...,0,2]-r[...,2,0],r[...,1,0]-r[...,0,1]),-1)
        b=(skew*axes).sum(-1);angles=torch.atan2(b,a)
        xyz=(C@root[...,None]).squeeze(-1);quat=matrix_to_quaternion(C@local[:,:,0]@C.T)
        result=torch.cat((xyz,quat,angles),-1)
        if single:result=result[0]
        return result.detach().cpu().numpy() if numpy else result

    def foot_bottom(self,motion):
        P=motion['posed_joints'];G=motion['global_rot_mats'];values=[]
        for item in self.constants['foot']:
            offset=torch.as_tensor(item['offset'],device=P.device,dtype=P.dtype);axis=torch.as_tensor(item['axis'],device=P.device,dtype=P.dtype)
            node=item['node'];center_y=P[...,node,1]+(G[...,node,:,:]@offset[...,None]).squeeze(-1)[...,1]
            direction_y=(G[...,node,:,:]@axis[...,None]).squeeze(-1)[...,1]
            values.append(center_y-item['radius']-item['half_length']*abs(direction_y))
        return torch.stack(values,-1)


class PhysicalG1BoundedDecoder(G1BoundedDecoder):
    def __init__(self,skeleton=None):
        skeleton=skeleton or build_skeleton(34);super().__init__(skeleton)
        self.converter=PhysicalG1Converter(skeleton)
        self.axes=self.converter.constants['axes'];self.offsets=np.swapaxes(self.converter.constants['rest'],-1,-2)

    def decode(self,motion):
        q,info=super().decode(motion);info['decoder']=PHYSICAL_VERSION;return q,info
