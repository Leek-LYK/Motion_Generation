#!/usr/bin/env python3
"""Separate raw FK, single-axis and bounded foot geometry for known tail failures.

The raw collider transform is calibrated from legal G1 qpos and must reproduce
independent random MuJoCo poses before any generated-motion result is reported.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation
from kimodo.skeleton.registry import build_skeleton
from training.m3 import atomic_json,sha
from training.g1_qpos import G1BoundedDecoder,nearest_angles


class RawColliderGeometry:
    def __init__(self):
        self.skeleton=build_skeleton(34);self.decoder=G1BoundedDecoder(self.skeleton)
        self.model=self.decoder.model;self.data=mujoco.MjData(self.model)
        self.C=self.decoder.converter.kimodo_to_mujoco_matrix.numpy().astype(float)
        self.geoms=[i for i in range(self.model.ngeom) if 'foot' in (mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,i) or '')
          and 'collision' in (mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,i) or '')]
        assert self.geoms
        zero=self.model.qpos0.copy();zero[2]=.8
        self.data.qpos[:]=zero;mujoco.mj_forward(self.model,self.data)
        # The frozen trajectory smoother requires a temporal sequence.
        motion=self.decoder.converter.qpos_to_motion_dict(np.tile(zero,(4,1)).astype(np.float32),source_fps=30)
        self.calibration=[]
        for geom in self.geoms:
            assert self.model.geom_type[geom]==mujoco.mjtGeom.mjGEOM_CAPSULE
            body=int(self.model.geom_bodyid[geom]);anchor=body
            while self.model.body_jntnum[anchor]==0:anchor=int(self.model.body_parentid[anchor])
            joint=int(self.model.body_jntadr[anchor]);name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,joint)
            bone=self.skeleton.bone_index[name.replace('_joint','_skel')]
            G=motion['global_rot_mats'][0,bone].numpy().astype(float);P=motion['posed_joints'][0,bone].numpy().astype(float)
            world_R=self.data.xmat[body].reshape(3,3).copy();world_P=self.data.xpos[body].copy()
            orientation=(self.C@G).T@world_R
            shift=world_R.T@(world_P-self.C@P)
            geom_R=Rotation.from_quat(self.model.geom_quat[geom],scalar_first=True).as_matrix()
            self.calibration.append((geom,bone,orientation,shift,geom_R))

    def raw(self,motion):
        P=np.asarray(motion['posed_joints']);G=np.asarray(motion['global_rot_mats']);bottom=[]
        for geom,bone,O,shift,geom_R in self.calibration:
            R=self.C@G[:,bone]@O;origin=(self.C@P[:,bone,:,None]).squeeze(-1)+(R@shift[:,None]).squeeze(-1)
            center=origin+(R@self.model.geom_pos[geom,:,None]).squeeze(-1)
            axis=(R@geom_R)[...,2]
            extent=self.model.geom_size[geom,0]+self.model.geom_size[geom,1]*abs(axis[:,2])
            bottom.append(center[:,2]-extent)
        return np.stack(bottom,1)

    def compiled(self,qpos):
        values=[]
        for q in qpos:
            self.data.qpos[:]=q;mujoco.mj_forward(self.model,self.data);row=[]
            for geom in self.geoms:
                extent=self.model.geom_size[geom,0]+self.model.geom_size[geom,1]*abs(self.data.geom_xmat[geom].reshape(3,3)[2,2])
                row.append(self.data.geom_xpos[geom,2]-extent)
            values.append(row)
        return np.asarray(values)

    def verify(self):
        rng=np.random.default_rng(6103);q=np.tile(self.model.qpos0,(24,1));q[:,:3]=rng.normal(0,.1,(24,3));q[:,2]+=.8
        q[:,3:7]=Rotation.random(24,random_state=rng).as_quat(scalar_first=True)
        for joint in self.decoder.joints:
            lo,hi=self.model.jnt_range[joint];q[:,self.model.jnt_qposadr[joint]]=rng.uniform(lo+.001,hi-.001,24)
        motion=self.decoder.converter.qpos_to_motion_dict(q.astype(np.float32),source_fps=30)
        predicted=self.raw(motion);actual=self.compiled(q)
        error=float(abs(predicted-actual).max())
        if error>2e-5:raise ValueError(f'Raw collider/MuJoCo mapping invalid: {error}')
        return error


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();torch.set_num_threads(4)
    geometry=RawColliderGeometry();mapping_error=geometry.verify();records=[]
    cases=[('970df39120445d57d1e534e8',1234,'heading0'),('7fceb592bfcc0370c1206c81',42,'heading0')]
    roots=[ROOT/'runs/m5_20261103'/name for name in ['A0_tiny_control','A1_tiny_fk']]
    roots += [ROOT/'runs/m5_research_20261103'/name for name in ['R2_cons_g30','R3_axis_g10','R4_axis_g30','S2_control_norm6','S2_control_fk','S2_treatment']]
    with torch.inference_mode():
        for run in roots:
            folder=run/'evaluations/generation_full_030000';rows=json.loads((folder/'per_sample.json').read_text())
            for sid,seed,condition in cases:
                row=next(r for r in rows if (r['sample_id'],r['seed'],r['condition'])==(sid,seed,condition));stem=folder/row['artifact_prefix']
                with np.load(stem.with_suffix('.npz')) as z:motion={k:z[k].copy() for k in z.files}
                q=np.loadtxt(stem.with_suffix('.csv'),delimiter=',');raw_bottom=geometry.raw(motion);bounded_bottom=geometry.compiled(q)
                decoder=geometry.decoder;r=decoder.offsets[None]@motion['local_rot_mats'][:,decoder.indices]
                unbounded=q.copy();unbounded[:,7:]=nearest_angles(r,decoder.axes,decoder.bounds,np.zeros(29,dtype=bool))
                axis_bottom=geometry.compiled(unbounded);t,g=np.unravel_index(bounded_bottom.argmin(),bounded_bottom.shape)
                record={'run':run.name,'sample_id':sid,'text':row['text'],'seed':seed,'condition':condition,
                  'worst_bounded_frame':int(t),'worst_geom':mujoco.mj_id2name(geometry.model,mujoco.mjtObj.mjOBJ_GEOM,int(geometry.geoms[g])),
                  'max_penetration_m':{'raw_fk':float(max(0,-raw_bottom.min())),'axis_only':float(max(0,-axis_bottom.min())),'bounded':float(max(0,-bounded_bottom.min()))},
                  'at_worst_frame':{'root_height_m':float(q[t,2]),'raw_bottom_m':float(raw_bottom[t,g]),'axis_bottom_m':float(axis_bottom[t,g]),'bounded_bottom_m':float(bounded_bottom[t,g])},
                  'root_height_trace_m':q[:,2].tolist(),'minimum_foot_bottom_trace_m':{'raw_fk':raw_bottom.min(1).tolist(),'axis_only':axis_bottom.min(1).tolist(),'bounded':bounded_bottom.min(1).tolist()}}
                records.append(record)
    atomic_json(args.output,{'passed':True,'mapping_verification_max_error_m':mapping_error,'records':records,
      'script_sha256':sha(__file__),'note':'Saved generations only; no optimizer updates, no root correction, no new model sampling. Projection stages are sequential, not additive.'})
    print('raw/compiled geometry mapping max error',mapping_error,flush=True)
    for r in records:print(r['run'],r['sample_id'],r['max_penetration_m'],r['at_worst_frame'],flush=True)


if __name__=='__main__':main()
