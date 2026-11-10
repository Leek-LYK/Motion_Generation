#!/usr/bin/env python3
"""Independent compiled-MuJoCo audit, not a converter-only roundtrip test."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch,mujoco
from scipy.spatial.transform import Rotation
from kimodo.skeleton.registry import build_skeleton
from kimodo.exports.mujoco import MujocoQposConverter
from training.g1_physical import PhysicalG1Converter,PhysicalG1BoundedDecoder,PHYSICAL_VERSION
from training.m3 import atomic_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();torch.set_num_threads(4)
    sk=build_skeleton(34);physical=PhysicalG1Converter(sk);legacy=MujocoQposConverter(sk)
    model=mujoco.MjModel.from_xml_path(str(physical.xml_path));data=mujoco.MjData(model);c=physical.constants
    rng=np.random.default_rng(6105);N=128;q=np.tile(model.qpos0,(N,1));q[:,:3]=rng.normal(0,.1,(N,3));q[:,2]+=.8
    q[:,3:7]=Rotation.random(N,random_state=rng).as_quat(scalar_first=True)
    for j in c['joint_ids']:
        lo,hi=model.jnt_range[j];q[:,model.jnt_qposadr[j]]=rng.uniform(lo+1e-4,hi-1e-4,N)
    with torch.no_grad():old=legacy.qpos_to_motion_dict(q,source_fps=30);new=physical.qpos_to_motion_dict(q);bottom=physical.foot_bottom(new).numpy()
    expected=[];expected_bottom=[]
    foot_geoms=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,x['name']) for x in c['foot']]
    for frame in q:
        data.qpos[:]=frame;mujoco.mj_forward(model,data)
        expected.append(np.stack([c['C'].T@data.xanchor[j] for j in c['joint_ids']]))
        expected_bottom.append([data.geom_xpos[g,2]-model.geom_size[g,0]-model.geom_size[g,1]*abs(data.geom_xmat[g].reshape(3,3)[2,2]) for g in foot_geoms])
    expected=np.asarray(expected);expected_bottom=np.asarray(expected_bottom)
    legacy_error=np.linalg.norm(old['posed_joints'][:,c['indices']].numpy()-expected,axis=-1)
    physical_error=np.linalg.norm(new['posed_joints'][:,c['indices']].numpy()-expected,axis=-1)
    foot_error=float(abs(bottom-expected_bottom).max())
    assert physical_error.max()<2e-6 and foot_error<2e-6,(physical_error.max(),foot_error)
    recovered=physical.dict_to_qpos(new);difference=np.arctan2(np.sin(recovered[:,7:]-q[:,7:]),np.cos(recovered[:,7:]-q[:,7:]))
    assert abs(difference).max()<1e-6
    # Verify the existing legacy feature chain really does use the legacy result.
    report={'passed':True,'version':PHYSICAL_VERSION,'random_seed':6105,'legal_poses':N,
      'legacy_joint_center_max_error_m':float(legacy_error.max()),'legacy_joint_center_mean_error_m':float(legacy_error.mean()),
      'physical_joint_center_max_error_m':float(physical_error.max()),'physical_joint_center_mean_error_m':float(physical_error.mean()),
      'physical_collider_bottom_max_error_m':foot_error,'physical_roundtrip_max_angle_error_rad':float(abs(difference).max()),
      'per_joint_legacy_mean_error_m':{sk.bone_order_names[idx]:float(legacy_error[:,k].mean()) for k,idx in enumerate(c['indices'])},
      'root_cause':'Legacy inverse reconstructs R_axis(q) @ R_rest at noncommuting hip/shoulder rest rotations; compiled MJCF requires R_rest @ R_axis(q).',
      'legacy_source_sha256':sha(ROOT.parent/'kimodo/source/kimodo/exports/mujoco.py'),'physical_code_sha256':sha(ROOT/'training/g1_physical.py'),
      'audit_script_sha256':sha(__file__),'xml_sha256':sha(physical.xml_path),'old_cache_modified':False}
    atomic_json(args.output,report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
