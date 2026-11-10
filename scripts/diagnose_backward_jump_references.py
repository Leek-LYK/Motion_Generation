#!/usr/bin/env python3
"""Compare foot flight in training references, held-out reference and controlled generations."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import mujoco
from kimodo.assets import skeleton_asset_path
from kimodo.exports.mujoco import MujocoQposConverter
from training.m3 import make_model,atomic_json,sha
from training.m4 import load_data
from training.m4_readiness import movement


def main():
    root=ROOT/'runs/m4_research_20261103';manifest=json.loads((ROOT/'runs/m4_20261102/data/BCD.json').read_text())
    texts={'A person standing in an upright stance jumps backwards.','A person jumps backward.',
           'The person facing forward jumps backwards.','The person jumps backwards while facing forward.'}
    selected=dict(manifest,train=[x for x in manifest['train'] if x['text'] in texts],
                  val=[x for x in manifest['val'] if x['text']=='The person standing upright jumps backward.'])
    network=make_model({'latent_dim':64,'ff_size':128,'num_layers':1,'num_heads':4})
    converter=MujocoQposConverter(network.motion_rep.skeleton)
    xml=skeleton_asset_path('g1skel34','xml','g1.xml');model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
    ids={side:[i for i in range(model.ngeom) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith(side+'_foot')
               and 'collision' in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '')] for side in ('left','right')}
    def measure(q):
        clearance=[]
        for frame in q:
            data.qpos[:]=frame;mujoco.mj_forward(model,data);bottom=[]
            for side in ('left','right'):
                values=[]
                for i in ids[side]:
                    if model.geom_type[i]!=mujoco.mjtGeom.mjGEOM_CAPSULE:raise ValueError('Unexpected foot collision geometry')
                    extent=model.geom_size[i,0]+model.geom_size[i,1]*abs(data.geom_xmat[i].reshape(3,3)[2,2])
                    values.append(data.geom_xpos[i,2]-extent)
                bottom.append(min(values))
            clearance.append(min(bottom))
        c=np.array(clearance)
        return {'frames':len(q),'root_height_q90_q10_m':float(np.quantile(q[:,2],.9)-np.quantile(q[:,2],.1)),
                'root_height_max_range_m':float(np.ptp(q[:,2])),'max_both_feet_clearance_m':float(c.max()),
                'both_feet_above_5mm_frames':int((c>.005).sum()),'both_feet_above_15mm_frames':int((c>.015).sum())}
    rows=[]
    with torch.inference_mode():
        for split in ['train','val']:
            for event in load_data(selected,split):
                decoded=network.motion_rep.inverse(event['x0'][None],is_normalized=True);q=converter.dict_to_qpos(decoded)[0]
                record={'kind':'reference','split':split,'sample_id':event['sample_id'],'text':event['text'],**measure(q)}
                rows.append(record);print(record,flush=True)
    for seed in [0,42,1234]:
        path=root/'R3_D_high_noise/controlled_three_seeds_cfg2'/f'seed_{seed}/jump_unseen_text_same_class.csv'
        record={'kind':'generated','seed':seed,**measure(np.loadtxt(path,delimiter=','))};rows.append(record);print(record,flush=True)
    atomic_json(root/'diagnostics/backward_jump_reference_semantics.json',{'records':rows,'kinematic_only':True,
                'xml_sha256':sha(xml),'script_sha256':sha(__file__),'plane':'original MJCF floor z=0; no physics stepping'})


if __name__=='__main__':main()
