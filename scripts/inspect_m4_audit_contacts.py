#!/usr/bin/env python3
"""Descriptive full-frame contact geometry for the consumed audit; no new gate or tuning."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import mujoco
from training.m3 import make_model,atomic_json,sha
from training.m4 import load_data
from training.g1_qpos import G1BoundedDecoder


def main():
    root=ROOT/'runs/m4_research_20261103';release=json.loads((root/'audit_release.json').read_text())
    assert json.loads((root/'audit_execution_status.json').read_text())['phase']=='audit_complete_requires_final_review'
    network=make_model({'latent_dim':64,'ff_size':128,'num_layers':1,'num_heads':4});decoder=G1BoundedDecoder(network.motion_rep.skeleton)
    model=decoder.model;data=mujoco.MjData(model)
    feet={side:[i for i in range(model.ngeom) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith(side+'_foot')
      and 'collision' in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '')] for side in ('left','right')}
    def measure(q):
        bottom=[];xy=[]
        for frame in q:
            data.qpos[:]=frame;mujoco.mj_forward(model,data);heights=[];centers=[]
            for ids in feet.values():
                values=[]
                for i in ids:
                    assert model.geom_type[i]==mujoco.mjtGeom.mjGEOM_CAPSULE
                    extent=model.geom_size[i,0]+model.geom_size[i,1]*abs(data.geom_xmat[i].reshape(3,3)[2,2])
                    values.append(data.geom_xpos[i,2]-extent)
                heights.append(min(values));centers.append(data.geom_xpos[ids,:2].mean(0).copy())
            bottom.append(heights);xy.append(centers)
        c=np.asarray(bottom);near=(c[:-1]<.02)&(c[1:]<.02)&(c[:-1]>-.02)&(c[1:]>-.02)
        speed=np.linalg.norm(np.diff(xy,axis=0),axis=-1)*30;slip=speed[near]
        return {'both_feet_clear_15mm_frames':int(np.sum(c.min(1)>.015)),
          'both_feet_max_clearance_m':float(c.min(1).max()),'max_foot_penetration_m':float(max(0,-c.min())),
          'foot_below_minus_5cm_frame_fraction':float(np.mean(c.min(1)<-.05)),
          'near_ground_foot_speed_p95_m_s':float(np.quantile(slip,.95)) if len(slip) else None}
    records=[]
    manifest=json.loads((root/'locked_audit.json').read_text())
    with torch.inference_mode():
        for event in load_data(manifest,'test_repetition'):
            raw=network.motion_rep.inverse(event['x0'][None],is_normalized=True)
            q,_=decoder.decode({k:v[0] for k,v in raw.items()})
            records.append({'run':'reference','condition':'native','group':event['group'],'sample_id':event['sample_id'],**measure(q)})
    for run in release['runs']:
        for condition in ['native','heading0']:
            folder=root/run/'locked_audit_v1'/condition
            for row in json.loads((folder/'per_sample.json').read_text()):
                q=np.loadtxt((folder/row['artifact_prefix']).with_suffix('.csv'),delimiter=',')
                records.append({'run':run,'condition':condition,'group':row['group'],'sample_id':row['sample_id'],'seed':row['seed'],
                  'amplitude_ratio':row['amplitude_ratio'],**measure(q)})
            print(run,condition,'contact geometry complete',flush=True)
    summary=[]
    for run,condition in [('reference','native')]+[(run,c) for run in release['runs'] for c in ['native','heading0']]:
        rows=[r for r in records if r['run']==run and r['condition']==condition];jump=[r for r in rows if r['group']=='jump']
        summary.append({'run':run,'condition':condition,'clips':len(rows),
          'jump_clips_with_at_least_2_dual_foot_flight_frames':sum(r['both_feet_clear_15mm_frames']>=2 for r in jump),
          'jump_clips':len(jump),'jump_median_dual_foot_flight_frames':float(np.median([r['both_feet_clear_15mm_frames'] for r in jump])),
          'max_foot_penetration_m':max(r['max_foot_penetration_m'] for r in rows),
          'median_clip_max_foot_penetration_m':float(np.median([r['max_foot_penetration_m'] for r in rows])),
          'mean_clip_fraction_below_minus_5cm':float(np.mean([r['foot_below_minus_5cm_frame_fraction'] for r in rows]))})
    result={'summary':summary,'records':records,'script_sha256':sha(__file__),'audit_release_sha256':sha(root/'audit_release.json'),
      'interpretation':'Descriptive geometry, not dynamics or an added/changed acceptance threshold. Height amplitude alone is not a semantic jump success score. Two frames at 30Hz over 15mm is only a transparent flight diagnostic; full flight duration and clearance retained.'}
    atomic_json(root/'audit_contact_geometry.json',result);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
