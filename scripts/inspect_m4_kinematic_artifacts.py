#!/usr/bin/env python3
"""Measure foot geometry and angular discontinuities, without physics or tuning."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import mujoco
import torch
from training.g1_qpos import G1BoundedDecoder
from training.m3 import atomic_json,make_model,sha
from training.m4 import load_data


def main():
    research=ROOT/'runs/m4_research_20261103'
    model=make_model({'latent_dim':64,'ff_size':128,'num_layers':1,'num_heads':4})
    decoder=G1BoundedDecoder(model.motion_rep.skeleton);mj=decoder.model;data=mujoco.MjData(mj)
    feet={s:[i for i in range(mj.ngeom) if (mujoco.mj_id2name(mj,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith(s+'_foot')
           and 'collision' in (mujoco.mj_id2name(mj,mujoco.mjtObj.mjOBJ_GEOM,i) or '')] for s in ('left','right')}
    assert all(feet.values())
    def measure(q):
        clearance=[];centers=[]
        for frame in q:
            data.qpos[:]=frame;mujoco.mj_forward(mj,data);bottom=[];xy=[]
            for ids in feet.values():
                values=[]
                for i in ids:
                    assert mj.geom_type[i]==mujoco.mjtGeom.mjGEOM_CAPSULE
                    extent=mj.geom_size[i,0]+mj.geom_size[i,1]*abs(data.geom_xmat[i].reshape(3,3)[2,2])
                    values.append(data.geom_xpos[i,2]-extent)
                bottom.append(min(values));xy.append(data.geom_xpos[ids,:2].mean(0).copy())
            clearance.append(bottom);centers.append(xy)
        c=np.asarray(clearance);xy=np.asarray(centers)
        speeds=np.linalg.norm(np.diff(xy,axis=0),axis=-1)*30
        contact=(c[:-1]<.02)&(c[1:]<.02)&(c[:-1]>-.02)&(c[1:]>-.02)
        slip=speeds[contact]
        angles=abs(np.diff(q[:,7:],axis=0))*30
        t,j=np.unravel_index(angles.argmax(),angles.shape)
        return {'frames':len(q),'max_foot_penetration_m':float(max(0,-c.min())),
                'frames_with_foot_below_minus_5cm_fraction':float(np.mean(c.min(1)<-.05)),
                'both_feet_clear_15mm_fraction':float(np.mean(c.min(1)>.015)),
                'near_ground_foot_intervals':int(contact.sum()),
                'near_ground_foot_xy_speed_p95_m_s':float(np.quantile(slip,.95)) if len(slip) else None,
                'max_joint_speed_rad_s':float(angles[t,j]),'worst_joint_transition_frames':[int(t),int(t+1)],
                'worst_joint':mujoco.mj_id2name(mj,mujoco.mjtObj.mjOBJ_JOINT,int(decoder.joints[j])),
                'hinge_interval_fraction_over_20_rad_s':float(np.mean(angles>20)),
                'hinge_interval_fraction_over_50_rad_s':float(np.mean(angles>50))}
    records=[]
    report=json.loads((research/'diagnostics/bounded_decoder_v1/report.json').read_text())
    for run,experiment in report['experiments'].items():
        for row in experiment['rows']:
            q=np.loadtxt(ROOT/row['bounded_qpos'],delimiter=',')
            records.append({'kind':'generated','run':run,'source':row['source'],
                            'qpos':row['bounded_qpos'],**measure(q)})
    manifest=json.loads((ROOT/'runs/m4_20261102/data/BCD.json').read_text())
    with torch.inference_mode():
        for event in load_data(manifest,'val'):
            raw=model.motion_rep.inverse(event['x0'][None],is_normalized=True)
            motion={k:v[0] for k,v in raw.items()};q,_=decoder.decode(motion)
            records.append({'kind':'reference','group':event['group'],'sample_id':event['sample_id'],**measure(q)})
    summaries={}
    for label in ['reference','R3_D_high_noise','R9_D_high_noise_seed20261104']:
        rows=[r for r in records if r.get('run',r['kind'])==label]
        summaries[label]={'clips':len(rows),'max_foot_penetration_m':max(r['max_foot_penetration_m'] for r in rows),
          'median_clip_max_foot_penetration_m':float(np.median([r['max_foot_penetration_m'] for r in rows])),
          'mean_clip_fraction_below_minus_5cm':float(np.mean([r['frames_with_foot_below_minus_5cm_fraction'] for r in rows])),
          'max_joint_speed_rad_s':max(r['max_joint_speed_rad_s'] for r in rows)}
    output={'records':records,'summary':summaries,'script_sha256':sha(__file__),
      'note':'Kinematic geometry only. Near-ground speed is a slip proxy, not contact-force or tracking success. Reference/generated text and duration differ; summaries are not paired scores. No audit data used.'}
    atomic_json(research/'diagnostics/kinematic_artifacts_v1.json',output)
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':main()
