#!/usr/bin/env python3
"""Render every fixed prompt-swap case and measure projection to G1's actual joint axes."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import argparse,json,hashlib,shutil,sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from kimodo.assets import skeleton_asset_path
from kimodo.skeleton.registry import build_skeleton
from kimodo.exports.mujoco import MujocoQposConverter
from training.m3 import atomic_json,sha
from training.m4_readiness import movement
from training.video import render
from training.video_archive import freeze_model
from training.g1_qpos import G1BoundedDecoder, DECODER_VERSION
from index_video_exports import build_index


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--probe-dir',type=Path);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--decoder',choices=['legacy','bounded-v1'],default='legacy')
    p.add_argument('--case',help='Optional exact group_cohort (or prefixed case) to render')
    p.add_argument('--report-name',default='candidate_video_review.json');args=p.parse_args()
    run=args.run_dir.resolve();config=json.loads((run/'run_config.json').read_text())
    evaluation=json.loads((run/'final_evaluation/report.json').read_text())
    checkpoint=run/'checkpoints/best_val_ema.pt';checksum=sha(checkpoint)
    if checksum!=evaluation['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    snapshot=freeze_model(run,checkpoint,checksum)
    guidance=1.0
    if args.probe_dir:
        probe=json.loads((args.probe_dir/'report.json').read_text())
        if probe['checkpoint_sha256']!=checksum:raise ValueError('Probe used a different checkpoint')
        cases=[x for x in probe['cases'] if x['seed']==args.seed]
        guidance=probe.get('guidance',1.0);source_dir=args.probe_dir/f'seed_{args.seed}'
    else:
        cases=json.loads((run/'final_evaluation/prompt_swap/manifest.json').read_text());source_dir=run/'final_evaluation/prompt_swap'
    skeleton=build_skeleton(34);converter=MujocoQposConverter(skeleton);records=[]
    bounded_decoder=G1BoundedDecoder(skeleton) if args.decoder=='bounded-v1' else None
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S_%f%z')
    base=ROOT/'exports'/run.relative_to(ROOT/'runs')/f"step_{evaluation['checkpoint_step']:06d}_ema_{checksum[:8]}"
    for index,case in enumerate(cases):
        name=case['group']+'_'+case['cohort'];source=source_dir/name
        if not source.with_suffix('.csv').exists():name=f'{index:02d}_'+name;source=source_dir/name
        if args.case and name!=args.case:continue
        suffix='__bounded_v1' if bounded_decoder else ''
        output=base/f'{stamp}__controlled_{name}__cfg{guidance:g}__seed{args.seed}{suffix}'
        output.mkdir(parents=True,exist_ok=False)
        qpos=np.loadtxt(source.with_suffix('.csv'),delimiter=',')
        with np.load(source.with_suffix('.npz'),allow_pickle=False) as archive:
            motion={k:torch.from_numpy(archive[k].copy()) for k in archive.files}
        decoder_diagnostics=None
        if bounded_decoder:
            qpos,decoder_diagnostics=bounded_decoder.decode(motion)
        rebuilt=converter.qpos_to_motion_dict(qpos,source_fps=30)
        rebuilt={k:torch.as_tensor(v) for k,v in rebuilt.items()}
        raw_amp=movement(motion,skeleton,case['group']);projected_amp=movement(rebuilt,skeleton,case['group'])
        joint_error=torch.linalg.vector_norm(rebuilt['posed_joints']-motion['posed_joints'],dim=-1)
        metadata={'schema':'kimodo-video-export-v1','status':'generating','exported_at':stamp,'timezone':'Asia/Shanghai',
          'training_run_id':str(run.relative_to(ROOT/'runs')),'checkpoint':str(checkpoint),'checkpoint_snapshot':str(snapshot),
          'checkpoint_sha256':checksum,'checkpoint_step':evaluation['checkpoint_step'],'weights':'ema','prompt':case['text'],
          'prompt_sha256':hashlib.sha256(case['text'].encode()).hexdigest(),'seed':args.seed,'frames':len(qpos),'fps':30,'duration_s':len(qpos)/30,
          'heading_radians':0,'ddim_steps':100,'guidance':guidance,'postprocess':bool(bounded_decoder),'physical_tracking_verified':False,
          'postprocess_label':'bounded hinge projection' if bounded_decoder else 'no postprocess',
          'decoder':DECODER_VERSION if bounded_decoder else 'upstream-euler',
          'decoder_sha256':sha(ROOT/'training/g1_qpos.py') if bounded_decoder else None,
          'decoder_diagnostics':decoder_diagnostics,
          'source_generated_csv':str(source.with_suffix('.csv')),'source_csv_sha256':sha(source.with_suffix('.csv')),
          'cohort':case['cohort'],'group':case['group'],'m2_fingerprint':config['identity']['m2_fingerprint'],
          'projection':{'mean_joint_difference_m':float(joint_error.mean()),'max_joint_difference_m':float(joint_error.max()),
                        'raw_amplitude':raw_amp,'projected_amplitude':projected_amp},
          'script_sha256':sha(__file__),'renderer_sha256':sha(ROOT/'training/video.py')}
        atomic_json(output/'manifest.json',metadata)
        if bounded_decoder:
            shutil.copyfile(source.with_suffix('.csv'),output/'qpos_legacy.csv')
            np.savetxt(output/'qpos.csv',qpos,delimiter=',')
        else:shutil.copyfile(source.with_suffix('.csv'),output/'qpos.csv')
        shutil.copyfile(source.with_suffix('.npz'),output/'motion.npz')
        metadata['render']=render(qpos,output/'replay.mp4',metadata,skeleton_asset_path('g1skel34','xml','g1.xml'))
        metadata['status']='complete';metadata['artifacts_sha256']={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'}
        atomic_json(output/'manifest.json',metadata);records.append({'case':name,'output':str(output),'projection':metadata['projection']})
        print(name,metadata['projection'],flush=True)
    if not records:raise ValueError('No matching cases')
    atomic_json(run/args.report_name,{'checkpoint_sha256':checksum,'guidance':guidance,'cases':records,
                'decoder':DECODER_VERSION if bounded_decoder else 'upstream-euler',
                'visual_assessment':'pending','physical_tracking_verified':False})
    build_index(ROOT/'exports')


if __name__=='__main__':main()
