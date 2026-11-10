#!/usr/bin/env python3
"""Archive retained audit outputs, including worst cases; never regenerates motion."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import argparse,json,sys,shutil,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from kimodo.assets import skeleton_asset_path
from training.m3 import atomic_json,sha
from training.video import render
from training.video_archive import freeze_model
from index_video_exports import build_index


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--selection',choices=['core','penetration'],default='core');args=parser.parse_args()
    root=ROOT/'runs/m4_research_20261103';status=json.loads((root/'audit_execution_status.json').read_text())
    if status['phase']!='audit_complete_requires_final_review':raise ValueError('Wait for all frozen conditions')
    release=json.loads((root/'audit_release.json').read_text());protocol=json.loads((root/'M5_entry_protocol.json').read_text())
    output_report=root/('audit_video_review_penetration.json' if args.selection=='penetration' else 'audit_video_review.json')
    if output_report.exists():raise FileExistsError(output_report)
    records=[];stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S_%f%z')
    for run_name in release['runs']:
        run=root/run_name;checkpoint=run/'checkpoints/best_val_ema.pt';checksum=sha(checkpoint)
        assert checksum in release['allowed_checkpoint_sha256'];snapshot=freeze_model(run,checkpoint,checksum)
        all_rows=[]
        for condition in ('native','heading0'):
            folder=run/'locked_audit_v1'/condition
            rows=json.loads((folder/'per_sample.json').read_text())
            all_rows.extend(dict(row,condition=condition,folder=folder) for row in rows)
        choices=[]
        for group in ('walk','jog','jump','raise_arms'):
            rows=sorted([r for r in all_rows if r['group']==group and r['reference_amplitude']>=protocol['reference_minimum'][group]],key=lambda r:r['amplitude_ratio'])
            choices.extend([('lowest_ratio',rows[0]),('median_ratio',rows[len(rows)//2])])
        worst=max(all_rows,key=lambda r:r['decoder_diagnostics']['bounded']['max_joint_speed_rad_s'])
        choices.append(('largest_joint_jump',worst))
        if args.selection=='penetration':
            contacts=json.loads((root/'audit_contact_geometry.json').read_text())['records']
            failure=max([r for r in contacts if r['run']==run_name],key=lambda r:r['max_foot_penetration_m'])
            row=next(r for r in all_rows if (r['condition'],r['sample_id'],r['seed'])==(failure['condition'],failure['sample_id'],failure['seed']))
            row['contact_geometry']=failure;choices=[('largest_foot_penetration',row)]
        for kind,row in choices:
            stem=row['folder']/row['artifact_prefix'];q=np.loadtxt(stem.with_suffix('.csv'),delimiter=',')
            suffix=f"{stamp}__audit_{kind}_{row['group']}_{row['condition']}__{row['sample_id']}__seed{row['seed']}__bounded_v1"
            out=ROOT/'exports'/run.relative_to(ROOT/'runs')/f'step_020000_ema_{checksum[:8]}'/suffix
            out.mkdir(parents=True,exist_ok=False)
            metadata={'schema':'kimodo-video-export-v1','status':'generating','exported_at':stamp,'timezone':'Asia/Shanghai',
              'training_run_id':str(run.relative_to(ROOT/'runs')),'checkpoint':str(checkpoint),'checkpoint_snapshot':str(snapshot),
              'checkpoint_sha256':checksum,'checkpoint_step':20000,'weights':'ema','prompt':row['text'],
              'prompt_sha256':hashlib.sha256(row['text'].encode()).hexdigest(),'seed':row['seed'],'frames':len(q),'fps':30,'duration_s':len(q)/30,
              'heading_radians':0 if row['condition']=='heading0' else None,'conditioning_heading_source':row['condition'],
              'ddim_steps':100,'guidance':2.0,'postprocess':True,'postprocess_label':'bounded hinge projection',
              'physical_tracking_verified':False,'decoder':'g1-nearest-bounded-hinge-v1','decoder_sha256':release['decoder_sha256'],
              'selection':kind,'group':row['group'],'cohort':'locked_final_audit','sample_id':row['sample_id'],
              'source_audit_csv':str(stem.with_suffix('.csv')),'source_csv_sha256':sha(stem.with_suffix('.csv')),
              'audit_release_sha256':sha(root/'audit_release.json'),'amplitude_ratio_raw':row['amplitude_ratio'],
              'contact_geometry':row.get('contact_geometry'),
              'decoder_diagnostics':row['decoder_diagnostics'],'script_sha256':sha(__file__),'renderer_sha256':sha(ROOT/'training/video.py')}
            atomic_json(out/'manifest.json',metadata)
            for source,target in [(stem.with_suffix('.csv'),'qpos.csv'),(stem.with_suffix('.npz'),'motion.npz'),(stem.with_name(stem.name+'_legacy.csv'),'qpos_legacy.csv')]:
                shutil.copyfile(source,out/target)
            source_dir=out/'source';source_dir.mkdir()
            for source in [Path(__file__),ROOT/'training/video.py',ROOT/'training/g1_qpos.py']:
                shutil.copyfile(source,source_dir/source.name)
            metadata['render']=render(q,out/'replay.mp4',metadata,skeleton_asset_path('g1skel34','xml','g1.xml'))
            metadata['status']='complete';metadata['artifacts_sha256']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f.name!='manifest.json'}
            atomic_json(out/'manifest.json',metadata)
            records.append({'run':run_name,'kind':kind,'group':row['group'],'sample_id':row['sample_id'],
                            'condition':row['condition'],'seed':row['seed'],'raw_amplitude_ratio':row['amplitude_ratio'],'output':str(out)})
            print(run_name,kind,row['group'],round(row['amplitude_ratio'],3),flush=True)
    rule='Per model: largest foot penetration across all audit samples. No resampling.' if args.selection=='penetration' else 'Per model: lowest and median eligible raw ratio in each class across both headings/three seeds, plus largest joint jump. No resampling.'
    atomic_json(output_report,{'audit_release_sha256':sha(root/'audit_release.json'),'selection_rule':rule,
                              'cases':records,'visual_review':'pending','physical_tracking_verified':False})
    build_index(ROOT/'exports')


if __name__=='__main__':main()
