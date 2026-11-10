#!/usr/bin/env python3
"""Archive exact retained physical-v2 evaluation qpos, with both export variants.

No diffusion sampling or network forward pass. The video and metric lineage stay
bound to the original evaluation, checkpoint, sample, heading and noise seed.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import argparse,hashlib,json,re,shutil,sys
from datetime import datetime,timezone,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from kimodo.assets import skeleton_asset_path
from training.m3 import atomic_json,sha
from training.m5_physical_evaluation import PhysicalGeometryMetrics
from training.video import render
from training.video_archive import freeze_model


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--step',type=int,required=True)
    p.add_argument('--sample-id',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--condition',choices=['native','heading0'],default='heading0')
    p.add_argument('--engineering-fixture',action='store_true');a=p.parse_args();run=a.run_dir.resolve();relative=run.relative_to(ROOT/'runs')
    if 'verification' in relative.parts and not a.engineering_fixture:raise ValueError('Verification paths require the engineering fixture flag')
    folder=run/'evaluations'/f'generation_full_{a.step:06d}';report=read(folder/'report.json');rows=read(folder/'per_sample.json')
    if report['step']!=a.step or report.get('weights')!='ema':raise ValueError('This tool requires an identified retained EMA evaluation')
    found=[r for r in rows if r['sample_id']==a.sample_id and r['seed']==a.seed and r['condition']==a.condition]
    if len(found)!=1:raise ValueError('Evaluation sample key is missing or ambiguous')
    row=found[0];stem=folder/row['artifact_prefix'];stem.resolve().relative_to(folder.resolve())
    checkpoint=run/'checkpoints'/f'step_{a.step:06d}.pt';checksum=sha(checkpoint);snapshot=freeze_model(run,checkpoint,checksum)
    state=torch.load(snapshot,map_location='cpu',weights_only=False)
    if state['step']!=a.step or state['identity']!=report['identity'] or not state['schema'].startswith('kimodo-m5-physical-'):raise ValueError('Evaluation/checkpoint identity differs')
    cfg=state['config'];identity=state['identity'];schema=state['schema'];del state
    for f,h in identity['stats_sha256'].items():
        if sha(run/f)!=h:raise ValueError('Training statistics changed: '+f)
    for f in ['training/m5_evaluation.py','training/m5_physical_evaluation.py']:
        if sha(ROOT/f)!=identity['code_sha256'][f]:raise ValueError('Foot metric implementation differs from this evaluation')
    bounded_path=stem.with_suffix('.csv');unbounded_path=stem.with_name(stem.name+'_unbounded.csv')
    bounded=np.loadtxt(bounded_path,delimiter=',');unbounded=np.loadtxt(unbounded_path,delimiter=',')
    if bounded.shape!=(row['frames'],36) or unbounded.shape!=bounded.shape or not np.isfinite(bounded).all() or not np.isfinite(unbounded).all():raise ValueError('Malformed retained qpos')
    metric=PhysicalGeometryMetrics();bm=metric.foot_metrics(bounded);um=metric.foot_metrics(unbounded)
    if abs(bm['max_penetration_m']-row['max_penetration_m'])>1e-10 or bm['penetration_over_5cm_fraction']!=row['penetration_over_5cm_fraction']:raise ValueError('Retained bounded qpos does not reproduce the reported penetration')
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%dT%H%M%S_%f%z');prompt=row['text'];prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()
    slug=re.sub(r'[^a-z0-9]+','_',prompt.lower()).strip('_')[:50].rstrip('_');namespace=ROOT/'exports'
    if a.engineering_fixture:namespace=namespace/'verification'
    out=namespace/relative/f'step_{a.step:06d}_ema_{checksum[:8]}'/f'{stamp}__{slug}__{prompt_hash[:8]}__seed{a.seed}__retained_{a.condition}'
    out.mkdir(parents=True,exist_ok=False);registered=read(run/'run_config.json');subset=read(run/'subset.json');complete=read(run/'COMPLETED.json') if (run/'COMPLETED.json').exists() else {}
    entry=next(e for e in subset['val'] if e['sample_id']==a.sample_id);xml=skeleton_asset_path('g1skel34','xml','g1.xml')
    inputs=Path(cfg['input_dir']);ready=read(inputs/'READY')
    if ready['fingerprint']!=identity['physical_input_fingerprint']:raise ValueError('Physical input identity differs')
    source_rows=pd.read_parquet(inputs/'samples.parquet',columns=['sample_id','text_id','text_row'])
    source_row=source_rows[source_rows.sample_id==a.sample_id]
    if len(source_row)!=1 or source_row.iloc[0].text_id!=entry['text_id']:raise ValueError('Physical text-row identity differs')
    text_row=int(source_row.iloc[0].text_row)
    code=['scripts/export_m5_evaluation_video.py','training/video.py','training/video_archive.py','training/g1_physical.py','training/m5_physical_evaluation.py','training/m5_evaluation.py']
    metadata={'schema':'kimodo-video-export-fixture-v1' if a.engineering_fixture else 'kimodo-video-export-v1','status':'rendering','exported_at':stamp,'timezone':'Asia/Shanghai',
      'artifact_role':'engineering_fixture' if a.engineering_fixture else 'M5_retained_development_evaluation','quality_accepted':False,'human_semantic_review':'pending',
      'training_run_id':str(relative),'training_run_path':str(run),'training_started_at_utc':registered.get('started_at_utc'),
      'training_finished_at_utc':complete.get('completed_at_utc'),'training_budget_completed':bool(complete),'training_configuration':cfg,
      'training_stage_start_step':cfg.get('stage_start_step',0),'training_lineage':read(run/'lineage.json') if (run/'lineage.json').exists() else {},
      'checkpoint':str(checkpoint),'checkpoint_snapshot':str(snapshot),'checkpoint_sha256':checksum,'checkpoint_step':a.step,'checkpoint_schema':schema,'weights':'ema',
      'prompt':prompt,'prompt_sha256':prompt_hash,'text_id':entry['text_id'],'text_row':text_row,'reference_sample_id':a.sample_id,
      'prompt_seen_in_training':any(e['text_id']==entry['text_id'] for e in subset['train']),'seed':a.seed,'fps':30,'frames':row['frames'],'duration_s':row['frames']/30,
      'heading_radians':row['heading'],'heading_condition':a.condition,'guidance':report['guidance'],'ddim_steps':report['ddim_steps'],
      'physical_version':report['physical_version'],'physical_input_fingerprint':identity['physical_input_fingerprint'],'training_view_fingerprint':identity['training_view_fingerprint'],
      'text_encoder_fingerprint':identity['text_config_fingerprint'],'normalization_files_sha256':identity['stats_sha256'],
      'robot_xml':str(xml),'robot_xml_sha256':sha(xml),'qpos_schema':'MuJoCo Z-up: xyz metres, quaternion wxyz, 29 native hinge angles rad',
      'postprocess':True,'postprocess_label':'physical-v2 bounded hinges; exact retained evaluation qpos',
      'replay_unbounded_label':'physical-v2 native hinge extraction without range limits; exact retained evaluation qpos',
      'new_diffusion_samples':0,'new_optimizer_updates':0,'trajectory_modified':False,'physical_tracking_verified':False,
      'source_evaluation':str(folder),'source_evaluation_sha256':sha(folder/'report.json'),'source_per_sample_sha256':sha(folder/'per_sample.json'),
      'source_artifact_sha256':{'bounded_qpos':sha(bounded_path),'unbounded_qpos':sha(unbounded_path)},
      'geometry_diagnostics':row,'bounded_foot_metrics':bm,'unbounded_foot_metrics':um,
      'metric_semantics':'Report penetration fields measure bounded native qpos; raw_fk_* measures full-rotation FK; legacy_limit_* measures unbounded hinge limit violations.',
      'source_code_sha256':{f:sha(ROOT/f) for f in code}}
    atomic_json(out/'manifest.json',metadata)
    try:
        for f in code:
            dest=out/'source'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/f,dest)
        for source,destination in [(bounded_path,'qpos.csv'),(unbounded_path,'qpos_unbounded.csv'),(stem.with_suffix('.npz'),'motion.npz'),(stem.with_name(stem.name+'_features.npy'),'normalized_features.npy')]:
            shutil.copyfile(source,out/destination)
            if sha(out/destination)!=sha(source):raise ValueError('Retained evidence changed while copying')
        atomic_json(out/'evaluation_row.json',row)
        np.savetxt(out/'root_trajectory.csv',np.column_stack([np.arange(row['frames'])/30,bounded[:,:3]]),delimiter=',',header='time_s,x_m,y_m,z_m',comments='')
        metadata['render']=render(bounded,out/'replay.mp4',metadata,xml)
        for name in ['poster.png','contact_sheet.png']:shutil.copyfile(out/name,out/('bounded_'+name))
        secondary=dict(metadata,postprocess=False,postprocess_label=metadata['replay_unbounded_label'])
        metadata['render_unbounded']=render(unbounded,out/'replay_unbounded.mp4',secondary,xml)
        # The shared renderer writes common poster names for each call.
        for name in ['poster.png','contact_sheet.png']:
            shutil.move(out/name,out/('unbounded_'+name));shutil.copyfile(out/('bounded_'+name),out/name)
        metadata['artifacts_sha256']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f.name!='manifest.json'}
        metadata['status']='complete';atomic_json(out/'manifest.json',metadata)
        if not a.engineering_fixture:
            from scripts.index_video_exports import build_index
            build_index(ROOT/'exports')
        print(json.dumps({'output':str(out),'bounded_video':str(out/'replay.mp4'),'unbounded_video':str(out/'replay_unbounded.mp4'),'new_diffusion_samples':0},indent=2),flush=True)
    except Exception as error:
        atomic_json(out/'manifest.json',dict(metadata,status='failed',error=repr(error)));raise


if __name__=='__main__':main()
