#!/usr/bin/env python3
"""Cached-text physical-v2 generation, MuJoCo replay and content-addressed archives."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,re,sys
from datetime import datetime,timezone,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from kimodo.assets import skeleton_asset_path
from training.m3 import atomic_json,batch_to,configure,make_model,sha,INPUTS
from training.m5_evaluation import sample_keyed
from training.m5_physical_losses import geometry
from training.m5_physical_evaluation import PhysicalGeometryMetrics
from training.g1_physical import PHYSICAL_VERSION
from training.video import render
from training.video_archive import freeze_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--checkpoint',default='latest.pt')
    choice=p.add_mutually_exclusive_group(required=True);choice.add_argument('--prompt');choice.add_argument('--sample-id')
    p.add_argument('--weights',choices=['raw','ema'],default='ema');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--duration',type=float);p.add_argument('--heading',type=float);p.add_argument('--guidance',type=float,choices=[1.,1.5,2.],default=2.)
    p.add_argument('--ddim-steps',type=int,default=100);p.add_argument('--decoder',choices=['bounded','unbounded'],default='bounded')
    p.add_argument('--engineering-fixture',action='store_true');p.add_argument('--blind-id',help='Optional rendered model alias; true identity stays in manifest')
    p.add_argument('--text-cache',type=Path,help='Optional separate verified cache with COMPLETE.json, rows.json and embeddings.npy')
    a=p.parse_args();run=a.run_dir.resolve();relative=run.relative_to(ROOT/'runs')
    if 'verification' in relative.parts and not a.engineering_fixture:raise ValueError('Verification models require --engineering-fixture and are not formal quality evidence')
    source=(run/'checkpoints'/a.checkpoint).resolve();source.relative_to(run/'checkpoints');checksum=sha(source)
    # Freeze before loading: a mutable latest.pt must never silently select new weights.
    snapshot=freeze_model(run,source,checksum);state=torch.load(snapshot,map_location='cpu',weights_only=False)
    if state.get('schema') not in ('kimodo-m5-physical-checkpoint-v2','kimodo-m5-physical-atan2-checkpoint-v1','kimodo-m5-physical-constraints-checkpoint-v1'):raise ValueError('This exporter requires physical-v2 input semantics')
    identity=state['identity'];cfg=state['config'];inputs=Path(cfg['input_dir']);ready=json.loads((inputs/'READY').read_text())
    if ready['fingerprint']!=identity['physical_input_fingerprint']:raise ValueError('Physical cache mismatch')
    stats=run/'stats/motion'
    for name,expected in identity.get('stats_sha256',{}).items():
        if sha(run/name)!=expected:raise ValueError('Run statistics changed: '+name)
    if json.loads((INPUTS/'text_encoder/COMPLETE').read_text())['fingerprint']!=identity['text_config_fingerprint']:raise ValueError('Text encoder mismatch')
    # Fixtures have a minimal identity; still bind their saved statistics by hash.
    stats_hashes={str(f.relative_to(run)):sha(f) for f in stats.rglob('*.npy')}
    rows=pd.read_parquet(inputs/'samples.parquet');reference=None;vector_cache=INPUTS/'text_encoder';vector_fingerprint=identity['text_config_fingerprint']
    if a.text_cache:
        if a.sample_id:raise ValueError('A separate text cache uses --prompt, not a motion sample ID')
        vector_cache=a.text_cache.resolve();proof=json.loads((vector_cache/'COMPLETE.json').read_text())
        if not proof['passed'] or proof['parent_encoder_fingerprint']!=identity['text_config_fingerprint']:
            raise ValueError('Separate text cache has a different encoder identity')
        if sha(vector_cache/'embeddings.npy')!=proof['embeddings_sha256'] or sha(vector_cache/'rows.json')!=proof['rows_sha256']:
            raise ValueError('Separate text cache bytes changed')
        matched=[r for r in json.loads((vector_cache/'rows.json').read_text()) if r['text']==a.prompt]
        if len(matched)!=1:raise ValueError('Prompt is not unique in the selected separate text cache')
        prompt=a.prompt;text_row=int(matched[0]['text_row']);text_id=matched[0]['text_sha256'];vector_fingerprint=proof['fingerprint']
    elif a.sample_id:
        matched=rows[rows.sample_id==a.sample_id]
        if len(matched)!=1:raise ValueError('Sample ID is not uniquely present in the audited cache')
        reference=matched.iloc[0];prompt=str(reference.text);text_row=int(reference.text_row);text_id=str(reference.text_id)
    else:
        prompt=a.prompt;texts=pd.read_parquet(INPUTS/'texts.parquet');matched=texts[texts.text==prompt]
        if len(matched)!=1:raise ValueError('Prompt is absent from the frozen text cache; encode it with the verified text pipeline separately')
        text=matched.iloc[0];text_row=int(text.text_row);text_id=str(text.text_id)
    frames=round(a.duration*30) if a.duration is not None else int(reference.num_frames) if reference is not None else 75
    heading=a.heading if a.heading is not None else float(reference.first_heading_angle) if reference is not None else 0.
    if not 30<=frames<=180 or not np.isfinite(heading) or not 2<=a.ddim_steps<=1000:raise ValueError('Use 1..6 seconds, finite heading and 2..1000 DDIM steps')
    embeddings=np.load(vector_cache/'embeddings.npy',mmap_mode='r');text=torch.from_numpy(np.array(embeddings[text_row],copy=True))
    configure(cfg['seed']);model=make_model(cfg['model'],stats).cuda().eval();model.load_state_dict(state['model' if a.weights=='raw' else 'ema'])
    sample={'x0':torch.zeros(frames,417),'text_feat':text,'length':frames,'first_heading_angle':torch.tensor(heading,dtype=torch.float32),
      'origin_xz':torch.zeros(2),'sample_id':a.sample_id or 'physical_video_shared_noise','text':prompt,'split':'inference'}
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%dT%H%M%S_%f%z');prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()
    slug=re.sub(r'[^a-z0-9]+','_',prompt.lower()).strip('_')[:50].rstrip('_')
    namespace=ROOT/'exports'
    if a.engineering_fixture:namespace=namespace/'verification'
    output=namespace/relative/f'step_{state["step"]:06d}_{a.weights}_{checksum[:8]}'/f'{stamp}__{slug}__{prompt_hash[:8]}__seed{a.seed}__physical_v2_{a.decoder}'
    output.mkdir(parents=True,exist_ok=False);run_config=json.loads((run/'run_config.json').read_text());subset=json.loads((run/'subset.json').read_text())
    completion=json.loads((run/'COMPLETED.json').read_text()) if (run/'COMPLETED.json').exists() else {}
    lineage=json.loads((run/'lineage.json').read_text()) if (run/'lineage.json').exists() else {}
    xml=skeleton_asset_path('g1skel34','xml','g1.xml');group=next((r['group'] for r in subset['val']+subset['train'] if r['sample_id']==a.sample_id),None)
    code=[Path(__file__),ROOT/'training/video.py',ROOT/'training/video_archive.py',ROOT/'training/g1_physical.py',ROOT/'training/m5_physical_losses.py',ROOT/'training/m5_physical_evaluation.py',ROOT/'training/m5_evaluation.py']
    metadata={'schema':'kimodo-video-export-fixture-v1' if a.engineering_fixture else 'kimodo-video-export-v1','status':'generating','exported_at':stamp,'timezone':'Asia/Shanghai',
      'artifact_role':'engineering_fixture' if a.engineering_fixture else 'M5_development_sample','quality_accepted':False,
      'training_run_id':str(relative),'training_run_path':str(run),'training_started_at_utc':run_config.get('started_at_utc'),
      'training_finished_at_utc':completion.get('completed_at_utc'),'training_budget_completed':bool(completion),'training_configuration':cfg,
      'training_stage_start_step':cfg.get('stage_start_step',0),'training_lineage':lineage,
      'checkpoint':str(source),'checkpoint_snapshot':str(snapshot),'checkpoint_sha256':checksum,'checkpoint_step':state['step'],'weights':a.weights,
      'checkpoint_schema':state['schema'],
      'prompt':prompt,'prompt_sha256':prompt_hash,'text_id':text_id,'text_row':text_row,'prompt_seen_in_training':any(r['text_id']==text_id for r in subset['train']),
      'reference_sample_id':a.sample_id,'sampling_key':sample['sample_id'],'seed':a.seed,'fps':30,'frames':frames,'duration_s':frames/30,'heading_radians':heading,
      'guidance':a.guidance,'ddim_steps':a.ddim_steps,'physical_version':PHYSICAL_VERSION,'physical_input_fingerprint':ready['fingerprint'],
      'training_view_fingerprint':identity['training_view_fingerprint'],'text_encoder_fingerprint':identity['text_config_fingerprint'],'normalization_files_sha256':stats_hashes,
      'text_vector_cache':str(vector_cache),'text_vector_cache_fingerprint':vector_fingerprint,
      'postprocess':a.decoder=='bounded','postprocess_label':'physical-v2 bounded hinges' if a.decoder=='bounded' else 'physical-v2 unbounded hinges',
      'robot_xml':str(xml),'robot_xml_sha256':sha(xml),'qpos_schema':'MuJoCo Z-up: xyz metres, quaternion wxyz, 29 native hinge angles rad',
      'rendered_model_label':a.blind_id or str(relative),'source_code_sha256':{str(f.relative_to(ROOT)):sha(f) for f in code},'physical_tracking_verified':False}
    atomic_json(output/'manifest.json',metadata)
    try:
        for f in code:
            dest=output/'source'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes())
        with torch.inference_mode():
            features=sample_keyed(model,batch_to([sample],'cuda:0'),[sample],a.seed,a.ddim_steps,a.guidance)
            _,_,decoded,fallback=geometry(model.motion_rep,features);motion={k:v[0].cpu() for k,v in decoded.items()}
            metrics,bounded,unbounded=PhysicalGeometryMetrics().measure(motion,group)
            qpos=bounded if a.decoder=='bounded' else unbounded
            np.savez_compressed(output/'motion.npz',**{k:v.numpy() for k,v in motion.items()});np.save(output/'normalized_features.npy',features[0].cpu().numpy())
            np.save(output/'text_embedding.npy',text.numpy());np.savetxt(output/'qpos.csv',qpos,delimiter=',');np.savetxt(output/'qpos_unbounded.csv',unbounded,delimiter=',')
        np.savetxt(output/'root_trajectory.csv',np.column_stack([np.arange(frames)/30,qpos[:,:3]]),delimiter=',',header='time_s,x_m,y_m,z_m',comments='')
        metadata.update(geometry_diagnostics=metrics,rotation_fallback_fraction=float(fallback.float().mean()))
        render_meta=dict(metadata,training_run_id=a.blind_id or metadata['training_run_id'])
        metadata['render']=render(qpos,output/'replay.mp4',render_meta,xml)
        metadata['artifacts_sha256']={str(f.relative_to(output)):sha(f) for f in output.rglob('*') if f.is_file() and f.name!='manifest.json'}
        metadata['status']='complete';atomic_json(output/'manifest.json',metadata)
        if not a.engineering_fixture:
            from scripts.index_video_exports import build_index
            build_index(ROOT/'exports')
        print(json.dumps({'output':str(output),'video':str(output/'replay.mp4'),'checkpoint_step':state['step'],'frames':frames,'engineering_fixture':a.engineering_fixture},indent=2),flush=True)
    except Exception as error:
        atomic_json(output/'manifest.json',dict(metadata,status='failed',error=repr(error)));raise


if __name__=='__main__':main()
