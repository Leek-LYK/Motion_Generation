#!/usr/bin/env python3
"""Frozen local official/R3 references with own statistics and explicit export contracts."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from omegaconf import OmegaConf
from kimodo.model.cfg import ClassifierFreeGuidedModel
from kimodo.motion_rep.feature_utils import compute_heading_angle
from training.m3 import atomic_json,configure,make_model,sha,validate_inputs,INPUTS,batch_to,forward
from training.m5_physical_data import PhysicalSubsetDataset
from training.m5_physical_losses import geometry,NativeHingeGeometry
from training.m5_physical_evaluation import evaluate_generation,PhysicalGeometryMetrics
from training.m5_research_losses import HingeGeometry
from training.g1_physical import PhysicalG1Converter
from training.g1_qpos import G1BoundedDecoder
from training.m5_evaluation import amplitude


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--kind',choices=['official','legacy-r3'],required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['val','monitor'],default='val');p.add_argument('--seeds',type=int,nargs='+',default=[0,42,1234])
    p.add_argument('--limit-per-class',type=int);p.add_argument('--steps',type=int,default=100);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--heading0-only',action='store_true');a=p.parse_args();root=a.experiment.resolve();out=a.output.resolve()
    if (out/'report.json').exists():raise FileExistsError('Reference already evaluated; preserve prior outputs')
    ready=validate_inputs();view=json.loads((root/'data/READY.json').read_text());manifest_path=root/'data/D_short_3500.json';manifest=json.loads(manifest_path.read_text())
    events=manifest[a.split]
    if a.limit_per_class is not None:
        groups=sorted({r['group'] for r in events});events=[r for g in groups for r in [v for v in events if v['group']==g][:a.limit_per_class]]
    checksum=sha(a.checkpoint);configure(3101);config_path=None
    if a.kind=='official':
        config_path=a.checkpoint.parent/'config.yaml';config=OmegaConf.load(config_path);d=config.denoiser
        architecture={k:int(d[k]) for k in ['latent_dim','ff_size','num_layers','num_heads']};stats=a.checkpoint.parent/'stats/motion'
        model=make_model(architecture,stats).cuda().eval();model.load_ckpt(str(a.checkpoint));step=0
    else:
        state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        if state['identity']['m2_fingerprint']!=ready['fingerprint']:raise ValueError('R3 input fingerprint mismatch')
        stats=INPUTS/'stats/motion';architecture=state['config']['model'];model=make_model(architecture,stats).cuda().eval();model.load_state_dict(state['ema']);step=state['step']
        del state
    if sha(a.checkpoint)!=checksum:raise ValueError('Reference checkpoint changed during load')
    # Dataset normalizes audited physical raw targets using EACH reference's own
    # stats. Model noise remains standard normal; reference GT is never passed as a condition.
    samples=PhysicalSubsetDataset(events,view['input_dir'],stats,cache_size=len(events))
    entries=samples.batch(range(min(3,len(samples))));batch=batch_to(entries,'cuda:0');rng=torch.Generator(device='cuda:0').manual_seed(7311)
    noisy=torch.randn(batch['x0'].shape,device='cuda:0',generator=rng);t=torch.full((len(entries),),499,device='cuda:0',dtype=torch.long)
    with torch.inference_mode():
        cond=forward(model,noisy,batch,t);null=forward(model,noisy,dict(batch,text_feat=torch.zeros_like(batch['text_feat'])),t);regular=null+2*(cond-null)
        wrapped=ClassifierFreeGuidedModel(model,cfg_type='separated')
        separated=wrapped((2.,2.),noisy,batch['x_pad_mask'],batch['text_feat'],batch['text_feat_pad_mask'],t,batch['first_heading_angle'])
        cfg_error=float(abs(regular-separated).max())
        relative_cfg_rms=float((regular-separated).square().mean().sqrt()/separated.square().mean().sqrt().clamp_min(1e-12))
        # Prove text-only CFG algebra with one shared batch/kernel. The official
        # 3B call and the matched protocol's separate B calls can round differently
        # in the 283M model; record that difference instead of claiming bitwise identity.
        zero=torch.zeros_like(batch['text_feat']);pad=batch['text_feat_pad_mask']
        joined=model(torch.cat([noisy]*3),torch.cat([batch['x_pad_mask']]*3),torch.cat([batch['text_feat'],zero,zero]),
          torch.cat([pad,False*pad,False*pad]),torch.cat([t]*3),first_heading_angle=torch.cat([batch['first_heading_angle']]*3))
        c3,n3,n31=joined.chunk(3);equivalent=n3+2*(c3-n3)
        algebra_error=float(abs(equivalent-separated).max());null_error=float(abs(n3-n31).max())
        assert torch.allclose(equivalent,separated,atol=1e-6,rtol=1e-6) and null_error<1e-6,(algebra_error,null_error)
        assert relative_cfg_rms<1e-4,relative_cfg_rms
        raw,fk,truth,_=geometry(model.motion_rep,batch['x0']);oracle_error=float(torch.linalg.vector_norm(fk-raw[...,5:107].reshape(*raw.shape[:2],34,3),dim=-1).masked_select(batch['x_pad_mask'][...,None]).max())
        assert oracle_error<1e-5,oracle_error
    out.mkdir(parents=True,exist_ok=True);metric_protocol=json.loads((root/'reference_review.json').read_text())
    result=evaluate_generation(model,samples,out,step,seeds=a.seeds,headings=(0.,) if a.heading0_only else (None,0.),batch_size=a.batch_size,steps=a.steps,
      retain=True,metric_thresholds=metric_protocol['metric_thresholds'],flags=json.loads((root/'data/development_flags.json').read_text()))
    native=NativeHingeGeometry(model.motion_rep.skeleton,'cuda:0');legacy=HingeGeometry(model.motion_rep.skeleton,'cuda:0');rows=json.loads((out/'per_sample.json').read_text())
    physical=PhysicalGeometryMetrics(metric_protocol['metric_thresholds']);old_decoder=G1BoundedDecoder(physical.skeleton);converter=PhysicalG1Converter(physical.skeleton)
    contract=[];deployment=[]
    with torch.inference_mode():
        for row in rows:
            stem=out/row['artifact_prefix']
            with np.load(stem.with_suffix('.npz'),allow_pickle=False) as z:motion={k:torch.from_numpy(z[k]) for k in z.files}
            local=motion['local_rot_mats'].cuda();native_energy=float(native.defects(local).mean());legacy_energy=float(legacy.defects(local)[0].mean())
            contract.append({'sample_id':row['sample_id'],'seed':row['seed'],'condition':row['condition'],'native_hinge_energy':native_energy,'legacy_frame_hinge_energy':legacy_energy})
            if a.kind=='legacy-r3':
                q,quality=old_decoder.decode(motion);np.savetxt(stem.with_name(stem.name+'_legacy_export.csv'),q,delimiter=',')
                physical_motion=converter.qpos_to_motion_dict(q.astype(np.float32))
                theta=compute_heading_angle(physical_motion['posed_joints'][None],physical.skeleton)[0]
                physical_motion['global_root_heading']=torch.stack((theta.cos(),theta.sin()),-1)
                deployment.append({'sample_id':row['sample_id'],'seed':row['seed'],'condition':row['condition'],'group':row['group'],
                  'compiled_legacy_export_amplitude':amplitude(physical_motion,physical.skeleton,row['group']),**physical.foot_metrics(q),
                  'root_translation_delta_vs_native_export_m':float(abs(q[:,:3]-np.loadtxt(stem.with_suffix('.csv'),delimiter=',')[:,:3]).max()),
                  'bounded_quality':quality['bounded']})
    identity={'kind':a.kind,'checkpoint':str(a.checkpoint.resolve()),'checkpoint_sha256':checksum,'parameters':sum(p.numel() for p in model.parameters()),
      'architecture':architecture,'statistics':str(stats.resolve()),'statistics_sha256':{str(f.relative_to(stats)):sha(f) for f in stats.rglob('*.npy')},
      'official_config_sha256':sha(config_path) if config_path else None,'physical_input_fingerprint':view['input_fingerprint'],
      'manifest_sha256':sha(manifest_path),'event_ids':[r['sample_id'] for r in events],'frozen_text_encoder_fingerprint':ready['text_config_fingerprint'],
      'regular_vs_separated_cfg_max_error':cfg_error,'physical_gt_inverse_max_error_m_with_reference_stats':oracle_error,
      'regular_vs_separated_relative_rms_error':relative_cfg_rms,'shared_batch_cfg_algebra_max_error':algebra_error,'shared_batch_null_branches_max_error':null_error,
      'source_sha256':{str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),ROOT/'training/m5_physical_evaluation.py',ROOT/'training/g1_physical.py',ROOT/'training/g1_qpos.py']},
      'scope':'Development capability reference, not a fair capacity/data matched baseline or independent final audit.',
      'pretraining_overlap':'Official pretraining overlap with benchmark/test groups has not been established.'}
    result.update(reference_identity=identity,checkpoint_sha256=checksum,official_reference=a.kind=='official',
      training_step_note='Not exposed by official checkpoint; report step=0 is an evaluation placeholder' if a.kind=='official' else 'Frozen R3 EMA checkpoint',
      coordinate_contract_summary={'mean_native_hinge_energy':float(np.mean([r['native_hinge_energy'] for r in contract])),
        'mean_legacy_frame_hinge_energy':float(np.mean([r['legacy_frame_hinge_energy'] for r in contract])),
        'interpretation':'Native and legacy-frame energies are diagnostics; lower energy alone does not prove a training representation contract.'})
    atomic_json(out/'coordinate_contract.json',{'records':contract,'reference_identity':identity})
    if deployment:
        atomic_json(out/'legacy_deployment.json',{'records':deployment,'scope':'Exact frozen R3 legacy bounded exporter, measured in compiled MuJoCo. Kept separate from the common native interpretation.',
          'historical_coordinate_targets_physically_incompatible':True,'mean_deep_frame_fraction':float(np.mean([r['penetration_over_5cm_fraction'] for r in deployment])),
          'max_penetration_m':max(r['max_penetration_m'] for r in deployment)})
    atomic_json(out/'report.json',result)
    atomic_json(out/'REFERENCE_COMPLETE.json',{'checkpoint_sha256':checksum,'report_sha256':sha(out/'report.json'),'events':len(events),'steps':a.steps,
      'seeds':a.seeds,'scope':identity['scope'],'M5_complete':False})
    print(json.dumps({'reference':identity,'coordinate_contract_summary':result['coordinate_contract_summary'],
      'all_finite':result['all_finite'],'all_valid_qpos':result['all_valid_qpos'],'core_amplitude_passed':result['core_amplitude_passed'],'M5_complete':False},indent=2),flush=True)


if __name__=='__main__':main()
