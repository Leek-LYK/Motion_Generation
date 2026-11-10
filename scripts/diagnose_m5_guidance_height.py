#!/usr/bin/env python3
"""Same-noise CFG sweep and exact final-step height decomposition on known failures."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from kimodo.model.diffusion import Diffusion,DDIMSampler
from training.m3 import atomic_json,configure,make_model,batch_to,forward,sha
from training.m4 import rotate_batch
from training.m5_physical_data import PhysicalSubsetDataset
from training.m5_physical_losses import geometry
from training.m5_physical_evaluation import PhysicalGeometryMetrics
from training.m5_evaluation import amplitude
from training.video_archive import freeze_model


@torch.inference_mode()
def trace(model,batch,events,seed,guidance,steps=100):
    device=batch['x0'].device;x=torch.zeros_like(batch['x0'])
    for i,event in enumerate(events):
        key=int(hashlib.sha256(f'{event["sample_id"]}:M5:{seed}'.encode()).hexdigest()[:15],16)
        rng=torch.Generator(device=device).manual_seed(key)
        x[i,:event['length']]=torch.randn((event['length'],417),device=device,generator=rng)
    diffusion=Diffusion(1000).to(device);sampler=DDIMSampler(diffusion);use,mapping=diffusion.space_timesteps(steps);use,mapping=use[:steps],mapping[:steps]
    for i in range(steps-1,-1,-1):
        t=torch.full((len(x),),i,device=device,dtype=torch.long)
        conditional=forward(model,x,batch,mapping[t]);null=forward(model,x,dict(batch,text_feat=torch.zeros_like(batch['text_feat'])),mapping[t])
        clean=conditional if guidance==1 else null+guidance*(conditional-null)
        x=sampler(use,x,clean,t).masked_fill(~batch['x_pad_mask'][...,None],0)
    if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite CFG diagnostic')
    raw={k:model.motion_rep.unnormalize(v.float()).cpu().numpy() for k,v in [('conditional',conditional),('null',null),('guided',clean)]}
    return x,raw


def height_summary(c,n,g):
    return {'conditional_range_m':[float(c.min()),float(c.max())],'null_range_m':[float(n.min()),float(n.max())],
      'guided_range_m':[float(g.min()),float(g.max())],'negative_guided_frames':int((g<0).sum()),
      'negative_conditional_frames':int((c<0).sum()),'negative_null_frames':int((n<0).sum()),
      'negative_frames_introduced_by_guidance':int(((g<0)&(c>=0)&(n>=0)).sum()),
      'conditional':c.tolist(),'null':n.tolist(),'guided':g.tolist()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--run',required=True);p.add_argument('--step',type=int,default=10000)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.experiment.resolve();run=root/a.run;out=a.output.resolve()
    if (out/'protocol.json').exists():raise FileExistsError('Diagnostic protocol already exists; inspect its job rather than restart')
    selection=['09ce5f8831c01b41e51ac947','b24cce82ab386569b168aa37'];manifest=json.loads((run/'subset.json').read_text())
    events=[next(r for r in manifest['val'] if r['sample_id']==sid) for sid in selection];parent=run/'checkpoints'/f'step_{a.step:06d}.pt';checksum=sha(parent)
    snapshot=freeze_model(run,parent,checksum);state=torch.load(snapshot,map_location='cpu',weights_only=False)
    if state['schema'] not in ('kimodo-m5-physical-checkpoint-v2','kimodo-m5-physical-atan2-checkpoint-v1','kimodo-m5-physical-constraints-checkpoint-v1') or state['step']!=a.step:raise ValueError('Wrong representation/step')
    for filename,expected in state['identity']['stats_sha256'].items():
        if sha(run/filename)!=expected:raise ValueError('Statistics changed')
    configure(state['config']['seed']);model=make_model(state['config']['model'],run/'stats/motion').cuda().eval();model.load_state_dict(state['ema'])
    view=json.loads((root/'data/READY.json').read_text())
    if view['input_fingerprint']!=state['identity']['physical_input_fingerprint']:raise ValueError('Physical input identity mismatch')
    samples=PhysicalSubsetDataset(events,view['input_dir'],run/'stats/motion',cache_size=2)
    protocol={'schema':'m5-guidance-height-diagnostic-v1','run':a.run,'step':a.step,'checkpoint_sha256':checksum,'checkpoint_snapshot':str(snapshot),
      'events':events,'seeds':[0,42,1234],'headings':[None,0.],'guidance':[1.,1.5,2.],'steps':100,'source_sha256':sha(__file__),
      'scope':'Known development failures, selected for diagnosis. No training changes, no candidate selection, no audit scores.',
      'batch_layout_note':'Two-case replay may differ numerically from the original 16-case evaluation layout; each CFG comparison here uses the same layout.',
      'M5_complete':False}
    atomic_json(out/'protocol.json',protocol);metric=PhysicalGeometryMetrics(state['config'].get('metric_thresholds'));records=[]
    with torch.inference_mode():
        for heading in (None,0.):
            batch=batch_to(samples.batch([0,1]),'cuda:0')
            if heading is not None:batch=rotate_batch(model,batch,heading-batch['first_heading_angle'])
            for seed in (0,42,1234):
                for cfg in (1.,1.5,2.):
                    generated,heights=trace(model,batch,samples.batch([0,1]),seed,cfg)
                    for i,event in enumerate(events):
                        n=event['num_frames'];features=generated[i:i+1,:n];raw,fk,motion,fallback=geometry(model.motion_rep,features);motion={k:v[0] for k,v in motion.items()}
                        info,q,unbounded=metric.measure(motion,event['group']);condition='native' if heading is None else 'heading0'
                        folder=out/condition/f'seed_{seed}'/f'cfg{cfg:g}';folder.mkdir(parents=True,exist_ok=True);stem=folder/event['sample_id']
                        np.save(stem.with_name(stem.name+'_features.npy'),features[0].cpu().numpy());np.savetxt(stem.with_suffix('.csv'),q,delimiter=',')
                        row={'sample_id':event['sample_id'],'text':event['text'],'group':event['group'],'seed':seed,'condition':condition,'guidance':cfg,'frames':n,
                          'artifact_prefix':str(stem.relative_to(out)),'generated_amplitude':amplitude(motion,model.motion_rep.skeleton,event['group']),**info,
                          'root_stage_height':height_summary(*[heights[k][i,:n,1] for k in ['conditional','null','guided']]),
                          'body_root_height':height_summary(*[heights[k][i,:n,6] for k in ['conditional','null','guided']])}
                        original=run/'evaluations'/f'generation_full_{a.step:06d}'/condition/f'seed_{seed}'/(event['sample_id']+'_features.npy')
                        if cfg==2 and original.exists():
                            old=model.motion_rep.unnormalize(torch.from_numpy(np.load(original))[None]).numpy()[0]
                            row['original_evaluation_height_replay_max_difference_m']=float(abs(old[:,[1,6]]-raw[0][:,[1,6]].cpu().numpy()).max())
                        records.append(row)
                    atomic_json(out/'progress.json',{'completed_clips':len(records),'target_clips':36,'run':a.run,'checkpoint_sha256':checksum})
                print('guidance trace',a.run,condition,'seed',seed,'complete',flush=True)
    summary={str(cfg):{'clips':len([r for r in records if r['guidance']==cfg]),
      'max_penetration_m':max(r['max_penetration_m'] for r in records if r['guidance']==cfg),
      'mean_deep_frame_fraction':float(np.mean([r['penetration_over_5cm_fraction'] for r in records if r['guidance']==cfg])),
      'root_guidance_introduced_negative_frames':sum(r['root_stage_height']['negative_frames_introduced_by_guidance'] for r in records if r['guidance']==cfg),
      'body_guidance_introduced_negative_frames':sum(r['body_root_height']['negative_frames_introduced_by_guidance'] for r in records if r['guidance']==cfg)} for cfg in (1.,1.5,2.)}
    atomic_json(out/'per_sample.json',records);atomic_json(out/'report.json',{'protocol_sha256':sha(out/'protocol.json'),'summary':summary,'M5_complete':False,
      'interpretation':'Guidance-introduced negatives count only frames whose conditional and null predictions were both nonnegative at that same final denoising state. Other failures can be inherited from the rollout.'})
    atomic_json(out/'COMPLETE.json',{'checkpoint_sha256':checksum,'report_sha256':sha(out/'report.json'),'generated_clips':36,'scope':protocol['scope']});print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
