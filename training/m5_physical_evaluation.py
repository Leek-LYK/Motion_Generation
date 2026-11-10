"""Independent physical-v2 geometry and complete, sample-keyed M5 development reports."""
import hashlib
from pathlib import Path
import numpy as np,torch,mujoco
from kimodo.model.diffusion import Diffusion
from training.m3 import atomic_json,batch_to,forward
from training.m4 import rotate_batch,per_sample_losses,TIMESTEPS
from training.m5_evaluation import GeometryMetrics,amplitude,sample_keyed,summarize
from training.g1_physical import PhysicalG1BoundedDecoder,PHYSICAL_VERSION
from training.m5_physical_losses import geometry,NativeHingeGeometry,objective


class PhysicalGeometryMetrics(GeometryMetrics):
    def __init__(self,rotation_thresholds=None):
        super().__init__(rotation_thresholds)
        self.decoder=PhysicalG1BoundedDecoder(self.skeleton);self.model=self.decoder.model;self.data=mujoco.MjData(self.model)

    def measure(self,motion,group=None):
        result,qpos,raw=super().measure(motion,group)
        rebuilt=self.decoder.converter.qpos_to_motion_dict(qpos.astype(np.float32))
        self.last_projection_errors=torch.linalg.vector_norm(rebuilt['posed_joints']-motion['posed_joints'].detach().cpu().float(),dim=-1).numpy()
        foot=self.decoder.converter.foot_bottom({k:v.detach().cpu().float() for k,v in motion.items()}).numpy()
        result.update(raw_fk_max_penetration_m=float(max(0,-foot.min())),raw_fk_penetration_over_5cm_fraction=float(np.mean(foot.min(-1)<-.05)))
        return result,qpos,raw


@torch.inference_mode()
def evaluate_denoising(model,samples,device,batch_size=16):
    training=model.training;model.eval();rep=model.motion_rep;hinge=NativeHingeGeometry(rep.skeleton,device);rows=[];diffusion=Diffusion(1000).to(device)
    for start in range(0,len(samples),batch_size):
        events=[samples[i] for i in range(start,min(len(samples),start+batch_size))];batch=batch_to(events,device);noise=torch.zeros_like(batch['x0'])
        for i,event in enumerate(events):
            seed=int(hashlib.sha256((event['sample_id']+':M5physicalval').encode()).hexdigest()[:12],16)
            rng=torch.Generator(device=device).manual_seed(seed)
            noise[i,:event['length']]=torch.randn((event['length'],417),device=device,generator=rng)
        for timestep in TIMESTEPS:
            t=torch.full((len(events),),timestep,device=device,dtype=torch.long)
            prediction=forward(model,diffusion.q_sample(batch['x0'],t,noise),batch,t)
            for i,event in enumerate(events):
                mask=batch['x_pad_mask'][i:i+1];loss,parts=objective(rep,prediction[i:i+1],batch['x0'][i:i+1],mask,{'loss_objective':'norm7'},hinge)
                values={k:float(v) for k,v in parts.items()}
                if not all(np.isfinite(v) for v in values.values()):raise FloatingPointError('Nonfinite physical validation')
                rows.append({'sample_id':event['sample_id'],'group':event['group'],'cohort':event.get('cohort','unknown'),'t':timestep,
                             'loss':values['norm6_comparison'],**values})
    groups={g:float(np.mean([r['loss'] for r in rows if r['group']==g])) for g in sorted({r['group'] for r in rows})}
    keys=[k for k in rows[0] if isinstance(rows[0][k],float)]
    strata={g+'/'+c:float(np.mean([r['loss'] for r in rows if r['group']==g and r['cohort']==c]))
            for g,c in sorted({(r['group'],r['cohort']) for r in rows})}
    result={'macro_loss':float(np.mean(list(strata.values()))),'groups':groups,'strata':strata,
      'parts':{k:float(np.mean([r[k] for r in rows])) for k in keys},
      'timesteps':{str(t):float(np.mean([r['loss'] for r in rows if r['t']==t])) for t in TIMESTEPS},
      'geometry_by_timestep':{str(t):{k:float(np.mean([r[k] for r in rows if r['t']==t])) for k in keys} for t in TIMESTEPS},
      'per_sample':rows,'physical_version':PHYSICAL_VERSION}
    model.train(training);return result


@torch.inference_mode()
def evaluate_generation(model,samples,output,step,seeds=(42,),headings=(None,0.),batch_size=16,steps=100,guidance=2.,retain=False,metric_thresholds=None,flags=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);rows=[];projection_errors=[];metrics=PhysicalGeometryMetrics(metric_thresholds)
    device=next(model.parameters()).device;training=model.training;model.eval();flags=flags or {}
    for heading in headings:
        condition='native' if heading is None else f'heading{heading:g}'
        for seed in seeds:
            for start in range(0,len(samples),batch_size):
                events=[samples[i] for i in range(start,min(len(samples),start+batch_size))];batch=batch_to(events,device)
                if heading is not None:batch=rotate_batch(model,batch,heading-batch['first_heading_angle'])
                generated=sample_keyed(model,batch,events,seed,steps,guidance)
                for i,event in enumerate(events):
                    n=event['length'];features=generated[i:i+1,:n];raw,fk,prediction,fallback=geometry(model.motion_rep,features)
                    _,_,reference,_=geometry(model.motion_rep,batch['x0'][i:i+1,:n])
                    motion={k:v[0] for k,v in prediction.items()};ref={k:v[0] for k,v in reference.items()}
                    actual=amplitude(motion,model.motion_rep.skeleton,event['group']);expected=amplitude(ref,model.motion_rep.skeleton,event['group'])
                    extra,qpos,unbounded=metrics.measure(motion,event['group'])
                    projection_errors.append(metrics.last_projection_errors.reshape(-1))
                    row={'sample_id':event['sample_id'],'group':event['group'],'cohort':event.get('cohort','unknown'),'text':event['text'],
                      'seed':seed,'heading':heading,'condition':condition,'frames':n,'generated_amplitude':actual,'reference_amplitude':expected,
                      'amplitude_ratio':actual/max(expected,1e-8),'fk_inconsistency_m':float(torch.linalg.vector_norm(fk-raw[...,5:107].reshape(1,n,34,3),dim=-1).mean()),
                      'finite':all(bool(torch.isfinite(v).all()) for v in motion.values()),'rotation_fallback_fraction':float(fallback.float().mean()),
                      **flags.get(event['sample_id'],{}),**extra}
                    if retain:
                        folder=output/condition/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True);stem=folder/event['sample_id']
                        np.savez_compressed(stem.with_suffix('.npz'),**{k:v.cpu().numpy() for k,v in motion.items()})
                        np.save(stem.with_name(stem.name+'_features.npy'),features[0].cpu().numpy())
                        np.savetxt(stem.with_suffix('.csv'),qpos,delimiter=',');np.savetxt(stem.with_name(stem.name+'_unbounded.csv'),unbounded,delimiter=',')
                        row['artifact_prefix']=str(stem.relative_to(output))
                    rows.append(row)
                atomic_json(output/'progress.json',{'step':step,'condition':condition,'seed':seed,'completed_clips':len(rows),'target_clips':len(samples)*len(seeds)*len(headings)})
            print(f'physical generation step={step} {condition} seed={seed} events={len(samples)}',flush=True)
    result=summarize(rows);conditions={}
    for condition in sorted({r['condition'] for r in rows}):
        selected=[r for r in rows if r['condition']==condition];conditions[condition]=summarize(selected)
        conditions[condition]['length_strata']={label:summarize(part) for label in ('short','extended') for part in [[r for r in selected if (r['frames']<=90)==(label=='short')]] if part}
    result.update(step=step,events=len(samples),generated_clips=len(rows),seeds=list(seeds),headings=list(headings),guidance=guidance,ddim_steps=steps,
      pooled_projection_node_p95_m=float(np.quantile(np.concatenate(projection_errors),.95)),
      conditions=conditions,physical_version=PHYSICAL_VERSION,raw_output_note='*_unbounded.csv uses physical native-hinge extraction; legacy quality keys mean unbounded output.',
      all_six_class_amplitude_passed=len(result['groups'])==6 and all(g['amplitude_passed'] for g in result['groups'].values()))
    flagged=[r for r in rows if r.get('unsupported_context') or r.get('reference_quality_flag')]
    if flagged:result['flagged_reference_stratum']=summarize(flagged)
    atomic_json(output/'per_sample.json',rows);atomic_json(output/'report.json',result);model.train(training);return result
