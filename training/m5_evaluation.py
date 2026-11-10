"""Sample-keyed DDIM and transparent six-class geometry evaluation (not semantic accuracy)."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
import mujoco
from kimodo.model.diffusion import Diffusion,DDIMSampler
from kimodo.skeleton.registry import build_skeleton
from training.m3 import atomic_json,batch_to,forward
from training.m4 import rotate_batch,evaluate_full
from training.m4_readiness import movement
from training.m5_losses import geometry
from training.g1_qpos import G1BoundedDecoder

MINIMUM={'walk':.3,'jog':.3,'jump':.05,'raise_arms':.05,'turn':.2,'squat':.05}


def amplitude(motion,skeleton,group):
    if group in ('walk','jog','jump','raise_arms'):return movement(motion,skeleton,group)
    if group=='squat':
        y=motion['root_positions'][:,1].detach().cpu().numpy();return float(np.quantile(y,.9)-np.quantile(y,.1))
    h=motion['global_root_heading'].detach().cpu().numpy()
    theta=np.unwrap(np.arctan2(h[:,1],h[:,0]));return float(np.quantile(theta,.9)-np.quantile(theta,.1))


@torch.inference_mode()
def sample_keyed(model,batch,events,seed,steps=100,guidance=2.,text_mode='real'):
    device=batch['x0'].device;x=torch.zeros_like(batch['x0'])
    for i,event in enumerate(events):
        key=int(hashlib.sha256(f'{event["sample_id"]}:M5:{seed}'.encode()).hexdigest()[:15],16)
        rng=torch.Generator(device=device).manual_seed(key)
        x[i,:event['length']]=torch.randn((event['length'],417),device=device,generator=rng)
    condition=dict(batch)
    if text_mode=='zero':condition['text_feat']=torch.zeros_like(batch['text_feat'])
    elif text_mode=='shuffled':condition['text_feat']=batch['text_feat'].roll(1,0)
    diffusion=Diffusion(1000).to(device);sampler=DDIMSampler(diffusion)
    use,mapping=diffusion.space_timesteps(steps);use,mapping=use[:steps],mapping[:steps]
    for i in range(steps-1,-1,-1):
        t=torch.full((len(x),),i,device=device,dtype=torch.long)
        conditional=forward(model,x,condition,mapping[t])
        if guidance==1:clean=conditional
        else:
            null=forward(model,x,dict(condition,text_feat=torch.zeros_like(batch['text_feat'])),mapping[t])
            clean=null+guidance*(conditional-null)
        x=sampler(use,x,clean,t).masked_fill(~batch['x_pad_mask'][...,None],0)
    if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite generation')
    return x


class GeometryMetrics:
    def __init__(self,rotation_thresholds=None):
        self.rotation_thresholds=rotation_thresholds or {}
        self.skeleton=build_skeleton(34);self.decoder=G1BoundedDecoder(self.skeleton)
        self.model=self.decoder.model;self.data=mujoco.MjData(self.model)
        self.feet={side:[i for i in range(self.model.ngeom) if (mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith(side+'_foot')
          and 'collision' in (mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,i) or '')] for side in ('left','right')}
        assert all(self.feet.values())

    def foot_metrics(self,qpos):
        heights=[];centers=[]
        for q in qpos:
            self.data.qpos[:]=q;mujoco.mj_forward(self.model,self.data);bottom=[];xy=[]
            for ids in self.feet.values():
                values=[]
                for i in ids:
                    if self.model.geom_type[i]!=mujoco.mjtGeom.mjGEOM_CAPSULE:raise ValueError('Unexpected foot collider')
                    extent=self.model.geom_size[i,0]+self.model.geom_size[i,1]*abs(self.data.geom_xmat[i].reshape(3,3)[2,2])
                    values.append(self.data.geom_xpos[i,2]-extent)
                bottom.append(min(values));xy.append(self.data.geom_xpos[ids,:2].mean(0).copy())
            heights.append(bottom);centers.append(xy)
        h=np.asarray(heights);speed=np.linalg.norm(np.diff(centers,axis=0),axis=-1)*30
        near=(abs(h[:-1])<.02)&(abs(h[1:])<.02);slip=speed[near]
        flight=h.min(1)>.015;longest=current=0
        for value in flight:
            current=current+1 if value else 0;longest=max(longest,current)
        return {'max_penetration_m':float(max(0,-h.min())),'penetration_over_5cm_fraction':float(np.mean(h.min(1)<-.05)),
                'dual_foot_flight_frames':int(flight.sum()),'longest_flight_run_frames':longest,
                'near_ground_speed_p95_m_s':float(np.quantile(slip,.95)) if len(slip) else None}

    def measure(self,motion,group=None):
        motion={k:v.detach().cpu().float() for k,v in motion.items()}
        bounded,quality=self.decoder.decode(motion)
        legacy=self.decoder.converter.dict_to_qpos({k:v[None] for k,v in motion.items()})[0]
        rebuilt=self.decoder.converter.qpos_to_motion_dict(bounded.astype(np.float32),source_fps=30)
        errors=torch.linalg.vector_norm(rebuilt['posed_joints']-motion['posed_joints'],dim=-1).numpy()
        r=motion['global_rot_mats'].numpy();rel=np.swapaxes(r[:-1],-1,-2)@r[1:]
        angles=np.arccos(np.clip((np.trace(rel,axis1=-2,axis2=-1)-1)/2,-1,1))
        threshold=self.rotation_thresholds.get(group,math.pi/4)
        valid=quality['bounded']['finite'] and quality['bounded']['quaternion_norm_max_error']<1e-4 and quality['bounded']['max_limit_violation_rad']<=1e-6
        return {'valid_qpos':bool(valid),'legacy_limit_violation_rad':quality['legacy']['max_limit_violation_rad'],
          'legacy_violating_frame_fraction':quality['legacy']['violating_frame_fraction'],
          'max_joint_speed_rad_s':quality['bounded']['max_joint_speed_rad_s'],
          'projection_node_mean_m':float(errors.mean()),'projection_node_p95_m':float(np.quantile(errors,.95)),
          'projection_node_max_m':float(errors.max()),'rotation_jump_max_rad':float(angles.max(initial=0)),
          'rotation_jump_over_45deg_fraction':float(np.mean(angles>math.pi/4)),
          'rotation_threshold_rad':float(threshold),'rotation_threshold_exceed_fraction':float(np.mean(angles>threshold)),
          **self.foot_metrics(bounded)},bounded,legacy


def summarize(rows):
    by_group={}
    for group in sorted({r['group'] for r in rows}):
        selected=[r for r in rows if r['group']==group];eligible=[r for r in selected if r['reference_amplitude']>=MINIMUM[group]]
        ratios=[r['amplitude_ratio'] for r in eligible]
        median=float(np.median(ratios)) if ratios else None;fraction=float(np.mean(np.asarray(ratios)>=.35)) if ratios else None
        by_group[group]={'events_with_seeds':len(selected),'eligible':len(eligible),'median_amplitude_ratio':median,'noncollapsed_fraction':fraction,
          'amplitude_passed':bool(ratios and .5<=median<=2 and fraction>=.75),
          'mean_fk_inconsistency_m':float(np.mean([r['fk_inconsistency_m'] for r in selected])),
          'mean_penetration_over_5cm_fraction':float(np.mean([r['penetration_over_5cm_fraction'] for r in selected])),
          'p95_clip_max_penetration_m':float(np.quantile([r['max_penetration_m'] for r in selected],.95)),
          'max_penetration_m':max(r['max_penetration_m'] for r in selected),
          'mean_projection_node_p95_m':float(np.mean([r['projection_node_p95_m'] for r in selected])),
          'max_projection_node_error_m':max(r['projection_node_max_m'] for r in selected),
          'mean_rotation_jump_over_45deg_fraction':float(np.mean([r['rotation_jump_over_45deg_fraction'] for r in selected])),
          'mean_rotation_threshold_exceed_fraction':float(np.mean([r['rotation_threshold_exceed_fraction'] for r in selected])),
          'flight_diagnostic_fraction':float(np.mean([r['longest_flight_run_frames']>=2 for r in selected])) if group=='jump' else None}
    core=[g for g in ('walk','jog','jump','raise_arms') if g in by_group]
    return {'groups':by_group,'core_amplitude_passed':len(core)==4 and all(by_group[g]['amplitude_passed'] for g in core),
       'all_valid_qpos':all(r['valid_qpos'] for r in rows),'all_finite':all(r['finite'] for r in rows),
       'geometry_score_m':float(np.mean([g['mean_fk_inconsistency_m']+g['p95_clip_max_penetration_m']+g['mean_projection_node_p95_m'] for g in by_group.values()])),
       'M5_complete':False,'semantic_review':'pending; amplitude and flight diagnostics are not semantic correctness'}


@torch.inference_mode()
def evaluate_generation(model,samples,output,step,seeds=(42,),headings=(None,0.),batch_size=16,steps=100,guidance=2.,retain=False,reference_rep=None,metric_thresholds=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);rows=[];metrics=GeometryMetrics(metric_thresholds);device=next(model.parameters()).device
    reference_rep=reference_rep or model.motion_rep;was_training=model.training;model.eval()
    for heading in headings:
        condition='native' if heading is None else f'heading{heading:g}'
        for seed in seeds:
            for start in range(0,len(samples),batch_size):
                events=[samples[i] for i in range(start,min(len(samples),start+batch_size))];batch=batch_to(events,device)
                if reference_rep is not model.motion_rep:
                    batch['x0']=model.motion_rep.normalize(reference_rep.unnormalize(batch['x0']))
                if heading is not None:batch=rotate_batch(model,batch,heading-batch['first_heading_angle'])
                generated=sample_keyed(model,batch,events,seed,steps,guidance)
                for i,event in enumerate(events):
                    n=event['length'];features=generated[i:i+1,:n]
                    raw,fk,prediction=geometry(model.motion_rep,features)
                    reference=model.motion_rep.inverse(batch['x0'][i:i+1,:n],is_normalized=True)
                    motion={k:v[0] for k,v in prediction.items()};ref={k:v[0] for k,v in reference.items()}
                    actual=amplitude(motion,model.motion_rep.skeleton,event['group']);expected=amplitude(ref,reference_rep.skeleton,event['group'])
                    extra,qpos,legacy=metrics.measure(motion,event['group'])
                    inconsistency=float(torch.linalg.vector_norm(fk-raw[...,5:107].reshape(1,n,34,3),dim=-1).mean())
                    row={'sample_id':event['sample_id'],'group':event['group'],'cohort':event.get('cohort','unknown'),
                      'text':event['text'],'seed':seed,'heading':heading,'condition':condition,'frames':n,
                      'generated_amplitude':actual,'reference_amplitude':expected,'amplitude_ratio':actual/max(expected,1e-8),
                      'fk_inconsistency_m':inconsistency,'finite':all(bool(torch.isfinite(v).all()) for v in motion.values()),**extra}
                    if retain:
                        folder=output/condition/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True);stem=folder/event['sample_id']
                        np.savez_compressed(stem.with_suffix('.npz'),**{k:v.cpu().numpy() for k,v in motion.items()})
                        np.save(stem.with_name(stem.name+'_features.npy'),features[0].cpu().numpy())
                        np.savetxt(stem.with_suffix('.csv'),qpos,delimiter=',');np.savetxt(stem.with_name(stem.name+'_legacy.csv'),legacy,delimiter=',')
                        row['artifact_prefix']=str(stem.relative_to(output))
                    rows.append(row)
                atomic_json(output/'progress.json',{'step':step,'condition':condition,'seed':seed,'completed_clips':len(rows),
                                                   'target_clips':len(samples)*len(seeds)*len(headings)})
            print(f'generation step={step} {condition} seed={seed} events={len(samples)}',flush=True)
    conditions={}
    for c in sorted({r['condition'] for r in rows}):
        selected=[r for r in rows if r['condition']==c];conditions[c]=summarize(selected)
        conditions[c]['length_strata']={label:summarize(part) for label in ('short','extended')
          for part in [[r for r in selected if (r['frames']<=90)==(label=='short')]] if part}
    result=summarize(rows);result.update(step=step,events=len(samples),generated_clips=len(rows),seeds=list(seeds),headings=list(headings),guidance=guidance,ddim_steps=steps,conditions=conditions)
    atomic_json(output/'per_sample.json',rows);atomic_json(output/'report.json',result);model.train(was_training);return result


def evaluate_denoising(model,samples,device,batch_size=16):
    result=evaluate_full(model,[samples[i] for i in range(len(samples))],device,batch_size)
    rows=result['per_sample'];result['groups']={g:float(np.mean([r['loss'] for r in rows if r['group']==g])) for g in sorted({r['group'] for r in rows})}
    return result
