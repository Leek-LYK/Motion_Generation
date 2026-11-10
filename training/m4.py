"""Controlled M4 experiments: text dropout, yaw augmentation, full validation and CFG."""
from collections import defaultdict
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import einops
import torch
from kimodo.model.diffusion import DDIMSampler, Diffusion
from kimodo.motion_rep.feature_utils import RotateFeatures
from training.m3 import (COMPONENTS, ROOT, Trainer, batch_to, forward, load_samples, losses, make_model,
                        sha, atomic_json)

GROUPS = ("walk", "jog", "jump", "raise_arms")
TIMESTEPS = (0, 249, 499, 749, 999)


def load_data(manifest, split):
    samples = load_samples(manifest, split)
    for sample, entry in zip(samples, manifest[split]):
        sample["cohort"] = entry.get("cohort", "train")
    return samples


def identity_for(manifest_path, ready):
    return {"m2_fingerprint": ready["fingerprint"], "text_config_fingerprint": ready["text_config_fingerprint"],
            "subset_sha256": sha(manifest_path), "training_code_sha256": sha(ROOT / "training/m3.py"),
            "m4_code_sha256": {name:sha(ROOT / name) for name in
                ("training/m4.py", "scripts/train_m4.py", "scripts/evaluate_m4.py")}}


def rotate_batch(model, batch, angles):
    rep = model.motion_rep
    raw = rep.unnormalize(batch["x0"])
    root,heading,positions,rotations,velocities,contacts=einops.unpack(raw,rep.ps,'batch time *')
    rotation=RotateFeatures(angles)
    def rotate_joint_vectors(v):
        # Upstream rotate_positions has ambiguous B/T broadcasting for [B,T,J,3].
        # Flatten temporal/joint axes so every sequence receives its own matrix.
        return rotation.rotate_positions(v.reshape(len(v),-1,3)).reshape_as(v)
    rotated,_=einops.pack([rotation.rotate_positions(root),rotation.rotate_2d_positions(heading),
                          rotate_joint_vectors(positions),rotation.rotate_6d_rotations(rotations),
                          rotate_joint_vectors(velocities),contacts],'batch time *')
    result = dict(batch)
    result["x0"] = rep.normalize(rotated).masked_fill(~batch["x_pad_mask"][..., None], 0)
    result["first_heading_angle"] = rep.get_root_heading_angle(rotated)[:, 0]
    return result


def per_sample_losses(prediction, target, mask):
    error = torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    values = {name:(error[..., a:b] * mask[..., None]).sum((1,2)) / (mask.sum(1) * (b-a))
              for name,(a,b) in COMPONENTS.items()}
    return torch.stack(list(values.values())).mean(0), values


class M4Trainer(Trainer):
    def __init__(self, config, samples, identity, device="cuda:0"):
        if config["network_dropout"] != 0 or config["precision"] != "float32":
            raise ValueError("This experiment protocol fixes network dropout=0 and precision=float32")
        if not 0 <= config["text_dropout"] <= 1 or not isinstance(config["heading_augmentation"], bool):
            raise ValueError("Invalid conditioning configuration")
        super().__init__(config, samples, identity, device)
        self.text_rng = torch.Generator(device=self.device).manual_seed(config["seed"] + 3)
        self.heading_rng = torch.Generator(device=self.device).manual_seed(config["seed"] + 4)
        self.best_raw = self.best_ema = self.early_best = float("inf")
        self.best_raw_step = self.best_ema_step = 0
        self.bad_evaluations = 0
        self.text_dropped = self.samples_seen = 0
        self.termination = None
        self.last_evaluation_step = -1

    def prepare_training_batch(self, batch):
        result = batch
        angles = torch.zeros(len(batch["x0"]), device=self.device)
        if self.config["heading_augmentation"]:
            angles = (torch.rand(len(batch["x0"]), device=self.device, generator=self.heading_rng)*2-1)*math.pi
            result = rotate_batch(self.model, result, angles)
        dropped = torch.zeros(len(batch["x0"]), device=self.device, dtype=torch.bool)
        if self.config["text_dropout"]:
            dropped = torch.rand(len(batch["x0"]), device=self.device, generator=self.text_rng) < self.config["text_dropout"]
            result = dict(result, text_feat=result["text_feat"].masked_fill(dropped[:, None, None], 0))
        return result, dropped, angles

    def update(self):
        self.model.train()
        indices = torch.randint(len(self.samples),(self.config["batch_size"],),generator=self.batch_rng).tolist()
        batch, dropped, angles = self.prepare_training_batch(batch_to([self.samples[i] for i in indices], self.device))
        t = torch.randint(1000,(len(indices),),device=self.device,generator=self.noise_rng)
        noise = torch.randn(batch["x0"].shape,device=self.device,generator=self.noise_rng)
        prediction = forward(self.model,self.diffusion.q_sample(batch["x0"],t,noise),batch,t)
        total, parts = losses(prediction,batch["x0"],batch["x_pad_mask"])
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite loss")
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        norms = {}
        for name,stage in (("root",self.model.root_model),("body",self.model.body_model)):
            grads = [p.grad for p in stage.parameters() if p.grad is not None]
            norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
            if not torch.isfinite(norm) or norm <= 0:
                raise FloatingPointError(f"Invalid {name} gradients")
            norms[name] = float(norm)
        grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config["grad_clip"],error_if_nonfinite=True)
        self.optimizer.step()
        self.step += 1
        self.samples_seen += len(indices)
        self.text_dropped += int(dropped.sum())
        if self.step % self.config["ema_every"] == 0:
            self.ema_updates += 1
            decay = min(self.config["ema_decay"],(1+self.ema_updates)/(10+self.ema_updates))
            with torch.no_grad():
                for a,p in zip(self.ema.parameters(),self.model.parameters()):
                    a.lerp_(p,1-decay)
        return {"loss":float(total.detach()),**{k:float(v.detach()) for k,v in parts.items()},
                "grad_norm":float(grad),"root_grad_norm":norms['root'],"body_grad_norm":norms['body'],
                "text_drop_fraction":float(dropped.float().mean()),"text_drop_fraction_cumulative":self.text_dropped/self.samples_seen,
                "heading_abs_delta_mean":float(angles.abs().mean()), "sample_exposures":self.samples_seen,
                "equivalent_passes":self.samples_seen/len(self.samples)}

    def record_validation(self, raw, ema):
        self.last_evaluation_step = self.step
        improved_raw, improved_ema = raw < self.best_raw, ema < self.best_ema
        if improved_raw:
            self.best_raw, self.best_raw_step = raw,self.step
        if improved_ema:
            self.best_ema, self.best_ema_step = ema,self.step
        if ema < self.early_best*(1-self.config["early_stop_relative_delta"]):
            self.early_best, self.bad_evaluations = ema,0
        elif self.step >= self.config["minimum_steps"]:
            self.bad_evaluations += 1
        return improved_raw,improved_ema

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        state={"schema":"kimodo-m4-checkpoint-v1","config":self.config,"identity":self.identity,
               "model":self.model.state_dict(),"ema":self.ema.state_dict(),"optimizer":self.optimizer.state_dict(),
               "step":self.step,"ema_updates":self.ema_updates,"best_raw":self.best_raw,"best_ema":self.best_ema,
               "best_raw_step":self.best_raw_step,"best_ema_step":self.best_ema_step,"early_best":self.early_best,
               "bad_evaluations":self.bad_evaluations,"samples_seen":self.samples_seen,"text_dropped":self.text_dropped,
               "last_evaluation_step":self.last_evaluation_step,
               "rngs":{name:getattr(self,name).get_state() for name in ('batch_rng','noise_rng','text_rng','heading_rng')},
               "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
               "python_rng":random.getstate(),"numpy_rng":np.random.get_state()}
        temp=path.with_suffix('.tmp')
        with temp.open('wb') as handle:
            torch.save(state,handle);handle.flush();os.fsync(handle.fileno())
        os.replace(temp,path)

    def restore(self,path):
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['schema']!='kimodo-m4-checkpoint-v1' or state['identity']!=self.identity or state['config']!=self.config:
            raise ValueError('Checkpoint identity/config mismatch')
        self.model.load_state_dict(state['model']);self.ema.load_state_dict(state['ema']);self.optimizer.load_state_dict(state['optimizer'])
        for key in ('step','ema_updates','best_raw','best_ema','best_raw_step','best_ema_step','early_best','bad_evaluations','samples_seen','text_dropped','last_evaluation_step'):
            setattr(self,key,state[key])
        for name,value in state['rngs'].items():getattr(self,name).set_state(value)
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])


@torch.inference_mode()
def evaluate_full(model,samples,device,batch_size=16,heading_offset=None):
    """Every event, every timestep; sample-keyed noise, equal class/cohort weighting."""
    training=model.training;model.eval();records=[]
    for start in range(0,len(samples),batch_size):
        chunk=samples[start:start+batch_size];batch=batch_to(chunk,device)
        if heading_offset is not None:
            batch=rotate_batch(model,batch,torch.full((len(chunk),),heading_offset,device=device))
        noise=torch.zeros_like(batch['x0'])
        for i,s in enumerate(chunk):
            seed=int(hashlib.sha256((s['sample_id']+':M4val').encode()).hexdigest()[:12],16)
            generator=torch.Generator(device=device).manual_seed(seed)
            noise[i,:s['length']]=torch.randn((s['length'],417),device=device,generator=generator)
        sums=torch.zeros(len(chunk),device=device);parts={k:torch.zeros_like(sums) for k in COMPONENTS};time_values={}
        diffusion=Diffusion(1000).to(device)
        for timestamp in TIMESTEPS:
            t=torch.full((len(chunk),),timestamp,device=device,dtype=torch.long)
            pred=forward(model,diffusion.q_sample(batch['x0'],t,noise),batch,t)
            loss,values=per_sample_losses(pred,batch['x0'],batch['x_pad_mask'])
            sums+=loss/len(TIMESTEPS);time_values[timestamp]=loss.cpu().tolist()
            for k,v in values.items():parts[k]+=v/len(TIMESTEPS)
        for i,s in enumerate(chunk):
            record={'sample_id':s['sample_id'],'group':s['group'],'cohort':s.get('cohort','train'),
                    'loss':float(sums[i]),**{k:float(v[i]) for k,v in parts.items()},
                    **{f't{t:03d}':v[i] for t,v in time_values.items()}}
            if not np.isfinite([v for v in record.values() if isinstance(v,float)]).all():
                raise FloatingPointError('Nonfinite validation')
            records.append(record)
    by_stratum=defaultdict(list)
    for r in records:by_stratum[r['group']+'/'+r['cohort']].append(r['loss'])
    strata={k:float(np.mean(v)) for k,v in by_stratum.items()}
    result={'macro_loss':float(np.mean(list(strata.values()))),'events':len(records),
            'strata':strata,'groups':{g:float(np.mean([r['loss'] for r in records if r['group']==g]))
                                    for g in GROUPS if any(r['group']==g for r in records)},
            'cohorts':{c:float(np.mean([r['loss'] for r in records if r['cohort']==c])) for c in {r['cohort'] for r in records}},
            'timesteps':{f't{t:03d}':float(np.mean([r[f't{t:03d}'] for r in records])) for t in TIMESTEPS},
            'parts':{k:float(np.mean([r[k] for r in records])) for k in COMPONENTS},'per_sample':records}
    model.train(training);return result


def cfg_prediction(model,x,batch,t,weight):
    conditional=forward(model,x,batch,t)
    if weight==1:
        return conditional
    unconditional=forward(model,x,dict(batch,text_feat=torch.zeros_like(batch['text_feat'])),t)
    return unconditional+weight*(conditional-unconditional)


@torch.inference_mode()
def sample(model,batch,seed,steps=100,guidance=1.0,text_dropout_trained=False,shared_noise=False):
    if guidance!=1 and not text_dropout_trained:
        raise ValueError('CFG requires a trained null-text branch')
    training=model.training;model.eval();device=batch['x0'].device
    generator=torch.Generator(device=device).manual_seed(seed)
    shape=batch['x0'].shape
    x=torch.randn((1 if shared_noise else shape[0],shape[1],shape[2]),device=device,generator=generator)
    if shared_noise:x=x.expand(shape).clone()
    diffusion=Diffusion(1000).to(device);sampler=DDIMSampler(diffusion)
    use,mapping=diffusion.space_timesteps(steps);use,mapping=use[:steps],mapping[:steps]
    for i in range(steps-1,-1,-1):
        t=torch.full((len(x),),i,device=device,dtype=torch.long)
        clean=cfg_prediction(model,x,batch,mapping[t],guidance)
        x=sampler(use,x,clean,t).masked_fill(~batch['x_pad_mask'][...,None],0)
    if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite generation')
    model.train(training);return x


@torch.inference_mode()
def export_preview(model,samples,output,seed=42,steps=100,guidance=1.0,text_dropout_trained=False,writer=None,step=0,tag='ema'):
    from kimodo.exports.mujoco import MujocoQposConverter
    from training.m3 import motion_image
    output=Path(output);output.mkdir(parents=True,exist_ok=True);device=next(model.parameters()).device
    batch=batch_to(samples,device);predicted=sample(model,batch,seed,steps,guidance,text_dropout_trained)
    converter=MujocoQposConverter(model.motion_rep.skeleton);records=[]
    for i,s in enumerate(samples):
        n=s['length'];motion=model.motion_rep.inverse(predicted[i:i+1,:n],is_normalized=True)
        reference=model.motion_rep.inverse(batch['x0'][i:i+1,:n],is_normalized=True)
        qpos=converter.dict_to_qpos(motion)[0]
        if qpos.shape!=(n,36) or not np.isfinite(qpos).all():raise FloatingPointError('Invalid qpos')
        name=f"{i:02d}_{s['group']}_{s.get('cohort','train')}"
        np.savez(output/f'{name}.npz',**{k:v[0].cpu().numpy() for k,v in motion.items()})
        np.savetxt(output/f'{name}.csv',qpos,delimiter=',')
        a,b=motion['posed_joints'][0],reference['posed_joints'][0]
        picture=motion_image(b.cpu().numpy(),a.cpu().numpy(),model.motion_rep.skeleton.joint_parents.cpu().tolist(),s['text'])
        picture.save(output/f'{name}.png')
        if writer:writer.add_image(f'samples/{tag}/{name}',np.array(picture),step,dataformats='HWC')
        records.append({'sample_id':s['sample_id'],'group':s['group'],'cohort':s.get('cohort','train'),'text':s['text'],
                        'seed':seed,'guidance':guidance,'frames':n,'heading':float(s['first_heading_angle']),
                        'world_mpjpe_m':float(torch.linalg.vector_norm(a-b,dim=-1).mean()),
                        'root_relative_mpjpe_m':float(torch.linalg.vector_norm((a-a[:,:1])-(b-b[:,:1]),dim=-1).mean()),
                        'postprocess':False,'finite_qpos':True})
    atomic_json(output/'metrics.json',records);return records
