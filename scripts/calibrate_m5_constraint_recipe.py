#!/usr/bin/env python3
"""Train-only gradient scales for bounded native hinges, self-consistency and foot ground loss.

Formal calibration requires all four frozen 40k runs to finish. An explicitly
labelled engineering fixture can exercise the tool at an earlier checkpoint;
its coefficients cannot be used as a formal recipe.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from kimodo.model.diffusion import Diffusion
from training.m3 import atomic_json,configure,make_model,batch_to,forward,sha
from training.m4 import rotate_batch
from training.m5_physical_data import PhysicalSubsetDataset
from training.m5_physical_losses import geometry,NativeHingeGeometry
from training.m5_constraint_losses import constraint_objective as objective
from training.g1_legality_loss import NativeLegalityLoss


def norm(gradients):
    values=[g.float().square().sum() for g in gradients if g is not None]
    return torch.sqrt(sum(values)) if values else torch.tensor(0.,device='cuda:0')


def dot(left,right):return sum((a*b).sum() for a,b in zip(left,right) if a is not None and b is not None)


def digest(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--parent-run',required=True)
    p.add_argument('--checkpoint',default='step_040000.pt');p.add_argument('--output',type=Path,required=True);p.add_argument('--engineering-fixture',action='store_true')
    a=p.parse_args();root=a.experiment.resolve();run=root/a.parent_run;out=a.output.resolve()
    if (out/'report.json').exists():raise FileExistsError('Calibration already recorded')
    if not a.engineering_fixture:
        launch=json.loads((root/'protocol/launch.json').read_text())
        for name in launch['jobs']:
            done=json.loads((root/name/'COMPLETED.json').read_text())
            if done['step']!=40000:raise ValueError('Wait for all four fixed 40k budgets')
        if not (root/'first_wave_assessment.json').exists():raise ValueError('First-wave assessment must precede formal calibration')
    parent=run/'checkpoints'/a.checkpoint;checksum=sha(parent);state=torch.load(parent,map_location='cpu',weights_only=False)
    if state.get('schema') not in ('kimodo-m5-physical-checkpoint-v2','kimodo-m5-physical-atan2-checkpoint-v1','kimodo-m5-physical-constraints-checkpoint-v1'):raise ValueError('Physical-v2 input parent required')
    if not a.engineering_fixture and state['step']!=40000:raise ValueError('Formal parent must be the fixed 40k endpoint')
    if not a.engineering_fixture:
        complete=json.loads((run/'COMPLETED.json').read_text())
        if complete['step']!=40000 or any(state['config'].get(k,0.) for k in ['consistency_weight','axis_weight','ground_weight','legality_weight']):raise ValueError('Formal recipe starts from a completed unregularized 40k parent')
    view=json.loads((root/'data/READY.json').read_text())
    if state['identity']['physical_input_fingerprint']!=view['input_fingerprint'] or state['identity']['training_view_fingerprint']!=view['fingerprint']:raise ValueError('Input identity mismatch')
    for name,expected in state['identity']['stats_sha256'].items():
        if sha(run/name)!=expected:raise ValueError('Statistics changed')
    configure(7401);model=make_model(state['config']['model'],run/'stats/motion').cuda().train();model.load_state_dict(state['model']);before=digest(model)
    if not a.engineering_fixture and sum(p.numel() for p in model.parameters())!=56755617:raise ValueError('Formal calibration uses the actual Small architecture')
    config=dict(state['config'],consistency_weight=0.,axis_weight=0.,ground_weight=0.,legality_weight=0.);manifest=json.loads((run/'subset.json').read_text());entries=manifest['train']
    if any(r['split']!='train' for r in entries):raise ValueError('Calibration cannot read held-out samples')
    dataset=PhysicalSubsetDataset(entries,view['input_dir'],run/'stats/motion',cache_size=128);pools={}
    for i,row in enumerate(entries):
        bucket=0 if row['num_frames']<=90 else 1 if row['num_frames']<=120 else 2;pools.setdefault(bucket,{}).setdefault(row['group'],[]).append(i)
    cpu=torch.Generator().manual_seed(7401);rng=torch.Generator(device='cuda:0').manual_seed(7402);diffusion=Diffusion(1000).cuda();hinge=NativeHingeGeometry(model.motion_rep.skeleton,'cuda:0');legality=NativeLegalityLoss('cuda:0')
    body=[p for p in model.body_model.parameters() if p.requires_grad];root_params=[p for p in model.root_model.parameters() if p.requires_grad];records=[]
    policy={'parent_checkpoint':str(parent),'parent_sha256':checksum,'step':state['step'],'samples':'training split only','batches':8,'batch_size':16,
      'seed_data':7401,'seed_noise_augmentation':7402,'base_objective':config['loss_objective'],'parent_precision':config['precision'],
      'extra_objectives':{'consistency':'Smoothed Euclidean FK/predicted-position metres','legality':'MJCF-native bounded-hinge matrix distance, including axis and legal range','ground':'Mean squared foot depth beyond 5mm, m^2'},
      'target_initial_body_gradient_fractions':[.1,.3],'coefficient_sanity_interval':[1e-8,1e3],
      'ground_truth_note':'Absolute 5mm ground regularization can be nonzero on noisy retargeted GT; record that conflict, do not assert its oracle loss is zero.',
      'engineering_fixture':a.engineering_fixture,'formal_coefficients_usable':False,'optimizer_updates':0,'script_sha256':sha(__file__)}
    atomic_json(out/'protocol.json',policy)
    for batch_index in range(8):
        buckets=sorted(pools);weights=torch.tensor([sum(len(v) for v in pools[b].values()) for b in buckets],dtype=torch.float64)
        bucket=buckets[int(torch.multinomial(weights,1,generator=cpu))];groups=sorted(pools[bucket]);choices=torch.randint(len(groups),(16,),generator=cpu)
        indices=[pools[bucket][groups[int(g)]][int(torch.randint(len(pools[bucket][groups[int(g)]]),(1,),generator=cpu))] for g in choices]
        samples=dataset.batch(indices);batch=batch_to(samples,'cuda:0');angles=(torch.rand(16,device='cuda:0',generator=rng)*2-1)*torch.pi;batch=rotate_batch(model,batch,angles)
        with torch.no_grad():
            _,truth=objective(model.motion_rep,batch['x0'],batch['x0'],batch['x_pad_mask'],config,hinge,legality)
            gt={'axis':float(truth['axis_defect']),'consistency_m':float(truth['fk_position_inconsistency_m']),'ground_m2':float(truth['ground_loss_m2']),'bounded_legality':float(truth['bounded_hinge_legality'])}
            if gt['axis']>1e-8 or gt['consistency_m']>2e-6 or gt['bounded_legality']>1e-8:raise ValueError('Training target violates physical-v2 contract')
        dropped=torch.rand(16,device='cuda:0',generator=rng)<config['text_dropout'];batch['text_feat']=batch['text_feat'].masked_fill(dropped[:,None,None],0)
        t=torch.randint(1000,(16,),device='cuda:0',generator=rng);replace=torch.rand(16,device='cuda:0',generator=rng)<config.get('high_noise_fraction',0.)
        t=torch.where(replace,torch.randint(800,1000,(16,),device='cuda:0',generator=rng),t);noise=torch.randn(batch['x0'].shape,device='cuda:0',generator=rng)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=config['precision']=='bfloat16'):
            pred=forward(model,diffusion.q_sample(batch['x0'],t,noise),batch,t)
        with torch.autocast(device_type='cuda',enabled=False):base,parts=objective(model.motion_rep,pred,batch['x0'],batch['x_pad_mask'],config,hinge,legality)
        terms={'base':base,'consistency':parts['consistency_loss_m'],'legality':parts['bounded_hinge_legality'],'ground':parts['ground_loss_m2']}
        gradients={k:torch.autograd.grad(value,body,allow_unused=True,retain_graph=True) for k,value in terms.items()};norms={k:float(norm(v)) for k,v in gradients.items()}
        if not all(np.isfinite(v) for v in norms.values()) or norms['base']<=1e-10:raise ValueError('Invalid gradient scales')
        root_norms={k:float(norm(torch.autograd.grad(value,root_params,allow_unused=True,retain_graph=True))) for k,value in terms.items() if k!='base'}
        if max(root_norms.values())>1e-6:raise ValueError('Unexpected added-loss gradient shortcut into Root stage')
        records.append({'batch':batch_index,'sample_ids':[s['sample_id'] for s in samples],'t':t.cpu().tolist(),'body_gradient_norms':norms,
          'extra_root_gradient_norms':root_norms,'ground_truth_terms':gt,'losses':{k:float(v.detach()) for k,v in terms.items()},
          'cosine_with_base':{k:float(dot(gradients['base'],gradients[k])/(norms['base']*norms[k])) if norms[k]>1e-10 else None for k in ['consistency','legality','ground']}})
        del gradients,pred,base,parts,terms;print('physical constraint calibration',batch_index+1,'/ 8',flush=True)
    coefficients={};usable=True
    for key in ['consistency','legality','ground']:
        active=[r['body_gradient_norms']['base']/r['body_gradient_norms'][key] for r in records if r['body_gradient_norms'][key]>1e-10]
        if len(active)<6:usable=False;coefficients[key]={'active_batches':len(active),'low':None,'medium':None};continue
        ratio=float(np.median(active));coefficients[key]={'active_batches':len(active),'low':.1*ratio,'medium':.3*ratio}
        usable=usable and all(1e-8<=coefficients[key][level]<=1e3 for level in ['low','medium'])
    unchanged=before==digest(model) and all(p.grad is None for p in model.parameters());assert unchanged and sha(parent)==checksum
    result=dict(policy,passed=bool(usable),formal_coefficients_usable=bool(usable and not a.engineering_fixture),coefficients=coefficients,records=records,
      parameters_unchanged=True,validation_or_test_scores_used=False,loss_code_sha256=sha(ROOT/'training/m5_constraint_losses.py'),legality_code_sha256=sha(ROOT/'training/g1_legality_loss.py'))
    atomic_json(out/'report.json',result);print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':main()
