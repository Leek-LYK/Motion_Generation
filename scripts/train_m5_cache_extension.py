#!/usr/bin/env python3
"""Versioned append-cache continuation runner with frozen qualified model statistics."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,fcntl,hashlib,json,shutil,signal,sys,time,traceback
from collections import Counter
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch.utils.tensorboard import SummaryWriter
from training.m3 import atomic_json,configure,sha,validate_inputs,INPUTS
from training.m5_cache_extension import AppendCompatibleConstraintTrainer as PhysicalTrainer
from training.m5_physical_data import PhysicalSubsetDataset
from training.m5_physical_evaluation import evaluate_denoising,evaluate_generation

CODE=('training/m3.py', 'training/m4.py', 'training/m4_research.py', 'training/m5.py', 'training/m5_losses.py', 'training/m5_evaluation.py', 'training/g1_qpos.py', 'training/g1_physical.py', 'training/m5_physical.py', 'training/m5_physical_losses.py', 'training/m5_physical_data.py', 'training/m5_physical_evaluation.py', 'scripts/train_m5_physical.py', 'training/adam_atan2.py', 'training/m5_atan2.py', 'training/g1_legality_loss.py', 'training/m5_constraint_losses.py', 'training/m5_constraints.py', 'scripts/train_m5_constraints.py', 'training/m5_cache_extension.py', 'scripts/train_m5_cache_extension.py')


def model_digest(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--subset',type=Path,required=True);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--parent',type=Path);parser.add_argument('--initial-weights',type=Path)
    parser.add_argument('--engineering-fixture',action='store_true')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.initial_weights or bool(args.parent)==bool(args.resume):raise ValueError('Choose exactly one of explicit full-state --parent or --resume')
    run=args.run_dir.resolve();run.relative_to(ROOT/'runs');run.mkdir(parents=True,exist_ok=True)
    lock=(run/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads(args.config.read_text());ready=validate_inputs();manifest=json.loads(args.subset.read_text())
    if bool(cfg.get('engineering_fixture',False))!=args.engineering_fixture:raise ValueError('Engineering fixture must be explicitly labelled in both config and CLI')
    target_step=cfg.get('training_stop_step',cfg['max_steps'])
    if not cfg['stage_start_step']<target_step<=cfg['max_steps']:raise ValueError('Invalid stage budget/schedule endpoint')
    assert manifest['m2_fingerprint']==ready['fingerprint']
    inputs=Path(cfg['input_dir']);stats_dir=Path(cfg['stats_dir']);view_path=Path(cfg['training_view'])
    view=json.loads(view_path.read_text());input_ready=json.loads((inputs/'READY').read_text())
    if stats_dir.resolve()!=(view_path.parent.parent/'stats/motion').resolve() or inputs.resolve()!=Path(view['input_dir']).resolve():
        raise ValueError('Configured statistics/input directory does not match the frozen training view')
    if not view['passed'] or view['input_fingerprint']!=input_ready['fingerprint'] or manifest['physical_cache_fingerprint']!=input_ready['fingerprint']:
        raise ValueError('Physical inputs/view/subset mismatch')
    for name,expected in view['stats_sha256'].items():
        if sha(view_path.parent.parent/name)!=expected:raise ValueError('Qualified statistics changed: '+name)
    expected=cfg.get('launch_code_sha256')
    if expected and expected!={f:sha(ROOT/f) for f in CODE}:raise ValueError('Launch source changed')
    identity={'cache_compatibility_proof_sha256':cfg['cache_compatibility_proof_sha256'], 'm2_fingerprint':ready['fingerprint'],'physical_input_fingerprint':input_ready['fingerprint'],
              'training_view_fingerprint':view['fingerprint'],'text_config_fingerprint':ready['text_config_fingerprint'],
              'subset_sha256':sha(args.subset),'stats_sha256':view['stats_sha256'],'code_sha256':{f:sha(ROOT/f) for f in CODE}}
    if cfg.get('subset_sha256',identity['subset_sha256'])!=identity['subset_sha256']:raise ValueError('Frozen subset changed')
    coefficients={k:cfg.get(k,0.) for k in ('consistency_weight','axis_weight','ground_weight','legality_weight')}
    if any(coefficients.values()) and not args.engineering_fixture:
        if not args.parent and not args.resume:raise ValueError('Constraint activation requires an explicit parent or resume')
        if cfg['stage_start_step']<40000 or not (view_path.parent.parent/'first_wave_assessment.json').exists():
            raise ValueError('Active constraint training requires completed first-wave evidence and a >=40k parent')
        activation_path=Path(cfg['constraint_activation_protocol']);activation=json.loads(activation_path.read_text())
        if not activation.get('passed') or activation['physical_input_fingerprint']!=identity['physical_input_fingerprint'] or activation['training_view_fingerprint']!=identity['training_view_fingerprint']:
            raise ValueError('Constraint activation protocol mismatch')
        entry=activation['runs'][cfg['stage']]
        source_sha=sha(args.parent) if args.parent else json.loads((run/'lineage.json').read_text())['parent_sha256'] if args.resume else None
        if entry['parent_sha256']!=source_sha or entry['coefficients']!=coefficients:raise ValueError('Constraint parent/coefficients differ from the frozen activation')

    if (run/'COMPLETED.json').exists():
        done=json.loads((run/'COMPLETED.json').read_text())
        if done['identity']!=identity or json.loads((run/'config.json').read_text())!=cfg or done['step']!=target_step:raise ValueError('Completed run identity/budget changed')
        print('Already complete; not accepted automatically',flush=True);return
    if (run/'run_config.json').exists() and not args.resume:raise FileExistsError('Explicit --resume required')
    configure(cfg['seed']);train=PhysicalSubsetDataset(manifest['train'],inputs,stats_dir,cache_size=cfg.get('cache_events',1024))
    val=PhysicalSubsetDataset(manifest['val'],inputs,stats_dir,cache_size=len(manifest['val']))
    monitor=PhysicalSubsetDataset(manifest.get('monitor',manifest['val']),inputs,stats_dir,cache_size=len(manifest['val']))
    trainer=PhysicalTrainer(cfg,train,identity,stats_dir);initialization='random';parent_hash=None;parent_changes={}
    if args.resume:trainer.restore(run/'checkpoints/latest.pt')
    elif args.parent or args.initial_weights:
        source=args.parent or args.initial_weights;parent_hash=sha(source)
        state=torch.load(source,map_location='cpu',weights_only=False)
        trainer.load_state(state,continuation=bool(args.parent),initialization_only=bool(args.initial_weights))
        initialization='full_state_continuation' if args.parent else 'matched_random_initialization'
        parent_changes={k:{'before':state['config'].get(k),'after':v} for k,v in cfg.items() if state['config'].get(k)!=v}
        del state
        if sha(source)!=parent_hash:raise ValueError('Parent changed during load')
    if not args.resume:
        if trainer.step!=cfg['stage_start_step']:raise ValueError('Declared stage start differs from the loaded model')
        atomic_json(run/'config.json',cfg);atomic_json(run/'subset.json',manifest)
        shutil.copytree(stats_dir,run/'stats/motion',dirs_exist_ok=True)
        source_dir=run/'source'
        for f in CODE:
            target=source_dir/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/f,target)
        atomic_json(run/'run_config.json',{'config':cfg,'identity':identity,'parameters':sum(p.numel() for p in trainer.model.parameters()),
          'train_samples':len(train),'val_samples':len(val),'initialization':initialization,'initial_model_sha256':model_digest(trainer.model),
          'started_at_utc':datetime.now(timezone.utc).isoformat(),'engineering_fixture':args.engineering_fixture})
        atomic_json(run/'lineage.json',{'parent_checkpoint':str(args.parent or args.initial_weights) if parent_hash else None,
          'parent_sha256':parent_hash,'initialization':initialization,'start_step':trainer.step,'configuration_changes':parent_changes,
          'optimizer_rng_preserved':bool(args.parent),'validation_history_reset_for_new_stage':bool(args.parent)})
        trainer.save(run/'checkpoints/initial_state.pt');trainer.save(run/'checkpoints/latest.pt')
    writer=SummaryWriter(str(run/'tensorboard'),flush_secs=10,purge_step=trainer.step+1 if args.resume else None)
    stop=[False];start=time.monotonic();last={};best_generation=None
    exposure={k:Counter() for k in ('groups','captions','length_bins')};trace_path=run/'sample_trace.jsonl'
    def count_exposure(indices):
        for index in indices:
            row=manifest['train'][index];exposure['groups'][row['group']]+=1;exposure['captions'][row['text_id']]+=1
            exposure['length_bins']['short' if row['num_frames']<=90 else 'medium' if row['num_frames']<=120 else 'long']+=1
    if args.resume:
        valid=[]
        if trace_path.exists():
            for line in trace_path.read_text().splitlines():
                try:record=json.loads(line)
                except json.JSONDecodeError:break
                if record['step']<=trainer.step:
                    if record['step']!=cfg['stage_start_step']+len(valid)+1:raise ValueError('Sample trace has missing or duplicate updates')
                    valid.append(record);count_exposure(record['indices'])
        if len(valid)!=trainer.step-cfg['stage_start_step']:raise ValueError('Checkpoint/sample-trace mismatch')
        trace_path.write_text(''.join(json.dumps(r)+'\n' for r in valid))
    trace=trace_path.open('a',buffering=1)
    def flush_trace():
        trace.flush();os.fsync(trace.fileno())
        atomic_json(run/'exposure_counts.json',{'step':trainer.step,'stage_start_step':cfg['stage_start_step'],
          'subset_sha256':identity['subset_sha256'],**{k:dict(v) for k,v in exposure.items()}})
    best_path=run/'best_generation.json'
    if best_path.exists():best_generation=json.loads(best_path.read_text())
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    def status(phase,**extra):
        atomic_json(run/'status.json',{'phase':phase,'pid':os.getpid(),'stage':cfg['stage'],'step':trainer.step,
          'target_steps':target_step,'stage_start_step':cfg['stage_start_step'],'stage_updates':trainer.stage_updates,
          'updated_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_s_this_process':time.monotonic()-start,'last_train':last,**extra})
    def denoise():
        flush_trace()
        status('denoising_validation');raw=evaluate_denoising(trainer.model,monitor,trainer.device,cfg.get('eval_batch_size',16))
        ema=evaluate_denoising(trainer.ema,monitor,trainer.device,cfg.get('eval_batch_size',16))
        a,b=trainer.record_validation(raw['macro_loss'],ema['macro_loss'])
        if a:trainer.save(run/'checkpoints/best_denoising_raw.pt')
        if b:trainer.save(run/'checkpoints/best_denoising_ema.pt')
        for label,result in [('raw',raw),('ema',ema)]:
            writer.add_scalar(f'validation/{label}/norm6_macro',result['macro_loss'],trainer.step)
            for family in ('groups','timesteps','parts'):
                for k,v in result[family].items():writer.add_scalar(f'validation/{label}/{family}/{k}',v,trainer.step)
        atomic_json(run/'evaluations'/f'denoising_{trainer.step:06d}.json',{'raw':raw,'ema':ema,'step':trainer.step})
        trainer.save(run/'checkpoints/latest.pt');writer.flush()
    def generation(full):
        nonlocal best_generation
        kind='full' if full else 'monitor';folder=run/'evaluations'/f'generation_{kind}_{trainer.step:06d}'
        if (folder/'report.json').exists():return
        status('generation_validation',evaluation=str(folder))
        result=evaluate_generation(trainer.ema,val if full else monitor,folder,trainer.step,
          seeds=cfg.get('full_seeds',[0,42,1234]) if full else [42],headings=(None,0.),
          batch_size=cfg.get('eval_batch_size',16),steps=cfg.get('ddim_steps',100),guidance=2.,retain=full,metric_thresholds=cfg.get('metric_thresholds'),flags=json.loads((view_path.parent/'development_flags.json').read_text()))
        result.update(identity=identity,weights='ema',semantic_quality_accepted=False)
        atomic_json(folder/'report.json',result)
        for group,value in result['groups'].items():
            for k,v in value.items():
                if isinstance(v,(int,float)) and not isinstance(v,bool):writer.add_scalar(f'generation/{kind}/{group}/{k}',v,trainer.step)
        writer.add_scalar(f'generation/{kind}/geometry_score_m',result['geometry_score_m'],trainer.step)
        if full:
            conditions=list(result['conditions'].values())
            qualifies=result['all_valid_qpos'] and result['all_finite'] and all(x['core_amplitude_passed'] for x in conditions)
            score=(0 if qualifies else 1,result['geometry_score_m'])
            if best_generation is None or score<tuple(best_generation['selection_tuple']):
                trainer.save(run/'checkpoints/best_generation.pt')
                best_generation={'step':trainer.step,'selection_tuple':list(score),'report':str(folder/'report.json'),
                  'selection_scope':'numerical provisional choice; semantic review remains required','checkpoint_sha256':sha(run/'checkpoints/best_generation.pt')}
                atomic_json(best_path,best_generation)
        writer.flush()
    writer.add_text('experiment/config',json.dumps(cfg,indent=2),trainer.step)
    try:
        status('training')
        if trainer.last_evaluation_step!=trainer.step:denoise()
        with (run/'metrics.jsonl').open('a',buffering=1) as metrics:
            while trainer.step<target_step and not stop[0]:
                # Peek and rewind only the explicit data RNG. The update consumes
                # exactly the same draw, preserving the verified M4 continuation.
                rng_before=trainer.batch_rng.get_state();indices=trainer.draw_indices();trainer.batch_rng.set_state(rng_before)
                begin=time.monotonic();last=trainer.update();last['step_seconds']=time.monotonic()-begin
                count_exposure(indices);trace.write(json.dumps({'step':trainer.step,'indices':indices})+'\n')
                if trainer.step%cfg['log_every']==0:
                    metrics.write(json.dumps(dict(step=trainer.step,**last))+'\n')
                    for k,v in last.items():writer.add_scalar('train/'+k,v,trainer.step)
                    writer.add_scalar('system/peak_cuda_gib',torch.cuda.max_memory_allocated()/1024**3,trainer.step)
                    status('training');print(f"step={trainer.step} loss={last['loss']:.6f} time={last['step_seconds']:.3f}s",flush=True)
                if trainer.step%cfg['eval_every']==0:denoise()
                if trainer.step%cfg['checkpoint_every']==0:
                    flush_trace()
                    trainer.save(run/'checkpoints'/f'step_{trainer.step:06d}.pt');trainer.save(run/'checkpoints/latest.pt')
                if trainer.step%cfg['full_generation_every']==0:generation(True)
                elif trainer.step%cfg['generation_every']==0:generation(False)
        flush_trace();trainer.save(run/'checkpoints/latest.pt')
        if stop[0]:status('stopped');return
        if trainer.last_evaluation_step!=trainer.step:denoise()
        generation(True);trainer.save(run/'checkpoints'/f'step_{trainer.step:06d}.pt');trainer.save(run/'checkpoints/latest.pt')
        completion={'engineering_fixture':args.engineering_fixture,'schedule_end_step':cfg['max_steps'],'training_stop_step':target_step,'status':'training_budget_completed_not_quality_acceptance','step':trainer.step,'stage':cfg['stage'],
          'identity':identity,'updates_this_stage':trainer.stage_updates,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
          'terminal_checkpoint_sha256':sha(run/'checkpoints/latest.pt'),'best_generation':best_generation,'M5_complete':False}
        atomic_json(run/'COMPLETED.json',completion);status('budget_completed');print(json.dumps(completion,indent=2),flush=True)
    except Exception as error:
        status('failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:trace.close();writer.close()


if __name__=='__main__':main()
