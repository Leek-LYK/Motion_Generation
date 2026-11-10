#!/usr/bin/env python3
"""Train one frozen M4 experiment with full validation, EMA selection and resumable early stop."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.utils.tensorboard import SummaryWriter
from training.m3 import INPUTS, atomic_json, configure, sha, validate_inputs
from training.m4 import M4Trainer, evaluate_full, export_preview, identity_for, load_data, GROUPS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--subset',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke-steps',type=int)
    args=p.parse_args()
    run=args.run_dir.resolve();run.relative_to(ROOT/'runs');run.mkdir(parents=True,exist_ok=True)
    lock=(run/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (run/'COMPLETED.json').exists():
        print('Run already complete',flush=True);return
    if (run/'run_config.json').exists() and not args.resume:
        raise FileExistsError('Use --resume or a new run')
    config=json.loads(args.config.read_text())
    if args.smoke_steps:
        config.update(max_steps=args.smoke_steps,minimum_steps=args.smoke_steps,eval_every=1,
                      sample_every=args.smoke_steps,log_every=1,preview_ddim_steps=4,smoke=True)
    atomic_json(run/'status.json',{'phase':'loading','pid':os.getpid(),'step':0,'target_steps':config['max_steps']})
    ready=validate_inputs();manifest=json.loads(args.subset.read_text())
    if manifest['m2_fingerprint']!=ready['fingerprint']:raise ValueError('Subset/M2 mismatch')
    configure(config['seed']);identity=identity_for(args.subset,ready)
    train,val=load_data(manifest,'train'),load_data(manifest,'val')
    if len(train)!=config['train_samples'] and not args.smoke_steps:raise ValueError('Training sample count mismatch')
    if len(val)!=64:raise ValueError('M4 requires the frozen complete 64-event validation set')
    trainer=M4Trainer(config,train,identity)
    latest=run/'checkpoints/latest.pt'
    if args.resume:trainer.restore(latest)
    else:
        atomic_json(run/'run_config.json',{'config':config,'identity':identity,'train_samples':len(train),'val_samples':len(val),
                    'parameters':sum(p.numel() for p in trainer.model.parameters()),'initialization':'independent random seed; ckpt_path=None',
                    'selection_metric':'EMA full-validation class/cohort macro denoising loss',
                    'human_semantic_review':manifest['human_semantic_review'],
                    'started_at_utc':datetime.now(timezone.utc).isoformat()})
        atomic_json(run/'subset.json',manifest)
        shutil.copytree(INPUTS/'stats/motion',run/'stats/motion')
    writer=SummaryWriter(str(run/'tensorboard'),flush_secs=10,purge_step=trainer.step+1 if args.resume else None)
    writer.add_text('experiment/config',json.dumps(config,indent=2),trainer.step)
    stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    start=time.monotonic();initial_step=trainer.step;last={}
    train_observation=[s for g in GROUPS for s in [r for r in train if r['group']==g][:8]]
    representatives=[next(s for s in val if s['group']==g and s['cohort']==cohort)
                     for g in GROUPS for cohort in ('seen_text_new_motion','unseen_text_same_class')]

    def status(phase,**extra):
        atomic_json(run/'status.json',{'phase':phase,'pid':os.getpid(),'stage':config['stage'],'step':trainer.step,
                    'target_steps':config['max_steps'],'elapsed_s_this_process':time.monotonic()-start,
                    'updated_at_utc':datetime.now(timezone.utc).isoformat(),'last_train':last,
                    'samples_seen':trainer.samples_seen,'bad_evaluations':trainer.bad_evaluations,
                    'best_val_ema':trainer.best_ema if trainer.best_ema!=float('inf') else None,
                    'best_val_ema_step':trainer.best_ema_step,**extra})

    def log_evaluation(prefix,result):
        writer.add_scalar(prefix+'/macro_loss',result['macro_loss'],trainer.step)
        writer.add_scalar(prefix+'/evaluated_events',result['events'],trainer.step)
        for family in ('groups','cohorts','strata','timesteps','parts'):
            for key,value in result[family].items():writer.add_scalar(prefix+'/'+family+'/'+key,value,trainer.step)

    def evaluate():
        status('evaluating')
        raw=evaluate_full(trainer.model,val,trainer.device)
        ema=evaluate_full(trainer.ema,val,trainer.device)
        observation=evaluate_full(trainer.ema,train_observation,trainer.device)
        assert raw['events']==ema['events']==len(val)
        improved_raw,improved_ema=trainer.record_validation(raw['macro_loss'],ema['macro_loss'])
        log_evaluation('validation_raw',raw);log_evaluation('validation_ema',ema);log_evaluation('train_observation_ema',observation)
        if improved_raw:trainer.save(run/'checkpoints/best_val_raw.pt')
        if improved_ema:trainer.save(run/'checkpoints/best_val_ema.pt')
        trainer.save(latest)
        report={'step':trainer.step,'raw':raw,'ema':ema,'train_observation_ema':observation}
        atomic_json(run/'evaluations'/f'step_{trainer.step:06d}.json',report)
        atomic_json(run/'last_evaluation.json',{'step':trainer.step,'raw_macro':raw['macro_loss'],'ema_macro':ema['macro_loss'],
                    'evaluated_events':len(val),'best_val_raw':trainer.best_raw,'best_val_ema':trainer.best_ema,
                    'best_val_ema_step':trainer.best_ema_step,'bad_evaluations':trainer.bad_evaluations})
        writer.flush();print(f"validation step={trainer.step} raw={raw['macro_loss']:.6f} ema={ema['macro_loss']:.6f} events={len(val)}",flush=True)

    def preview():
        status('sampling')
        for w in config['preview_guidance']:
            records=export_preview(trainer.ema,representatives,run/'samples'/f'step_{trainer.step:06d}'/f'ema_cfg{w:g}',
                        seed=42,steps=config['preview_ddim_steps'],guidance=w,text_dropout_trained=config['text_dropout']>0,
                        writer=writer,step=trainer.step,tag=f'ema_cfg{w:g}')
            writer.add_scalar(f'generation/ema_cfg{w:g}/paired_world_mpjpe_m',sum(r['world_mpjpe_m'] for r in records)/len(records),trainer.step)
        writer.flush()

    try:
        if trainer.last_evaluation_step!=trainer.step:evaluate()
        if trainer.step==0:preview()
        with (run/'metrics.jsonl').open('a',buffering=1) as handle:
            while trainer.step<config['max_steps'] and not stop[0]:
                if trainer.step>=config['minimum_steps'] and trainer.bad_evaluations>=config['early_stop_patience']:
                    break
                tick=time.monotonic();last=trainer.update();last['step_seconds']=time.monotonic()-tick
                if trainer.step%config['log_every']==0:
                    for key,value in last.items():writer.add_scalar('train/'+key,value,trainer.step)
                    writer.add_scalar('system/peak_cuda_gib',torch.cuda.max_memory_allocated()/1024**3,trainer.step)
                    handle.write(json.dumps({'step':trainer.step,**last})+'\n');status('training')
                    print(f"step={trainer.step} loss={last['loss']:.6f} drop={last['text_drop_fraction_cumulative']:.3f} yaw={last['heading_abs_delta_mean']:.3f}",flush=True)
                if trainer.step%config['eval_every']==0:evaluate()
                if trainer.step%config['sample_every']==0:preview()
        if trainer.last_evaluation_step!=trainer.step:evaluate()
        trainer.save(latest)
        if stop[0]:
            status('stopped');return
        if trainer.step%config['sample_every']:preview()
        reason='budget_completed' if trainer.step>=config['max_steps'] else 'early_stopped'
        result={'phase':reason,'stage':config['stage'],'step':trainer.step,'best_val_raw':trainer.best_raw,
                'best_val_ema':trainer.best_ema,'best_val_raw_step':trainer.best_raw_step,'best_val_ema_step':trainer.best_ema_step,
                'best_val_ema_checkpoint_sha256':sha(run/'checkpoints/best_val_ema.pt'),'identity':identity,
                'samples_seen':trainer.samples_seen,'updates_this_process':trainer.step-initial_step,
                'completed_at_utc':datetime.now(timezone.utc).isoformat()}
        status(reason);writer.flush();atomic_json(run/'COMPLETED.json',result)
    except Exception as error:
        status('failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:writer.close()


if __name__=='__main__':main()
