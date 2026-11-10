#!/usr/bin/env python3
"""Prepare a late-yaw continuation while resetting only the validation selector."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from training.m3 import atomic_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args()
    run=args.run_dir.resolve();config=json.loads((run/'config.json').read_text())
    if json.loads((run/'continuation.json').read_text())['source_step']!=20000:
        raise ValueError('This curriculum requires the full 20000-step B warmup')
    existing=torch.load(run/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    if existing['step']!=20000 or (run/'COMPLETED.json').exists():
        raise ValueError('Refuse to reconfigure a curriculum that has already trained')
    config.update(heading_augmentation=True,heading_augmentation_probability=0.5,text_dropout=0.0,
        minimum_steps=30000,max_steps=30000,selection_metric='equal native/rotated90 macro validation',
        curriculum='20000 B steps, then 10000 steps with 50% yaw augmentation; no text dropout or loss change')
    identity_extra={name:sha(ROOT/name) for name in ['training/m4_orientation.py','scripts/train_m4_orientation.py']}
    for name in ['latest.pt','best_val_raw.pt','best_val_ema.pt']:
        path=run/'checkpoints'/name;state=torch.load(path,map_location='cpu',weights_only=False)
        state['config']=config;state['identity']['orientation_code_sha256']=identity_extra
        state.update(best_raw=float('inf'),best_ema=float('inf'),early_best=float('inf'),best_raw_step=0,
                     best_ema_step=0,bad_evaluations=0,last_evaluation_step=-1)
        torch.save(state,path)
    context=json.loads((run/'run_config.json').read_text());context['config']=config
    context['identity']['orientation_code_sha256']=identity_extra;context['selection_metric']=config['selection_metric']
    atomic_json(run/'config.json',config);atomic_json(run/'run_config.json',context)
    provenance=json.loads((run/'continuation.json').read_text())
    provenance.update(heading_augmentation_probability=.5,text_dropout=0.0,validation_history_reset=True,
        reset_reason='Native-only and native/rotated balanced validation have different selection criteria',
        model_optimizer_and_existing_rngs_preserved=True)
    atomic_json(run/'continuation.json',provenance)
    atomic_json(run/'orientation_prepared.json',{'source_step':20000,'configuration':config,'code_sha256':identity_extra})
    print('Late-heading continuation prepared',run,flush=True)


if __name__=='__main__':main()
