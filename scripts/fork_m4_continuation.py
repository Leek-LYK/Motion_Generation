#!/usr/bin/env python3
"""Fork an auditable full-state continuation without altering the parent run."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from training.m3 import atomic_json,sha,validate_inputs


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--stage',required=True)
    p.add_argument('--max-steps',type=int,default=20000);args=p.parse_args()
    parent=args.parent.resolve();run=args.run_dir.resolve();run.relative_to(ROOT/'runs')
    if run.exists():raise FileExistsError(run)
    ready=validate_inputs();source=parent/'checkpoints/latest.pt';source_hash=sha(source)
    state=torch.load(source,map_location='cpu',weights_only=False)
    if state['identity']['m2_fingerprint']!=ready['fingerprint']:raise ValueError('Data version mismatch')
    if args.max_steps<=state['step']:raise ValueError('Continuation budget must exceed source step')
    config=dict(state['config']);config.update(stage=args.stage,max_steps=args.max_steps,minimum_steps=10000,
        eval_every=500,sample_every=2500,log_every=50,early_stop_patience=12,preview_guidance=[1.0])
    changes={k:{'before':state['config'].get(k),'after':v} for k,v in config.items() if state['config'].get(k)!=v}
    run.mkdir(parents=True);(run/'checkpoints').mkdir()
    for name in ('latest.pt','best_val_raw.pt','best_val_ema.pt'):
        original=parent/'checkpoints'/name
        checkpoint=torch.load(original,map_location='cpu',weights_only=False)
        checkpoint['config']=config
        torch.save(checkpoint,run/'checkpoints'/name)
    shutil.copytree(parent/'stats',run/'stats');shutil.copyfile(parent/'subset.json',run/'subset.json')
    for name in ('last_evaluation.json',):
        if (parent/name).exists():shutil.copyfile(parent/name,run/name)
    context=json.loads((parent/'run_config.json').read_text())
    context.update(config=config,initialization='Full-state continuation: model, EMA, optimizer, all RNGs and validation history',
                   parent_run=str(parent),parent_checkpoint_sha256=source_hash,source_step=state['step'],
                   started_at_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(run/'run_config.json',context);atomic_json(run/'config.json',config)
    atomic_json(run/'continuation.json',{'parent_run':str(parent),'parent_checkpoint':str(source),'parent_sha256':source_hash,
        'source_step':state['step'],'configuration_changes':changes,'weights_optimizer_rng_modified':False,
        'script_sha256':sha(__file__)})
    assert sha(source)==source_hash
    print(json.dumps({'run':str(run),'source_step':state['step'],'max_steps':args.max_steps,'changes':changes},indent=2),flush=True)


if __name__=='__main__':main()
