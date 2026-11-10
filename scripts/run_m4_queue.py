#!/usr/bin/env python3
"""Durable A -> B -> C -> D train/evaluate queue; no quality-based stage skipping."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data_pipeline.numeric_common import atomic_json, sha_file

STAGES=[('A','A_256','A.json'),('B','B_1024','BCD.json'),('C','C_1024_textdrop','BCD.json'),('D','D_1024_heading','BCD.json')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    experiment=args.experiment.resolve();experiment.relative_to(ROOT/'runs')
    target=experiment/'queue_smoke' if args.smoke else experiment
    target.mkdir(parents=True,exist_ok=True)
    lock=(target/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not json.loads((experiment/'data/audit.json').read_text())['passed']:raise ValueError('Data audit failed')
    if not args.smoke:
        verification=json.loads((experiment/'verification/report.json').read_text())
        if not verification['passed']:raise ValueError('Numerical checks not passed')
        for name,digest in verification['m4_code_sha256'].items():
            if sha_file(ROOT/name)!=digest:raise ValueError('Numerical verification is stale: '+name)
        smoke=json.loads((experiment/'queue_smoke/QUEUE_COMPLETE.json').read_text())
        if not smoke['passed'] or smoke['stages']!=['A','B','C','D']:raise ValueError('Four-stage smoke not complete')
    protocol_path=target/'protocol.json'
    protocol={'stages':[s[0] for s in STAGES],
              'configs':{s[0]:sha_file(ROOT/'configs'/f'm4_{s[0]}.json') for s in STAGES},
              'manifests':{name:sha_file(experiment/'data'/name) for name in ('A.json','BCD.json')},
              'code':{name:sha_file(ROOT/name) for name in ('training/m4.py','scripts/train_m4.py','scripts/evaluate_m4.py','scripts/run_m4_queue.py')}}
    if not args.smoke and smoke['protocol']!=protocol:
        raise ValueError('Queue smoke used a different protocol')
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())!=protocol:raise ValueError('Queue protocol changed; use a new experiment')
    else:atomic_json(protocol_path,protocol)
    stopped=[False];child=[None];completed=[]
    def stop(*_):
        stopped[0]=True
        if child[0] is not None and child[0].poll() is None:child[0].send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def status(phase,**extra):
        atomic_json(target/'queue_status.json',{'phase':phase,'pid':os.getpid(),'completed_stages':completed,
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),**extra})
    def launch(command,log,phase,stage):
        with log.open('ab',buffering=0) as stream:
            child[0]=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            status(phase,stage=stage,child_pid=child[0].pid,command=command)
            print(f"{stage} {phase}: pid={child[0].pid}",flush=True)
            code=child[0].wait()
        if stopped[0]:return False
        if code:raise RuntimeError(f'{stage} {phase} failed (exit {code}); see {log}')
        return True
    try:
        for stage,name,manifest_name in STAGES:
            if stopped[0]:break
            run=target/name;subset=experiment/'data'/manifest_name
            if not (run/'COMPLETED.json').exists():
                command=[sys.executable,'-B','-u',str(ROOT/'scripts/train_m4.py'),'--config',str(ROOT/'configs'/f'm4_{stage}.json'),
                         '--subset',str(subset),'--run-dir',str(run)]
                if args.smoke:command+=['--smoke-steps','2']
                if (run/'checkpoints/latest.pt').exists():command+=['--resume']
                if not launch(command,target/f'{stage}_train.log','training',stage):break
                if not (run/'COMPLETED.json').exists():raise RuntimeError('Trainer exited without completion marker')
            if not (run/'final_evaluation/COMPLETE.json').exists():
                command=[sys.executable,'-B','-u',str(ROOT/'scripts/evaluate_m4.py'),'--run-dir',str(run)]
                if args.smoke:command.append('--smoke')
                if not launch(command,target/f'{stage}_evaluation.log','evaluating',stage):break
            result=json.loads((run/'final_evaluation/COMPLETE.json').read_text())
            if not result['passed'] or result['smoke']!=args.smoke:raise ValueError('Invalid stage evaluation')
            if sha_file(run/'final_evaluation/report.json')!=result['report_sha256']:raise ValueError('Evaluation report changed')
            train_result=json.loads((run/'COMPLETED.json').read_text())
            if result['checkpoint_sha256']!=sha_file(run/'checkpoints/best_val_ema.pt'):raise ValueError('Selected checkpoint changed')
            completed.append(stage)
            atomic_json(target/'comparison.json',{s:json.loads((target/n/'final_evaluation/report.json').read_text())
                         for s,n,_ in STAGES if s in completed})
        status('stopped' if stopped[0] else 'completed')
        if not stopped[0]:atomic_json(target/'QUEUE_COMPLETE.json',{'passed':True,'stages':completed,'smoke':args.smoke,'protocol':protocol})
    except Exception as error:
        status('failed',error=repr(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
