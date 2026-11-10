#!/usr/bin/env python3
"""Bounded continuation controls; train, evaluate, and measure amplitude before further decisions."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data_pipeline.numeric_common import atomic_json


def main():
    experiment=ROOT/'runs/m4_research_20261103'
    lock=(experiment/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stopped=[False];child=[None];done=[]
    def stop(*_):
        stopped[0]=True
        if child[0] is not None and child[0].poll() is None:child[0].send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def status(phase,**extra):
        atomic_json(experiment/'queue_status.json',{'phase':phase,'pid':os.getpid(),'completed_controls':done,
                    'updated_at_utc':datetime.now(timezone.utc).isoformat(),**extra})
    try:
        for name in ['R1_D_continue','R2_B_continue']:
            run=experiment/name
            tasks=[('training',run/'COMPLETED.json',[sys.executable,'-B','-u',str(ROOT/'scripts/train_m4.py'),
                    '--config',str(run/'config.json'),'--subset',str(run/'subset.json'),'--run-dir',str(run),'--resume']),
                   ('evaluation',run/'final_evaluation/COMPLETE.json',[sys.executable,'-B','-u',str(ROOT/'scripts/evaluate_m4.py'),'--run-dir',str(run)]),
                   ('amplitude',run/'readiness/report.json',[sys.executable,'-B','-u',str(ROOT/'scripts/assess_m4_readiness.py'),
                    '--run-dir',str(run),'--output',str(run/'readiness')])]
            for phase,marker,command in tasks:
                if stopped[0]:break
                if marker.exists():continue
                with (experiment/f'{name}_{phase}.log').open('ab',buffering=0) as log:
                    child[0]=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    status(phase,run=name,child_pid=child[0].pid);print(name,phase,child[0].pid,flush=True)
                    code=child[0].wait()
                if stopped[0]:break
                if code or not marker.exists():raise RuntimeError(f'{name}/{phase} incomplete: exit {code}')
            if stopped[0]:break
            done.append(name)
        status('stopped' if stopped[0] else 'controls_completed')
    except Exception as error:
        status('failed',error=repr(error));raise


if __name__=='__main__':main()
