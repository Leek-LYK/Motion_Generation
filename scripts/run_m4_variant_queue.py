#!/usr/bin/env python3
"""Run next single-variable ablations only when continuation controls fail the amplitude gate."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data_pipeline.numeric_common import atomic_json,sha_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--force',action='store_true')
    p.add_argument('--order',choices=['high-first','mse-first'],default='high-first');args=p.parse_args()
    root=ROOT/'runs/m4_research_20261103';lock=(root/'variants.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verification=json.loads((root/'variant_verification/report.json').read_text())
    if not verification['passed'] or verification['research_source_sha256']!=sha_file(ROOT/'training/m4_research.py'):
        raise ValueError('Research implementation verification is stale')
    stopped=[False];child=[None];completed=[]
    def stop(*_):
        stopped[0]=True
        if child[0] is not None and child[0].poll() is None:child[0].send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def status(phase,**extra):
        atomic_json(root/'variants_status.json',{'phase':phase,'pid':os.getpid(),'completed_variants':completed,
                    'updated_at_utc':datetime.now(timezone.utc).isoformat(),**extra})
    status('waiting_for_controls')
    while not stopped[0]:
        state=json.loads((root/'queue_status.json').read_text())
        if state['phase']=='controls_completed':break
        if state['phase'] in ('failed','stopped'):
            status('controls_require_attention',control_status=state);return
        time.sleep(10)
    if stopped[0]:status('stopped');return
    controls=[name for name in ('R1_D_continue','R2_B_continue')
              if json.loads((root/name/'readiness/report.json').read_text())['amplitude_gate_passed']]
    if controls and not args.force:
        status('candidate_requires_independent_verification',candidates=controls);return
    try:
        order=['R3_D_high_noise','R4_D_mse'] if args.order=='high-first' else ['R4_D_mse','R3_D_high_noise']
        atomic_json(root/'variants_execution_order.json',{'order':order,'force':args.force,
                    'reason':'Development candidate did not satisfy all entry evidence; prioritize a single-variable loss test when requested',
                    'entry_thresholds_changed':False,'audit_used':False})
        for name in order:
            run=root/name
            tasks=[('training',run/'COMPLETED.json',[sys.executable,'-B','-u',str(ROOT/'scripts/train_m4_research.py'),
                    '--config',str(run/'config.json'),'--subset',str(run/'subset.json'),'--run-dir',str(run),'--resume']),
                   ('evaluation',run/'final_evaluation/COMPLETE.json',[sys.executable,'-B','-u',str(ROOT/'scripts/evaluate_m4.py'),'--run-dir',str(run)]),
                   ('amplitude',run/'readiness/report.json',[sys.executable,'-B','-u',str(ROOT/'scripts/assess_m4_readiness.py'),
                    '--run-dir',str(run),'--output',str(run/'readiness')])]
            for phase,marker,command in tasks:
                if stopped[0]:break
                if marker.exists():continue
                with (root/f'{name}_{phase}.log').open('ab',buffering=0) as log:
                    child[0]=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    status(phase,run=name,child_pid=child[0].pid);print(name,phase,child[0].pid,flush=True)
                    code=child[0].wait()
                if stopped[0]:break
                if code or not marker.exists():raise RuntimeError(f'{name}/{phase} failed: {code}')
            if stopped[0]:status('stopped');return
            completed.append(name)
            if json.loads((run/'readiness/report.json').read_text())['amplitude_gate_passed']:
                status('candidate_requires_independent_verification',candidates=[name]);return
        status('variants_completed_research_still_needed')
    except Exception as error:
        status('failed',error=repr(error));raise


if __name__=='__main__':main()
