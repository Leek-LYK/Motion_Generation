#!/usr/bin/env python3
"""Wait for the registered independent-seed run, then assess without touching the locked audit."""
import json,os,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data_pipeline.numeric_common import atomic_json


def main():
    root=ROOT/'runs/m4_research_20261103';run=root/'R5_B_repeat_seed20261104';stopped=[False];child=[None]
    def stop(*_):
        stopped[0]=True
        if child[0] is not None and child[0].poll() is None:child[0].send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def status(phase,**extra):
        atomic_json(root/'repeat_assessment_status.json',{'phase':phase,'pid':os.getpid(),
                    'updated_at_utc':datetime.now(timezone.utc).isoformat(),**extra})
    status('waiting_for_registered_training')
    while not (run/'COMPLETED.json').exists() and not stopped[0]:
        registered=json.loads((root/'repeat_seed_process.json').read_text())
        process=Path('/proc')/str(registered['pid'])
        if not process.exists():status('training_process_missing_without_completion');return
        cmd=process.joinpath('cmdline').read_bytes().decode().replace('\0',' ')
        if 'train_m4.py' not in cmd or str(run) not in cmd:status('registered_pid_no_longer_matches_training');return
        time.sleep(10)
    if stopped[0]:status('stopped');return
    tasks=[('evaluation',run/'final_evaluation/COMPLETE.json',[sys.executable,'-B','-u',str(ROOT/'scripts/evaluate_m4.py'),'--run-dir',str(run)]),
           ('native_amplitude',run/'readiness/report.json',[sys.executable,'-B','-u',str(ROOT/'scripts/assess_m4_readiness.py'),'--run-dir',str(run),'--output',str(run/'readiness')]),
           ('heading0_amplitude',run/'readiness_heading0/report.json',[sys.executable,'-B','-u',str(ROOT/'scripts/assess_m4_readiness.py'),'--run-dir',str(run),'--output',str(run/'readiness_heading0'),'--heading','0'])]
    try:
        for phase,marker,command in tasks:
            if marker.exists():continue
            with (root/f'repeat_{phase}.log').open('ab',buffering=0) as log:
                child[0]=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                status(phase,child_pid=child[0].pid);code=child[0].wait()
            if stopped[0]:status('stopped');return
            if code or not marker.exists():raise RuntimeError(f'{phase} failed: {code}')
        status('development_assessment_completed',audit_released=False)
    except Exception as error:status('failed',error=repr(error));raise


if __name__=='__main__':main()
