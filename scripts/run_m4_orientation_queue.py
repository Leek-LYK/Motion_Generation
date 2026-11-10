#!/usr/bin/env python3
"""Late-heading curricula for the original and independent-seed B models."""
import os,json,fcntl,signal,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data_pipeline.numeric_common import atomic_json,sha_file


def main():
    root=ROOT/'runs/m4_research_20261103';lock=(root/'orientation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verification=json.loads((root/'orientation_verification/report.json').read_text())
    smoke=json.loads((root/'orientation_smoke/COMPLETED.json').read_text())
    if not verification['passed'] or verification['module_sha256']!=sha_file(ROOT/'training/m4_orientation.py') or smoke['step']!=2:
        raise ValueError('Late-heading verification missing')
    if smoke['identity']['orientation_code_sha256']['scripts/train_m4_orientation.py']!=sha_file(ROOT/'scripts/train_m4_orientation.py'):
        raise ValueError('Runner smoke verification stale')
    repeat=root/'R5_B_repeat_seed20261104'
    if json.loads((repeat/'COMPLETED.json').read_text())['step']!=20000:raise ValueError('Independent warmup has not reached 20000 steps')
    replica=root/'R7_B_late_heading_seed20261104'
    if not replica.exists():
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/fork_m4_continuation.py'),'--parent',str(repeat),'--run-dir',str(replica),
                        '--stage','R7_B_late_heading_seed20261104','--max-steps','30000'],cwd=ROOT,check=True)
    if not (replica/'orientation_prepared.json').exists():
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/prepare_m4_orientation.py'),'--run-dir',str(replica)],cwd=ROOT,check=True)
    stopped=[False];child=[None];done=[]
    def stop(*_):
        stopped[0]=True
        if child[0] is not None and child[0].poll() is None:child[0].send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def status(phase,**extra):
        atomic_json(root/'orientation_status.json',{'phase':phase,'pid':os.getpid(),'completed':done,
                    'updated_at_utc':datetime.now(timezone.utc).isoformat(),**extra})
    try:
        for name in ['R6_B_late_heading','R7_B_late_heading_seed20261104']:
            run=root/name
            tasks=[('training',run/'COMPLETED.json',[sys.executable,'-B','-u',str(ROOT/'scripts/train_m4_orientation.py'),
                    '--config',str(run/'config.json'),'--subset',str(run/'subset.json'),'--run-dir',str(run),'--resume']),
                   ('evaluation',run/'final_evaluation/COMPLETE.json',[sys.executable,'-B','-u',str(ROOT/'scripts/evaluate_m4.py'),'--run-dir',str(run)]),
                   ('native_amplitude',run/'readiness/report.json',[sys.executable,'-B','-u',str(ROOT/'scripts/assess_m4_readiness.py'),
                    '--run-dir',str(run),'--output',str(run/'readiness')]),
                   ('heading0_amplitude',run/'readiness_heading0/report.json',[sys.executable,'-B','-u',str(ROOT/'scripts/assess_m4_readiness.py'),
                    '--run-dir',str(run),'--output',str(run/'readiness_heading0'),'--heading','0']),
                   ('controlled_text',run/'controlled_three_seeds/report.json',[sys.executable,'-B','-u',str(ROOT/'scripts/probe_m4_controlled_text.py'),'--run-dir',str(run)])]
            for phase,marker,command in tasks:
                if stopped[0]:break
                if marker.exists():continue
                with (root/f'{name}_{phase}.log').open('ab',buffering=0) as log:
                    child[0]=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    status(phase,run=name,child_pid=child[0].pid)
                    with (root/'orientation_launch_history.jsonl').open('a') as history:
                        history.write(json.dumps({'run':name,'phase':phase,'pid':child[0].pid,'command':command,'started_at_utc':datetime.now(timezone.utc).isoformat()})+'\n')
                    print(name,phase,child[0].pid,flush=True);code=child[0].wait()
                if stopped[0]:status('stopped');return
                if code or not marker.exists():raise RuntimeError(f'{name}/{phase} failed: {code}')
            done.append(name)
        status('curricula_completed_audit_still_locked')
    except Exception as error:status('failed',error=repr(error));raise


if __name__=='__main__':main()
