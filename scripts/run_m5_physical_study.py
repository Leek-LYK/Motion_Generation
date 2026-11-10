#!/usr/bin/env python3
"""Bounded parallel physical-v2 first wave; dependent curriculum requires assessment."""
import argparse,fcntl,json,os,signal,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.m3 import atomic_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();root=a.experiment.resolve()
    lock=(root/'study.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=root/'protocol/launch.json';protocol=json.loads(path.read_text());workers={};logs={};finished=[];stop=[False]
    for filename,expected in protocol['code_sha256'].items():
        if sha(ROOT/filename)!=expected:raise ValueError('Frozen source changed: '+filename)
    for name,expected in protocol['config_sha256'].items():
        if sha(root/'configs'/(name+'.json'))!=expected:raise ValueError('Frozen config changed: '+name)
    subset=Path(protocol['subset'])
    if sha(subset)!=protocol['subset_sha256']:raise ValueError('Frozen subset changed')
    def status(phase,**extra):
        atomic_json(root/'study_status.json',{'phase':phase,'pid':os.getpid(),'updated_at_utc':datetime.now(timezone.utc).isoformat(),
          'active':{name:p.pid for name,p in workers.items() if p.poll() is None},'completed':finished,'protocol_sha256':sha(path),'M5_complete':False,**extra})
    def halt(*_):
        stop[0]=True
        for process in workers.values():
            if process.poll() is None:process.send_signal(signal.SIGTERM)
    for s in [signal.SIGTERM,signal.SIGINT]:signal.signal(s,halt)
    try:
        status('preflight')
        result=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]
        total,free=[int(x.strip()) for x in result.split(',')]
        if free<max(int(total*.20),protocol['minimum_free_memory_mib']):raise RuntimeError('Insufficient current free GPU memory')
        waiting=list(protocol['launch_order'])
        while waiting or any(p.poll() is None for p in workers.values()):
            if stop[0]:break
            for name,process in list(workers.items()):
                code=process.poll()
                if code is None or name in finished:continue
                if code!=0:raise RuntimeError(f'{name} exited {code}; see {root/(name+".log")}')
                done=root/name/'COMPLETED.json'
                if not done.exists():raise RuntimeError(name+' exited without completed training evidence')
                finished.append(name);print('completed',name,flush=True)
            for name in waiting[:]:
                if sum(p.poll() is None for p in workers.values())>=protocol['max_parallel']:break
                job=protocol['jobs'][name];initial=Path(job['initial_weights']) if job.get('initial_weights') else None
                if initial and not initial.exists():
                    parent=workers.get('B1_small_norm7')
                    if parent is not None and parent.poll() is not None:raise RuntimeError('B1 exited without its shared initial checkpoint')
                    continue
                command=[sys.executable,'-B','-u',str(ROOT/'scripts/train_m5_physical.py'),'--config',str(root/'configs'/(name+'.json')),
                  '--subset',str(subset),'--run-dir',str(root/name)]
                if (root/name/'run_config.json').exists():command+=['--resume']
                elif initial:command+=['--initial-weights',str(initial)]
                logs[name]=(root/(name+'.log')).open('ab',buffering=0)
                workers[name]=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=logs[name],stderr=subprocess.STDOUT)
                waiting.remove(name);print('launched',name,workers[name].pid,flush=True)
            status('training_first_wave',waiting=waiting);time.sleep(5)
        if stop[0]:
            for process in workers.values():
                if process.poll() is None:process.wait()
            status('stopped');return
        # Poll again after the last process exits; do not assume terminal success.
        for name,process in workers.items():
            if process.returncode!=0 or not (root/name/'COMPLETED.json').exists():raise RuntimeError('Incomplete '+name)
            if name not in finished:finished.append(name)
        status('assessing_first_wave')
        command=[sys.executable,'-B','-u',str(ROOT/'scripts/assess_m5_physical_first_wave.py'),'--experiment',str(root)]
        with (root/'first_wave_assessment.log').open('ab',buffering=0) as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        assessment=json.loads((root/'first_wave_assessment.json').read_text());status('first_wave_complete_pending_curriculum_review',assessment=assessment)
    except Exception as error:
        halt()
        for process in workers.values():
            if process.poll() is None:process.wait()
        status('failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        for handle in logs.values():handle.close()


if __name__=='__main__':main()
