#!/usr/bin/env python3
"""Bounded controlled M5 geometry trials; explicit adoption/resume and terminal checks."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,fcntl,json,signal,subprocess,sys,time,traceback,shlex
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.m3 import atomic_json,sha
from training.m5_geometry_study import evidence,compare,combination_for


def read(path):return json.loads(path.read_text()) if path.exists() else {}


def is_live(pid,run):
    if not pid:return False
    out=subprocess.run(['ps','-p',str(pid),'-o','args='],capture_output=True,text=True).stdout
    try:
        args=shlex.split(out);return '--run-dir' in args and Path(args[args.index('--run-dir')+1]).resolve()==run.resolve()
    except (ValueError,IndexError):return False


def main():
    p=argparse.ArgumentParser();p.add_argument('--study-dir',type=Path,required=True);p.add_argument('--engineering-fixture',action='store_true');a=p.parse_args();root=a.study_dir.resolve()
    lock=(root/'study.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);protocol=read(root/'protocol.json')
    if bool(protocol['engineering_fixture'])!=a.engineering_fixture:raise ValueError('Fixture scope differs')
    for filename,expected in protocol['code_sha256'].items():
        if sha(ROOT/filename)!=expected:raise ValueError('Study source drift: '+filename)
    for filename,expected in protocol.get('orchestration_code_sha256',{}).items():
        if sha(ROOT/filename)!=expected:raise ValueError('Study orchestration source drift: '+filename)
    if sha(Path(protocol['parent_checkpoint']))!=protocol['parent_sha256'] or sha(Path(protocol['subset']))!=protocol['subset_sha256']:raise ValueError('Parent or subset changed')
    for name,expected in protocol['config_sha256'].items():
        if sha(root/'configs'/(name+'.json'))!=expected:raise ValueError('Configuration changed: '+name)
    prior_active=read(root/'study_status.json').get('active',{})
    workers={};logs={};completed=[];stop=[False]
    def status(phase,**more):
        active={name:entry['pid'] for name,entry in workers.items() if is_live(entry['pid'],root/name)}
        atomic_json(root/'study_status.json',{'phase':phase,'pid':os.getpid(),'updated_at_utc':datetime.now(timezone.utc).isoformat(),'active':active,
          'completed':completed,'protocol_sha256':sha(root/'protocol.json'),'engineering_fixture':a.engineering_fixture,'M5_complete':False,**more})
    def halt(*_):
        stop[0]=True
        for name,entry in workers.items():
            if is_live(entry['pid'],root/name):os.kill(entry['pid'],signal.SIGTERM)
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,halt)
    def running(name,entry):
        process=entry.get('process')
        return process is not None and process.poll() is None or is_live(entry['pid'],root/name)
    def checked(name):
        run=root/name;result=evidence(run,protocol['target_step']);cfg=read(run/'config.json')
        if cfg!=read(root/'configs'/(name+'.json')) or result['report']['identity']['code_sha256']!=protocol['code_sha256'] or result['report']['identity']['subset_sha256']!=protocol['subset_sha256']:
            raise ValueError('Completed trial differs from frozen configuration/code/data: '+name)
        if read(run/'lineage.json')['parent_sha256']!=protocol['parent_sha256']:raise ValueError('Trial parent differs: '+name)
        return result
    def run_jobs(names):
        waiting=list(names)
        while waiting or any(running(n,e) for n,e in workers.items() if n not in completed):
            if stop[0]:raise InterruptedError('Study stopped')
            for name,entry in list(workers.items()):
                if name in completed or is_live(entry['pid'],root/name):continue
                process=entry.get('process')
                if process is not None and process.poll() is None:continue
                if process is not None and process.returncode!=0:raise RuntimeError(name+' failed; see '+str(root/(name+'.log')))
                if not (root/name/'COMPLETED.json').exists():raise RuntimeError(name+' handle disappeared without terminal evidence')
                checked(name);completed.append(name)
            for name in waiting[:]:
                if sum(running(n,e) for n,e in workers.items() if n not in completed)>=protocol['max_parallel']:break
                run=root/name
                if (run/'COMPLETED.json').exists():checked(name);completed.append(name);waiting.remove(name);continue
                old=read(run/'status.json')
                if old.get('phase')=='failed':raise RuntimeError('Preserved failed trial requires inspection: '+name)
                old_pid=old.get('pid',prior_active.get(name))
                if is_live(old_pid,run):
                    if (run/'config.json').exists() and read(run/'config.json')!=read(root/'configs'/(name+'.json')):raise ValueError('Live trial configuration differs: '+name)
                    workers[name]={'pid':old_pid,'process':None,'adopted':True};waiting.remove(name);continue
                cmd=[sys.executable,'-B','-u',str(ROOT/'scripts/train_m5_constraints.py'),'--config',str(root/'configs'/(name+'.json')),
                  '--subset',protocol['subset'],'--run-dir',str(run)]
                if (run/'run_config.json').exists():
                    if not (run/'checkpoints/latest.pt').exists():raise RuntimeError('Missing recovery checkpoint: '+name)
                    cmd+=['--resume']
                else:cmd+=['--parent',protocol['parent_checkpoint']]
                if a.engineering_fixture:cmd+=['--engineering-fixture']
                logs[name]=(root/(name+'.log')).open('ab',buffering=0);process=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=logs[name],stderr=subprocess.STDOUT)
                workers[name]={'pid':process.pid,'process':process,'adopted':False};waiting.remove(name);print('launched',name,process.pid,flush=True)
            status('training_trials',waiting=waiting);time.sleep(2)
        for name in names:
            entry=workers.get(name)
            if entry and entry.get('process') is not None:
                if entry['process'].wait()!=0:raise RuntimeError('Terminal trial error: '+name)
            checked(name)
            if name not in completed:completed.append(name)
    try:
        run_jobs(protocol['first_wave'])
        initial=[read(root/name/'run_config.json')['initial_model_sha256'] for name in protocol['first_wave']]
        streams=[sha(root/name/'sample_trace.jsonl') for name in protocol['first_wave']]
        if len(set(initial))!=1 or len(set(streams))!=1:raise ValueError('Matched source weights/sample streams diverged')
        control=checked('G0_control')
        assessments={name:compare(checked(name),control) for name in protocol['first_wave'] if name!='G0_control'}
        atomic_json(root/'assessments/first_wave.json',assessments);combo=combination_for(assessments)
        if combo:
            status('single_factor_evidence_ready',combination=combo);run_jobs([combo]);assessments[combo]=compare(checked(combo),control)
        candidates=[(v['metrics']['geometry_score_m'],name) for name,v in assessments.items() if v['joint_development_passed']]
        decision={'provisional_recipe':min(candidates)[1] if candidates else None,'combination':combo,'assessments':assessments,
          'control_checkpoint_sha256':control['checkpoint_sha256'],'protocol_sha256':sha(root/'protocol.json'),'engineering_fixture':a.engineering_fixture,
          'M5_complete':False,'D_E_started':False,'audit_model_scores_read':False,
          'remaining':'Complete selected full curriculum, exact matched Tiny, independent Small seed and semantic/final audit requirements. Numerical recipe selection is provisional.'}
        atomic_json(root/'research_decision.json',decision);status('research_budget_complete',decision=str(root/'research_decision.json'))
    except InterruptedError:status('stopped')
    except Exception as error:
        halt();status('failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        for entry in workers.values():
            process=entry.get('process')
            if process is not None and process.poll() is None:process.wait()
        for handle in logs.values():handle.close()


if __name__=='__main__':main()
