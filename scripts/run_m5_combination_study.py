#!/usr/bin/env python3
"""Parallel two-coefficient/two-training-seed combination study, frozen at launch."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.m3 import atomic_json,sha
from training.m5_research_protocol import evidence,compare


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);args=p.parse_args();root=args.experiment.resolve()
    lock=(root/'study.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=json.loads((root/'protocol.json').read_text())
    for f,h in protocol['code_sha256'].items():
        if sha(ROOT/f)!=h:raise ValueError(f'Frozen study code changed: {f}')
    for name,h in protocol['job_sha256'].items():
        if sha(root/'jobs'/f'{name}.json')!=h:raise ValueError('Frozen job changed')
    stop=[False];active={};complete=[];waiting=list(protocol['jobs'])
    def status(phase,**details):
        rows=[]
        for name,proc in active.items():
            row={'name':name,'pid':proc.pid}
            for field,path in [('job',root/name/'job_status.json'),('trainer',root/name/'status.json')]:
                if path.exists():row[field]=json.loads(path.read_text())
            rows.append(row)
        atomic_json(root/'study_status.json',{'phase':phase,'pid':os.getpid(),'updated_at_utc':datetime.now(timezone.utc).isoformat(),
          'active':rows,'completed':complete,'waiting':waiting,'M5_complete':False,'protocol_sha256':sha(root/'protocol.json'),**details})
    def halt(*_):
        stop[0]=True
        for proc in active.values():
            if proc.poll() is None:proc.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    try:
        while waiting or active:
            if stop[0]:
                for proc in active.values():proc.wait()
                active.clear();status('stopped');return
            for name in list(waiting):
                marker=root/name/'JOB_COMPLETE.json'
                if marker.exists():
                    if json.loads(marker.read_text())['protocol_sha256']!=sha(root/'protocol.json'):raise ValueError('Stale completed job')
                    waiting.remove(name);complete.append(name);continue
                if len(active)>=protocol['max_parallel']:break
                free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0])
                if free<24576:break
                command=[sys.executable,'-B','-u',str(ROOT/'scripts/run_m5_research_job.py'),'--experiment',str(root),'--job',str(root/'jobs'/f'{name}.json')]
                with (root/f'{name}_launcher.log').open('ab',buffering=0) as log:
                    proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                active[name]=proc;waiting.remove(name);print('launched',name,proc.pid,flush=True)
            for name,proc in list(active.items()):
                code=proc.poll()
                if code is not None:
                    del active[name]
                    if code or not (root/name/'JOB_COMPLETE.json').exists():raise RuntimeError(f'{name} failed ({code})')
                    complete.append(name)
            status('parallel_combinations')
            if waiting or active:time.sleep(10)
        results={}
        for name,spec in protocol['jobs'].items():
            controls=protocol['controls'][spec['parent_key']]
            base6=evidence(controls['norm6_run'],controls['norm6_diagnostic'])
            base_fk=evidence(controls['fk_run'],controls['fk_diagnostic'])
            result=compare(evidence(root/name),base6,base_fk)
            result.update(recipe=spec['recipe'],training_seed=spec['training_seed'])
            atomic_json(root/'assessments'/f'{name}.json',result);results[name]=result
        recipes={}
        for recipe in ('low_cons_weak_axis','medium_cons_weak_axis'):
            selected=[v for v in results.values() if v['recipe']==recipe]
            recipes[recipe]={'both_training_seeds_passed':len(selected)==2 and all(v['joint_gate_passed'] for v in selected),
              'runs':[v['run'] for v in selected],'mean_geometry_score_m':sum(v['metrics']['geometry_score_m'] for v in selected)/len(selected)}
        eligible=[(r,v) for r,v in recipes.items() if v['both_training_seeds_passed']]
        chosen=min(eligible,key=lambda x:x[1]['mean_geometry_score_m'])[0] if eligible else None
        decision={'recipes':recipes,'chosen_recipe':chosen,'results':results,'M5_complete':False,'locked_audit_accessed':False,
          'phase':'two_seed_combination_candidate' if chosen else 'no_two_seed_combination_candidate',
          'next_step':'Validate the Small-stage implementation and freeze the selected training recipe; otherwise diagnose failed cases before a new bounded study.'}
        atomic_json(root/'study_decision.json',decision);status(decision['phase'])
    except Exception as error:
        halt()
        for proc in active.values():proc.wait()
        active.clear();status('failed',error=repr(error));raise


if __name__=='__main__':main()
