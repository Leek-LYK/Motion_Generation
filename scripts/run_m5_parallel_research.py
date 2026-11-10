#!/usr/bin/env python3
"""Bounded parallel independent trials, followed by evidence-dependent stages."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.m3 import atomic_json,sha
from training.m5_research_protocol import configuration,evidence,compare


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);args=p.parse_args();root=args.experiment.resolve()
    lock=(root/'parallel.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=json.loads((root/'protocol.json').read_text());stop=[False];active={};completed=[]
    for f,h in protocol['code_sha256'].items():
        if sha(ROOT/f)!=h:raise ValueError(f'Frozen code changed: {f}')
    for name,h in protocol['first_wave_job_sha256'].items():
        if sha(root/'jobs'/f'{name}.json')!=h:raise ValueError(f'Frozen first-wave job changed: {name}')
    for path,h in protocol['baseline_evidence_sha256'].items():
        if sha(Path(path))!=h:raise ValueError(f'Frozen baseline evidence changed: {path}')
    def status(phase,**details):
        states=[]
        for name,process in active.items():
            row={'name':name,'pid':process.pid};path=root/name/'job_status.json'
            if path.exists():row['job']=json.loads(path.read_text())
            path=root/name/'status.json'
            if path.exists():row['trainer']=json.loads(path.read_text())
            states.append(row)
        atomic_json(root/'parallel_status.json',{'phase':phase,'pid':os.getpid(),'updated_at_utc':datetime.now(timezone.utc).isoformat(),
          'active':states,'completed':completed,'max_parallel':protocol['max_parallel'],'protocol_sha256':sha(root/'protocol.json'),**details})
    def halt(*_):
        stop[0]=True
        for process in active.values():
            if process.poll() is None:process.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    def put_job(name,cfg,parent_key):
        parent=protocol['parents'][parent_key];config=root/'configs'/f'{name}.json';job_path=root/'jobs'/f'{name}.json'
        if config.exists():
            if json.loads(config.read_text())!=cfg:raise ValueError('Immutable configuration changed')
        else:atomic_json(config,cfg)
        job={'name':name,'config_path':str(config),'config_sha256':sha(config),'parent':parent['path'],'parent_sha256':parent['sha256']}
        for path,value in [(job_path,job)]:
            if path.exists():
                if json.loads(path.read_text())!=value:raise ValueError('Immutable job/config changed')
            else:atomic_json(path,value)
        return name
    def run_wave(names,label):
        waiting=list(names)
        while waiting or active:
            if stop[0]:
                status('stopping');
                for process in active.values():process.wait()
                active.clear();raise InterruptedError('Stopped')
            for name in list(waiting):
                if (root/name/'JOB_COMPLETE.json').exists():
                    waiting.remove(name)
                    if name not in completed:completed.append(name)
                    continue
                if len(active)>=protocol['max_parallel']:break
                free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0])
                if free<protocol['minimum_free_memory_mib_for_launch']:break
                log=(root/f'{name}_launcher.log').open('ab',buffering=0)
                command=[sys.executable,'-B','-u',str(ROOT/'scripts/run_m5_research_job.py'),'--experiment',str(root),'--job',str(root/'jobs'/f'{name}.json')]
                process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);log.close()
                active[name]=process;waiting.remove(name);print('launched',name,process.pid,flush=True)
            failure=None
            for name,process in list(active.items()):
                code=process.poll()
                if code is not None:
                    del active[name]
                    if code or not (root/name/'JOB_COMPLETE.json').exists():failure=f'{name} failed ({code})'
                    else:completed.append(name)
            status(label,waiting=waiting)
            if failure:
                halt()
                for process in active.values():process.wait()
                active.clear();raise RuntimeError(failure)
            if waiting or active:time.sleep(10)
    def review(name,control6,control_fk):
        candidate=evidence(root/name);result=compare(candidate,control6,control_fk)
        atomic_json(root/'assessments'/f'{name}.json',result);return candidate,result
    def choose(candidates):return min(candidates,key=lambda x:(x[1]['metrics']['geometry_score_m'],x[0]))
    try:
        baseline6=evidence(ROOT/'runs/m5_20261103/A0_tiny_control',root/'diagnostics/A0_tiny_control/report.json')
        baseline_fk=evidence(ROOT/'runs/m5_20261103/A1_tiny_fk',root/'diagnostics/A1_tiny_fk/report.json')
        run_wave(protocol['first_wave'],'independent_first_wave')
        reviewed={name:review(name,baseline6,baseline_fk)[1] for name in protocol['first_wave']}
        atomic_json(root/'assessments/first_wave.json',reviewed)
        eligible=[(name,result) for name,result in reviewed.items() if result['joint_gate_passed']]
        if not eligible:
            cons=[(n,r) for n,r in reviewed.items() if n.startswith('R1_') or n.startswith('R2_')]
            axes=[(n,r) for n,r in reviewed.items() if n.startswith('R3_') or n.startswith('R4_')]
            cons=[x for x in cons if x[1]['directional_evidence']['consistency']]
            axes=[x for x in axes if x[1]['directional_evidence']['axis']]
            if cons and axes:
                c=min(cons,key=lambda x:(x[1]['metrics']['mean_fk_inconsistency_m'],x[0]));a=min(axes,key=lambda x:(x[1]['metrics']['mean_projection_node_p95_m'],x[0]))
                cc=json.loads((root/'configs'/f'{c[0]}.json').read_text());ac=json.loads((root/'configs'/f'{a[0]}.json').read_text())
                cfg=configuration('R5_joint',cc['consistency_weight'],ac['axis_weight'])
                atomic_json(root/'assessments/combination_basis.json',{'consistency_source':c[0],'axis_source':a[0],
                  'reason':'Both independent directions improved their own declared target; combined trial starts from the original R3 parent, not from a selected trained variant.'})
                put_job('R5_joint',cfg,'R3');run_wave(['R5_joint'],'dependent_combination')
                _,result=review('R5_joint',baseline6,baseline_fk)
                if result['joint_gate_passed']:eligible.append(('R5_joint',result))
        if not eligible:
            status('research_complete_no_joint_candidate',M5_complete=False,M5_main_matrix_resumed=False);return
        chosen,result=choose(eligible);cfg=json.loads((root/'configs'/f'{chosen}.json').read_text())
        atomic_json(root/'assessments/candidate.json',{'chosen':chosen,'checkpoint_sha256':sha(root/chosen/'checkpoints/latest.pt'),
          'consistency_weight':cfg['consistency_weight'],'axis_weight':cfg['axis_weight'],'assessment':result})
        names=[put_job('S2_control_norm6',configuration('S2_control_norm6',seed=20261104,norm6_control=True),'R9'),
               put_job('S2_control_fk',configuration('S2_control_fk',seed=20261104),'R9'),
               put_job('S2_treatment',configuration('S2_treatment',cfg['consistency_weight'],cfg['axis_weight'],seed=20261104),'R9')]
        run_wave(names,'dependent_independent_seed_repeat')
        repeat6=evidence(root/'S2_control_norm6');repeatfk=evidence(root/'S2_control_fk')
        _,repeat_result=review('S2_treatment',repeat6,repeatfk)
        atomic_json(root/'research_decision.json',{'primary_candidate':chosen,'repeat_candidate':'S2_treatment',
          'repeat_joint_gate_passed':repeat_result['joint_gate_passed'],'primary_assessment':result,'repeat_assessment':repeat_result,
          'M5_complete':False,'M5_main_matrix_resumed':False,'locked_audit_accessed':False,
          'next_step':'Review the candidate and adapt/freeze the Small-stage recipe before resuming B; the original failed protocol/results remain unchanged.'})
        status('repeat_completed_candidate_for_M5_handoff' if repeat_result['joint_gate_passed'] else 'repeat_failed_joint_gate',M5_complete=False,M5_main_matrix_resumed=False)
    except InterruptedError:status('stopped')
    except Exception as error:status('failed',error=repr(error));raise


if __name__=='__main__':main()
