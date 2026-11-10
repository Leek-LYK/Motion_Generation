#!/usr/bin/env python3
"""Repeat the full D warmup -> high-noise curriculum with its frozen CFG=2 evaluation."""
import json,os,subprocess,sys,time,signal,fcntl
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data_pipeline.numeric_common import atomic_json,sha_file


def main():
    root=ROOT/'runs/m4_research_20261103';lock=(root/'high_noise_repeat.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    warm=root/'R8_D_warmup_seed20261104';run=root/'R9_D_high_noise_seed20261104';stop=[False];child=[None]
    def halt(*_):
        stop[0]=True
        if child[0] is not None and child[0].poll() is None:child[0].send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    def status(phase,**extra):
        atomic_json(root/'high_noise_repeat_status.json',{'phase':phase,'pid':os.getpid(),'updated_at_utc':datetime.now(timezone.utc).isoformat(),**extra})
    status('waiting_for_independent_D_warmup')
    while not (warm/'COMPLETED.json').exists() and not stop[0]:
        registered=json.loads((root/'D_repeat_warmup_process.json').read_text());p=Path('/proc')/str(registered['pid'])
        if not p.exists():status('warmup_process_missing_without_completion');return
        time.sleep(10)
    if stop[0]:status('stopped');return
    if json.loads((warm/'COMPLETED.json').read_text())['step']!=5000:raise ValueError('Warmup budget mismatch')
    if not run.exists():
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/fork_m4_continuation.py'),'--parent',str(warm),'--run-dir',str(run),
                        '--stage','R9_D_high_noise_seed20261104','--max-steps','20000'],cwd=ROOT,check=True)
    if not (run/'replication_prepared.json').exists():
        import torch
        source_config=json.loads((root/'R3_D_high_noise/config.json').read_text())
        config=dict(source_config,seed=20261104,stage='R9_D_high_noise_seed20261104')
        code={name:sha_file(ROOT/name) for name in ['training/m4_research.py','scripts/train_m4_research.py']}
        for filename in ['latest.pt','best_val_raw.pt','best_val_ema.pt']:
            path=run/'checkpoints'/filename;s=torch.load(path,map_location='cpu',weights_only=False)
            if s['step']>5000:raise ValueError('Do not reconfigure a trained repeat')
            s['config']=config;s['identity']['research_code_sha256']=code;torch.save(s,path)
        context=json.loads((run/'run_config.json').read_text());context['config']=config;context['identity']['research_code_sha256']=code
        atomic_json(run/'config.json',config);atomic_json(run/'run_config.json',context)
        atomic_json(run/'replication_prepared.json',{'primary_run':'R3_D_high_noise','warmup':str(warm),'source_step':5000,
                    'training_seed':20261104,'inference_guidance':2.0,'configuration_changes_from_primary':['seed','stage']})
    tasks=[('training',run/'COMPLETED.json',[sys.executable,'-B','-u',str(ROOT/'scripts/train_m4_research.py'),
                '--config',str(run/'config.json'),'--subset',str(run/'subset.json'),'--run-dir',str(run),'--resume']),
           ('evaluation',run/'final_evaluation/COMPLETE.json',[sys.executable,'-B','-u',str(ROOT/'scripts/evaluate_m4.py'),'--run-dir',str(run)]),
           ('native_cfg2',run/'readiness_cfg2/report.json',[sys.executable,'-B','-u',str(ROOT/'scripts/assess_m4_readiness.py'),
                '--run-dir',str(run),'--guidance','2','--output',str(run/'readiness_cfg2')]),
           ('heading0_cfg2',run/'readiness_heading0_cfg2/report.json',[sys.executable,'-B','-u',str(ROOT/'scripts/assess_m4_readiness.py'),
                '--run-dir',str(run),'--guidance','2','--heading','0','--output',str(run/'readiness_heading0_cfg2')]),
           ('controlled_text_cfg2',run/'controlled_three_seeds_cfg2/report.json',[sys.executable,'-B','-u',str(ROOT/'scripts/probe_m4_controlled_text.py'),
                '--run-dir',str(run),'--guidance','2','--output',str(run/'controlled_three_seeds_cfg2')])]
    try:
        for phase,marker,command in tasks:
            if stop[0]:break
            if marker.exists():continue
            with (root/f'R9_{phase}.log').open('ab',buffering=0) as log:
                child[0]=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                status(phase,child_pid=child[0].pid);print(phase,child[0].pid,flush=True);code=child[0].wait()
            if stop[0]:break
            if code or not marker.exists():raise RuntimeError(f'{phase} failed: {code}')
        status('stopped' if stop[0] else 'replication_completed_audit_locked')
    except Exception as error:status('failed',error=repr(error));raise


if __name__=='__main__':main()
