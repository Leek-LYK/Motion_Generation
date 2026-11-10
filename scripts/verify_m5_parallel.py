#!/usr/bin/env python3
"""Run four short independent runner fixtures concurrently and verify shared draws."""
import os,json,sys,subprocess,time
from pathlib import Path
import argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.m3 import atomic_json,sha
from training.m5_research_protocol import configuration


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);args=p.parse_args();root=args.experiment.resolve()
    calibration=json.loads((root/'calibration/report.json').read_text());assert calibration['passed']
    if calibration['loss_code_sha256']!=sha(ROOT/'training/m5_research_losses.py'):raise ValueError('Calibration code changed')
    manifest=json.loads((ROOT/'runs/m5_20261103/data/D_anchor_1024.json').read_text())
    val=[next(r for r in manifest['val'] if r['group']==g) for g in ('walk','jog','jump','raise_arms')]
    fixture=dict(manifest,val=val,monitor=val);out=root/'verification/parallel_pilot';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'subset.json',fixture);coeff=calibration['coefficients']
    cases=[('C_low',coeff['consistency']['low'],0.),('C_medium',coeff['consistency']['medium'],0.),
           ('H_low',0.,coeff['axis']['low']),('H_medium',0.,coeff['axis']['medium'])]
    processes=[];start=time.monotonic();parent=ROOT/'runs/m4_research_20261103/R3_D_high_noise/checkpoints/latest.pt'
    for name,c,a in cases:
        cfg=configuration('pilot_'+name,c,a);cfg.update(max_steps=20016,minimum_steps=20016,eval_every=100000,generation_every=100000,
          full_generation_every=100000,checkpoint_every=100000,log_every=1,ddim_steps=4,full_seeds=[42],eval_batch_size=4,verification_only=True)
        path=out/(name+'.json');atomic_json(path,cfg)
        command=[sys.executable,'-B','-u',str(ROOT/'scripts/train_m5_research.py'),'--config',str(path),'--subset',str(out/'subset.json'),
          '--run-dir',str(out/name),'--parent',str(parent)]
        log=(out/(name+'.log')).open('wb',buffering=0)
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);log.close()
        processes.append((name,process));print('pilot launched',name,process.pid,flush=True)
    for name,process in processes:
        code=process.wait()
        if code:raise RuntimeError(f'{name} failed: {code}; see {out/name}.log')
    checks={};traces=[]
    for name,_ in processes:
        run=out/name;completion=json.loads((run/'COMPLETED.json').read_text());assert completion['step']==20016 and completion['updates_this_stage']==16
        report=json.loads((run/'evaluations/generation_full_020016/report.json').read_text());assert report['all_finite'] and report['all_valid_qpos']
        traces.append((run/'sample_trace.jsonl').read_bytes())
        checks[name]={'checkpoint_sha256':completion['terminal_checkpoint_sha256'],'updates':16,'finite_exports':True,
          'source_sha256':completion['identity']['code_sha256']}
    assert all(v==traces[0] for v in traces)
    original=(ROOT/'runs/m5_20261103/A1_tiny_fk/sample_trace.jsonl').read_bytes().splitlines(keepends=True)
    assert traces[0]==b''.join(original[:16])
    result={'passed':True,'concurrent_training_processes':4,'updates_per_process':16,'total_updates':64,
      'elapsed_wall_s':time.monotonic()-start,'sample_traces_equal_to_historical_A1_prefix':True,'checks':checks,
      'source_code_sha256':{f:sha(ROOT/f) for f in ['scripts/verify_m5_parallel.py','scripts/train_m5_research.py','training/m5_research.py',
        'training/m5_research_losses.py','training/m5_research_evaluation.py']},'note':'Fixtures verify execution only; outputs are not ranked as experiments.'}
    atomic_json(root/'verification/parallel_report.json',result);print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2),flush=True)


if __name__=='__main__':main()
