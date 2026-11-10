#!/usr/bin/env python3
"""One isolated research job: train, diagnose, then matched-noise text controls."""
import argparse,json,os,signal,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.m3 import atomic_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--job',type=Path,required=True);args=p.parse_args()
    root=args.experiment.resolve();job=json.loads(args.job.read_text());run=root/job['name'];stop=[False];child=[None]
    protocol=json.loads((root/'protocol.json').read_text())
    if job['parent_sha256'] not in {v['sha256'] for v in protocol['parents'].values()}:raise ValueError('Parent is not an allowed frozen source')
    for f,h in protocol['code_sha256'].items():
        if sha(ROOT/f)!=h:raise ValueError(f'Frozen research code changed: {f}')
    if sha(Path(job['parent']))!=job['parent_sha256']:raise ValueError('Parent checkpoint changed')
    if sha(Path(job['config_path']))!=job['config_sha256']:raise ValueError('Frozen job configuration changed')
    if sha(Path(protocol['manifest']))!=protocol['manifest_sha256']:raise ValueError('Anchor manifest changed')
    run.mkdir(parents=True,exist_ok=True)
    def status(phase,**details):atomic_json(run/'job_status.json',{'phase':phase,'pid':os.getpid(),'updated_at_utc':datetime.now(timezone.utc).isoformat(),**details})
    def halt(*_):
        stop[0]=True
        if child[0] is not None and child[0].poll() is None:child[0].send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    train=[sys.executable,'-B','-u',str(ROOT/'scripts/train_m5_research.py'),'--config',job['config_path'],'--subset',protocol['manifest'],'--run-dir',str(run)]
    if (run/'run_config.json').exists():train+=['--resume']
    else:train+=['--parent',job['parent']]
    tasks=[('training',run/'COMPLETED.json',train),
      ('geometry_diagnostics',run/'diagnostics_terminal/report.json',[sys.executable,'-B','-u',str(ROOT/'scripts/diagnose_m5_geometry.py'),
        '--run-dir',str(run),'--output',str(run/'diagnostics_terminal')]),
      ('text_controls',run/'controlled_terminal/report.json',[sys.executable,'-B','-u',str(ROOT/'scripts/probe_m5_text.py'),'--run-dir',str(run)])]
    try:
        for phase,marker,command in tasks:
            if stop[0]:break
            if marker.exists():continue
            with (run/f'{phase}.log').open('ab',buffering=0) as log:
                child[0]=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                status(phase,child_pid=child[0].pid,command=command);code=child[0].wait()
            if stop[0]:break
            if code or not marker.exists():raise RuntimeError(f'{phase} failed: {code}')
        if stop[0]:status('stopped');return
        done=json.loads((run/'COMPLETED.json').read_text());checkpoint=run/'checkpoints/latest.pt';checksum=sha(checkpoint)
        if done['step']!=30000 or checksum!=done['terminal_checkpoint_sha256']:raise ValueError('Terminal state mismatch')
        for name in ('diagnostics_terminal/report.json','controlled_terminal/report.json'):
            if json.loads((run/name).read_text())['checkpoint_sha256']!=checksum:raise ValueError('Assessment/checkpoint mismatch')
        atomic_json(run/'JOB_COMPLETE.json',{'phase':'research_job_complete_not_M5_acceptance','checkpoint_sha256':checksum,
          'completed_at_utc':datetime.now(timezone.utc).isoformat(),'job_sha256':sha(args.job),'protocol_sha256':sha(root/'protocol.json')})
        status('complete')
    except Exception as error:status('failed',error=repr(error));raise


if __name__=='__main__':main()
