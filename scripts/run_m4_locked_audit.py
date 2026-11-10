#!/usr/bin/env python3
"""Evaluate all predeclared M4 final-audit conditions sequentially, no model selection."""
import os,json,hashlib,subprocess,sys,fcntl
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data_pipeline.numeric_common import atomic_json


def main():
    root=ROOT/'runs/m4_research_20261103';release_path=root/'audit_release.json'
    release=json.loads(release_path.read_text());lock=(root/'audit_execution.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(phase,**details):
        atomic_json(root/'audit_execution_status.json',{'phase':phase,'pid':os.getpid(),
          'updated_at_utc':datetime.now(timezone.utc).isoformat(),
          'release_sha256':hashlib.sha256(release_path.read_bytes()).hexdigest(),**details})
    results=[]
    try:
        for run in release['runs']:
            for heading in release['conditioning_headings']:
                label='native' if heading is None else 'heading0'
                out=root/run/'locked_audit_v1'/label;marker=out/'report.json'
                if not marker.exists():
                    command=[sys.executable,'-B','-u',str(ROOT/'scripts/assess_m4_readiness.py'),
                      '--run-dir',str(root/run),'--output',str(out),'--guidance',str(release['guidance']),
                      '--manifest',str(root/'locked_audit.json'),'--split','test_repetition',
                      '--seeds',*map(str,release['sampling_seeds']),'--decoder',release['decoder'],'--save-samples']
                    if heading is not None:command+=['--heading',str(heading)]
                    log_path=root/f'audit_{run}_{label}.log'
                    with log_path.open('ab',buffering=0) as log:
                        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                        status('running',run=run,condition=label,child_pid=process.pid,command=command)
                        print(run,label,process.pid,flush=True);code=process.wait()
                    if code or not marker.exists():raise RuntimeError(f'{run}/{label} failed: {code}; inspect {log_path}')
                report=json.loads(marker.read_text())
                assert report['checkpoint_sha256'] in release['allowed_checkpoint_sha256']
                assert report['source_manifest_sha256']==release['audit_manifest_sha256']
                assert report['protocol_sha256']==release['protocol_sha256']
                assert report['seeds']==release['sampling_seeds'] and report['conditioning_heading']==heading
                results.append({'run':run,'condition':label,'report':str(marker),
                  'amplitude_gate_passed':report['amplitude_gate_passed'],'all_exported_qpos_valid':report['all_exported_qpos_valid']})
        status('audit_complete_requires_final_review',results=results)
    except Exception as error:
        status('failed',error=repr(error),completed=results);raise


if __name__=='__main__':main()
