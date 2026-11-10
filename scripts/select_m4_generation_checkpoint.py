#!/usr/bin/env python3
"""Select among pre-saved milestones using unchanged development gates, never the locked audit."""
import argparse,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data_pipeline.numeric_common import atomic_json,sha_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args()
    run=args.run_dir.resolve();output=run/'generation_selection';output.mkdir(exist_ok=True)
    protocol={'version':1,'candidate_steps':[22500,25000,27500,30000],
              'rule':'Smallest saved step passing unchanged native and heading0 amplitude gates with all arrays/qpos finite',
              'development_only':True,'audit_used':False,
              'entry_protocol_sha256':sha_file(ROOT/'runs/m4_research_20261103/M5_entry_protocol.json')}
    if (output/'protocol.json').exists():
        if json.loads((output/'protocol.json').read_text())!=protocol:raise ValueError('Selection protocol changed')
    else:atomic_json(output/'protocol.json',protocol)
    rows=[]
    for step in protocol['candidate_steps']:
        checkpoint=f'step_{step:06d}.pt';file=run/'checkpoints'/checkpoint
        if not file.exists():raise FileNotFoundError(file)
        row={'step':step,'checkpoint':str(file),'checkpoint_sha256':sha_file(file)}
        for view in ['native','heading0']:
            folder=output/f'step_{step:06d}'/view;report_path=folder/'report.json'
            if not report_path.exists():
                if folder.exists():folder.rename(folder.with_name(view+'_incomplete_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')))
                command=[sys.executable,'-B','-u',str(ROOT/'scripts/assess_m4_readiness.py'),'--run-dir',str(run),
                    '--checkpoint',checkpoint,'--output',str(folder)]
                if view=='heading0':command+=['--heading','0']
                folder.parent.mkdir(parents=True,exist_ok=True)
                with (folder.parent/f'{view}.log').open('ab',buffering=0) as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            report=json.loads(report_path.read_text())
            if report['checkpoint_sha256']!=row['checkpoint_sha256']:raise ValueError('Checkpoint changed')
            row[view]={'passed':report['amplitude_gate_passed'] and report['all_finite'],'groups':report['groups'],'report':str(report_path)}
            print('step',step,view,row[view]['passed'],flush=True)
            if not row[view]['passed']:
                break
        row['passed']=row.get('native',{}).get('passed',False) and row.get('heading0',{}).get('passed',False)
        rows.append(row)
        atomic_json(output/'selection.json',{'protocol':protocol,'results':rows,'selected_checkpoint':row['checkpoint'] if row['passed'] else None,
                                             'M5_ready':False,'remaining':'Independent-seed, qualitative review and locked audit'})
        if row['passed']:return
    print('No milestone passed both development gates',flush=True)


if __name__=='__main__':main()
