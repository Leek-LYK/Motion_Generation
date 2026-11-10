#!/usr/bin/env python3
"""Development-only CFG sweep; thresholds and locked audit remain unchanged."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data_pipeline.numeric_common import atomic_json,sha_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args()
    run=args.run_dir.resolve();parent=run/'cfg_readiness';parent.mkdir(exist_ok=True)
    config=json.loads((run/'run_config.json').read_text())['config']
    protocol_hash=sha_file(ROOT/'runs/m4_research_20261103/M5_entry_protocol.json')
    weights=[1.0,1.5,2.0] if config['text_dropout']>0 else [1.0]
    reports={}
    for weight in weights:
        if weight==1 and (run/'readiness/report.json').exists():
            path=run/'readiness/report.json'
        else:
            out=parent/f'cfg{weight:g}';path=out/'report.json'
            if not path.exists():
                command=[sys.executable,'-B','-u',str(ROOT/'scripts/assess_m4_readiness.py'),'--run-dir',str(run),
                         '--output',str(out),'--guidance',str(weight)]
                with (parent/f'cfg{weight:g}.log').open('ab',buffering=0) as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads(path.read_text())
        if report['checkpoint_sha256']!=sha_file(run/'checkpoints/best_val_ema.pt'):
            raise ValueError('Do not mix gate results from different checkpoints')
        if report['protocol_sha256']!=protocol_hash:raise ValueError('Entry thresholds changed')
        if report.get('source_split','val')!='val':raise ValueError('CFG selection may only use development data')
        reports[str(weight)]=report
        print('cfg',weight,'amplitude pass',report['amplitude_gate_passed'],flush=True)
        if report['amplitude_gate_passed'] and report['all_finite']:
            atomic_json(parent/'selection.json',{'development_only':True,'selected_guidance':weight,
                'amplitude_candidate_exists':True,'M5_ready':False,'reports':reports,
                'decision':'Smallest tested guidance passing unchanged gate; stronger guidance need not be evaluated'})
            return
    passing=[w for w in weights if reports[str(w)]['amplitude_gate_passed'] and reports[str(w)]['all_finite']]
    atomic_json(parent/'selection.json',{'development_only':True,'selected_guidance':min(passing) if passing else None,
                'amplitude_candidate_exists':bool(passing),'M5_ready':False,'reports':reports,
                'decision':'Prefer smallest guidance satisfying unchanged amplitude gate; still requires other evidence'})


if __name__=='__main__':main()
