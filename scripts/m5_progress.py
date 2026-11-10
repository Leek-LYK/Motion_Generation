#!/usr/bin/env python3
"""Read-only current M5 stage, training step and routing decisions."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,default=ROOT/'runs/m5_20261103');args=p.parse_args();root=args.experiment
status=root/'queue_status.json'
if not status.exists():print('M5-0 preparation; formal queue not yet launched')
else:
    d=json.loads(status.read_text());print(json.dumps(d,ensure_ascii=False,indent=2))
    if d.get('run'):
        run=root/d['run'];s=run/'status.json'
        if s.exists():
            current=json.loads(s.read_text());print('Current runner:',json.dumps(current,ensure_ascii=False,indent=2))
            evaluation=current.get('evaluation')
            if evaluation and (Path(evaluation)/'progress.json').exists():print('Generation:',(Path(evaluation)/'progress.json').read_text())
for name in ['data/READY.json','benchmark/report.json','verification/report.json','M5_exit_decision.json']:
    path=root/name
    if path.exists():
        value=json.loads(path.read_text());print(name,{k:value[k] for k in ('passed','actual_counts','recommended_microbatch','M5_complete','phase') if k in value})
print('TensorBoard: http://localhost:6009 (forward port 6009 in the IDE when remote)')
