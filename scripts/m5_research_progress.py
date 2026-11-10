#!/usr/bin/env python3
"""Read-only status for the parallel M5-A research experiment."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,default=ROOT/'runs/m5_research_20261103');args=p.parse_args();root=args.experiment
path=root/'parallel_status.json'
if not path.exists():print('Preparation in progress; formal parallel controller has not started.')
else:
    state=json.loads(path.read_text());print('Phase:',state['phase'],'controller PID:',state['pid'],'completed:',state['completed'])
    for entry in state['active']:
        trainer=entry.get('trainer',{});print(entry['name'],'job PID',entry['pid'],entry.get('job',{}).get('phase'),
          'step',trainer.get('step'),'/',trainer.get('target_steps'),'loss',trainer.get('last_train',{}).get('loss'))
    if 'error' in state:print('ERROR:',state['error'])
for name in ['assessments/first_wave.json','assessments/candidate.json','research_decision.json']:
    path=root/name
    if path.exists():print(name,json.loads(path.read_text()))
print('TensorBoard: http://localhost:6010; original M5-A results remain on port 6009.')
