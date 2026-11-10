#!/usr/bin/env python3
"""Read-only status for the four-stage experiment queue."""
import argparse
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    experiment=ROOT/'runs/m4_20261102'
    while True:
        queue=experiment/'queue_status.json'
        print('QUEUE:',json.loads(queue.read_text()) if queue.exists() else 'not started',flush=True)
        for name in ['A_256','B_1024','C_1024_textdrop','D_1024_heading']:
            path=experiment/name/'status.json'
            if not path.exists():print(name,'pending');continue
            d=json.loads(path.read_text())
            print(name,{k:d[k] for k in ('phase','step','target_steps','best_val_ema','best_val_ema_step','bad_evaluations') if k in d})
        if not args.watch:break
        time.sleep(10)


if __name__=='__main__':main()
