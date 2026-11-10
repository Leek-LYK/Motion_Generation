#!/usr/bin/env python3
"""Assess valid CFG choices after each variant's primary evaluation, without interrupting training."""
import json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data_pipeline.numeric_common import atomic_json


def main():
    root=ROOT/'runs/m4_research_20261103'
    def status(phase,**extra):
        atomic_json(root/'guidance_status.json',{'phase':phase,'pid':os.getpid(),'updated_at_utc':datetime.now(timezone.utc).isoformat(),**extra})
    for name in ['R4_D_mse','R3_D_high_noise']:
        run=root/name;status('waiting_for_variant_evaluation',run=name)
        while not (run/'readiness/report.json').exists():
            queue=json.loads((root/'variants_status.json').read_text())
            process=Path('/proc')/str(queue['pid'])
            if queue['phase'] in ['failed','stopped','controls_require_attention']:
                status('variant_requires_attention',queue=queue);return
            if not process.exists():
                status('variant_queue_terminal_before_this_variant',queue=queue);return
            time.sleep(10)
        with (root/f'{name}_guidance.log').open('ab',buffering=0) as log:
            child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts/check_m4_cfg_gate.py'),'--run-dir',str(run)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            status('testing_guidance',run=name,child_pid=child.pid);code=child.wait()
        if code:status('failed',run=name,exit_code=code);return
        selection=json.loads((run/'cfg_readiness/selection.json').read_text())
        if selection['amplitude_candidate_exists']:
            status('candidate_requires_remaining_evidence',run=name,guidance=selection['selected_guidance']);return
    status('guidance_sweeps_completed_no_candidate')


if __name__=='__main__':main()
