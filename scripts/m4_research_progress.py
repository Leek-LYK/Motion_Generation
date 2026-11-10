#!/usr/bin/env python3
"""Read-only summary of ongoing M4 research and evidence gates."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
r=ROOT/'runs/m4_research_20261103'
for name in ['queue_status.json','variants_status.json','repeat_assessment_status.json','orientation_status.json','guidance_status.json','high_noise_repeat_status.json','audit_execution_status.json']:
    path=r/name
    print(name,json.loads(path.read_text()) if path.exists() else 'pending')
for name in ['R1_D_continue','R2_B_continue','R3_D_high_noise','R4_D_mse','R5_B_repeat_seed20261104','R6_B_late_heading','R7_B_late_heading_seed20261104','R8_D_warmup_seed20261104','R9_D_high_noise_seed20261104']:
    run=r/name;status=run/'status.json'
    if not status.exists():print(name,'prepared, not started');continue
    d=json.loads(status.read_text())
    print(name,{k:d[k] for k in ['phase','step','target_steps','best_val_ema','best_val_ema_step'] if k in d})
    path=run/'readiness/report.json'
    if path.exists():
        gate=json.loads(path.read_text());print('  amplitude gate',gate['amplitude_gate_passed'],'M5_ready',gate['M5_ready'])
    path=run/'readiness_heading0/report.json'
    if path.exists():
        gate=json.loads(path.read_text());print('  heading0 amplitude gate',gate['amplitude_gate_passed'],'seeds',gate['seeds'])
    for relative in ['cfg_readiness/cfg2/report.json','readiness_cfg2/report.json','readiness_heading0_cfg2/report.json','locked_audit_v1/native/report.json','locked_audit_v1/heading0/report.json']:
        path=run/relative
        if path.exists():
            gate=json.loads(path.read_text());print(' ',relative,'amplitude gate',gate['amplitude_gate_passed'],'CFG',gate['guidance'],'seeds',gate['seeds'])
decision=r/'M5_entry_decision.json'
if decision.exists():
    value=json.loads(decision.read_text());print('Final engineering entry decision:',value.get('M5_ready'),value.get('status'))
else:print('Final engineering entry decision: pending; M5 has not started.')
