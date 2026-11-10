#!/usr/bin/env python3
"""Freeze combination trials from existing single-factor priors, no threshold changes."""
import argparse,json,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.m3 import atomic_json,sha,validate_inputs
from training.m5_research_protocol import configuration


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);args=p.parse_args();root=args.experiment.resolve()
    if (root/'protocol.json').exists():raise FileExistsError('Protocol already frozen')
    ready=validate_inputs();prior=ROOT/'runs/m5_research_20261103';original=json.loads((prior/'protocol.json').read_text());calibration=json.loads((prior/'calibration/report.json').read_text())
    for name,h in original['code_sha256'].items():assert sha(ROOT/name)==h,name
    coeff=calibration['coefficients'];parents=original['parents'];controls={
      'R3':{'norm6_run':str(ROOT/'runs/m5_20261103/A0_tiny_control'),'norm6_diagnostic':str(prior/'diagnostics/A0_tiny_control/report.json'),
            'fk_run':str(ROOT/'runs/m5_20261103/A1_tiny_fk'),'fk_diagnostic':str(prior/'diagnostics/A1_tiny_fk/report.json')},
      'R9':{'norm6_run':str(prior/'S2_control_norm6'),'norm6_diagnostic':str(prior/'S2_control_norm6/diagnostics_terminal/report.json'),
            'fk_run':str(prior/'S2_control_fk'),'fk_diagnostic':str(prior/'S2_control_fk/diagnostics_terminal/report.json')}}
    jobs={};jobhash={}
    for level,recipe in [('low','low_cons_weak_axis'),('medium','medium_cons_weak_axis')]:
        for parent_key,tag in [('R3','S1'),('R9','S2')]:
            name=f'J_{level}_{tag}';parent=parents[parent_key]
            cfg=configuration(name,coeff['consistency'][level],coeff['axis']['low'],seed=parent['training_seed'])
            path=root/'configs'/f'{name}.json';atomic_json(path,cfg)
            job={'name':name,'config_path':str(path),'config_sha256':sha(path),'parent':parent['path'],'parent_sha256':parent['sha256']}
            jp=root/'jobs'/f'{name}.json';atomic_json(jp,job);jobhash[name]=sha(jp)
            jobs[name]={'recipe':recipe,'parent_key':parent_key,'training_seed':parent['training_seed']}
    code=dict(original['code_sha256']);code.update({f:sha(ROOT/f) for f in ['scripts/run_m5_combination_study.py','scripts/prepare_m5_combinations.py','scripts/diagnose_m5_contact_tail.py']})
    for name in code:
        target=root/'source_at_launch'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    protocol={'schema':'m5-combination-study-v3','frozen_at_utc':datetime.now(timezone.utc).isoformat(),'m2_fingerprint':ready['fingerprint'],
      'manifest':original['manifest'],'manifest_sha256':original['manifest_sha256'],'parents':parents,'controls':controls,
      'code_sha256':code,'jobs':jobs,'job_sha256':jobhash,'max_parallel':4,'new_updates_per_job':10000,'total_new_updates':40000,
      'prior_protocol_sha256':sha(prior/'protocol.json'),'prior_repeat_decision_sha256':sha(prior/'research_decision.json'),
      'criteria':'Unchanged compare() gates from the preceding v2 study; both independent training seeds must pass.',
      'selection':'No new coefficient search: combine the two already-calibrated consistency levels with the previously selected weak axis weight.',
      'scope':'M5 short-motion engineering research; no audit release or M6 training.'}
    atomic_json(root/'protocol.json',protocol);print(json.dumps({'jobs':jobs,'total_new_updates':40000,'protocol_sha256':sha(root/'protocol.json')},indent=2),flush=True)


if __name__=='__main__':main()
