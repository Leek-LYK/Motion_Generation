#!/usr/bin/env python3
"""Freeze bounded independent trials and dependency rules after diagnostic verification."""
import argparse,json,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.m3 import atomic_json,sha,validate_inputs
from training.m5_research_protocol import configuration


def read(p):return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);args=p.parse_args();root=args.experiment.resolve()
    if (root/'protocol.json').exists():raise FileExistsError('Research protocol already frozen')
    ready=validate_inputs();old=read(ROOT/'runs/m5_20261103/protocol/launch.json')
    for name,h in old['code_sha256'].items():assert sha(ROOT/name)==h,name
    verify=read(root/'verification/report.json');calibration=read(root/'calibration/report.json');parallel=read(root/'verification/parallel_report.json');logic=read(root/'verification/decision_logic.json')
    assert verify['passed'] and calibration['passed'] and parallel['passed'] and logic['passed']
    for name,h in verify['code_sha256'].items():assert sha(ROOT/name)==h,name
    for name,h in parallel['source_code_sha256'].items():assert sha(ROOT/name)==h,name
    assert calibration['loss_code_sha256']==sha(ROOT/'training/m5_research_losses.py')
    assert logic['code_sha256']==sha(ROOT/'training/m5_research_protocol.py')
    coeff=calibration['coefficients'];parents={}
    for key,folder in [('R3','R3_D_high_noise'),('R9','R9_D_high_noise_seed20261104')]:
        run=ROOT/'runs/m4_research_20261103'/folder;path=run/'checkpoints/latest.pt';done=read(run/'COMPLETED.json');assert done['step']==20000
        parents[key]={'path':str(path),'sha256':sha(path),'source_step':20000,'training_seed':read(run/'config.json')['seed']}
    assert parents['R3']['sha256']==calibration['parent_sha256']
    cases=[('R1_cons_g10',coeff['consistency']['low'],0.),('R2_cons_g30',coeff['consistency']['medium'],0.),
           ('R3_axis_g10',0.,coeff['axis']['low']),('R4_axis_g30',0.,coeff['axis']['medium'])]
    jobs={}
    for name,c,a in cases:
        cfg=configuration(name,c,a);config=root/'configs'/f'{name}.json';atomic_json(config,cfg)
        job={'name':name,'config_path':str(config),'config_sha256':sha(config),
             'parent':parents['R3']['path'],'parent_sha256':parents['R3']['sha256']}
        path=root/'jobs'/f'{name}.json';atomic_json(path,job);jobs[name]=sha(path)
    added=[*list((ROOT/'training').glob('m5_research*.py')),*list((ROOT/'scripts').glob('*m5*research*.py')),
           ROOT/'scripts/diagnose_m5_geometry.py',ROOT/'scripts/verify_m5_parallel.py']
    hashes=dict(old['code_sha256']);hashes.update({str(f.relative_to(ROOT)):sha(f) for f in added})
    snapshot=root/'source_at_launch'
    for name in hashes:
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    manifest=ROOT/'runs/m5_20261103/data/D_anchor_1024.json'
    baseline_files=[ROOT/'runs/m5_20261103'/n/'evaluations/generation_full_030000/report.json' for n in ['A0_tiny_control','A1_tiny_fk','A1b_tiny_fk025']]
    baseline_files += [root/'diagnostics'/n/'report.json' for n in ['A0_tiny_control','A1_tiny_fk','A1b_tiny_fk025']]
    protocol={'schema':'m5-a-research-v2','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
      'scope':'Investigate supervised FK versus self-consistency and G1 hinge feasibility; retain the original failed M5-A protocol unchanged.',
      'authorization':'User explicitly requested independent experiments in parallel, with dependent experiments sequenced.',
      'm2_fingerprint':ready['fingerprint'],'manifest':str(manifest),'manifest_sha256':sha(manifest),'parents':parents,
      'first_wave':[x[0] for x in cases],'first_wave_job_sha256':jobs,'max_parallel':4,'minimum_free_memory_mib_for_launch':24576,
      'new_updates_per_run':10000,'first_wave_updates':40000,'conditional_combination_updates':10000,
      'conditional_repeat_updates':30000,'maximum_total_research_updates':80000,
      'calibrated_coefficients':coeff,'calibration_sha256':sha(root/'calibration/report.json'),
      'consistency_term':'mean(sqrt(||FK_position - predicted_position||^2 + (0.001m)^2) - 0.001m), masked valid frames/joints',
      'axis_term':'mean(0.5 * ||(O_f2q @ R_local) @ hinge_axis - hinge_axis||^2), 29 real hinges; no clamping/limits/smoothing in the loss',
      'base_objective':'Unchanged norm6 + 1.0 * supervised normalized FK to ground truth',
      'calibration_policy':'Fixed coefficients from median body-parameter gradient norm ratios on 8 training-only batches; target initial fractions 0.1 and 0.3.',
      'comparisons':{'terminal_step':30000,'generation_seeds':[0,42,1234],'headings':[None,0.0],'guidance':2,'ddim_steps':100,
         'historical_self_gate':'<= 0.9 * same-seed norm6 continuation control; original A0 threshold is not reduced',
         'supervised_fk_guard':'mean per heading <=1.10 * same-seed FK control and each fixed t <=1.25 * control',
         'norm6_guard':'mean fixed-t denoising <=1.05 * FK control',
         'projection_guard':'macro mean per-clip node p95 <=1.05 * norm6 control',
         'penetration_guard':'deep-frame macro <=1.1 * norm6 control + 0.0001; max penetration <=0.10m',
         'rotation_guard':'reference-calibrated exceedance macro <=1.1 * norm6 control +0.00001',
         'core_guard':'all four classes pass original amplitude thresholds at both headings, fraction regression no more than 5 percentage points'},
      'dependency_rules':[
         'Wait for all four independent runs before selection; do not choose by finish time.',
         'If one trial passes joint checks, select minimum frozen geometry score; otherwise combine only if consistency reduces self error at least 5% vs A1 and the axis direction reduces axis energy 20% and projection p95 10%, with supervised/core guards.',
         'Combination restarts from the original R3 20k full state, using the selected single-factor coefficients; it is not a merge of weights.',
         'Only after a primary joint candidate passes: run norm6 control, FK control and fixed-weight treatment concurrently from independent R9 20k full states.',
         'Repeat uses the same numeric coefficients, no recalibration to the second seed.',
         'No locked audit access, no automatic bypass of the original main-matrix failure; emit evidence and a recipe for subsequent Small-stage review.'],
      'code_sha256':hashes,'baseline_evidence_sha256':{str(p):sha(p) for p in baseline_files},
      'verification_sha256':{name:sha(root/name) for name in ['verification/report.json','verification/parallel_report.json','verification/decision_logic.json']},
      'M5_complete':False,'locked_audit_accessed':False}
    atomic_json(root/'protocol.json',protocol)
    atomic_json(root/'parallel_status.json',{'phase':'ready_to_launch','first_wave':protocol['first_wave'],'max_parallel':4,'M5_complete':False})
    print(json.dumps({'ready':True,'first_wave':protocol['first_wave'],'coefficients':coeff,'maximum_total_updates':80000,'protocol_sha256':sha(root/'protocol.json')},indent=2),flush=True)


if __name__=='__main__':main()
