#!/usr/bin/env python3
"""Decision logic and four sequential real-Small CLI fixtures, excluded from ranking."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import ast,copy,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.m3 import atomic_json,sha
from training.m5_data import GROUPS
from training.m5_geometry_study import build_study,compare,combination_for


def synthetic_evidence():
    group={g:{'events_with_seeds':20,'eligible':20,'median_amplitude_ratio':1.,'noncollapsed_fraction':1.,'amplitude_passed':True,
      'p95_clip_max_penetration_m':.03,'max_penetration_m':.075,'mean_penetration_over_5cm_fraction':.005,'mean_projection_node_p95_m':.025,
      'max_projection_node_error_m':.1,'mean_rotation_threshold_exceed_fraction':.0005} for g in GROUPS}
    base={'groups':group,'core_amplitude_passed':True,'all_finite':True,'all_valid_qpos':True}
    report=dict(base,pooled_projection_node_p95_m=.05,
      conditions={h:dict(copy.deepcopy(base),length_strata={'short':copy.deepcopy(base)}) for h in ['native','heading0']})
    metrics={'mean_fk_inconsistency_m':.04,'mean_projection_node_p95_m':.05,'mean_penetration_over_5cm_fraction':.02,'mean_rotation_threshold_exceed_fraction':.002,
      'pooled_projection_node_p95_m':.05,'max_penetration_m':.15,'max_projection_node_error_m':.2,'geometry_score_m':.2,'norm6_validation':.1,'supervised_fk_validation':.1}
    denoising={'geometry_by_timestep':{str(t):{'fk':.1} for t in [0,249,499,749,999]}}
    return {'run':'SYNTHETIC','checkpoint_sha256':'SYNTHETIC','metrics':metrics,'report':report,'denoising':denoising}


def main():
    root=ROOT/'runs/m5_physical_20261103';out=root/'verification/geometry_study_fixture'
    if (out/'verification_report.json').exists():raise FileExistsError('Existing verification must be inspected, not overwritten')
    control=synthetic_evidence();candidate=copy.deepcopy(control)
    for k in ['mean_fk_inconsistency_m','mean_projection_node_p95_m','mean_penetration_over_5cm_fraction','mean_rotation_threshold_exceed_fraction',
      'pooled_projection_node_p95_m','max_penetration_m','max_projection_node_error_m']:candidate['metrics'][k]*=.5
    positive=compare(candidate,control);assert positive['joint_development_passed'] and not positive['M5_complete']
    bad=copy.deepcopy(candidate);bad['metrics']['max_penetration_m']=.1000001;assert not compare(bad,control)['joint_development_passed']
    still=copy.deepcopy(candidate)
    for value in still['report']['conditions'].values():
        value['core_amplitude_passed']=False;value['length_strata']['short']['core_amplitude_passed']=False;value['groups']['squat']['noncollapsed_fraction']=0.
    assert not compare(still,control)['joint_development_passed']
    first={n:copy.deepcopy(positive) for n in ['G1_consistency','G2_legality','G3_ground']};assert combination_for(first)=='G4_all'
    first['G3_ground']['directional_evidence']['ground']=False;assert combination_for(first)=='G4_consistency_legality'
    first['G2_legality']['directional_evidence']['legality']=False;assert combination_for(first) is None
    source=root/'verification/parallel_pilot/C';parent=source/'checkpoints/step_000016.pt'
    # These are deliberately synthetic fixture constants. The real train-gradient
    # calibration tool is separately verified; no formal-calibration claim is made.
    calibration={'passed':True,'engineering_fixture':True,'formal_coefficients_usable':False,'parent_sha256':sha(parent),'step':16,
      'coefficient_provenance':'Synthetic engineering constants, not a parent-calibrated formal recipe',
      'coefficients':{k:{'low':v} for k,v in [('consistency',.01),('legality',.01),('ground',.1)]}}
    worker=ROOT/'scripts/train_m5_constraints.py';tree=ast.parse(worker.read_text());files=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CODE' for t in n.targets))
    code={f:sha(ROOT/f) for f in files};build_study(source,out,parent,calibration,code,fixture=True)
    with (out/'controller.log').open('wb') as log:
        result=subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/run_m5_geometry_study.py'),'--study-dir',str(out),'--engineering-fixture'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Geometry fixture failed; see '+str(out/'controller.log'))
    decision=json.loads((out/'research_decision.json').read_text());assert not decision['M5_complete'] and decision['engineering_fixture']
    for name in ['G0_control','G1_consistency','G2_legality','G3_ground']:
        d=json.loads((out/name/'COMPLETED.json').read_text());assert d['engineering_fixture'] and d['step']==18 and d['updates_this_stage']==2
    proof={'passed':True,'fixtures':'Synthetic decision logic plus four real-Small two-update engineering jobs. Not model-quality or human evidence.',
      'real_small_updates':8,'parallelism':1,'checks':{'positive_and_negative_joint_guards':True,'stationary_or_new_class_regression_rejected':True,
        'single_factor_combination_dependency':True,'parent_initializations_and_sampling_match':True,'controller_cli_serialization_and_terminal_evidence':True,
        'engineering_decision_cannot_complete_M5':True},'code_sha256':{p:sha(ROOT/p) for p in ['training/m5_geometry_study.py','scripts/run_m5_geometry_study.py','scripts/verify_m5_geometry_study.py']}}
    atomic_json(out/'verification_report.json',proof);print(json.dumps(proof,indent=2),flush=True)


if __name__=='__main__':main()
