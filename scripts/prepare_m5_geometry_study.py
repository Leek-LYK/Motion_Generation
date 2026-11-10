#!/usr/bin/env python3
"""Prepare a bounded geometry study only from completed frozen 40k evidence."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ast,json,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from training.m3 import atomic_json,sha
from training.m5_geometry_study import build_study


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--parent-run',default='C_small_raw7')
    p.add_argument('--study-dir',type=Path,required=True);p.add_argument('--launch',action='store_true');a=p.parse_args();source_root=a.experiment.resolve();out=a.study_dir.resolve()
    out.relative_to(ROOT/'runs');intent_path=source_root/'protocol/bounded_geometry_research_v1.json';intent=json.loads(intent_path.read_text())
    if a.parent_run!='C_small_raw7' or intent['maximum_new_updates']!=50000 or intent['new_updates_each']!=10000:raise ValueError('Research parent/budget differs from the declared intent')
    launch_path=source_root/'protocol/launch.json';main=json.loads(launch_path.read_text());completed={}
    for name in main['jobs']:
        run=source_root/name;done=json.loads((run/'COMPLETED.json').read_text())
        if done['step']!=40000 or sha(run/'checkpoints/latest.pt')!=done['terminal_checkpoint_sha256']:raise ValueError('Original endpoint incomplete or changed: '+name)
        if sha(source_root/'configs'/(name+'.json'))!=main['config_sha256'][name] or json.loads((run/'config.json').read_text())!=json.loads((source_root/'configs'/(name+'.json')).read_text()):raise ValueError('Original configuration changed: '+name)
        completed[name]=done
    if not (source_root/'first_wave_assessment.json').exists():raise ValueError('Wait for the frozen first-wave assessment')
    assessment=json.loads((source_root/'first_wave_assessment.json').read_text())
    if assessment.get('protocol_sha256')!=sha(launch_path) or assessment.get('phase')!='first_wave_assessed':raise ValueError('First-wave assessment is stale or has another protocol')
    if set(assessment['metrics'])!=set(completed) or any(assessment['metrics'][name]['checkpoint_sha256']!=done['terminal_checkpoint_sha256'] for name,done in completed.items()):raise ValueError('Assessment/terminal checkpoint lineage differs')
    if (out/'activation.json').exists():raise FileExistsError('Prepared study already exists; inspect/reuse its controller rather than overwriting')
    verified=json.loads((source_root/'verification/geometry_study_fixture/verification_report.json').read_text());assert verified['passed']
    for f,h in verified['code_sha256'].items():
        if f=='scripts/run_m5_geometry_study.py':continue # Its added source guard is verified by the completed-fixture replay below.
        if sha(ROOT/f)!=h:raise ValueError('Verified orchestration code changed: '+f)
    guard_path=source_root/'verification/geometry_study_source_guard/report.json';guard=json.loads(guard_path.read_text())
    if not guard.get('passed') or guard['original_fixture_verification_sha256']!=sha(source_root/'verification/geometry_study_fixture/verification_report.json'):raise ValueError('Source-guard replay evidence is missing or stale')
    for f,h in guard['code_sha256'].items():
        if sha(ROOT/f)!=h:raise ValueError('Guard-verified orchestration source changed: '+f)
    source=source_root/a.parent_run;done=json.loads((source/'COMPLETED.json').read_text());parent=source/'checkpoints/latest.pt';checksum=sha(parent)
    state=torch.load(parent,map_location='cpu',weights_only=False)
    if state['step']!=40000 or done['step']!=40000 or checksum!=done['terminal_checkpoint_sha256']:raise ValueError('Research parent is not the fixed 40k terminal state')
    if any(state['config'].get(k,0.) for k in ['consistency_weight','axis_weight','ground_weight','legality_weight']):raise ValueError('The parent already uses extra constraints')
    del state;out.mkdir(parents=True,exist_ok=True)
    calibration_dir=out/'calibration'
    if not (calibration_dir/'report.json').exists():
        command=[sys.executable,'-B','-u',str(ROOT/'scripts/calibrate_m5_constraint_recipe.py'),'--experiment',str(source_root),
          '--parent-run',a.parent_run,'--checkpoint','latest.pt','--output',str(calibration_dir)]
        with (out/'calibration.log').open('ab',buffering=0) as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    calibration=json.loads((calibration_dir/'report.json').read_text())
    if not calibration.get('passed') or not calibration.get('formal_coefficients_usable') or calibration['parent_sha256']!=checksum:raise ValueError('Calibration unusable; preserve it and inspect before training')
    if calibration['script_sha256']!=sha(ROOT/'scripts/calibrate_m5_constraint_recipe.py'):raise ValueError('Calibration procedure changed')
    snapshot=out/'parent_snapshot/checkpoint.pt';snapshot.parent.mkdir(parents=True,exist_ok=True)
    if not snapshot.exists():shutil.copyfile(parent,snapshot)
    if sha(snapshot)!=checksum or sha(parent)!=checksum:raise ValueError('Parent changed during snapshot')
    for f in ['config.json','run_config.json','subset.json','lineage.json']:
        if (source/f).exists():shutil.copyfile(source/f,snapshot.parent/f)
    shutil.copytree(source/'stats/motion',snapshot.parent/'stats/motion',dirs_exist_ok=True)
    worker=ROOT/'scripts/train_m5_constraints.py';tree=ast.parse(worker.read_text());files=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CODE' for t in n.targets))
    code={f:sha(ROOT/f) for f in files};plan=build_study(source,out,snapshot,calibration,code)
    plan.update(frozen_at_utc=datetime.now(timezone.utc).isoformat(),calibration_sha256=sha(calibration_dir/'report.json'),
      original_first_wave_assessment_sha256=sha(source_root/'first_wave_assessment.json'),original_main_gate_results_preserved=True,
      orchestration_code_sha256={f:sha(ROOT/f) for f in ['training/m5_geometry_study.py','training/m5_protocol.py','scripts/run_m5_geometry_study.py','scripts/prepare_m5_geometry_study.py','scripts/calibrate_m5_constraint_recipe.py']},
      research_intent_sha256=sha(intent_path),budget_accounting={'prior_registered_new_updates':255000,'maximum_additional_new_updates':50000,'potential_total_new_updates':305000,'formal_new_updates_completed':0},
      parent_choice_scope='Additional M5 research from a declared raw7 parent, not relabelling the original failed FK diagnostic as passed.')
    atomic_json(out/'protocol.json',plan)
    for f in [*code,*plan['orchestration_code_sha256']]:
        destination=out/'source_at_launch'/f;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/f,destination)
    print(json.dumps({'prepared':str(out),'new_updates_first_wave':plan['new_updates_first_wave'],'maximum_new_updates':plan['maximum_new_updates'],'M5_complete':False},indent=2),flush=True)
    if a.launch:
        if (out/'controller_process.json').exists():raise FileExistsError('Study controller already registered; inspect the existing handle')
        command=[sys.executable,'-B','-u',str(ROOT/'scripts/run_m5_geometry_study.py'),'--study-dir',str(out)]
        with (out/'controller.log').open('ab',buffering=0) as log:process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(out/'controller_process.json',{'pid':process.pid,'command':command,'started_at_utc':datetime.now(timezone.utc).isoformat()});print('launched controller',process.pid,flush=True)


if __name__=='__main__':main()
