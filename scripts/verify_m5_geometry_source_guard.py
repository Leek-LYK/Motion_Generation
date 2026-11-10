#!/usr/bin/env python3
"""Replay completed engineering evidence and reject orchestration source drift.

No optimizer updates: the four real-Small engineering jobs are reused read-only.
"""
import json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.m3 import atomic_json,sha


def read(path):return json.loads(path.read_text())


def main():
    verification=ROOT/'runs/m5_physical_20261103/verification';original=verification/'geometry_study_fixture';out=verification/'geometry_study_source_guard'
    if out.exists():raise FileExistsError('Preserve existing source-guard evidence')
    proof=read(original/'verification_report.json');assert proof['passed'] and proof['real_small_updates']==8
    for f,h in proof['code_sha256'].items():
        if sha(original/'source_at_validation'/f)!=h:raise ValueError('Original fixture source archive changed')
    protocol=read(original/'protocol.json');source_files=['training/m5_geometry_study.py','training/m5_protocol.py','scripts/run_m5_geometry_study.py',
      'scripts/prepare_m5_geometry_study.py','scripts/calibrate_m5_constraint_recipe.py','scripts/verify_m5_geometry_source_guard.py']
    code={f:sha(ROOT/f) for f in source_files};protocol['orchestration_code_sha256']=code
    completed=read(original/'study_status.json')['completed'];before={n:sha(original/n/'checkpoints/latest.pt') for n in completed}
    for f,h in protocol['code_sha256'].items():
        if sha(ROOT/f)!=h:raise ValueError('Engineering worker code changed')
    replay=out/'replay';replay.mkdir(parents=True);shutil.copytree(original/'configs',replay/'configs')
    for n in completed:(replay/n).symlink_to(original/n,target_is_directory=True)
    atomic_json(replay/'protocol.json',protocol)
    def run(directory):
        result=subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/run_m5_geometry_study.py'),'--study-dir',str(directory),'--engineering-fixture'],
          cwd=ROOT,capture_output=True,text=True)
        (directory/'controller.log').write_text(result.stdout+result.stderr);return result
    result=run(replay)
    if result.returncode:raise RuntimeError('Completed-fixture replay failed; see '+str(replay/'controller.log'))
    decision=read(replay/'research_decision.json');prior=read(original/'research_decision.json')
    assert decision['engineering_fixture'] and not decision['M5_complete'] and decision['provisional_recipe']==prior['provisional_recipe'] and decision['combination']==prior['combination']
    for n in prior['assessments']:
        for key in ['metrics','joint_checks','directional_evidence','full_final_geometry_passed']:
            assert decision['assessments'][n][key]==prior['assessments'][n][key]
    rejected=out/'rejected';rejected.mkdir();bad=json.loads(json.dumps(protocol));bad['orchestration_code_sha256']['training/m5_geometry_study.py']='0'*64
    atomic_json(rejected/'protocol.json',bad);result=run(rejected)
    assert result.returncode!=0 and 'Study orchestration source drift' in result.stderr
    assert not any((rejected/n).exists() for n in protocol['first_wave'])
    assert before=={n:sha(original/n/'checkpoints/latest.pt') for n in completed}
    report={'passed':True,'new_optimizer_updates':0,'source_archive_verified':True,'completed_small_evidence_replay_identical':True,
      'changed_orchestration_source_rejected_before_workers':True,'original_fixture_states_unchanged':True,
      'original_fixture_verification_sha256':sha(original/'verification_report.json'),'original_fixture_protocol_sha256':sha(original/'protocol.json'),
      'original_fixture_updates':8,'code_sha256':code,'engineering_fixture':True,'M5_complete':False}
    atomic_json(out/'report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
