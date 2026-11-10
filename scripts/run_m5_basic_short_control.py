#!/usr/bin/env python3
"""Matched short40->60 control for the selected basicv3 uniform/G1 foundation."""
import argparse
import ast
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_m5_length_research import read, write, sha, now, process_live

BASE = ROOT / 'runs/m5_physical_20261103'
SCOPE = ROOT / 'runs/m5_basic_scope_20261104'
DEFAULT = ROOT / 'runs/m5_basic_curriculum_20261104'
NAME = 'D0_short_uniform'
D1 = ROOT / 'runs/m5_curriculum_factorial_20261104/T2_cons_uniform'


def validate(study):
    plan = read(study / 'protocol.json')
    for file, expected in plan['protected_sha256'].items():
        if sha(file) != expected: raise ValueError('Frozen basic-curriculum source/input changed: ' + file)
    if sha(study / 'configs' / (NAME + '.json')) != plan['config_sha256'][NAME]: raise ValueError('Matched short configuration changed')
    return plan


def start(study):
    if study.exists(): raise FileExistsError('Preserve existing matched-control study')
    decision_path = SCOPE / 'current_trials_assessments/decision_v3.json'; decision = read(decision_path)
    if decision['provisional_basic_recipe'] != 'T2_cons_uniform': raise ValueError('This version only implements the actually selected T2 foundation')
    current = read(SCOPE / 'CURRENT_POLICY.json'); policy = Path(current['policy'])
    assert sha(policy) == decision['policy_sha256'] == current['policy_sha256']
    reference = read(SCOPE / 'assessments_revision2/T2_cons_uniform/report.json'); cfg0 = read(D1 / 'config.json')
    assert reference['checkpoint_sha256'] == read(D1 / 'COMPLETED.json')['terminal_checkpoint_sha256'] == sha(D1 / 'checkpoints/latest.pt')
    assert cfg0['high_noise_fraction'] == 0 and cfg0['consistency_weight'] == .012261994051407363
    assert cfg0['network_dropout'] == 0 and cfg0['stage_start_step'] == cfg0['lr_schedule_origin_step'] == 40000
    assert cfg0['training_stop_step'] == cfg0['max_steps'] == cfg0['lr_schedule_end_step'] == 60000
    assert all(cfg0.get(k, 0.) == 0 for k in ('axis_weight', 'ground_weight', 'legality_weight'))
    source = ROOT / 'runs/m5_curriculum_factorial_20261104/parent_snapshot/checkpoint.pt'; parent_sha = sha(source)
    assert parent_sha == '5502e10cd0425ca0c349b8ce0f0008e165cc3213623f7a7550571b7759e1f001'
    short = BASE / 'data/D_short_3500.json'; mixed = read(D1 / 'subset.json'); manifest = read(short)
    assert manifest['val'] == mixed['val'] and manifest['monitor'] == mixed['monitor']
    assert len(manifest['train']) == 3404 and all(60 <= r['num_frames'] <= 90 for r in manifest['train'])
    for file, expected in cfg0['launch_code_sha256'].items(): assert sha(ROOT / file) == expected
    study.mkdir(parents=True); snapshot = study / 'parent_snapshot/checkpoint.pt'; snapshot.parent.mkdir()
    shutil.copyfile(source, snapshot); assert sha(snapshot) == parent_sha
    subset = study / 'data/short.json'; subset.parent.mkdir(); shutil.copyfile(short, subset)
    view = read(cfg0['training_view']); coefficients = {k: cfg0.get(k, 0.) for k in ('consistency_weight', 'axis_weight', 'ground_weight', 'legality_weight')}
    activation = study / 'activation.json'
    write(activation, {'schema': 'm5-constraint-activation-v1', 'passed': True, 'engineering_fixture': False,
        'physical_input_fingerprint': view['input_fingerprint'], 'training_view_fingerprint': view['fingerprint'],
        'runs': {NAME: {'parent_sha256': parent_sha, 'coefficients': coefficients}}})
    cfg = dict(cfg0, stage=NAME, subset_sha256=sha(subset), train_samples=len(manifest['train']), constraint_activation_protocol=str(activation))
    changes = {k for k in set(cfg) | set(cfg0) if cfg.get(k) != cfg0.get(k)}
    assert changes <= {'stage', 'subset_sha256', 'train_samples', 'constraint_activation_protocol'}
    config = study / 'configs' / (NAME + '.json'); write(config, cfg)
    files = [ROOT / p for p in cfg0['launch_code_sha256']]
    files += [Path(__file__).resolve(), ROOT / 'scripts/run_m5_length_research.py', ROOT / 'scripts/assess_m5_basic_scope_v3.py',
        ROOT / 'training/m5_basic_scope.py', ROOT / 'runs/m5_geometry_20261104/verification/verify_inheritance.py',
        policy, SCOPE / 'cohorts.json', decision_path, SCOPE / 'assessments_revision2/T2_cons_uniform/report.json',
        D1 / 'config.json', D1 / 'COMPLETED.json', D1 / 'checkpoints/latest.pt', D1 / 'subset.json',
        D1 / 'evaluations/generation_full_060000/report.json', D1 / 'evaluations/generation_full_060000/per_sample.json',
        D1 / 'evaluations/denoising_060000.json', snapshot, subset, activation, Path(cfg0['training_view']),
        Path(cfg0['input_dir']) / 'features.npy', Path(cfg0['input_dir']) / 'READY', Path(cfg0['input_dir']) / 'samples.parquet']
    files += [BASE / path for path in view['stats_sha256']]
    plan = {'schema': 'm5-basic-matched-short-control-v1', 'frozen_at_utc': now(),
        'first_wave': [NAME], 'parent_checkpoint': str(snapshot), 'parent_sha256': parent_sha,
        'reused_mixed_run': str(D1), 'reused_mixed_checkpoint_sha256': reference['checkpoint_sha256'],
        'stage_start_step': 40000, 'target_step': 60000, 'maximum_new_updates': 20000,
        'reused_mixed_updates': 20000, 'nominal_matched_pair_updates': 40000,
        'registered_total_physical_updates_before': 495000, 'registered_total_physical_updates': 515000,
        'subset': str(subset), 'config_sha256': {NAME: sha(config)}, 'protected_sha256': {str(p): sha(p) for p in set(files)},
        'policy': str(policy), 'policy_sha256': sha(policy), 'same_algorithm_except_training_events': True,
        'scope': 'Required matchedshortcontrol forT2uniform/G1basicfoundation. Existinghigh-noiseshortcontrolcannotreplaceit.',
        'fixed_comparison': 'Both branches C40->60,AdamW/raw7/G1/uniformt,batch64,BF16/FP32,sameLRschedule,stats,augmentation,textdropandEMA. Short3404vsMixed3366 eventcurriculum.',
        'automatic_E_launch': False, 'final_recipe_frozen': False, 'seed3102_quality_scores_used': False,
        'final_audit_model_scores_read': False, 'M5_complete': False}
    write(study / 'protocol.json', plan)
    for file in files:
        if file.suffix == '.py' and file.is_relative_to(ROOT):
            target = study / 'source_at_launch' / file.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(file, target)
    validate(study)
    write(study / 'verification/preflight.json', {'passed': True, 'verified_at_utc': now(),
          'matched_setting_differences': sorted(changes), 'all_nondata_hyperparameters_matched': True,
          'same_validation_and_monitor': True, 'new_optimizer_updates': 0, 'M5_complete': False})
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    run = study / NAME; command = [str(ROOT / '.venv/bin/python'), '-B', '-u', str(ROOT / 'scripts/train_m5_constraints.py'),
        '--config', str(config), '--subset', str(subset), '--run-dir', str(run), '--parent', str(snapshot)]
    lock = (study / 'launch.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (study / (NAME + '.log')).open('ab', buffering=0) as log:
        worker = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {'pid': worker.pid, 'start_ticks': Path('/proc', str(worker.pid), 'stat').read_text().rsplit(')', 1)[1].split()[19],
              'command': command, 'run_dir': str(run), 'started_at_utc': now()}
    write(study / 'processes.json', {NAME: record})
    observer_command = [str(ROOT / '.venv/bin/python'), '-B', '-u', str(Path(__file__).resolve()), 'assess', '--study-dir', str(study), '--wait']
    with (study / 'assessment.log').open('ab', buffering=0) as log:
        observer = subprocess.Popen(observer_command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(study / 'assessment_process.json', {'pid': observer.pid, 'command': observer_command, 'started_at_utc': now()})
    write(study / 'study_status.json', {'phase': 'training', 'active': {NAME: worker.pid}, 'completed': [], 'updated_at_utc': now(), 'M5_complete': False})
    registry = BASE / 'runtime/dependent_studies.json'; registered = read(registry)
    registered['studies'].append({'study_dir': str(study), 'role': 'basic_v3_matched_uniform_short_control',
        'protocol_sha256': sha(study / 'protocol.json'), 'registered_at_utc': now()}); write(registry, registered)
    ledger_path = BASE / 'M5_requirement_ledger.json'; ledger = read(ledger_path)
    assert ledger['registered_total_physical_update_budget'] == 495000
    ledger['registered_total_physical_update_budget'] = 515000; ledger['registered_update_budget_components']['basic_matched_uniform_short_control'] = 20000
    ledger['basic_matched_short_control'] = {'study': '../' + study.name, 'run': NAME, 'pid': worker.pid,
        'reused_D1': str(D1), 'maximum_new_updates': 20000, 'protocol_sha256': sha(study / 'protocol.json'), 'M5_complete': False}
    ledger['updated_at_utc'] = now(); write(ledger_path, ledger)
    print({'worker_pid': worker.pid, 'assessor_pid': observer.pid, 'new_updates': 20000, 'formal_registered_total': 515000}, flush=True)


def assess(study, wait):
    plan = validate(study); run = study / NAME; process = read(study / 'processes.json')[NAME]
    while not (run / 'COMPLETED.json').exists():
        if not process_live(process, run):
            write(study / 'study_status.json', {'phase': 'requires_inspection', 'missing_handles': [NAME], 'M5_complete': False})
            raise RuntimeError('Matchedshortworker terminalbeforecompletion; no automaticrestart')
        write(study / 'study_status.json', {'phase': 'training_or_evaluating', 'active': {NAME: process['pid']},
              'completed': [], 'updated_at_utc': now(), 'M5_complete': False})
        if not wait: print({'active': process['pid']}); return
        time.sleep(30)
    validate(study)
    from scripts.assess_m5_basic_scope_v3 import assess as evaluate_basic, compare
    short = evaluate_basic(NAME, run, study / 'basic_v3' / NAME, read(plan['policy']), read(SCOPE / 'cohorts.json'))
    mixed = read(SCOPE / 'assessments_revision2/T2_cons_uniform/report.json')
    assert mixed['checkpoint_sha256'] == plan['reused_mixed_checkpoint_sha256']
    result = compare(mixed, short)
    result.update(schema='m5-basic-uniform-matched-length-decision-v1', assessed_at_utc=now(),
        protocol_sha256=sha(study / 'protocol.json'), policy_sha256=plan['policy_sha256'],
        matched_pair_setting_verified=True, actual_new_updates=read(run / 'COMPLETED.json')['updates_this_stage'],
        reused_mixed_updates=20000, automatic_E_launch=False, final_recipe_frozen=False,
        seed3102_quality_scores_used=False, final_audit_model_scores_read=False, M5_complete=False)
    write(study / 'research_decision.json', result)
    write(study / 'study_status.json', {'phase': 'matched_control_budget_complete', 'active': {}, 'completed': [NAME], 'updated_at_utc': now(), 'M5_complete': False})
    print({'matched_basic_length_guards_passed': result['research_guards_passed_v3'], 'failed_checks': result['failed_checks']}, flush=True)


def verify(study, wait):
    import numpy as np
    import torch
    plan = validate(study); run = study / NAME; process = read(study / 'processes.json')[NAME]
    output = study / 'verification/startup.json'
    if output.exists(): raise FileExistsError('Preserve startup verification')
    while True:
        trace = []
        if (run / 'sample_trace.jsonl').exists():
            with (run / 'sample_trace.jsonl').open() as stream:
                for line in stream:
                    try: trace.append(__import__('json').loads(line))
                    except ValueError: break
                    if len(trace) == 100: break
        if len(trace) == 100: break
        if not process_live(process, run): raise RuntimeError('Shortworker handle missingbefore100draws')
        if not wait: print({'waiting_for_100': True}); return
        time.sleep(10)
    methods = ROOT / 'runs/m5_geometry_20261104/verification/verify_inheritance.py'
    assert sha(methods) == 'a271055eff7f072db234a756d6ad06f0da785219dd16061a9efb66650c7df792'
    node = next(n for n in ast.parse(methods.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'fingerprint')
    namespace = {'hashlib': hashlib, 'np': np, 'torch': torch}; exec(compile(ast.Module(body=[node], type_ignores=[]), str(methods), 'exec'), namespace)
    fingerprint = namespace['fingerprint']; parts = ('model', 'ema', 'optimizer', 'rngs', 'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng')
    source = torch.load(plan['parent_checkpoint'], map_location='cpu', weights_only=False)
    initial = torch.load(run / 'checkpoints/initial_state.pt', map_location='cpu', weights_only=False)
    assert {k: fingerprint(source[k]) for k in parts} == {k: fingerprint(initial[k]) for k in parts}
    assert initial['stage_updates'] == 0 and initial['step'] == 40000
    from training.m5 import M5Trainer
    class Sampler: pass
    s = Sampler(); s.config = read(run / 'config.json'); s.samples = read(run / 'subset.json')['train']; s.pools = {}
    s.batch_rng = torch.Generator().set_state(source['rngs']['batch_rng'])
    for i, row in enumerate(s.samples): s.pools.setdefault(0, {}).setdefault(row['group'], []).append(i)
    expected = [{'step': 40001 + i, 'indices': M5Trainer.draw_indices(s)} for i in range(100)]; assert expected == trace
    import json
    metrics = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]; last = metrics[-1]
    assert last['high_noise_replacement_fraction'] == 0 and all(np.isfinite(last[k]) and last[k] > 0 for k in ('loss', 'root_grad_norm', 'body_grad_norm'))
    initial_ema = read(run / 'evaluations/denoising_040000.json')['ema']['macro_loss']
    original = read(ROOT / 'runs/m5_geometry_20261104/G1_consistency/evaluations/denoising_040000.json')['ema']['macro_loss']
    assert abs(initial_ema - original) < 1e-8
    validate(study)
    write(output, {'passed': True, 'verified_at_utc': now(), 'all8_parent_state_parts_identical': True,
        'first100_real_draws_match_parent_rng': True, 'uniform_time_no_high_noise_replacement': True,
        'initial_ema_validation_matches_old_short_parent': True, 'positive_finite_root_body_gradients': True,
        'last_verified_update': last['step'], 'new_optimizer_updates_by_verifier': 0,
        'source_sha256': sha(Path(__file__)), 'protocol_sha256': sha(study / 'protocol.json'), 'M5_complete': False})
    print({'startup_verified': True, 'through': last['step']}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('command', choices=('start', 'assess', 'verify-startup'))
    p.add_argument('--study-dir', type=Path, default=DEFAULT); p.add_argument('--wait', action='store_true'); a = p.parse_args()
    study = a.study_dir.resolve(); study.relative_to(ROOT / 'runs')
    if a.command == 'start': start(study)
    elif a.command == 'assess': assess(study, a.wait)
    else: verify(study, a.wait)
