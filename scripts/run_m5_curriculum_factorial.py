#!/usr/bin/env python3
"""Bounded same-data consistency x timestep-sampling factorial after failed D60."""
import argparse
import ast
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_m5_length_research import read, write, sha, now, process_live

BASE = ROOT / 'runs/m5_physical_20261103'
LENGTH = ROOT / 'runs/m5_length_research_20261104'
DEFAULT = ROOT / 'runs/m5_curriculum_factorial_20261104'
RUNS = {'T0_raw_uniform': (False, False), 'T1_raw_high': (False, True), 'T2_cons_uniform': (True, False)}
CONTROL = 'T3_cons_high'
PARTS = ('model', 'ema', 'optimizer', 'rngs', 'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng')


def validate(study):
    plan = read(study / 'protocol.json')
    for file, expected in plan['protected_sha256'].items():
        if sha(file) != expected:
            raise ValueError('Frozen factorial source/evidence changed: ' + file)
    for name in RUNS:
        if sha(study / 'configs' / (name + '.json')) != plan['config_sha256'][name]:
            raise ValueError('Frozen factorial configuration changed')
    return plan


def prepare(study):
    if study.exists():
        raise FileExistsError('Preserve the existing study')
    decision = read(LENGTH / 'research_decision.json')
    if decision['length_direction_supported'] or decision['checks']['extended_core_both_headings']:
        raise ValueError('Registered long-motion failure trigger is absent')
    for name in ('L0_short_G1_control', 'L1_mixed_G1'):
        done = read(LENGTH / name / 'COMPLETED.json')
        assert done['step'] == 60000 and sha(LENGTH / name / 'checkpoints/latest.pt') == done['terminal_checkpoint_sha256']
    control = LENGTH / 'L1_mixed_G1'; cfg0 = read(control / 'config.json'); subset = control / 'subset.json'
    lineage = read(control / 'lineage.json'); parent = Path(lineage['parent_checkpoint'])
    assert sha(parent) == lineage['parent_sha256'] == '5502e10cd0425ca0c349b8ce0f0008e165cc3213623f7a7550571b7759e1f001'
    for file, expected in cfg0['launch_code_sha256'].items():
        assert sha(ROOT / file) == expected
    assert cfg0['stage_start_step'] == cfg0['lr_schedule_origin_step'] == 40000
    assert cfg0['max_steps'] == cfg0['lr_schedule_end_step'] == cfg0['training_stop_step'] == 60000
    assert cfg0['high_noise_fraction'] == .5 and cfg0['consistency_weight'] == .012261994051407363
    assert all(cfg0.get(k, 0.) == 0 for k in ('axis_weight', 'ground_weight', 'legality_weight'))
    study.mkdir(parents=True); snapshot = study / 'parent_snapshot/checkpoint.pt'
    snapshot.parent.mkdir(); shutil.copyfile(parent, snapshot); assert sha(snapshot) == lineage['parent_sha256']
    target_subset = study / 'data/mixed_small.json'; target_subset.parent.mkdir(); shutil.copyfile(subset, target_subset)
    assert sha(target_subset) == cfg0['subset_sha256']
    view = read(cfg0['training_view'])
    activation = {'schema': 'm5-constraint-activation-v1', 'passed': True, 'engineering_fixture': False,
                  'physical_input_fingerprint': view['input_fingerprint'], 'training_view_fingerprint': view['fingerprint'], 'runs': {}}
    config_hashes = {}
    for name, (consistency, high) in RUNS.items():
        cfg = dict(cfg0, stage=name, consistency_weight=cfg0['consistency_weight'] if consistency else 0.,
                   high_noise_fraction=.5 if high else 0., constraint_activation_protocol=str(study / 'activation.json'),
                   time_sampling='Uniform t=0..999 in this 40k->60k stage' if not high else cfg0['time_sampling'])
        changes = {k for k in set(cfg) | set(cfg0) if cfg.get(k) != cfg0.get(k)}
        assert changes <= {'stage', 'consistency_weight', 'high_noise_fraction', 'constraint_activation_protocol', 'time_sampling'}
        path = study / 'configs' / (name + '.json'); write(path, cfg); config_hashes[name] = sha(path)
        activation['runs'][name] = {'parent_sha256': sha(snapshot), 'coefficients': {
            k: cfg.get(k, 0.) for k in ('consistency_weight', 'axis_weight', 'ground_weight', 'legality_weight')}}
    write(study / 'activation.json', activation)
    files = [ROOT / p for p in cfg0['launch_code_sha256']]
    files += [Path(__file__).resolve(), ROOT / 'scripts/run_m5_length_research.py',
              ROOT / 'training/m5_length_study.py', ROOT / 'training/m5_geometry_study.py', ROOT / 'training/m5_protocol.py',
              LENGTH / 'research_decision.json', LENGTH / 'protocol.json', control / 'config.json',
              control / 'COMPLETED.json', control / 'lineage.json', subset, target_subset, snapshot, study / 'activation.json',
              BASE / 'protocol/M5_acceptance_protocol_v2.json', BASE / 'data/READY.json',
              BASE / 'verification/paper_engineering_audit_20261104/root_representation.json',
              LENGTH / 'diagnostics/root_conditioning_050000.json', LENGTH / 'diagnostics/full_050000.json']
    files += [BASE / p for p in view['stats_sha256']]
    plan = {'schema': 'm5-mixed-consistency-time-factorial-v1', 'frozen_at_utc': now(), 'first_wave': list(RUNS),
            'conditional_combinations': {}, 'parent_checkpoint': str(snapshot), 'parent_sha256': sha(snapshot),
            'subset': str(target_subset), 'subset_sha256': sha(target_subset), 'reused_control': str(control),
            'reused_control_checkpoint_sha256': read(control / 'COMPLETED.json')['terminal_checkpoint_sha256'],
            'factors': {'consistency_weight': [0., .012261994051407363], 'high_noise_fraction': [0., .5]},
            'factor_cells': {**{n: {'consistency': c, 'high_noise': h} for n, (c, h) in RUNS.items()},
                             CONTROL: {'consistency': True, 'high_noise': True}},
            'stage_start_step': 40000, 'target_step': 60000, 'new_updates_per_run': 20000,
            'maximum_new_updates': 60000, 'nominal_factorial_updates': 80000, 'reused_updates': 20000,
            'max_parallel': 3, 'registered_total_physical_updates_before': 375000,
            'registered_total_physical_updates': 435000, 'config_sha256': config_hashes,
            'protected_sha256': {str(p): sha(p) for p in files}, 'engineering_fixture': False,
            'trigger': 'Both D research endpoints completed; mixed long/core guard and maximum projection regression failed.',
            'fixed_comparison': 'All four cells have the same C40 parent, mixed3366 data, 40->60 LR schedule, batch/precision, optimizer and augmentations. Only existing consistency and high-noise switches differ.',
            'decision_rule': 'Use the predeclared short+extended core, six-class full amplitude, <=5pp regression, norm6/FK guards and <=10pct key-geometry regression vs T3. Rank qualifying development candidates by the existing geometry_score_m. Final absolute geometry is reported separately. No automatic E or M5 entry.',
            'factor_effects': 'Report consistency/high-noise main effects and their difference-of-differences interaction on the fixed60k metrics. One training seed; no claim of independent replication.',
            'source_final_audit_model_scores_read': False, 'seed3102_quality_scores_used': False,
            'automatic_E_launch': False, 'M5_complete': False}
    write(study / 'protocol.json', plan)
    for p in files:
        if p.suffix == '.py' and p.is_relative_to(ROOT):
            dest = study / 'source_at_launch' / p.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
    validate(study)
    write(study / 'verification/preflight.json', {'passed': True, 'verified_at_utc': now(),
          'protocol_sha256': sha(study / 'protocol.json'), 'completed60k_dependency_verified': True,
          'original_length_failure_preserved': True, 'new_optimizer_updates': 0, 'M5_complete': False})
    print('Prepared three additional factorial cells, 60k new updates.', flush=True)


def launch(study):
    lock = (study / 'launch.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = validate(study)
    if (study / 'processes.json').exists():
        raise FileExistsError('One-shot launch: inspect existing worker handles')
    free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()[0])
    if free < 24576:
        raise RuntimeError('Insufficient measured free GPU memory')
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    records = {}
    for name in RUNS:
        run = study / name
        if run.exists():
            raise FileExistsError('Run exists; do not overwrite')
        command = [str(ROOT / '.venv/bin/python'), '-B', '-u', str(ROOT / 'scripts/train_m5_constraints.py'),
                   '--config', str(study / 'configs' / (name + '.json')), '--subset', plan['subset'],
                   '--run-dir', str(run), '--parent', plan['parent_checkpoint']]
        with (study / (name + '.log')).open('ab', buffering=0) as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        ticks = Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        records[name] = {'pid': proc.pid, 'start_ticks': ticks, 'command': command, 'run_dir': str(run), 'started_at_utc': now()}
        write(study / 'processes.json', records)
    write(study / 'study_status.json', {'phase': 'training', 'active': {n: r['pid'] for n, r in records.items()},
          'completed': [], 'M5_complete': False, 'updated_at_utc': now()})
    command = [str(ROOT / '.venv/bin/python'), '-B', '-u', str(Path(__file__).resolve()), 'assess', '--study-dir', str(study), '--wait']
    with (study / 'assessment.log').open('ab', buffering=0) as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    write(study / 'assessment_process.json', {'pid': proc.pid, 'command': command, 'started_at_utc': now()})
    print(json.dumps({'workers': {n: r['pid'] for n, r in records.items()}, 'assessor_pid': proc.pid}), flush=True)


def assess(study, wait=False):
    plan = validate(study); processes = read(study / 'processes.json')
    while True:
        done = [n for n in RUNS if (study / n / 'COMPLETED.json').exists()]
        active = {n: processes[n]['pid'] for n in RUNS if n not in done and process_live(processes[n], study / n)}
        missing = [n for n in RUNS if n not in done and n not in active]
        if len(done) == 3:
            break
        write(study / 'study_status.json', {'phase': 'requires_inspection' if missing else 'training_or_evaluating',
              'completed': done, 'active': active, 'missing_handles': missing, 'updated_at_utc': now(), 'M5_complete': False})
        if missing:
            raise RuntimeError('Worker handles missing before completion; no automatic restart: ' + str(missing))
        if not wait:
            print({'active': active, 'completed': done}); return
        time.sleep(30)
    from training.m5_geometry_study import evidence
    from training.m5_length_study import compare
    values = {name: evidence(study / name, 60000) for name in RUNS}
    values[CONTROL] = evidence(Path(plan['reused_control']), 60000)
    assert values[CONTROL]['checkpoint_sha256'] == plan['reused_control_checkpoint_sha256']
    assessments = {}
    for name in RUNS:
        value = compare(values[name], values[CONTROL])
        value['motion_and_geometry_guards_passed'] = value.pop('length_direction_supported')
        value['relative_changes_vs_control'] = value.pop('mixed_relative_changes')
        value['metrics'] = {'candidate': value['metrics']['mixed'], 'control': value['metrics']['short']}
        value['amplitude_by_stratum'] = {'candidate': value['amplitude_by_stratum']['mixed'], 'control': value['amplitude_by_stratum']['short']}
        value['full_final_absolute_geometry'] = {'candidate': value['full_final_absolute_geometry']['mixed'], 'control': value['full_final_absolute_geometry']['short']}
        value['scope'] = 'Same-data fixed60k consistency/time factorial; no length-only interpretation or final M5 acceptance.'
        assessments[name] = value
    qualified = [n for n, value in assessments.items() if value['motion_and_geometry_guards_passed']]
    selected = min(qualified, key=lambda n: values[n]['metrics']['geometry_score_m']) if qualified else None
    effects = {}
    for key in values[CONTROL]['metrics']:
        a, b, c, d = (values[n]['metrics'][key] for n in (*RUNS, CONTROL))
        effects[key] = {'consistency_main_effect': .5 * ((c - a) + (d - b)),
                        'high_noise_main_effect': .5 * ((b - a) + (d - c)),
                        'interaction': (d - b) - (c - a)}
    result = {'protocol_sha256': sha(study / 'protocol.json'), 'assessed_at_utc': now(), 'assessments': assessments,
              'provisional_motion_preserving_recipe': selected, 'factor_effects_on_metrics': effects,
              'checkpoints': {n: v['checkpoint_sha256'] for n, v in values.items()},
              'actual_new_updates': sum(read(study / n / 'COMPLETED.json')['updates_this_stage'] for n in RUNS),
              'reused_control_updates': 20000, 'training_seeds': [3101], 'automatic_E_launch': False,
              'seed3102_quality_scores_used': False, 'final_audit_model_scores_read': False, 'M5_complete': False}
    assert result['actual_new_updates'] == 60000
    if (study / 'research_decision.json').exists():
        old = read(study / 'research_decision.json')
        assert all(old[k] == v for k, v in result.items() if k != 'assessed_at_utc')
    else:
        write(study / 'research_decision.json', result)
    write(study / 'study_status.json', {'phase': 'research_budget_complete', 'completed': list(RUNS),
          'active': {}, 'updated_at_utc': now(), 'M5_complete': False})
    print(json.dumps(result, indent=2), flush=True)


def verify_startup(study):
    import numpy as np
    import torch
    from training.m5 import M5Trainer
    plan = validate(study); process = read(study / 'processes.json')
    source = ROOT / 'runs/m5_geometry_20261104/verification/verify_inheritance.py'
    assert sha(source) == 'a271055eff7f072db234a756d6ad06f0da785219dd16061a9efb66650c7df792'
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'fingerprint')
    namespace = {'hashlib': hashlib, 'np': np, 'torch': torch}; exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    fingerprint = namespace['fingerprint']; parent = torch.load(plan['parent_checkpoint'], map_location='cpu', weights_only=False)
    expected = {k: fingerprint(parent[k]) for k in PARTS}; subset = read(plan['subset'])
    class Sampler:
        pass
    sampler = Sampler(); sampler.config = read(study / 'configs/T0_raw_uniform.json'); sampler.samples = subset['train']; sampler.pools = {}
    sampler.batch_rng = torch.Generator().set_state(parent['rngs']['batch_rng'])
    for i, row in enumerate(subset['train']):
        bucket = 0 if row['num_frames'] <= 90 else 1 if row['num_frames'] <= 120 else 2
        sampler.pools.setdefault(bucket, {}).setdefault(row['group'], []).append(i)
    draws = [{'step': 40001 + i, 'indices': M5Trainer.draw_indices(sampler)} for i in range(100)]
    with (Path(plan['reused_control']) / 'sample_trace.jsonl').open() as stream:
        assert [json.loads(next(stream)) for _ in range(100)] == draws
    results = {}
    for name, (_, high) in RUNS.items():
        run = study / name; assert process_live(process[name], run) or (run / 'COMPLETED.json').exists()
        initial = torch.load(run / 'checkpoints/initial_state.pt', map_location='cpu', weights_only=False)
        assert {k: fingerprint(initial[k]) for k in PARTS} == expected
        assert initial['stage_updates'] == 0 and all(initial[k] == parent[k] for k in ('step', 'ema_updates', 'samples_seen', 'text_dropped'))
        with (run / 'sample_trace.jsonl').open() as stream:
            assert [json.loads(next(stream)) for _ in range(100)] == draws
        rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines(keepends=True) if line.endswith('\n')]
        assert rows[-1]['step'] >= 40100 and all(np.isfinite(rows[-1][k]) and rows[-1][k] > 0 for k in ('loss', 'root_grad_norm', 'body_grad_norm'))
        assert all((r['high_noise_replacement_fraction'] > 0) if high else (r['high_noise_replacement_fraction'] == 0) for r in rows)
        initial_loss = read(run / 'evaluations/denoising_040000.json')['ema']['macro_loss']
        reference_loss = read(Path(plan['reused_control']) / 'evaluations/denoising_040000.json')['ema']['macro_loss']
        assert abs(initial_loss - reference_loss) <= 1e-8
        results[name] = {'verified_step': rows[-1]['step'], 'all_parent_parts_identical': True,
                         'first100_draws_match_control': True, 'noise_replacement_matches_registered_switch': True,
                         'initial_ema_max_abs_difference': abs(initial_loss - reference_loss),
                         'initial_state_sha256': sha(run / 'checkpoints/initial_state.pt')}
        del initial
    write(study / 'verification/startup.json', {'passed': True, 'verified_at_utc': now(), 'runs': results,
          'parent_parts_fingerprints': expected, 'protocol_sha256': sha(study / 'protocol.json'),
          'new_optimizer_updates_by_verifier': 0, 'M5_complete': False})
    print({'passed': True, 'steps': {n: v['verified_step'] for n, v in results.items()}}, flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('prepare', 'launch', 'assess', 'verify-startup'))
    p.add_argument('--study-dir', type=Path, default=DEFAULT); p.add_argument('--wait', action='store_true'); a = p.parse_args()
    study = a.study_dir.resolve(); study.relative_to(ROOT / 'runs')
    if a.action == 'assess':
        assess(study, a.wait)
    else:
        {'prepare': prepare, 'launch': launch, 'verify-startup': verify_startup}[a.action](study)


if __name__ == '__main__':
    main()
