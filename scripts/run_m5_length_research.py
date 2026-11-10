#!/usr/bin/env python3
"""Prepare, launch or verify the bounded G1 short/mixed length-course study."""
import argparse
import ast
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'runs/m5_physical_20261103'
GEOMETRY = ROOT / 'runs/m5_geometry_20261104'
DEFAULT = ROOT / 'runs/m5_length_research_20261104'
NAMES = ('L0_short_G1_control', 'L1_mixed_G1')
PARTS = ('model', 'ema', 'optimizer', 'rngs', 'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng')
ORCHESTRATION = ('scripts/run_m5_length_research.py', 'scripts/assess_m5_length_research.py',
                 'training/m5_length_study.py', 'training/m5_geometry_study.py', 'training/m5_protocol.py')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def process_live(record, run):
    try:
        args = Path(f"/proc/{record['pid']}/cmdline").read_bytes().split(b'\0')
        start = Path(f"/proc/{record['pid']}/stat").read_text().rsplit(')', 1)[1].split()[19]
        return (start == record['start_ticks'] and b'--run-dir' in args and
                Path(os.fsdecode(args[args.index(b'--run-dir') + 1])).resolve() == Path(run).resolve())
    except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
        return False


def validate(study):
    plan = read(study / 'protocol.json')
    for path, expected in plan['protected_sha256'].items():
        if sha(Path(path)) != expected:
            raise ValueError('Frozen study evidence/source changed: ' + path)
    for name in NAMES:
        entry = plan['runs'][name]
        for key in ('config', 'subset', 'parent'):
            if sha(entry[key]) != entry[key + '_sha256']:
                raise ValueError(name + ': frozen ' + key + ' changed')
        cfg = read(entry['config'])
        if not (cfg['training_stop_step'] == cfg['minimum_steps'] == cfg['max_steps'] == 60000 and
                cfg['lr_schedule_origin_step'] == 40000 and cfg['lr_schedule_end_step'] == 60000):
            raise ValueError('Wrong fixed endpoint/LR origin')
    return plan


def prepare(study):
    if study.exists():
        raise FileExistsError('Preserve the existing study; do not prepare over it')
    source_plan = read(GEOMETRY / 'protocol.json')
    decision = read(GEOMETRY / 'research_decision.json')
    g1 = decision['assessments']['G1_consistency']
    g4 = decision['assessments']['G4_all']
    if not (all(g1['common_guards'].values()) and g1['joint_checks']['all_six_amplitude_both_headings']
            and g1['joint_checks']['full_core_fraction_no_regression']
            and not g4['joint_development_passed'] and not g4['common_guards']['short_core_fraction_no_regression']):
        raise ValueError('Registered research trigger is not present')
    readiness = read(BASE / 'verification/curriculum_data_readiness.json')
    engineering = read(GEOMETRY / 'verification/long_sequence_contract/report.json')
    prelaunch = BASE / 'verification/length_research_prelaunch/report.json'
    if not readiness['passed'] or not engineering['passed'] or not read(prelaunch)['passed']:
        raise ValueError('Mixed-data engineering is not ready')
    cfg0 = read(GEOMETRY / 'configs/G1_consistency.json')
    assert cfg0 == read(GEOMETRY / 'G1_consistency/config.json')
    for relative, expected in cfg0['launch_code_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Frozen worker source changed: ' + relative)
    g1_done = read(GEOMETRY / 'G1_consistency/COMPLETED.json')
    c_done = read(BASE / 'C_small_raw7/COMPLETED.json')
    if (g1_done['step'], c_done['step']) != (50000, 40000):
        raise ValueError('Wrong fixed parent checkpoints')
    common_parent = source_plan['parent_sha256']
    if (read(GEOMETRY / 'G1_consistency/lineage.json')['parent_sha256'] != common_parent or
            c_done['terminal_checkpoint_sha256'] != common_parent):
        raise ValueError('The reused prefix did not start from the common 40k parent')
    allowed = {'stage', 'stage_start_step', 'minimum_steps', 'training_stop_step', 'constraint_activation_protocol'}
    coefficients = {k: cfg0.get(k, 0.) for k in ('consistency_weight', 'axis_weight', 'ground_weight', 'legality_weight')}
    assert coefficients == {'consistency_weight': .012261994051407363, 'axis_weight': 0., 'ground_weight': 0., 'legality_weight': 0.}
    assert cfg0['lr_schedule_origin_step'] == 40000 and cfg0['lr_schedule_end_step'] == 60000
    assert cfg0['training_stop_step'] == 50000 and cfg0['warmup_steps'] == 500
    assert g1_done['identity']['code_sha256'] == cfg0['launch_code_sha256']
    prefix_trace = GEOMETRY / 'G1_consistency/sample_trace.jsonl'
    prefix_steps = [json.loads(line)['step'] for line in prefix_trace.read_text().splitlines()]
    assert prefix_steps == list(range(40001, 50001))
    prefix_metrics = GEOMETRY / 'G1_consistency/metrics.jsonl'
    lr_records = [json.loads(line) for line in prefix_metrics.read_text().splitlines()]
    assert [row['step'] for row in lr_records] == list(range(40050, 50001, 50))
    for row in lr_records:
        local = row['step'] - 1 - 40000
        expected_lr = (2e-5 * (local + 1) / 500 if local < 500 else
                       2e-6 + (2e-5 - 2e-6) * .5 * (1 + math.cos(math.pi * (local - 500) / 19500)))
        assert abs(row['learning_rate'] - expected_lr) <= 1e-15
    manifests = [BASE / 'data/D_short_3500.json', BASE / 'data/D_mixed_small_3500.json']
    data = [read(p) for p in manifests]
    assert len(data[0]['train']) == 3404 and len(data[1]['train']) == 3366
    assert all(data[0][key] == data[1][key] for key in ('val', 'monitor'))
    for path in manifests:
        meta = readiness['manifests'][path.stem]
        assert sha(path) == meta['manifest_sha256']
    view_path = Path(cfg0['training_view']); view = read(view_path)
    assert view['passed'] and read(Path(cfg0['input_dir']) / 'READY')['fingerprint'] == view['input_fingerprint']
    for relative, expected in view['stats_sha256'].items():
        assert sha(BASE / relative) == expected
    study.mkdir(parents=True)
    activation = {'schema': 'm5-constraint-activation-v1', 'passed': True, 'engineering_fixture': False,
                  'physical_input_fingerprint': view['input_fingerprint'], 'training_view_fingerprint': view['fingerprint'],
                  'scope': 'G1-only length-course research; previous failed gates are preserved.', 'runs': {}}
    runs = {}
    parents = [GEOMETRY / 'G1_consistency/checkpoints/step_050000.pt', Path(source_plan['parent_checkpoint'])]
    expected_parents = [g1_done['terminal_checkpoint_sha256'], common_parent]
    for i, name in enumerate(NAMES):
        assert sha(parents[i]) == expected_parents[i]
        snapshot = study / 'parent_snapshots' / ('G1_step050000.pt' if i == 0 else 'C_step040000.pt')
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parents[i], snapshot)
        assert sha(snapshot) == expected_parents[i]
        subset = study / 'data' / manifests[i].name
        subset.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(manifests[i], subset)
        cfg = dict(cfg0, stage=name, stage_start_step=50000 if i == 0 else 40000,
                   training_stop_step=60000, minimum_steps=60000,
                   constraint_activation_protocol=str(study / 'activation.json'),
                   subset_sha256=sha(subset), train_samples=len(data[i]['train']))
        differences = {k for k in set(cfg0) | set(cfg) if cfg0.get(k) != cfg.get(k)}
        assert differences <= allowed | ({'subset_sha256', 'train_samples'} if i else set())
        config = study / 'configs' / (name + '.json'); write(config, cfg)
        activation['runs'][name] = {'parent_sha256': expected_parents[i], 'coefficients': coefficients}
        runs[name] = {'config': str(config), 'config_sha256': sha(config), 'subset': str(subset),
                      'subset_sha256': sha(subset), 'parent': str(snapshot), 'parent_sha256': sha(snapshot),
                      'source_parent': str(parents[i]), 'source_start_step': cfg['stage_start_step'],
                      'target_step': 60000, 'new_updates': 10000 if i == 0 else 20000,
                      'reused_prefix_updates': 10000 if i == 0 else 0}
    write(study / 'activation.json', activation)
    paths = [ROOT / f for f in (*cfg0['launch_code_sha256'], *ORCHESTRATION)]
    paths += [GEOMETRY / 'protocol.json', GEOMETRY / 'research_decision.json',
              GEOMETRY / 'G1_consistency/config.json', GEOMETRY / 'G1_consistency/lineage.json',
              GEOMETRY / 'G1_consistency/COMPLETED.json', GEOMETRY / 'calibration/report.json',
              GEOMETRY / 'verification/startup_inheritance.json', prefix_trace, prefix_metrics,
              GEOMETRY / 'verification/long_sequence_contract/report.json', BASE / 'first_wave_assessment.json',
              BASE / 'verification/curriculum_data_readiness.json', BASE / 'protocol/M5_acceptance_protocol_v2.json',
              prelaunch, prelaunch.parent / 'verify.py',
              view_path, Path(cfg0['input_dir']) / 'READY', study / 'activation.json']
    paths += [BASE / f for f in view['stats_sha256']]
    budget_files = [p for folder in (BASE, GEOMETRY) for p in folder.glob('*/COMPLETED.json')]
    previous_updates = sum(read(p)['updates_this_stage'] for p in budget_files)
    assert previous_updates == 305000
    paths += budget_files
    protected = {str(path): sha(path) for path in paths}
    plan = {'schema': 'm5-g1-length-research-v1', 'frozen_at_utc': now(), 'first_wave': list(NAMES),
            'conditional_combinations': {}, 'runs': runs, 'target_step': 60000,
            'maximum_new_updates': 30000, 'nominal_pair_updates': 40000, 'reused_prefix_updates': 10000,
            'previous_physical_updates': previous_updates, 'registered_total_physical_updates': previous_updates + 30000,
            'common_40k_parent_sha256': common_parent, 'protected_sha256': protected,
            'max_parallel': 2, 'engineering_fixture': False, 'code_sha256': cfg0['launch_code_sha256'],
            'scope': 'Bounded research of the required short/mixed event-length course. Provisional G1 baseline; no claim of formal D/E or M5 admission.',
            'admission_evidence': {'G1_common_guards': g1['common_guards'], 'G1_failed_absolute_checks': g1['failed_joint_checks'],
                                   'G4_failed_checks': g4['failed_joint_checks'], 'original_failed_gates_preserved': True},
            'prefix_reuse': 'L0 uses archived C40->G1_50 full state; continues 50->60 without LR warmup restart. L1 starts C40 and runs mixed 40->60. Both conceptual courses share C40 and the same 40->60 LR/loss/optimizer settings.',
            'data_interpretation': readiness['length_curriculum_change'],
            'comparison': {'endpoint': 60000, 'events': 300, 'seeds': [0, 42, 1234], 'headings': [None, 0.],
                           'ddim_steps': 100, 'guidance': 2., 'short_and_extended_core_required': True,
                           'core_and_new_class_fraction_regression_tolerance': .05,
                           'norm6_guard_ratio': 1.05, 'fk_mean_guard_ratio': 1.1, 'fk_each_t_guard_ratio': 1.25,
                           'key_geometry_guard_ratio': 1.1, 'distance_additive_tolerance_m': 1e-4,
                           'deep_fraction_additive_tolerance': 1e-4, 'rotation_fraction_additive_tolerance': 1e-5,
                           'absolute_final_geometry_thresholds_unchanged': True},
            'automatic_E_launch': False, 'M5_complete': False, 'final_audit_model_scores_read': False}
    write(study / 'protocol.json', plan)
    for path in paths:
        if path.is_relative_to(ROOT) and path.parts[-1].endswith('.py'):
            dest = study / 'source_at_launch' / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, dest)
    validate(study)
    write(study / 'verification/preflight.json', {'passed': True, 'verified_at_utc': now(),
          'protocol_sha256': sha(study / 'protocol.json'), 'parent_and_source_and_data_hashes_verified': True,
          'same_ordered_validation_and_monitor': True, 'G1_prefix_reuse_config_compatible': True,
          'reused_prefix_trace_updates': len(prefix_steps), 'reused_prefix_lr_records_checked': len(lr_records),
          'lr_warmup_restarts_at_50k': False, 'new_optimizer_updates': 0, 'M5_complete': False})
    print(json.dumps({'prepared': str(study), 'protocol_sha256': sha(study / 'protocol.json'),
                      'new_update_budget': 30000}, indent=2), flush=True)


def launch(study):
    lock = (study / 'launch.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = validate(study)
    if (study / 'processes.json').exists():
        raise FileExistsError('Launch is one-shot; inspect existing handles instead of restarting')
    free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()[0])
    if free < 24576:
        raise RuntimeError('Insufficient measured GPU memory for the registered pair')
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    records = {}
    for name in NAMES:
        entry = plan['runs'][name]; run = study / name
        if run.exists():
            raise FileExistsError('Do not overwrite a run: ' + str(run))
        command = [str(ROOT / '.venv/bin/python'), '-B', '-u', str(ROOT / 'scripts/train_m5_constraints.py'),
                   '--config', entry['config'], '--subset', entry['subset'], '--run-dir', str(run), '--parent', entry['parent']]
        with (study / (name + '.log')).open('ab', buffering=0) as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ticks = Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        records[name] = {'pid': proc.pid, 'start_ticks': ticks, 'command': command,
                         'started_at_utc': now(), 'run_dir': str(run)}
        write(study / 'processes.json', records)
    write(study / 'study_status.json', {'phase': 'training', 'active': {n: v['pid'] for n, v in records.items()},
          'completed': [], 'maximum_additional_updates': 30000, 'updated_at_utc': now(), 'M5_complete': False})
    command = [str(ROOT / '.venv/bin/python'), '-B', '-u', str(ROOT / 'scripts/assess_m5_length_research.py'),
               '--study-dir', str(study), '--wait']
    with (study / 'assessment_waiter.log').open('ab', buffering=0) as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(study / 'assessment_process.json', {'pid': proc.pid, 'command': command, 'started_at_utc': now()})
    print(json.dumps({'workers': records, 'assessment_waiter_pid': proc.pid}, indent=2), flush=True)


def verify_startup(study):
    import numpy as np
    import torch
    from training.m5 import M5Trainer
    plan = validate(study); records = read(study / 'processes.json')
    source = GEOMETRY / 'verification/verify_inheritance.py'
    assert sha(source) == 'a271055eff7f072db234a756d6ad06f0da785219dd16061a9efb66650c7df792'
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'fingerprint')
    namespace = {'hashlib': hashlib, 'np': np, 'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    fingerprint = namespace['fingerprint']; result = {}
    for name in NAMES:
        entry = plan['runs'][name]; run = study / name; cfg = read(entry['config'])
        assert process_live(records[name], run) or (run / 'COMPLETED.json').exists()
        parent = torch.load(entry['parent'], map_location='cpu', weights_only=False)
        initial = torch.load(run / 'checkpoints/initial_state.pt', map_location='cpu', weights_only=False)
        before = {k: fingerprint(parent[k]) for k in PARTS}; after = {k: fingerprint(initial[k]) for k in PARTS}
        assert before == after
        for k in ('step', 'ema_updates', 'samples_seen', 'text_dropped'):
            assert initial[k] == parent[k]
        assert initial['stage_updates'] == 0 and initial['optimizer_kind'] == 'adamw-fp32-v1'
        assert parent['step'] == entry['source_start_step']
        subset = read(entry['subset'])
        class Sampler:
            pass
        sampler = Sampler(); sampler.config = cfg; sampler.samples = subset['train']; sampler.pools = {}
        sampler.batch_rng = torch.Generator().set_state(parent['rngs']['batch_rng'])
        for i, row in enumerate(subset['train']):
            bucket = 0 if row['num_frames'] <= 90 else 1 if row['num_frames'] <= 120 else 2
            sampler.pools.setdefault(bucket, {}).setdefault(row['group'], []).append(i)
        expected = [{'step': parent['step'] + i + 1, 'indices': M5Trainer.draw_indices(sampler)} for i in range(100)]
        with (run / 'sample_trace.jsonl').open() as stream:
            actual = [json.loads(next(stream)) for _ in range(100)]
        assert actual == expected
        lineage = read(run / 'lineage.json'); assert lineage['parent_sha256'] == entry['parent_sha256'] and lineage['optimizer_rng_preserved']
        lines = (run / 'metrics.jsonl').read_text().splitlines(keepends=True)
        values = [json.loads(line) for line in lines if line.endswith('\n')]
        latest = values[-1]; assert latest['step'] >= entry['source_start_step'] + 100
        assert all(math.isfinite(latest[k]) and latest[k] > 0 for k in ('loss', 'root_grad_norm', 'body_grad_norm'))
        initial_ema = read(run / f"evaluations/denoising_{parent['step']:06d}.json")['ema']['macro_loss']
        reference_file = GEOMETRY / 'G1_consistency/evaluations' / f"denoising_{parent['step']:06d}.json"
        reference_ema = read(reference_file)['ema']['macro_loss']; assert abs(initial_ema - reference_ema) <= 1e-8
        local = latest['step'] - 1 - cfg['lr_schedule_origin_step']; warm = cfg['warmup_steps']
        lr = (cfg['learning_rate'] * (local + 1) / warm if local < warm else
              cfg['minimum_learning_rate'] + (cfg['learning_rate'] - cfg['minimum_learning_rate']) * .5 *
              (1 + math.cos(math.pi * (local - warm) / (60000 - 40000 - warm))))
        assert abs(latest['learning_rate'] - lr) <= 1e-15
        result[name] = {'parent_sha256': entry['parent_sha256'], 'parts_fingerprints': before,
                        'global_counters_preserved': True, 'first_100_draws_replayed': True,
                        'initial_ema_max_absolute_difference': abs(initial_ema - reference_ema),
                        'verified_step': latest['step'], 'verified_learning_rate': lr,
                        'initial_state_sha256': sha(run / 'checkpoints/initial_state.pt')}
        del parent, initial
    earlier = read(GEOMETRY / 'verification/startup_inheritance.json')
    assert earlier['passed'] and result['L1_mixed_G1']['parts_fingerprints'] == earlier['parent_parts_fingerprints']
    write(study / 'verification/startup_inheritance.json', {'passed': True, 'verified_at_utc': now(),
          'protocol_sha256': sha(study / 'protocol.json'), 'runs': result,
          'conceptual_common_40k_parent_sha256': plan['common_40k_parent_sha256'],
          'local_initial_steps_differ_because_short_prefix_is_reused': True,
          'new_optimizer_updates_by_verifier': 0, 'M5_complete': False})
    print(json.dumps({'passed': True, 'steps': {n: v['verified_step'] for n, v in result.items()}}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('prepare', 'launch', 'verify-startup'))
    parser.add_argument('--study-dir', type=Path, default=DEFAULT); args = parser.parse_args()
    study = args.study_dir.resolve(); study.relative_to(ROOT / 'runs')
    {'prepare': prepare, 'launch': launch, 'verify-startup': verify_startup}[args.action](study)


if __name__ == '__main__':
    main()
