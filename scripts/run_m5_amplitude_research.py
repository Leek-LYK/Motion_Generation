#!/usr/bin/env python3
"""Bounded dropout/data/amplitude-sampling singles after the fixed60k factorial."""
import argparse
import ast
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_m5_length_research import read, write, sha, now, process_live

BASE = ROOT / 'runs/m5_physical_20261103'
FACTORIAL = ROOT / 'runs/m5_curriculum_factorial_20261104'
EXTENSION = ROOT / 'runs/m5_amplitude_data_20261104'
DEFAULT = ROOT / 'runs/m5_amplitude_regularization_20261104'
DESIGN = ROOT / 'runs/m5_amplitude_regularization_20261104_design.json'
RUNNERS = {'R1_dropout10': 'scripts/train_m5_dropout.py',
           'R2_jump_data19': 'scripts/train_m5_cache_extension.py',
           'R3_jump_mix50': 'scripts/train_m5_amplitude_sampling.py'}
PARTS = ('model', 'ema', 'optimizer', 'rngs', 'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng')


def runner_code(runner):
    for node in ast.parse((ROOT / runner).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CODE' for t in node.targets):
            return {name: sha(ROOT / name) for name in ast.literal_eval(node.value)}
    raise ValueError('Runner has no literal CODE contract')


def validate(study):
    plan = read(study / 'protocol.json')
    for file, digest in plan['protected_sha256'].items():
        if sha(file) != digest: raise ValueError('Frozen research input/source changed: ' + file)
    for name in RUNNERS:
        if sha(study / 'configs' / (name + '.json')) != plan['config_sha256'][name]:
            raise ValueError('Research configuration changed: ' + name)
    return plan


def prepare(study):
    if study.exists(): raise FileExistsError('Preserve the existing study')
    design = read(DESIGN)
    if design['source_sha256'] != sha(Path(__file__)) or design['maximum_new_updates'] != 60000:
        raise ValueError('Predeclared study design/source differs')
    # All dependency endpoints and the original decision must exist before choosing
    # a foundation. No partial-generation ranking or seed3102 quality is consulted.
    for name in ('T0_raw_uniform', 'T1_raw_high', 'T2_cons_uniform'):
        if not (FACTORIAL / name / 'COMPLETED.json').exists():
            raise RuntimeError('Fixed60k dependency still active: ' + name)
    decision = read(FACTORIAL / 'research_decision.json'); dependency = read(FACTORIAL / 'protocol.json')
    for file, digest in dependency['protected_sha256'].items():
        if sha(file) != digest: raise ValueError('Factorial dependency evidence changed: ' + file)
    chosen = decision['provisional_motion_preserving_recipe']
    if chosen is None:
        base_run = Path(dependency['reused_control']); foundation = 'T3_cons_high'
        reason = 'No factorial alternative passed all registered guards; retain the prior failedT3 as research reference,without claiming admission.'
    else:
        if chosen not in ('T0_raw_uniform', 'T1_raw_high', 'T2_cons_uniform') or not decision['assessments'][chosen]['motion_and_geometry_guards_passed']:
            raise ValueError('Unqualified factorial foundation selected')
        base_run = FACTORIAL / chosen; foundation = chosen
        reason = 'Use the sole predeclared factorial selection at the fixed60k endpoint; absolute geometry and all M5 requirements remain separate.'
    from training.m5_geometry_study import evidence
    baseline = evidence(base_run, 60000); cfg0 = read(base_run / 'config.json')
    assert baseline['checkpoint_sha256'] == decision['checkpoints'][foundation]
    assert cfg0['stage_start_step'] == cfg0['lr_schedule_origin_step'] == 40000
    assert cfg0['max_steps'] == cfg0['lr_schedule_end_step'] == cfg0['training_stop_step'] == 60000
    assert cfg0['warmup_steps'] == 500 and cfg0['network_dropout'] == 0
    assert cfg0['high_noise_fraction'] in (0., .5) and cfg0['consistency_weight'] in (0., .012261994051407363)
    assert all(cfg0.get(k, 0.) == 0 for k in ('axis_weight', 'ground_weight', 'legality_weight'))
    sources = {'dropout': BASE / 'verification/dropout_runtime_20261104/cli_verification.json',
               'cache_extension': BASE / 'verification/cache_extension_runtime_20261104/cli_verification.json',
               'amplitude_sampling': BASE / 'verification/amplitude_sampling_runtime_20261104/cli_verification.json'}
    for name, path in sources.items():
        if not read(path)['passed']: raise ValueError('Engineering verification absent: ' + name)
    old_subset = ROOT / 'runs/m5_physical_20261103/data/D_mixed_small_3500.json'
    new_subset = EXTENSION / 'data/mixed_small_plus19.json'; old = read(old_subset); new = read(new_subset)
    assert len(old['train']) == 3366 and len(new['train']) == 3385 and new['train'][:3366] == old['train']
    assert old['val'] == new['val'] and old['monitor'] == new['monitor'] and len(old['val']) == 300 and len(old['monitor']) == 91
    assert max(Counter(r['text_id'] for r in new['train']).values()) <= 20
    assert all(r['split'] == 'train' for r in old['train'] + new['train'])
    view = read(cfg0['training_view']); new_view = read(EXTENSION / 'data/READY.json')
    assert view['stats_sha256'] == new_view['stats_sha256']
    proof_path = EXTENSION / 'cache_compatibility_proof.json'; proof = read(proof_path)
    assert proof['source_cache_fingerprint'] == view['input_fingerprint']
    assert proof['target_cache_fingerprint'] == new_view['input_fingerprint']
    parent = Path(dependency['parent_checkpoint']); parent_sha = sha(parent)
    assert parent_sha == dependency['parent_sha256'] == '5502e10cd0425ca0c349b8ce0f0008e165cc3213623f7a7550571b7759e1f001'
    # Verify the successful tested implementations before committing the protocol.
    checks = ((BASE / 'verification/dropout_runtime_20261104/cli/config.json', RUNNERS['R1_dropout10']),
              (BASE / 'verification/cache_extension_runtime_20261104/cli/config.json', RUNNERS['R2_jump_data19']),
              (BASE / 'verification/amplitude_sampling_runtime_20261104/cli/config.json', RUNNERS['R3_jump_mix50']))
    for config, runner in checks:
        if read(config)['launch_code_sha256'] != runner_code(runner): raise ValueError('Tested runner implementation changed')
    study.mkdir(parents=True); snapshot = study / 'parent_snapshot/checkpoint.pt'; snapshot.parent.mkdir()
    shutil.copyfile(parent, snapshot); assert sha(snapshot) == parent_sha
    data = study / 'data'; data.mkdir(); shutil.copyfile(old_subset, data / 'mixed_control.json'); shutil.copyfile(new_subset, data / 'mixed_plus19.json')
    coefficients = {k: cfg0.get(k, 0.) for k in ('consistency_weight', 'axis_weight', 'ground_weight', 'legality_weight')}
    for label, ready, names in (('original', view, ('R1_dropout10', 'R3_jump_mix50')), ('extended', new_view, ('R2_jump_data19',))):
        write(study / ('activation_' + label + '.json'), {'schema': 'm5-constraint-activation-v1', 'passed': True,
            'engineering_fixture': False, 'physical_input_fingerprint': ready['input_fingerprint'],
            'training_view_fingerprint': ready['fingerprint'], 'runs': {
                name: {'parent_sha256': parent_sha, 'coefficients': coefficients} for name in names}})
    configs = {}; run_subsets = {}; codes = {}; pool_path = BASE / 'data_candidates/medium_jump_sampling_20261104/pools.json'
    for name, runner in RUNNERS.items():
        subset = data / ('mixed_plus19.json' if name == 'R2_jump_data19' else 'mixed_control.json')
        cfg = dict(cfg0, stage=name, constraint_activation_protocol=str(study / ('activation_extended.json' if name == 'R2_jump_data19' else 'activation_original.json')),
                   launch_code_sha256=runner_code(runner), subset_sha256=sha(subset), train_samples=len(read(subset)['train']))
        if name == 'R1_dropout10': cfg['network_dropout'] = .1
        elif name == 'R2_jump_data19':
            cfg.update(input_dir=new_view['input_dir'], stats_dir=str(EXTENSION / 'stats/motion'),
                training_view=str(EXTENSION / 'data/READY.json'), cache_compatibility_proof=str(proof_path),
                cache_compatibility_proof_sha256=sha(proof_path))
        else:
            cfg.update(amplitude_pool_file=str(pool_path), amplitude_pool_sha256=sha(pool_path),
                       medium_jump_amplitude_sampling_enabled=True, medium_jump_high_probability=.5)
        path = study / 'configs' / (name + '.json'); write(path, cfg); configs[name] = sha(path)
        run_subsets[name] = str(subset); codes.update(cfg['launch_code_sha256'])
    remaining = read(FACTORIAL / 'remaining_matrix_requirements.json')
    remaining.update(study_scope='Three independent M5 singles fromC40 to60,not final M5 admission.',
                     original_E0_E1_replaced=False, final_training_recipe_frozen=False, M5_complete=False)
    write(study / 'remaining_matrix_requirements.json', remaining)
    files = [ROOT / f for f in codes]
    files += [Path(__file__).resolve(), ROOT / 'scripts/run_m5_length_research.py', ROOT / 'training/m5_geometry_study.py',
              ROOT / 'training/m5_length_study.py', ROOT / 'training/m5_protocol.py', DESIGN,
              FACTORIAL / 'protocol.json', FACTORIAL / 'research_decision.json', FACTORIAL / 'remaining_matrix_requirements.json',
              base_run / 'config.json', base_run / 'COMPLETED.json', base_run / 'lineage.json', base_run / 'subset.json',
              base_run / 'checkpoints/latest.pt', base_run / 'evaluations/generation_full_060000/report.json',
              base_run / 'evaluations/generation_full_060000/per_sample.json', base_run / 'evaluations/denoising_060000.json',
              BASE / 'protocol/M5_acceptance_protocol_v2.json', Path(cfg0['training_view']), EXTENSION / 'data/READY.json',
              Path(view['input_dir']) / 'features.npy', Path(new_view['input_dir']) / 'features.npy',
              snapshot, data / 'mixed_control.json', data / 'mixed_plus19.json', proof_path, pool_path,
              study / 'activation_original.json', study / 'activation_extended.json', study / 'remaining_matrix_requirements.json']
    files += list(sources.values())
    files += [BASE / p for p in view['stats_sha256']] + [EXTENSION / p for p in new_view['stats_sha256']]
    files += [Path(p) for p in proof['files_sha256']] + [Path(p) for p in read(pool_path)['files_sha256']]
    plan = {'schema': 'm5-amplitude-regularization-research-v1', 'frozen_at_utc': now(),
        'first_wave': list(RUNNERS), 'runner_by_run': RUNNERS, 'subset_by_run': run_subsets,
        'parent_checkpoint': str(snapshot), 'parent_sha256': parent_sha, 'reused_control': str(base_run),
        'reused_control_checkpoint_sha256': baseline['checkpoint_sha256'], 'foundation_cell': foundation,
        'foundation_rule_reason': reason, 'factorial_decision_sha256': sha(FACTORIAL / 'research_decision.json'),
        'target_step': 60000, 'stage_start_step': 40000, 'new_updates_per_run': 20000,
        'maximum_new_updates': 60000, 'nominal_updates_including_reused_control': 80000, 'reused_updates': 20000,
        'registered_total_physical_updates_before': 435000, 'registered_total_physical_updates': 495000,
        'max_parallel': 3, 'config_sha256': configs, 'protected_sha256': {str(p): sha(p) for p in set(files)},
        'single_changes': {'R1_dropout10': 'Transformerattention/feedforward dropout0->0.1 only in40->60; not a claim of fullpaperphase1 reproduction.',
            'R2_jump_data19': 'Append19 qualifiedmediumhighjump events; exact originalfeatureprefix and qualified modelstats preserved.',
            'R3_jump_mix50': 'Originaldata retained; within91-120framejump only,draw highGTamplitude>=0.20m withprobability0.5,usingaseparatecheckpointedRNG.'},
        'decision_rule': 'Atfixed60 compareall fullshort/extended/sixclass amplitude,<=5ppsemanticproxy amplitude regressions,norm6/FK guards and<=10pctkeygeometryregressions using theunchangedlengthcompare. Rankonlyqualifyingcandidatesbygeometry_score_m. Absolutegeometrystillrequiredseparately.',
        'automatic_combination': False, 'automatic_E_launch': False, 'final_recipe_frozen': False,
        'seed3102_quality_scores_used': False, 'final_audit_model_scores_read': False,
        'human_feedback_scope': 'Prior one-human12prompt40kdevelopment feedback motivatesdiagnosis; notusedas60k orfinalsemanticacceptance.', 'M5_complete': False}
    write(study / 'protocol.json', plan)
    for file in files:
        if file.suffix == '.py' and file.is_relative_to(ROOT):
            target = study / 'source_at_launch' / file.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(file, target)
    validate(study)
    write(study / 'verification/preflight.json', {'passed': True, 'verified_at_utc': now(),
        'protocol_sha256': sha(study / 'protocol.json'), 'dependency_fixed60k_complete': True,
        'training_stats_unchanged': True, 'candidate_val_and_monitor_identical': True,
        'all_three_versioned_runners_engineering_verified': True, 'new_optimizer_updates': 0, 'M5_complete': False})
    print({'foundation': foundation, 'new_update_budget': 60000, 'M5_complete': False}, flush=True)


def launch(study):
    lock = (study / 'launch.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = validate(study)
    if (study / 'processes.json').exists(): raise FileExistsError('One-shot launch; inspect existing handles')
    free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()[0])
    if free < 24576: raise RuntimeError('Insufficient measured free GPU memory')
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    processes = {}
    for name, runner in RUNNERS.items():
        run = study / name
        if run.exists(): raise FileExistsError('Run already exists')
        command = [str(ROOT / '.venv/bin/python'), '-B', '-u', str(ROOT / runner), '--config',
                   str(study / 'configs' / (name + '.json')), '--subset', plan['subset_by_run'][name],
                   '--run-dir', str(run), '--parent', plan['parent_checkpoint']]
        with (study / (name + '.log')).open('ab', buffering=0) as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ticks = Path('/proc', str(proc.pid), 'stat').read_text().rsplit(')', 1)[1].split()[19]
        processes[name] = {'pid': proc.pid, 'start_ticks': ticks, 'command': command, 'run_dir': str(run), 'started_at_utc': now()}
        write(study / 'processes.json', processes)
    write(study / 'study_status.json', {'phase': 'training', 'active': {k: v['pid'] for k, v in processes.items()},
          'completed': [], 'updated_at_utc': now(), 'M5_complete': False})
    command = [str(ROOT / '.venv/bin/python'), '-B', '-u', str(Path(__file__).resolve()), 'assess', '--study-dir', str(study), '--wait']
    with (study / 'assessment.log').open('ab', buffering=0) as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(study / 'assessment_process.json', {'pid': proc.pid, 'command': command, 'started_at_utc': now()})
    print({'workers': {k: v['pid'] for k, v in processes.items()}, 'assessor_pid': proc.pid}, flush=True)


def assess(study, wait=False):
    plan = validate(study); processes = read(study / 'processes.json')
    while True:
        done = [n for n in RUNNERS if (study / n / 'COMPLETED.json').exists()]
        active = {n: processes[n]['pid'] for n in RUNNERS if n not in done and process_live(processes[n], study / n)}
        missing = [n for n in RUNNERS if n not in done and n not in active]
        if len(done) == 3: break
        write(study / 'study_status.json', {'phase': 'requires_inspection' if missing else 'training_or_evaluating',
              'completed': done, 'active': active, 'missing_handles': missing, 'updated_at_utc': now(), 'M5_complete': False})
        if missing: raise RuntimeError('Training handles missing; no automatic restart: ' + str(missing))
        if not wait: print({'active': active, 'completed': done}); return
        time.sleep(30)
    from training.m5_geometry_study import evidence
    from training.m5_length_study import compare
    control = evidence(plan['reused_control'], 60000); assessments = {}; checkpoints = {}
    assert control['checkpoint_sha256'] == plan['reused_control_checkpoint_sha256']
    reference_rows = read(Path(plan['reused_control']) / 'evaluations/generation_full_060000/per_sample.json')
    key = lambda r: (r['sample_id'], r['seed'], r['condition'])
    refs = {key(r): r for r in reference_rows}; assert len(refs) == len(reference_rows) == 1800
    for name in RUNNERS:
        candidate = evidence(study / name, 60000)
        rows = read(study / name / 'evaluations/generation_full_060000/per_sample.json')
        assert len(rows) == 1800 and len({key(r) for r in rows}) == 1800 and {key(r) for r in rows} == set(refs)
        assert all(r['frames'] == refs[key(r)]['frames'] and abs(r['reference_amplitude'] - refs[key(r)]['reference_amplitude']) < 2e-6 for r in rows)
        value = compare(candidate, control); value['research_guards_passed'] = value.pop('length_direction_supported')
        value['relative_changes_vs_control'] = value.pop('mixed_relative_changes')
        for field in ('metrics', 'amplitude_by_stratum', 'full_final_absolute_geometry'):
            value[field] = {'candidate': value[field]['mixed'], 'control': value[field]['short']}
        value['scope'] = 'Fixed60k single-variable development research; all1800samepairedkeys/GTreferences verified. No length-only,finalhumanorM5 admission claim.'
        assessments[name] = value; checkpoints[name] = candidate['checkpoint_sha256']
    qualifying = [n for n, v in assessments.items() if v['research_guards_passed']]
    selected = min(qualifying, key=lambda n: assessments[n]['metrics']['candidate']['geometry_score_m']) if qualifying else None
    result = {'assessed_at_utc': now(), 'protocol_sha256': sha(study / 'protocol.json'),
        'foundation_cell': plan['foundation_cell'], 'assessments': assessments,
        'provisional_guard_passing_direction': selected, 'checkpoint_sha256': checkpoints,
        'actual_new_updates': sum(read(study / n / 'COMPLETED.json')['updates_this_stage'] for n in RUNNERS),
        'reused_control_updates': 20000, 'training_seeds': [3101], 'automatic_combination': False,
        'automatic_E_launch': False, 'final_recipe_frozen': False, 'seed3102_quality_scores_used': False,
        'final_audit_model_scores_read': False, 'M5_complete': False}
    assert result['actual_new_updates'] == 60000
    if (study / 'research_decision.json').exists():
        old = read(study / 'research_decision.json'); assert all(old[k] == v for k, v in result.items() if k != 'assessed_at_utc')
    else: write(study / 'research_decision.json', result)
    write(study / 'study_status.json', {'phase': 'research_budget_complete', 'completed': list(RUNNERS),
          'active': {}, 'updated_at_utc': now(), 'M5_complete': False}); print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('command', choices=('prepare', 'launch', 'assess'))
    parser.add_argument('--study-dir', type=Path, default=DEFAULT); parser.add_argument('--wait', action='store_true'); args = parser.parse_args()
    study = args.study_dir.resolve(); study.relative_to(ROOT / 'runs')
    if args.command == 'prepare': prepare(study)
    elif args.command == 'launch': launch(study)
    else: assess(study, args.wait)


if __name__ == '__main__': main()
