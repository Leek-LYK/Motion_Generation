#!/usr/bin/env python3
"""Paired inference-only CFG sweep on frozen T2EMA/basic180; no training changes."""
import argparse
import fcntl
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
DEFAULT = ROOT / 'runs/m5_basic_guidance_20261104'
MODEL_RUN = ROOT / 'runs/m5_curriculum_factorial_20261104/T2_cons_uniform'
SETTINGS = {'G10_cfg1': 1., 'G15_cfg1p5': 1.5, 'G20_control_cfg2': 2.}


def validate(study):
    plan = read(study / 'protocol.json')
    for file, expected in plan['protected_sha256'].items():
        if sha(file) != expected: raise ValueError('Frozen guidance-study source/input changed: ' + file)
    return plan


def start(study):
    if study.exists(): raise FileExistsError('Preserve existing guidance study')
    cfg = read(MODEL_RUN / 'config.json'); done = read(MODEL_RUN / 'COMPLETED.json')
    checkpoint = MODEL_RUN / 'checkpoints/latest.pt'
    assert done['step'] == 60000 and sha(checkpoint) == done['terminal_checkpoint_sha256']
    policy = read(SCOPE / 'CURRENT_POLICY.json'); cohorts = read(policy['cohorts'])
    assert sha(policy['policy']) == policy['policy_sha256'] and sha(policy['cohorts']) == policy['cohorts_sha256']
    ids = {k for k, v in cohorts['development']['classifications'].items() if v['difficulty'] == 'basic'}
    source_subset = MODEL_RUN / 'subset.json'; entries = [r for r in read(source_subset)['val'] if r['sample_id'] in ids]
    assert len(entries) == len(ids) == 180
    study.mkdir(parents=True); snapshot = study / 'model_snapshot/checkpoint.pt'; snapshot.parent.mkdir()
    shutil.copyfile(checkpoint, snapshot); assert sha(snapshot) == done['terminal_checkpoint_sha256']
    shutil.copytree(MODEL_RUN / 'stats/motion', study / 'stats/motion')
    for relative, expected in done['identity']['stats_sha256'].items(): assert sha(study / relative) == expected
    write(study / 'model_config.json', cfg); write(study / 'events.json', entries)
    fields = [ROOT / file for file in cfg['launch_code_sha256']]
    fields += [Path(__file__).resolve(), ROOT / 'scripts/run_m5_length_research.py', ROOT / 'scripts/assess_m5_basic_scope_v3.py',
        ROOT / 'training/m5_basic_scope.py', Path(policy['policy']), Path(policy['cohorts']), snapshot,
        study / 'model_config.json', study / 'events.json', source_subset, MODEL_RUN / 'COMPLETED.json',
        SCOPE / 'assessments_revision2/T2_cons_uniform/report.json', BASE / 'data/development_flags.json',
        Path(cfg['input_dir']) / 'READY', Path(cfg['input_dir']) / 'samples.parquet', Path(cfg['training_view'])]
    fields += [study / file for file in done['identity']['stats_sha256']]
    plan = {'schema': 'm5-basic-frozen-model-guidance-v1', 'frozen_at_utc': now(),
        'model_run': str(MODEL_RUN), 'checkpoint': str(snapshot), 'checkpoint_sha256': done['terminal_checkpoint_sha256'],
        'weights': 'ema', 'model_step': 60000, 'settings': SETTINGS,
        'events': 180, 'seeds': [0, 42, 1234], 'headings': [None, 0.], 'ddim_steps': 100,
        'batch_size': 16, 'event_order': 'Original300eventorder filtered by frozen basiccohort; same order/layoutfor all settings.',
        'initial_noise': 'Originalsample_keyed sid:M5:seed; shared across all CFG values.',
        'new_generation_clips': 3240, 'paired_event_seed_heading_keys': 1080,
        'control_regenerated': True, 'legacy_cfg2_outputs_unchanged': True,
        'policy': policy['policy'], 'policy_sha256': policy['policy_sha256'], 'cohorts_sha256': policy['cohorts_sha256'],
        'input_cache': cfg['input_dir'], 'input_cache_fingerprint': done['identity']['physical_input_fingerprint'],
        'protected_sha256': {str(p): sha(p) for p in set(fields)},
        'decision_rule': 'CompareagainsttheregeneratedCFG2controlwithsamebatchlayout. Requireallbasicmotion/5ppguardsand10pctgeometryguardsplus>=5pctimprovementinbothmeanFKselfandpoolednodeP95. Rankqualifyingguidancebyexistingbasicgeometryscore. Selectionprovisionaluntilhuman/finalcoursevalidation.',
        'scope': 'Inference-only basicdevelopment180; no newcomplex-onlyrepair,optimizerupdate,normalizationfit,filteredtrajectoryorM6conditions.',
        'all300_program_validity_under_new_cfg_proven': False,
        'automatic_guidance_adoption_or_E_launch': False, 'registered_formal_update_budget_unchanged': 515000,
        'new_optimizer_updates': 0, 'seed3102_quality_scores_read': False, 'final_audit_model_scores_read': False, 'M5_complete': False}
    write(study / 'protocol.json', plan)
    for file in fields:
        if file.suffix == '.py' and file.is_relative_to(ROOT):
            dest = study / 'source_at_launch' / file.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(file, dest)
    validate(study)
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    records = {}
    for name in SETTINGS:
        run = study / name; command = [str(ROOT / '.venv/bin/python'), '-B', '-u', str(Path(__file__).resolve()),
            'worker', '--study-dir', str(study), '--setting', name, '--run-dir', str(run)]
        with (study / (name + '.log')).open('ab', buffering=0) as log:
            p = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        records[name] = {'pid': p.pid, 'start_ticks': Path('/proc', str(p.pid), 'stat').read_text().rsplit(')', 1)[1].split()[19],
                         'command': command, 'run_dir': str(run), 'started_at_utc': now()}
        write(study / 'processes.json', records)
    command = [str(ROOT / '.venv/bin/python'), '-B', '-u', str(Path(__file__).resolve()), 'assess', '--study-dir', str(study), '--wait']
    with (study / 'assessment.log').open('ab', buffering=0) as log:
        p = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(study / 'assessment_process.json', {'pid': p.pid, 'command': command, 'started_at_utc': now()})
    write(study / 'status.json', {'phase': 'inference_running', 'workers': {k: v['pid'] for k, v in records.items()},
          'target_generated_clips': 3240, 'new_optimizer_updates': 0, 'updated_at_utc': now(), 'M5_complete': False})
    print({'workers': {k: v['pid'] for k, v in records.items()}, 'observer': p.pid, 'new_generation_clips': 3240}, flush=True)


def digest(model):
    import hashlib
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        result.update(name.encode()); result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def worker(study, name, output):
    import numpy as np
    import torch
    from training.m3 import configure, make_model
    from training.m5_physical_data import PhysicalSubsetDataset
    from training.m5_physical_evaluation import evaluate_generation
    from scripts.assess_m5_basic_scope_v3 import basic_summary, LENGTH_CORE
    plan = validate(study)
    if name not in SETTINGS or output != study / name: raise ValueError('Unknown inference task/path')
    if (output / 'COMPLETED.json').exists(): raise FileExistsError('Preserve completed sampling')
    output.mkdir(parents=True, exist_ok=True); lock = (output / 'run.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = read(study / 'model_config.json'); configure(cfg['seed'])
    state = torch.load(plan['checkpoint'], map_location='cpu', weights_only=False)
    assert state['step'] == 60000 and state['identity']['physical_input_fingerprint'] == plan['input_cache_fingerprint']
    model = make_model(cfg['model'], study / 'stats/motion').cuda().eval().requires_grad_(False)
    model.load_state_dict(state['ema']); identity = state['identity']; del state
    before = digest(model); entries = read(study / 'events.json')
    samples = PhysicalSubsetDataset(entries, cfg['input_dir'], study / 'stats/motion', cache_size=180)
    result = evaluate_generation(model, samples, output / 'generation', 60000,
        seeds=(0, 42, 1234), headings=(None, 0.), batch_size=16, steps=100, guidance=SETTINGS[name], retain=True,
        metric_thresholds=cfg['metric_thresholds'], flags=read(BASE / 'data/development_flags.json'))
    rows = read(output / 'generation/per_sample.json'); assert len(rows) == 1080
    assert len({(r['sample_id'], r['seed'], r['condition']) for r in rows}) == 1080
    basic = basic_summary(rows); conditions = {}
    for heading in ('native', 'heading0'):
        part = [r for r in rows if r['condition'] == heading]; conditions[heading] = basic_summary(part)
        conditions[heading]['length_strata'] = {s: basic_summary([r for r in part if (r['frames'] <= 90) == (s == 'short')], LENGTH_CORE) for s in ('short', 'extended')}
    groups = list(basic['groups'].values()); metrics = {key: float(np.mean([g[key] for g in groups])) for key in (
        'mean_fk_inconsistency_m', 'mean_projection_node_p95_m', 'mean_penetration_over_5cm_fraction', 'mean_rotation_threshold_exceed_fraction')}
    metrics.update(pooled_projection_node_p95_m=result['pooled_projection_node_p95_m'],
        max_penetration_m=max(g['max_penetration_m'] for g in groups),
        max_projection_node_error_m=max(g['max_projection_node_error_m'] for g in groups), geometry_score_m=basic['geometry_score_m'])
    absolute = all(g['p95_clip_max_penetration_m'] <= .05 and g['max_penetration_m'] <= .1 and
        g['mean_penetration_over_5cm_fraction'] <= .01 and g['mean_projection_node_p95_m'] <= .03 and
        g['max_projection_node_error_m'] <= .1 and g['mean_rotation_threshold_exceed_fraction'] <= .001 for g in groups)
    report = {'schema': 'm5-basic-guidance-frozen-model-report-v1', 'setting': name, 'guidance': SETTINGS[name],
        'checkpoint_sha256': plan['checkpoint_sha256'], 'weights': 'ema', 'model_step': 60000,
        'identity': identity, 'policy_sha256': plan['policy_sha256'], 'cohorts_sha256': plan['cohorts_sha256'],
        'basic_events': 180, 'generated_clips': 1080, 'ddim_steps': 100, 'batch_size': 16,
        'conditions': conditions, 'groups': basic['groups'], 'basic_metrics': metrics,
        'basic_motion_both_headings_and_lengths': all(c['basic_amplitude_passed'] and all(s['basic_amplitude_passed'] for s in c['length_strata'].values()) for c in conditions.values()),
        'basic_absolute_geometry_passed': absolute and metrics['pooled_projection_node_p95_m'] <= .03,
        'denoising_basic': read(SCOPE / 'assessments_revision2/T2_cons_uniform/report.json')['denoising_basic'],
        'numerical_validity_all_prompts': result['all_finite'] and result['all_valid_qpos'],
        'numerical_validity_scope': 'All180basicprompts inthisstudy; complex120notregeneratedundertheseCFGsettings.',
        'all300_program_validity_proven': name == 'G20_control_cfg2',
        'model_unchanged': before == digest(model), 'new_optimizer_updates': 0, 'new_DDIM_samples': 1080,
        'sampling_source_sha256': sha(ROOT / 'training/m5_evaluation.py'),
        'physical_metric_source_sha256': sha(ROOT / 'training/m5_physical_evaluation.py'),
        'source_sha256': sha(Path(__file__)), 'final_audit_model_scores_read': False, 'M5_complete': False}
    assert report['model_unchanged']; write(output / 'report.json', report)
    write(output / 'COMPLETED.json', {'completed_at_utc': now(), 'report_sha256': sha(output / 'report.json'),
          'checkpoint_sha256': plan['checkpoint_sha256'], 'new_optimizer_updates': 0, 'new_DDIM_samples': 1080, 'M5_complete': False})
    print({'setting': name, 'metrics': metrics, 'basic_motion': report['basic_motion_both_headings_and_lengths']}, flush=True)


def assess(study, wait):
    plan = validate(study); processes = read(study / 'processes.json')
    while True:
        done = [n for n in SETTINGS if (study / n / 'COMPLETED.json').exists()]
        pending = [n for n in SETTINGS if n not in done]
        missing = [n for n in pending if not process_live(processes[n], study / n)]
        write(study / 'status.json', {'phase': 'requires_inspection' if missing else 'inference_running',
            'completed': done, 'active': {n: processes[n]['pid'] for n in pending if n not in missing},
            'missing_live_handles': missing, 'new_optimizer_updates': 0, 'updated_at_utc': now(), 'M5_complete': False})
        if missing: raise RuntimeError('Inference worker terminalbeforecompletion; no automaticrestart: ' + str(missing))
        if not pending: break
        if not wait: print({'pending': pending}); return
        time.sleep(30)
    from scripts.assess_m5_basic_scope_v3 import compare
    reports = {n: read(study / n / 'report.json') for n in SETTINGS}; control = reports['G20_control_cfg2']; comparisons = {}
    for name in ('G10_cfg1', 'G15_cfg1p5'):
        value = compare(reports[name], control); a, b = reports[name]['basic_metrics'], control['basic_metrics']
        gains = {k: a[k] <= .95 * b[k] for k in ('mean_fk_inconsistency_m', 'pooled_projection_node_p95_m')}
        value.update(both_named_geometry_improved_ge5pct=gains, numeric_guidance_candidate=value['research_guards_passed_v3'] and all(gains.values()))
        comparisons[name] = value
    qualified = [n for n, v in comparisons.items() if v['numeric_guidance_candidate']]
    selected = min(qualified, key=lambda n: reports[n]['basic_metrics']['geometry_score_m']) if qualified else None
    original = read(SCOPE / 'assessments_revision2/T2_cons_uniform/report.json')['basic_metrics']
    results = {'schema': 'm5-basic-guidance-study-decision-v1', 'assessed_at_utc': now(),
        'protocol_sha256': sha(study / 'protocol.json'), 'checkpoint_sha256': plan['checkpoint_sha256'],
        'comparisons_vs_same_layout_cfg2': comparisons, 'provisional_numeric_guidance_setting': selected,
        'provisional_guidance': SETTINGS[selected] if selected else 2.,
        'metrics_by_setting': {n: r['basic_metrics'] for n, r in reports.items()},
        'regenerated_cfg2_vs_legacy_basic_metrics': {k: control['basic_metrics'][k] - original[k] for k in original},
        'actual_new_DDIM_samples': 3240, 'paired_keys': 1080,
        'semantic_quality_not_proven_by_geometry': True, 'human_review_still_required': True,
        'automatic_guidance_adoption_or_training': False, 'original_control_outputs_modified': False,
        'source_sha256': sha(Path(__file__)), 'new_optimizer_updates': 0,
        'registered_formal_training_budget_unchanged': 515000, 'final_audit_model_scores_read': False, 'M5_complete': False}
    validate(study); write(study / 'decision.json', results)
    write(study / 'status.json', {'phase': 'sampling_budget_complete', 'completed': list(SETTINGS),
          'new_DDIM_samples': 3240, 'new_optimizer_updates': 0, 'updated_at_utc': now(), 'M5_complete': False})
    print({'numeric_guidance_candidate': results['provisional_guidance'], 'M5_complete': False}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('command', choices=('start', 'worker', 'assess'))
    p.add_argument('--study-dir', type=Path, default=DEFAULT); p.add_argument('--setting', choices=tuple(SETTINGS))
    p.add_argument('--run-dir', type=Path); p.add_argument('--wait', action='store_true'); a = p.parse_args()
    study = a.study_dir.resolve(); study.relative_to(ROOT / 'runs')
    if a.command == 'start': start(study)
    elif a.command == 'worker': worker(study, a.setting, a.run_dir.resolve())
    else: assess(study, a.wait)
