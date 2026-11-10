#!/usr/bin/env python3
"""Sidecar M5v3 basic-motion assessment on exact retained development outputs."""
import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from training.g1_physical import PhysicalG1Converter
from training.m5_basic_scope import BASIC_GROUPS
from training.m5_evaluation import summarize
from scripts.run_m5_length_research import read, write, sha, now

SCOPE = ROOT / 'runs/m5_basic_scope_20261104'
POLICY_FILE = SCOPE / 'M5_acceptance_protocol_v3_basic_revision2.json'
LENGTH_CORE = ('walk', 'jog', 'raise_arms')
RUNS = {'L0_short': ROOT / 'runs/m5_length_research_20261104/L0_short_G1_control',
        'T0_raw_uniform': ROOT / 'runs/m5_curriculum_factorial_20261104/T0_raw_uniform',
        'T1_raw_high': ROOT / 'runs/m5_curriculum_factorial_20261104/T1_raw_high',
        'T2_cons_uniform': ROOT / 'runs/m5_curriculum_factorial_20261104/T2_cons_uniform',
        'T3_cons_high': ROOT / 'runs/m5_length_research_20261104/L1_mixed_G1'}


def basic_summary(rows, required_groups=BASIC_GROUPS):
    if not rows: return {'coverage': False, 'basic_amplitude_passed': False, 'groups': {}}
    result = summarize(rows); result.pop('core_amplitude_passed', None)
    result['required_amplitude_groups'] = list(required_groups)
    result['coverage'] = set(required_groups) <= set(result['groups'])
    result['basic_amplitude_passed'] = result['coverage'] and all(result['groups'][g]['amplitude_passed'] for g in required_groups)
    return result


def denoising_summary(denoising, basic_ids):
    rows = [r for r in denoising['per_sample'] if r['sample_id'] in basic_ids]
    if not rows: raise ValueError('No basic events in the frozen denoising monitor')
    strata = sorted({(r['group'], r['cohort']) for r in rows})
    macro = float(np.mean([np.mean([r['loss'] for r in rows if (r['group'], r['cohort']) == key]) for key in strata]))
    return {'macro_loss': macro, 'supervised_fk_mean': float(np.mean([r['fk'] for r in rows])),
            'fk_by_t': {str(t): float(np.mean([r['fk'] for r in rows if r['t'] == t])) for t in (0, 249, 499, 749, 999)},
            'events': len({r['sample_id'] for r in rows}), 'rows': len(rows), 'scope': 'Basic subset of original91event denoising monitor.'}


def assess(label, run, output, policy, cohorts):
    if (output / 'report.json').exists(): raise FileExistsError('Preserve existing v3 assessment: ' + label)
    done = read(run / 'COMPLETED.json'); step = done['step']; assert step == 60000
    checkpoint = run / 'checkpoints/latest.pt'; assert sha(checkpoint) == done['terminal_checkpoint_sha256']
    folder = run / 'evaluations/generation_full_060000'; legacy = read(folder / 'report.json'); rows = read(folder / 'per_sample.json')
    assert legacy['identity'] == done['identity'] and legacy['step'] == 60000 and len(rows) == 1800
    regenerated = summarize(rows)
    for group, values in regenerated['groups'].items():
        for key in ('mean_fk_inconsistency_m', 'mean_projection_node_p95_m', 'max_penetration_m',
                    'mean_rotation_threshold_exceed_fraction', 'noncollapsed_fraction'):
            a, b = values[key], legacy['groups'][group][key]
            assert a is b is None or a is not None and b is not None and abs(a - b) < 1e-10
    classes = cohorts['development']['classifications']; basic_ids = {k for k, v in classes.items() if v['difficulty'] == 'basic'}
    selected = [r for r in rows if r['sample_id'] in basic_ids]; extended = [r for r in rows if r['sample_id'] not in basic_ids]
    assert len(selected) + len(extended) == 1800 and {r['sample_id'] for r in selected} == basic_ids
    assert all(classes[r['sample_id']]['group'] == r['group'] for r in rows)
    output.mkdir(parents=True, exist_ok=True); converter = PhysicalG1Converter(); errors = []; by_group = defaultdict(list)
    input_hashes = {}; max_replay_difference = 0.
    for start in range(0, len(selected), 16):
        part = selected[start:start + 16]; qpos = []; originals = []
        for row in part:
            stem = folder / row['artifact_prefix']; qpath = stem.with_suffix('.csv'); mpath = stem.with_suffix('.npz')
            qb = qpath.read_bytes(); mb = mpath.read_bytes()
            import hashlib
            input_hashes[str(qpath)] = hashlib.sha256(qb).hexdigest(); input_hashes[str(mpath)] = hashlib.sha256(mb).hexdigest()
            q = np.loadtxt(io.BytesIO(qb), delimiter=',').astype(np.float32)
            with np.load(io.BytesIO(mb)) as archive: positions = archive['posed_joints']
            assert q.shape == (row['frames'], 36) and positions.shape == (row['frames'], 34, 3)
            qpos.append(q); originals.append(positions)
        width = max(len(q) for q in qpos); batch = np.stack([np.pad(q, ((0, width - len(q)), (0, 0)), mode='edge') for q in qpos])
        with torch.no_grad(): rebuilt = converter.qpos_to_motion_dict(batch)['posed_joints'].numpy()
        for i, row in enumerate(part):
            value = np.linalg.norm(rebuilt[i, :len(qpos[i])] - originals[i], axis=-1)
            difference = max(abs(float(value.max()) - row['projection_node_max_m']),
                             abs(float(np.quantile(value, .95)) - row['projection_node_p95_m']))
            assert difference < 5e-6, (label, row['sample_id'], difference)
            max_replay_difference = max(max_replay_difference, difference); errors.append(value.reshape(-1)); by_group[row['group']].append(value.reshape(-1))
        write(output / 'progress.json', {'phase': 'exact_basic_node_pool_replay', 'clips': min(start + 16, len(selected)),
              'target_clips': len(selected), 'updated_at_utc': now(), 'new_optimizer_updates': 0})
    pooled = np.concatenate(errors); assert np.isfinite(pooled).all()
    np.save(output / 'basic_projection_errors.npy', pooled)
    result = basic_summary(selected); result['conditions'] = {}
    for heading in ('native', 'heading0'):
        part = [r for r in selected if r['condition'] == heading]; summary = basic_summary(part)
        summary['length_strata'] = {label: basic_summary([r for r in part if (r['frames'] <= 90) == (label == 'short')], LENGTH_CORE)
                                   for label in ('short', 'extended')}
        result['conditions'][heading] = summary
    values = list(result['groups'].values())
    metrics = {key: float(np.mean([g[key] for g in values])) for key in (
        'mean_fk_inconsistency_m', 'mean_projection_node_p95_m', 'mean_penetration_over_5cm_fraction', 'mean_rotation_threshold_exceed_fraction')}
    metrics.update(pooled_projection_node_p95_m=float(np.quantile(pooled, .95)),
                   max_penetration_m=max(g['max_penetration_m'] for g in values),
                   max_projection_node_error_m=max(g['max_projection_node_error_m'] for g in values),
                   geometry_score_m=result['geometry_score_m'])
    geometry_checks = {g: {'p95_clip_foot_le5cm': v['p95_clip_max_penetration_m'] <= .05,
        'max_foot_le10cm': v['max_penetration_m'] <= .1, 'deep_frame_fraction_le1pct': v['mean_penetration_over_5cm_fraction'] <= .01,
        'mean_clip_node_p95_le3cm': v['mean_projection_node_p95_m'] <= .03,
        'max_node_le10cm': v['max_projection_node_error_m'] <= .1,
        'reference_rotation_exceed_le0p1pct': v['mean_rotation_threshold_exceed_fraction'] <= .001} for g, v in result['groups'].items()}
    basic_motion = all(c['basic_amplitude_passed'] and all(s['basic_amplitude_passed'] for s in c['length_strata'].values())
                       for c in result['conditions'].values())
    result.update(schema='m5-basic-development-assessment-v3', label=label, run=str(run), step=step,
        assessed_at_utc=now(), policy_sha256=sha(POLICY_FILE),
        cohorts_sha256=sha(SCOPE / 'cohorts.json'), checkpoint_sha256=done['terminal_checkpoint_sha256'], weights='ema',
        original_full_events=300, original_full_clips=1800, basic_events=len(basic_ids), basic_clips=len(selected),
        extended_events=300-len(basic_ids), extended_clips=len(extended),
        numerical_validity_all_prompts=legacy['all_finite'] and legacy['all_valid_qpos'],
        basic_motion_both_headings_and_lengths=basic_motion, basic_metrics=metrics,
        basic_absolute_geometry_checks=geometry_checks,
        basic_absolute_geometry_passed=all(all(c.values()) for c in geometry_checks.values()) and metrics['pooled_projection_node_p95_m'] <= .03,
        pooled_projection_p95_by_basic_group={g: float(np.quantile(np.concatenate(v), .95)) for g, v in by_group.items()},
        pooled_node_error_count=len(pooled), pooled_error_replay_max_difference_m=max_replay_difference,
        denoising_basic=denoising_summary(read(run / 'evaluations/denoising_060000.json')['ema'], basic_ids),
        extended_report_only=summarize(extended), extended_quality_blocks_M5=False,
        source_sha256=sha(Path(__file__)), physical_adapter_sha256=sha(ROOT / 'training/g1_physical.py'),
        original_report_sha256=sha(folder / 'report.json'), original_per_sample_sha256=sha(folder / 'per_sample.json'),
        input_artifact_sha256=input_hashes, new_optimizer_updates=0, new_DDIM_samples=0,
        seed3102_quality_scores_read=False, final_audit_model_scores_read=False, human_gate_passed=False, M5_complete=False)
    write(output / 'report.json', result); print(label, metrics, 'basic_motion', basic_motion, flush=True)
    return result


def compare(candidate, control):
    a, b = candidate['basic_metrics'], control['basic_metrics']; regression = []
    for heading in ('native', 'heading0'):
        for stratum in (None, 'short'):
            x = candidate['conditions'][heading]; y = control['conditions'][heading]
            if stratum: x = x['length_strata'][stratum]; y = y['length_strata'][stratum]
            for group in (LENGTH_CORE if stratum else BASIC_GROUPS):
                u = x['groups'].get(group, {}).get('noncollapsed_fraction'); v = y['groups'].get(group, {}).get('noncollapsed_fraction')
                if u is None or v is None or u + .05 + 1e-12 < v: regression.append(f'{heading}/{stratum}/{group}')
    tolerances = {'mean_fk_inconsistency_m': 1e-4, 'pooled_projection_node_p95_m': 1e-4,
        'mean_projection_node_p95_m': 1e-4, 'max_penetration_m': 1e-4, 'max_projection_node_error_m': 1e-4,
        'mean_penetration_over_5cm_fraction': 1e-4, 'mean_rotation_threshold_exceed_fraction': 1e-5}
    geometry = {k: a[k] <= 1.1 * b[k] + tol for k, tol in tolerances.items()}
    da, db = candidate['denoising_basic'], control['denoising_basic']
    checks = {'finite_valid_all_prompts': candidate['numerical_validity_all_prompts'],
        'basic_motion_both_headings_lengths': candidate['basic_motion_both_headings_and_lengths'],
        'basic_amplitude_no_regression_over5pp': not regression, 'key_basic_geometry_no_regression_over10pct': all(geometry.values()),
        'basic_norm6_guard': da['macro_loss'] <= 1.05 * db['macro_loss'],
        'basic_supervised_fk_guard': da['supervised_fk_mean'] <= 1.1 * db['supervised_fk_mean'],
        'basic_supervised_fk_any_t_guard': all(da['fk_by_t'][k] <= 1.25 * db['fk_by_t'][k] for k in da['fk_by_t'])}
    return {'checks': checks, 'research_guards_passed_v3': all(checks.values()),
        'failed_checks': [k for k, value in checks.items() if not value], 'basic_geometry_regression_checks': geometry,
        'basic_amplitude_regression_strata': regression,
        'relative_basic_geometry_changes': {k: a[k] / b[k] - 1 if b[k] else None for k in a},
        'basic_absolute_geometry_passed': candidate['basic_absolute_geometry_passed'],
        'complex_quality_blocks_decision': False, 'M5_complete': False}


def main(output):
    torch.set_num_threads(4); policy = read(POLICY_FILE); cohorts = read(SCOPE / 'cohorts.json')
    assert sha(ROOT / 'training/m5_basic_scope.py') == policy['classifier_sha256']
    assert sha(SCOPE / 'cohorts.json') == policy['cohorts_sha256']
    if (output / 'development_decision_v3.json').exists(): raise FileExistsError('Preserve completed scope assessment')
    reports = {}
    for label, run in RUNS.items():
        folder = output / label
        if (folder / 'report.json').exists():
            r = read(folder / 'report.json'); assert r['source_sha256'] == sha(Path(__file__)) and r['policy_sha256'] == sha(POLICY_FILE)
            reports[label] = r
        else: reports[label] = assess(label, run, folder, policy, cohorts)
    comparisons = {label: compare(r, reports['T3_cons_high']) for label, r in reports.items() if label not in ('T3_cons_high', 'L0_short')}
    result = {'schema': 'm5-basic-development-decision-v3', 'created_at_utc': now(),
        'policy_sha256': sha(POLICY_FILE),
        'matched_original_D1_vs_D0': compare(reports['T3_cons_high'], reports['L0_short']),
        'factorial_alternatives_vs_T3': comparisons,
        'new_noise_or_dropout_setting_still_needs_matched_short_control': True,
        'model_summaries': {k: {'basic_metrics': r['basic_metrics'], 'basic_motion': r['basic_motion_both_headings_and_lengths'],
            'basic_absolute_geometry_passed': r['basic_absolute_geometry_passed'], 'checkpoint_sha256': r['checkpoint_sha256']} for k, r in reports.items()},
        'user_authorized_scope_change': True, 'original_v2_results_modified': False,
        'new_optimizer_updates': 0, 'new_DDIM_samples': 0, 'final_audit_model_scores_read': False,
        'source_sha256': sha(Path(__file__)), 'M5_complete': False}
    write(output / 'development_decision_v3.json', result); print(result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output-dir', type=Path, default=SCOPE / 'assessments_revision2')
    main(p.parse_args().output_dir.resolve())
