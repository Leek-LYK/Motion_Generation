#!/usr/bin/env python3
"""Descriptive paired 50k milestone; the registered 60k decision stays separate."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from training.m5_review_statistics import source_clusters
from scripts.run_m5_length_research import read, write, sha, now, validate

GROUPS = ('walk', 'jog', 'jump', 'raise_arms', 'turn', 'squat')


def metrics(report, denoising):
    groups = list(report['groups'].values())
    return {'mean_fk_self_m': float(np.mean([r['mean_fk_inconsistency_m'] for r in groups])),
            'pooled_projection_p95_m': report['pooled_projection_node_p95_m'],
            'macro_mean_clip_projection_p95_m': float(np.mean([r['mean_projection_node_p95_m'] for r in groups])),
            'max_penetration_m': max(r['max_penetration_m'] for r in groups),
            'max_node_projection_m': max(r['max_projection_node_error_m'] for r in groups),
            'deep_frame_macro_fraction': float(np.mean([r['mean_penetration_over_5cm_fraction'] for r in groups])),
            'rotation_exceedance_macro_fraction': float(np.mean([r['mean_rotation_threshold_exceed_fraction'] for r in groups])),
            'ema_norm6': denoising['macro_loss'], 'ema_supervised_fk': denoising['parts']['fk']}


def paired_intervals(rows, clusters):
    labels = sorted(set(clusters.values())); index = {v: i for i, v in enumerate(labels)}
    denominator = np.zeros((len(labels), 6)); numerator = np.zeros((len(labels), 6, 2))
    fields = ('fk_inconsistency_m', 'projection_node_p95_m')
    for key, first in rows['short'].items():
        second = rows['mixed'][key]; i = index[clusters[key[0]]]; j = GROUPS.index(first['group'])
        denominator[i, j] += 1
        for k, field in enumerate(fields):
            numerator[i, j, k] += second[field] - first[field]
    assert (denominator > 0).sum(0).min() >= 2
    rng = np.random.default_rng(51173); batches = []; rejected = 0
    while sum(len(x) for x in batches) < 5000:
        count = min(1000, 5000 - sum(len(x) for x in batches))
        weights = rng.multinomial(len(labels), np.full(len(labels), 1. / len(labels)), size=count)
        d = weights @ denominator; good = (d > 0).all(1); rejected += int((~good).sum())
        if good.any():
            n = np.einsum('bc,cgf->bgf', weights[good], numerator)
            batches.append((n / d[good, :, None]).mean(1))
    values = np.concatenate(batches, axis=0)[:5000]
    point = (numerator.sum(0) / denominator.sum(0)[:, None]).mean(0)
    return {'source_clusters': len(labels), 'draws': 5000, 'seed': 51173,
            'estimand': 'Equal-six-class macro of paired clip metrics; mixed minus short. Repeated generation seeds/headings remain within each source cluster.',
            'fk_self_difference_m': {'point': float(point[0]), 'ci95': np.quantile(values[:, 0], [.025, .975]).tolist()},
            'mean_clip_projection_p95_difference_m': {'point': float(point[1]), 'ci95': np.quantile(values[:, 1], [.025, .975]).tolist()},
            'pooled_projection_p95_ci_computed': False,
            'note': 'A confidence interval for mean per-clip P95 is not an interval for pooled node P95.',
            'empty_class_draws_rejected': rejected}


def main():
    study = ROOT / 'runs/m5_length_research_20261104'; plan = validate(study)
    output = study / 'diagnostics/full_050000.json'
    if output.exists():
        raise FileExistsError('Preserve existing midpoint analysis')
    runs = {'short': ROOT / 'runs/m5_geometry_20261104/G1_consistency', 'mixed': study / 'L1_mixed_G1'}
    evidence = {}; reports = {}; rows = {}; configs = {}; datasets = {}; denoising = {}
    for label, run in runs.items():
        checkpoint = run / 'checkpoints/step_050000.pt'; checksum = sha(checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        folder = run / 'evaluations/generation_full_050000'; report = read(folder / 'report.json')
        cfg = read(run / 'config.json'); subset = read(run / 'subset.json')
        assert state['step'] == report['step'] == 50000 and state['stage_updates'] == 10000
        assert state['config'] == cfg and state['identity'] == report['identity']
        assert read(run / 'lineage.json')['parent_sha256'] == plan['common_40k_parent_sha256']
        assert report['events'] == 300 and report['generated_clips'] == 1800
        assert report['seeds'] == [0, 42, 1234] and report['headings'] == [None, 0.]
        assert report['guidance'] == 2. and report['ddim_steps'] == 100 and report['weights'] == 'ema'
        records = read(folder / 'per_sample.json')
        paired = {(r['sample_id'], r['seed'], r['condition']): r for r in records}
        expected = {(e['sample_id'], seed, h) for e in subset['val'] for seed in (0, 42, 1234) for h in ('native', 'heading0')}
        assert len(records) == len(paired) == 1800 and set(paired) == expected
        reports[label] = report; rows[label] = paired; configs[label] = cfg; datasets[label] = subset
        denoising[label] = read(run / 'evaluations/denoising_050000.json')['ema']
        evidence[label] = {'run': str(run), 'checkpoint_sha256': checksum, 'checkpoint_step': state['step'],
                           'checkpoint_stage_updates': state['stage_updates'], 'report_sha256': sha(folder / 'report.json'),
                           'per_sample_sha256': sha(folder / 'per_sample.json')}
        del state
    assert datasets['short']['val'] == datasets['mixed']['val'] and datasets['short']['monitor'] == datasets['mixed']['monitor']
    changes = {k for k in set(configs['short']) | set(configs['mixed']) if configs['short'].get(k) != configs['mixed'].get(k)}
    assert changes <= {'stage', 'subset_sha256', 'train_samples', 'minimum_steps', 'training_stop_step', 'constraint_activation_protocol'}
    for key in reports['short']['identity']:
        if key != 'subset_sha256':
            assert reports['short']['identity'][key] == reports['mixed']['identity'][key]
    for key, a in rows['short'].items():
        b = rows['mixed'][key]
        assert all(a[k] == b[k] for k in ('group', 'text', 'frames', 'reference_amplitude', 'heading', 'unsupported_context', 'reference_quality_flag'))
    summary = {label: metrics(reports[label], denoising[label]) for label in runs}
    clusters = source_clusters(datasets['short']['val']); intervals = paired_intervals(rows, clusters)
    assert abs(intervals['fk_self_difference_m']['point'] - (summary['mixed']['mean_fk_self_m'] - summary['short']['mean_fk_self_m'])) < 1e-8
    assert abs(intervals['mean_clip_projection_p95_difference_m']['point'] - (summary['mixed']['macro_mean_clip_projection_p95_m'] - summary['short']['macro_mean_clip_projection_p95_m'])) < 1e-8
    amplitude = {label: {h: {s: {'core_passed': v['core_amplitude_passed'],
        'groups': {g: {k: row[k] for k in ('eligible', 'noncollapsed_fraction', 'median_amplitude_ratio', 'amplitude_passed')}
                   for g, row in v['groups'].items()}} for s, v in condition['length_strata'].items()}
        for h, condition in report['conditions'].items()} for label, report in reports.items()}
    result = {'schema': 'm5-length-full-midpoint-v1', 'analyzed_at_utc': now(), 'step': 50000,
              'protocol_sha256': sha(study / 'protocol.json'), 'evidence': evidence, 'matched_clip_keys': 1800,
              'configuration_differences': sorted(changes), 'metrics': summary,
              'mixed_relative_changes': {k: summary['mixed'][k] / summary['short'][k] - 1. if summary['short'][k] else None for k in summary['short']},
              'amplitude': amplitude, 'paired_source_cluster_bootstrap': intervals,
              'all_finite_and_valid_qpos': all(r['all_finite'] and r['all_valid_qpos'] for r in reports.values()),
              'interim_only': True, 'fixed_decision_endpoint': 60000, 'endpoint_admission_evaluated': False,
              'scope': 'One training seed and a fixed intermediate milestone; 300 development events, 3 generation seeds, 2 headings. No early recipe choice or automatic E.',
              'source_sha256': sha(Path(__file__)), 'new_optimizer_updates': 0, 'new_diffusion_samples': 0,
              'seed3102_quality_scores_read': False, 'final_audit_model_scores_read': False, 'M5_complete': False}
    write(output, result)
    print({'output': str(output), 'metrics': summary, 'paired_intervals': intervals}, flush=True)


if __name__ == '__main__':
    main()
