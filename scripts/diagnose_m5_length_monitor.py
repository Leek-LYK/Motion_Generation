#!/usr/bin/env python3
"""Compare equal-step length-course monitors, including the reused short prefix."""
import argparse
import json
import sys
from pathlib import Path
from statistics import mean
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m5_length_research import read, write, sha, now, validate


def summarize(report):
    groups = list(report['groups'].values())
    return {
        'mean_fk_self_m': mean(v['mean_fk_inconsistency_m'] for v in groups),
        'pooled_node_projection_p95_m': report['pooled_projection_node_p95_m'],
        'mean_clip_node_projection_p95_m': mean(v['mean_projection_node_p95_m'] for v in groups),
        'max_penetration_m': max(v['max_penetration_m'] for v in groups),
        'max_node_projection_m': max(v['max_projection_node_error_m'] for v in groups),
        'deep_frame_macro_fraction': mean(v['mean_penetration_over_5cm_fraction'] for v in groups),
        'rotation_exceedance_macro_fraction': mean(v['mean_rotation_threshold_exceed_fraction'] for v in groups),
    }


def amplitude(report):
    return {heading: {name: {
        'core_passed': value['core_amplitude_passed'],
        'groups': {group: {key: row[key] for key in
                          ('events_with_seeds', 'eligible', 'noncollapsed_fraction', 'median_amplitude_ratio', 'amplitude_passed')}
                   for group, row in value['groups'].items()}}
        for name, value in condition['length_strata'].items()}
        for heading, condition in report['conditions'].items()}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--step', type=int, required=True)
    p.add_argument('--study-dir', type=Path, default=ROOT / 'runs/m5_length_research_20261104')
    args = p.parse_args(); study = args.study_dir.resolve(); plan = validate(study)
    if args.step < 42000 or args.step >= 60000 or args.step % 2000:
        raise ValueError('Use a registered intermediate monitor; the fixed 60k endpoint has its own assessor')
    short = ROOT / 'runs/m5_geometry_20261104/G1_consistency' if args.step < 50000 else study / 'L0_short_G1_control'
    mixed = study / 'L1_mixed_G1'
    configs = {label: read(run / 'config.json') for label, run in (('short', short), ('mixed', mixed))}
    changes = {key for key in set(configs['short']) | set(configs['mixed']) if configs['short'].get(key) != configs['mixed'].get(key)}
    allowed = {'stage', 'stage_start_step', 'minimum_steps', 'training_stop_step', 'subset_sha256',
               'train_samples', 'constraint_activation_protocol'}
    if not changes <= allowed:
        raise ValueError('Unexpected algorithm/config difference: ' + str(changes - allowed))
    for label, run in (('short', short), ('mixed', mixed)):
        expected_parent = plan['common_40k_parent_sha256'] if args.step < 50000 or label == 'mixed' else plan['runs']['L0_short_G1_control']['parent_sha256']
        if read(run / 'lineage.json')['parent_sha256'] != expected_parent:
            raise ValueError('Monitor does not belong to the registered course')
    reports = {}; records = {}; paths = {}
    for label, run in (('short', short), ('mixed', mixed)):
        folder = run / 'evaluations' / f'generation_monitor_{args.step:06d}'
        report = read(folder / 'report.json'); rows = read(folder / 'per_sample.json')
        if not (report['step'] == args.step and report['events'] == 91 and report['generated_clips'] == 182 and
                report['seeds'] == [42] and report['headings'] == [None, 0.] and report['guidance'] == 2. and report['ddim_steps'] == 100):
            raise ValueError('Changed monitor scope')
        records[label] = {(r['sample_id'], r['seed'], r['condition']): r for r in rows}
        expected = {(r['sample_id'], 42, heading) for r in read(run / 'subset.json')['monitor'] for heading in ('native', 'heading0')}
        if len(rows) != 182 or len(records[label]) != 182 or set(records[label]) != expected:
            raise ValueError('Missing or duplicate monitor keys')
        reports[label] = report
        paths[label] = {'run': str(run), 'report_sha256': sha(folder / 'report.json'),
                        'per_sample_sha256': sha(folder / 'per_sample.json'),
                        'denoising_sha256': sha(run / 'evaluations' / f'denoising_{args.step:06d}.json')}
    if set(records['short']) != set(records['mixed']):
        raise ValueError('Paired monitor keys differ')
    for key in records['short']:
        a, b = records['short'][key], records['mixed'][key]
        if any(a[k] != b[k] for k in ('group', 'text', 'frames', 'reference_amplitude', 'heading', 'unsupported_context', 'reference_quality_flag')):
            raise ValueError('Paired source metadata differs')
    for key in reports['short']['identity']:
        if key != 'subset_sha256' and reports['short']['identity'][key] != reports['mixed']['identity'][key]:
            raise ValueError('Unexpected evaluation identity difference: ' + key)
    metrics = {label: summarize(report) for label, report in reports.items()}
    denoising = {label: read(Path(paths[label]['run']) / 'evaluations' / f'denoising_{args.step:06d}.json')['ema'] for label in paths}
    result = {'schema': 'm5-length-paired-monitor-v1', 'analyzed_at_utc': now(), 'step': args.step,
        'protocol_sha256': sha(study / 'protocol.json'), 'sources': paths,
        'matched_source_and_noise_keys': 182, 'events': 91, 'sample_seeds': [42], 'headings': 2,
        'configuration_differences': sorted(changes), 'short_uses_archived_reused_prefix': args.step < 50000,
        'metrics': metrics, 'mixed_relative_changes': {
            k: metrics['mixed'][k] / metrics['short'][k] - 1 if metrics['short'][k] else None for k in metrics['short']},
        'amplitude': {label: amplitude(report) for label, report in reports.items()},
        'ema_denoising': {label: {'norm6_macro': value['macro_loss'], 'supervised_fk': value['parts']['fk']} for label, value in denoising.items()},
        'all_finite_and_valid_qpos': all(r['all_finite'] and r['all_valid_qpos'] for r in reports.values()),
        'interim_only': True, 'endpoint_admission_evaluated': False,
        'checkpoint_binding': 'Frozen-worker EMA monitor reports identify the global step; no separate 42k/44k-style checkpoint is claimed. Fixed full endpoint checkpoints remain required.',
        'scope': 'One training seed, one generation seed, 91-event development monitor. Compare equal global steps; event/length distributions differ. No early recipe selection.',
        'new_optimizer_updates': 0, 'final_audit_model_scores_read': False, 'M5_complete': False}
    output = study / 'diagnostics' / f'monitor_{args.step:06d}.json'
    if output.exists():
        old = read(output)
        if any(old[k] != result[k] for k in result if k != 'analyzed_at_utc'):
            raise ValueError('Preserved monitor analysis changed')
        print('Existing matched monitor evidence preserved:', output); return
    write(output, result)
    print(json.dumps({'step': args.step, 'metrics': metrics,
                      'core_pass_by_stratum': {label: {h: {s: v['core_passed'] for s, v in cond.items()} for h, cond in value.items()}
                                               for label, value in result['amplitude'].items()}}, indent=2))


if __name__ == '__main__':
    main()
