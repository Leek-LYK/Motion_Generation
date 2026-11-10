#!/usr/bin/env python3
"""Describe existing GT amplitude bins and completed60k outputs; no gate changes."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.run_m5_length_research import read, write, sha, now
from training.m5_evaluation import MINIMUM, summarize


def bucket(n):
    return 'short' if n <= 90 else 'medium' if n <= 120 else 'long'


def profile(rows):
    result = {}
    for group in ('jump', 'raise_arms'):
        result[group] = {}
        for length in ('short', 'medium', 'long'):
            selected = [r for r in rows if r['group'] == group and r['length_bin'] == length]
            values = np.array([r['amplitude'] for r in selected]); eligible = values[values >= MINIMUM[group]]
            result[group][length] = {'all_events': len(values), 'eligible_events': len(eligible),
                'eligible_amplitude_p10_p50_p90_m': np.quantile(eligible, [.1, .5, .9]).tolist() if len(eligible) else None,
                'numeric_vertical_range_ge10cm': int((values >= .1).sum()) if group == 'jump' else None,
                'numeric_vertical_range_ge20cm': int((values >= .2).sum()) if group == 'jump' else None,
                'eligible_vertical_range_ge20cm_fraction': float((eligible >= .2).mean()) if group == 'jump' and len(eligible) else None}
    return result


def main():
    base = ROOT / 'runs/m5_physical_20261103'; folder = base / 'diagnostics/amplitude_coverage'; output = folder / 'detail.json'
    if output.exists():
        raise FileExistsError('Preserve the completed descriptive audit')
    audit = read(folder / 'report.json'); assert audit['passed']
    assert sha(folder / 'per_event.json') == audit['per_event_sha256']
    rows = read(folder / 'per_event.json'); by_id = {r['sample_id']: r for r in rows}
    short = read(base / 'data/D_short_3500.json'); small = read(base / 'data/D_mixed_small_3500.json'); large = read(base / 'data/D_mixed_large_4500.json')
    small_ids = {r['sample_id'] for r in small['train']}; extra_ids = {r['sample_id'] for r in large['train']} - small_ids
    subsets = {'short_train': [by_id[r['sample_id']] for r in short['train']],
               'mixed_small_train': [by_id[r['sample_id']] for r in small['train']],
               'mixed_large_train': [by_id[r['sample_id']] for r in large['train']],
               'expansion_new961': [by_id[s] for s in sorted(extra_ids)],
               'development': [r for r in rows if r['split'] == 'development']}
    details = {name: profile(r) for name, r in subsets.items()}; models = {}; model_sources = {}
    for name in ('L0_short_G1_control', 'L1_mixed_G1'):
        run = ROOT / 'runs/m5_length_research_20261104' / name; done = read(run / 'COMPLETED.json')
        report_dir = run / 'evaluations/generation_full_060000'; report = read(report_dir / 'report.json')
        assert done['step'] == 60000 and report['identity'] == done['identity']
        assert sha(run / 'checkpoints/step_060000.pt') == done['terminal_checkpoint_sha256']
        generated = read(report_dir / 'per_sample.json'); assert len(generated) == 1800
        models[name] = {}
        for heading in ('heading0', 'native'):
            models[name][heading] = {}
            for length in ('short', 'medium', 'long'):
                part = [r for r in generated if r['condition'] == heading and r['group'] == 'jump' and bucket(r['frames']) == length]
                # Reuse the actual frozen evaluator, including its .35 ratio threshold.
                value = summarize(part)['groups']['jump']
                models[name][heading][length] = {k: value[k] for k in ('events_with_seeds', 'eligible',
                    'median_amplitude_ratio', 'noncollapsed_fraction', 'amplitude_passed')}
            reconstructed = summarize([r for r in generated if r['condition'] == heading and r['group'] == 'jump' and r['frames'] > 90])['groups']['jump']
            original = report['conditions'][heading]['length_strata']['extended']['groups']['jump']
            assert reconstructed['noncollapsed_fraction'] == original['noncollapsed_fraction']
            assert reconstructed['median_amplitude_ratio'] == original['median_amplitude_ratio']
        model_sources[name] = {'checkpoint_sha256': done['terminal_checkpoint_sha256'], 'report_sha256': sha(report_dir / 'report.json'),
                               'per_sample_sha256': sha(report_dir / 'per_sample.json')}
    training = details['mixed_small_train']['jump']['medium']; dev = details['development']['jump']['medium']
    expansion = details['expansion_new961']['jump']['medium']
    result = {'schema': 'm5-amplitude-coverage-detail-v1', 'completed_at_utc': now(), 'passed': True,
        'ground_truth_profiles': details, 'completed60k_jump_diagnostics': models, 'model_sources': model_sources,
        'medium_jump_eligible_median_dev_over_train': dev['eligible_amplitude_p10_p50_p90_m'][1] / training['eligible_amplitude_p10_p50_p90_m'][1],
        'new961_medium_jump_events': expansion['all_events'],
        'new961_medium_jump_vertical_range_ge20cm_events': expansion['numeric_vertical_range_ge20cm'],
        'noncollapse_definition': 'Frozen summarize: reference>=0.05m for jump; generated/reference>=0.35; class median in[0.5,2], fraction>=0.75.',
        'scope': 'Descriptive split and data coverage on development. Original short/extended gates unchanged; no independence claimed for repeated generation seeds or 23 medium development motions.',
        'data_change_or_sampler_change': False, 'automatic_reweighting': False,
        'reference_metric_note': 'Jump amplitude is the P90-P10 root-height range, not peak flight height or proof of complete takeoff/landing.',
        'source_sha256': sha(Path(__file__)), 'frozen_evaluator_sha256': sha(ROOT / 'training/m5_evaluation.py'),
        'audit_sha256': sha(folder / 'report.json'), 'new_optimizer_updates': 0, 'new_diffusion_samples': 0,
        'seed3102_quality_scores_read': False, 'final_audit_model_scores_read': False, 'M5_complete': False}
    write(output, result)
    print({'output': str(output), 'medium_jump_gt_median_ratio': result['medium_jump_eligible_median_dev_over_train'],
           'added_medium_jumps_ge20cm': result['new961_medium_jump_vertical_range_ge20cm_events'],
           'mixed_completed60k': models['L1_mixed_G1']}, flush=True)


if __name__ == '__main__':
    main()
