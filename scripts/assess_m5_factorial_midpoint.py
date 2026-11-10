#!/usr/bin/env python3
"""Paired four-cell50k description; does not replace the frozen60k assessor."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import torch
from scripts.run_m5_length_research import read, write, sha, now
from scripts.run_m5_curriculum_factorial import validate, RUNS, CONTROL
from scripts.assess_m5_length_midpoint import metrics


def main():
    study = ROOT / 'runs/m5_curriculum_factorial_20261104'; plan = validate(study)
    output = study / 'diagnostics/full_050000.json'
    if output.exists():
        raise FileExistsError('Preserve completed midpoint evidence')
    runs = {name: study / name for name in RUNS}; runs[CONTROL] = Path(plan['reused_control'])
    reports = {}; rows = {}; evidence = {}; values = {}; reference_subset = None
    for name, run in runs.items():
        folder = run / 'evaluations/generation_full_050000'; report = read(folder / 'report.json')
        checkpoint = run / 'checkpoints/step_050000.pt'; checksum = sha(checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False); cfg = read(run / 'config.json')
        assert state['step'] == report['step'] == 50000 and state['stage_updates'] == 10000
        assert state['config'] == cfg and state['identity'] == report['identity']
        assert read(run / 'lineage.json')['parent_sha256'] == plan['parent_sha256']
        assert report['events'] == 300 and report['generated_clips'] == 1800
        assert report['seeds'] == [0, 42, 1234] and report['headings'] == [None, 0.]
        assert report['ddim_steps'] == 100 and report['guidance'] == 2. and report['weights'] == 'ema'
        subset = read(run / 'subset.json')
        if reference_subset is None:
            reference_subset = subset
        else:
            assert subset == reference_subset
        generated = read(folder / 'per_sample.json')
        lookup = {(r['sample_id'], r['seed'], r['condition']): r for r in generated}
        expected = {(r['sample_id'], s, h) for r in subset['val'] for s in (0, 42, 1234) for h in ('native', 'heading0')}
        assert len(generated) == len(lookup) == 1800 and set(lookup) == expected
        reports[name] = report; rows[name] = lookup
        values[name] = metrics(report, read(run / 'evaluations/denoising_050000.json')['ema'])
        evidence[name] = {'run': str(run), 'checkpoint_sha256': checksum, 'report_sha256': sha(folder / 'report.json'),
                          'per_sample_sha256': sha(folder / 'per_sample.json'),
                          'factor_values': {'consistency_weight': cfg['consistency_weight'], 'high_noise_fraction': cfg['high_noise_fraction']}}
        del state
    baseline = rows[CONTROL]
    for name in RUNS:
        assert reports[name]['identity'] == reports[CONTROL]['identity']
        for key, row in rows[name].items():
            assert all(row[k] == baseline[key][k] for k in ('group', 'text', 'frames', 'reference_amplitude', 'heading'))
    effects = {}
    for metric in values[CONTROL]:
        a, b, c, d = (values[n][metric] for n in (*RUNS, CONTROL))
        effects[metric] = {'consistency_main_effect': .5 * ((c - a) + (d - b)),
                           'high_noise_main_effect': .5 * ((b - a) + (d - c)),
                           'interaction': (d - b) - (c - a)}
    amplitude = {name: {head: {length: {'core_amplitude_passed': item['core_amplitude_passed'],
        'groups': {g: {k: r[k] for k in ('eligible', 'median_amplitude_ratio', 'noncollapsed_fraction', 'amplitude_passed')}
                   for g, r in item['groups'].items()}} for length, item in value['length_strata'].items()}
        for head, value in report['conditions'].items()} for name, report in reports.items()}
    result = {'schema': 'm5-four-cell-fixed50k-midpoint-v1', 'analyzed_at_utc': now(), 'step': 50000,
        'protocol_sha256': sha(study / 'protocol.json'), 'evidence': evidence, 'paired_clip_keys_per_cell': 1800,
        'metrics': values, 'amplitude': amplitude, 'factor_effects_on_metrics': effects,
        'effects_scope': 'Descriptive fixed-milestone point contrasts, one training seed. No confidence intervals or causal generalization across training seeds.',
        'interim_only': True, 'fixed_decision_endpoint': 60000, 'endpoint_admission_evaluated': False,
        'recipe_selected': False, 'automatic_E_launch': False, 'source_sha256': sha(Path(__file__)),
        'new_optimizer_updates': 0, 'new_diffusion_samples': 0, 'seed3102_quality_scores_read': False,
        'final_audit_model_scores_read': False, 'M5_complete': False}
    write(output, result); print({'output': str(output), 'metrics': values}, flush=True)


if __name__ == '__main__':
    main()
