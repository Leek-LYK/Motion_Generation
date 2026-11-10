#!/usr/bin/env python3
"""Audit train/development GT amplitude coverage without model inference or filtering."""
import sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from kimodo.skeleton.registry import build_skeleton
from training.m5_physical_losses import stable_cont6d
from training.m5_evaluation import amplitude, MINIMUM
from scripts.run_m5_length_research import read, write, sha, now

GROUPS = ('walk', 'jog', 'jump', 'raise_arms', 'turn', 'squat')


def length_bin(n):
    return 'short' if n <= 90 else 'medium' if n <= 120 else 'long'


def direct_amplitude(raw, skeleton, group):
    positions = raw[:, 5:107].astype(np.float64).reshape(len(raw), 34, 3)
    positions[..., 0] += raw[:, None, 0]; positions[..., 2] += raw[:, None, 2]
    root = positions[:, skeleton.root_idx]
    rotation = None
    if group in ('walk', 'jog'):
        xy = root[:, [0, 2]]
        smooth = np.stack([np.convolve(np.pad(xy[:, i], (2, 2), mode='edge'), np.array([1, 2, 3, 2, 1]) / 9., mode='valid') for i in range(2)], -1)
        value = float(np.linalg.norm(np.diff(smooth, axis=0), axis=-1).sum())
    elif group in ('jump', 'squat'):
        value = float(np.quantile(root[:, 1], .9) - np.quantile(root[:, 1], .1))
    elif group == 'turn':
        heading = raw[:, 3:5]; angle = np.unwrap(np.arctan2(heading[:, 1], heading[:, 0]))
        value = float(np.quantile(angle, .9) - np.quantile(angle, .1))
    else:
        first = 107 + 6 * skeleton.root_idx
        rotation = stable_cont6d(torch.from_numpy(np.array(raw[:, first:first + 6], copy=True)))[0].numpy()
        wrists = [skeleton.bone_index['left_wrist_yaw_skel'], skeleton.bone_index['right_wrist_yaw_skel']]
        local = np.einsum('tji,tkj->tki', rotation, positions[:, wrists] - root[:, None])
        ranges = np.quantile(local, .9, axis=0) - np.quantile(local, .1, axis=0)
        value = float(np.linalg.norm(ranges, axis=-1).mean())
    return value, positions, root, rotation


def summarize(rows):
    table = {}
    for group in GROUPS:
        table[group] = {}
        for bucket in ('short', 'medium', 'long', 'all'):
            selected = [r for r in rows if r['group'] == group and (bucket == 'all' or r['length_bin'] == bucket)]
            values = np.array([r['amplitude'] for r in selected])
            table[group][bucket] = {'events': len(selected), 'eligible_events': int((values >= MINIMUM[group]).sum()),
                'minimum_reference_amplitude': MINIMUM[group], 'units': 'radians' if group == 'turn' else 'metres',
                'eligible_fraction': float((values >= MINIMUM[group]).mean()) if len(values) else None,
                'quantiles_p10_p50_p90': np.quantile(values, [.1, .5, .9]).tolist() if len(values) else None,
                'near_eligibility_boundary_2e_minus5': int((abs(values - MINIMUM[group]) <= 2e-5).sum())}
    return table


def main():
    base = ROOT / 'runs/m5_physical_20261103'; output = base / 'diagnostics/amplitude_coverage'
    if (output / 'report.json').exists():
        raise FileExistsError('Preserve completed coverage evidence')
    view = read(base / 'data/READY.json'); inputs = Path(view['input_dir'])
    manifests = {name: read(base / 'data' / file) for name, file in (
        ('short', 'D_short_3500.json'), ('mixed_small', 'D_mixed_small_3500.json'), ('mixed_large', 'D_mixed_large_4500.json'))}
    assert all(m['val'] == manifests['short']['val'] for m in manifests.values())
    events = {}
    for manifest in manifests.values():
        for row in manifest['train']:
            if row['sample_id'] in events:
                assert events[row['sample_id']]['group'] == row['group']
            events[row['sample_id']] = row
    train_ids = set(events); assert len(train_ids) == 5993
    for row in manifests['short']['val']:
        assert row['sample_id'] not in events; events[row['sample_id']] = row
    metadata_path = inputs / 'samples.parquet'
    metadata = pd.read_parquet(metadata_path, columns=['sample_id', 'frame_offset', 'num_frames']).set_index('sample_id').to_dict('index')
    features = np.load(inputs / 'features.npy', mmap_mode='r', allow_pickle=False)
    reference_dir = ROOT / 'runs/m5_length_research_20261104/L1_mixed_G1/evaluations/generation_full_060000'
    reference_rows = [r for r in read(reference_dir / 'per_sample.json') if r['condition'] == 'native' and r['seed'] == 0]
    references = {r['sample_id']: r['reference_amplitude'] for r in reference_rows}; assert len(references) == 300
    torch.set_num_threads(4); skeleton = build_skeleton(34); records = {}; reference_max_error = 0.; function_max_error = 0.
    with torch.inference_mode():
        for count, (sample_id, event) in enumerate(sorted(events.items()), 1):
            row = metadata[sample_id]; start, n = int(row['frame_offset']), int(row['num_frames'])
            raw = np.array(features[start:start + n], copy=True)
            value, positions, root, rotation = direct_amplitude(raw, skeleton, event['group'])
            assert np.isfinite(value) and value >= 0 and n == event['num_frames']
            if sample_id in references:
                reference_max_error = max(reference_max_error, abs(value - references[sample_id]))
                assert abs(value - references[sample_id]) <= 2e-5
                motion = {'root_positions': torch.from_numpy(root), 'posed_joints': torch.from_numpy(positions),
                          'global_root_heading': torch.from_numpy(raw[:, 3:5])}
                if rotation is not None:
                    assert skeleton.root_idx == 0
                    motion['global_rot_mats'] = torch.from_numpy(rotation[:, None])
                frozen = amplitude(motion, skeleton, event['group'])
                function_max_error = max(function_max_error, abs(value - frozen)); assert abs(value - frozen) <= 2e-5
                # Keep frozen-evaluation reference amplitudes for the development distribution.
                value = references[sample_id]
            records[sample_id] = {'sample_id': sample_id, 'group': event['group'], 'frames': n,
                'length_bin': length_bin(n), 'split': 'train' if sample_id in train_ids else 'development',
                'amplitude': value, 'eligible': value >= MINIMUM[event['group']],
                'source_group_id': event.get('source_group_id'), 'motion_id': event.get('motion_id')}
            if count % 500 == 0:
                write(output / 'progress.json', {'processed_events': count, 'target_events': len(events), 'updated_at_utc': now()})
                print('Profiled GT amplitude', count, '/', len(events), flush=True)
    profiles = {name: summarize([records[r['sample_id']] for r in manifest['train']]) for name, manifest in manifests.items()}
    profiles['development'] = summarize([records[r['sample_id']] for r in manifests['short']['val']])
    small = {r['sample_id'] for r in manifests['mixed_small']['train']}; large = {r['sample_id'] for r in manifests['mixed_large']['train']}
    assert small < large and len(large - small) == 961
    profiles['expansion_new961'] = summarize([records[s] for s in sorted(large - small)])
    exposure = {}
    for name, manifest in manifests.items():
        buckets = Counter(length_bin(r['num_frames']) for r in manifest['train']); total = len(manifest['train'])
        exposure[name] = {}
        for group in GROUPS:
            fractions = {b: profiles[name][group][b]['eligible_fraction'] for b in buckets}
            assert all(v is not None for v in fractions.values())
            exposure[name][group] = {'bucket_probabilities': {b: n / total for b, n in buckets.items()},
                'expected_eligible_sample_fraction': sum(n / total * fractions[b] for b, n in buckets.items()),
                'scope': 'Expected sample exposure under count-weighted bucket then uniform class/event draws; not an exact gradient or loss weighting.'}
    write(output / 'per_event.json', list(records.values()))
    result = {'schema': 'm5-gt-amplitude-coverage-v1', 'completed_at_utc': now(), 'passed': True,
        'unique_train_events': 5993, 'development_events': 300, 'new_expansion_events': 961,
        'profiles': profiles, 'sampler_exposure_expectations': exposure,
        'frozen_development_reference_reconstruction_max_abs': reference_max_error,
        'frozen_amplitude_function_replay_max_abs': function_max_error,
        'source_sha256': sha(Path(__file__)), 'input_fingerprint': view['input_fingerprint'],
        'samples_metadata_sha256': sha(metadata_path), 'reference_per_sample_sha256': sha(reference_dir / 'per_sample.json'),
        'manifest_sha256': {name: sha(base / 'data' / file) for name, file in (
            ('short', 'D_short_3500.json'), ('mixed_small', 'D_mixed_small_3500.json'), ('mixed_large', 'D_mixed_large_4500.json'))},
        'per_event_sha256': sha(output / 'per_event.json'),
        'eligibility_policy': 'Existing numerical MINIMUM only; no semantic correctness, takeoff/flight/landing proof, or filtering of any data.',
        'model_instantiated': False, 'new_optimizer_updates': 0, 'new_diffusion_samples': 0,
        'data_or_sampling_modified': False, 'seed3102_quality_scores_read': False,
        'final_audit_model_scores_read': False, 'M5_complete': False}
    write(output / 'report.json', result); print('Saved amplitude coverage report:', output / 'report.json', flush=True)


if __name__ == '__main__':
    main()
