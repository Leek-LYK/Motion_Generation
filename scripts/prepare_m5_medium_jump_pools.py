#!/usr/bin/env python3
"""Bind train-only medium-jump amplitude pools to an ordered physical subset."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_m5_length_research import read, write, sha, now


def entries_fingerprint(entries):
    fields = ('sample_id', 'text_id', 'text', 'motion_id', 'split', 'num_frames', 'group')
    value = [{k: row[k] for k in fields} for row in entries]
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--subset', type=Path, required=True)
    p.add_argument('--training-view', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--high-amplitude-cutoff-m', type=float, default=.2); a = p.parse_args()
    if a.output.exists(): raise FileExistsError('Preserve existing train-pool definition')
    if not np.isfinite(a.high_amplitude_cutoff_m) or a.high_amplitude_cutoff_m <= 0: raise ValueError('Invalid sampling cutoff')
    subset = read(a.subset); view = read(a.training_view); inputs = Path(view['input_dir'])
    ready = read(inputs / 'READY'); config = read(inputs / 'config.json')
    assert ready['passed'] and view['passed'] and ready['fingerprint'] == view['input_fingerprint'] == config['fingerprint'] == subset['physical_cache_fingerprint']
    assert config['fps'] == 30 and config['feature_dim'] == 417 and sha(inputs / 'features.npy') == ready['feature_sha256']
    assert sha(inputs / 'samples.parquet') == ready['samples_sha256']
    rows = pd.read_parquet(inputs / 'samples.parquet').set_index('sample_id').to_dict('index')
    features = np.load(inputs / 'features.npy', mmap_mode='r', allow_pickle=False)
    train = subset['train']; assert len({r['sample_id'] for r in train}) == len(train)
    eligible = []; high = []; low = []
    for index, entry in enumerate(train):
        assert entry['split'] == rows[entry['sample_id']]['split'] == 'train'
        for key in ('text_id', 'text', 'motion_id', 'num_frames'): assert entry[key] == rows[entry['sample_id']][key]
        if entry['group'] != 'jump' or not 91 <= entry['num_frames'] <= 120: continue
        row = rows[entry['sample_id']]; offset = int(row['frame_offset']); n = int(row['num_frames'])
        raw = np.asarray(features[offset:offset + n]); assert raw.shape == (n, 417) and np.isfinite(raw).all()
        # The frozen jump metric uses posed pelvis Y. The smooth-root Y is an
        # independently retained equivalent value for GT, checked here only.
        assert abs(raw[:, 1] - raw[:, 6]).max() < 2e-5
        amplitude = float(np.quantile(raw[:, 6], .9) - np.quantile(raw[:, 6], .1))
        is_high = amplitude >= a.high_amplitude_cutoff_m
        (high if is_high else low).append(index)
        eligible.append({'index': index, 'sample_id': entry['sample_id'], 'text_id': entry['text_id'],
                         'motion_id': entry['motion_id'], 'frames': n, 'amplitude_m': amplitude, 'high': is_high})
    if not high or not low: raise ValueError('Both train-only high/low pools are required')
    descriptor = {'schema': 'm5-medium-jump-amplitude-pools-v1',
        'training_entries_fingerprint': entries_fingerprint(train), 'subset_sha256': sha(a.subset),
        'physical_input_fingerprint': view['input_fingerprint'], 'training_view_fingerprint': view['fingerprint'],
        'input_dir': str(inputs.resolve()), 'train_events': len(train), 'eligible_events': eligible,
        'high_indices': high, 'low_indices': low, 'high_amplitude_cutoff_m': a.high_amplitude_cutoff_m,
        'eligibility': {'group': 'jump', 'minimum_frames': 91, 'maximum_frames': 120},
        'amplitude_definition': 'P90-P10 of original physical GT posed-pelvis Y (raw feature6),metres. Sampling cutoff only; acceptance thresholds unchanged.',
        'features_sha256': ready['feature_sha256'], 'train_only': True, 'statistics_refitted': False,
        'all_original_low_amplitude_events_retained': True}
    fingerprint = hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    files = [a.subset.resolve(), a.training_view.resolve(), inputs / 'config.json', inputs / 'READY', inputs / 'samples.parquet', Path(__file__).resolve()]
    result = {'descriptor': descriptor, 'fingerprint': fingerprint, 'prepared_at_utc': now(),
        'files_sha256': {str(path): sha(path) for path in files}, 'high_count': len(high), 'low_count': len(low),
        'natural_high_fraction': len(high) / (len(high) + len(low)), 'model_training_started': False,
        'development_or_audit_model_scores_read': False, 'new_optimizer_updates': 0, 'M5_complete': False}
    write(a.output, result); print({k: result[k] for k in ('fingerprint', 'high_count', 'low_count', 'natural_high_fraction')}, flush=True)


if __name__ == '__main__': main()
