#!/usr/bin/env python3
"""Read-only availability screen in unused M2 train sources; not a training view.

Only the invariant vertical root block is used from the legacy M2 cache. New
body features and native-G1 quality qualification remain mandatory before use.
"""
import re
import sys
from collections import OrderedDict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from training.m3 import INPUTS
from training.m5_data import classify, group_keys, rank
from scripts.run_m5_length_research import read, write, sha, now


def main():
    out = ROOT / 'runs/m5_physical_20261103/diagnostics/amplitude_coverage/source_candidates.json'
    if out.exists():
        raise FileExistsError('Preserve the completed source screen')
    base = ROOT / 'runs/m5_physical_20261103'; manifests = [read(base / 'data' / name) for name in
        ('D_short_3500.json', 'D_mixed_small_3500.json', 'D_mixed_large_4500.json')]
    current_train = {r['sample_id']: r for m in manifests for r in m['train']}
    held = read(ROOT / 'runs/m5_20261103/data/locked_audit.json')['events']
    reserved_keys = set().union(*(group_keys(r) for r in [*manifests[0]['val'], *held]))
    # Repetition families are split-leakage keys, not a reason to collapse distinct
    # performances within the same training split (same rule as prepare_m5_data).
    used_train_keys = {k for r in current_train.values() for k in group_keys(r) if k[0] != 'validation_group_id'}
    held_text = set(manifests[0]['heldout_text_ids'])
    view = read(base / 'data/READY.json'); unsupported = re.compile(view['unsupported_context_pattern'], re.I)
    metadata_path = INPUTS / 'train.parquet'; frame = pd.read_parquet(metadata_path)
    frame = frame[frame.num_frames.between(91, 120) & frame.text.str.contains(r'\bjump(?:s|ing)?\b', case=False, regex=True, na=False)].copy()
    original_text_rows = len(frame)
    motion_path = ROOT / 'data/processed/g1_text_motion_v1/motions.parquet'
    motion = pd.read_parquet(motion_path, columns=['motion_id', 'source_group_id', 'validation_group_id', 'take_name', 'actor_uid', 'is_mirror'])
    frame = frame.merge(motion, on='motion_id', validate='many_to_one')
    frame = frame[~frame.is_mirror & ~frame.text_id.isin(held_text)]
    mapped = OrderedDict(); candidates = []; counters = {'root_text_prefilter_rows': original_text_rows}
    for row in frame.to_dict('records'):
        assert row['split'] == 'train'
        group, _ = classify(row['text'])
        if group != 'jump' or unsupported.search(row['text']):
            continue
        keys = group_keys(row)
        if keys & reserved_keys:
            continue
        train_keys = {k for k in keys if k[0] != 'validation_group_id'}
        if train_keys & used_train_keys:
            continue
        path = (INPUTS / row['feature_path']).resolve(); path.relative_to(INPUTS.resolve())
        if path not in mapped:
            mapped[path] = np.load(path, mmap_mode='r', allow_pickle=False)
            if len(mapped) > 8:
                mapped.popitem(last=False)
        start, n = int(row['frame_offset']), int(row['num_frames'])
        height = np.asarray(mapped[path][start:start + n, 1])
        value = float(np.quantile(height, .9) - np.quantile(height, .1))
        if value < .2:
            continue
        candidates.append({k: row[k] for k in ('sample_id', 'text_id', 'text', 'split', 'motion_id', 'npz_path',
            'source_sha256', 'source_group_id', 'validation_group_id', 'take_name', 'actor_uid',
            'start_frame', 'end_frame_exclusive', 'num_frames', 'feature_path', 'frame_offset')} |
            {'root_height_p90_p10_m': value, 'group': 'jump', 'native_body_quality_qualified': False})
    selected = []; used = set()
    for row in sorted(candidates, key=lambda r: rank(r['sample_id'], 'm5-new-medium-high-jump-availability-v1')):
        keys = {k for k in group_keys(row) if k[0] != 'validation_group_id'}
        if not keys & used:
            selected.append(row); used.update(keys)
    assert not any(group_keys(r) & reserved_keys for r in selected)
    assert not any(({k for k in group_keys(r) if k[0] != 'validation_group_id'} & used_train_keys) for r in selected)
    result = {'schema': 'm5-medium-high-jump-source-availability-v1', 'screened_at_utc': now(),
        'target': 'train-only nonmirror91-120frame jump; P90-P10 vertical root range>=0.20m',
        'prefilter_counts': counters, 'candidate_rows_after_metadata_and_height_checks': len(candidates),
        'new_source_motion_unique_candidates': len(selected), 'candidates': selected,
        'group_policy': 'Disjoint all frozen development/audit group keys and current train source/motion keys; one new event per source/motion. Validation-family keys are not interpreted as independent performances within train.',
        'height_source': 'Legacy M2 smooth_root_pos Y; unchanged by root planar smoothing and the physical-v2 hinge-order repair. No legacy body targets are reused.',
        'input_m2_fingerprint': read(INPUTS / 'READY')['fingerprint'],
        'metadata_sha256': sha(metadata_path), 'motion_metadata_sha256': sha(motion_path), 'source_sha256': sha(Path(__file__)),
        'usable_training_samples_proven': False,
        'pending': 'Rebuild physical-v2 body features from native qpos; verify source hash, hinge limits and the same GT foot-depth rules. Freeze any new view and matched sampling comparison before training.',
        'training_inputs_or_sampling_modified': False, 'new_optimizer_updates': 0, 'new_diffusion_samples': 0,
        'seed3102_quality_scores_read': False, 'final_audit_model_scores_read': False, 'M5_complete': False}
    write(out, result); print({k: result[k] for k in ('candidate_rows_after_metadata_and_height_checks', 'new_source_motion_unique_candidates', 'usable_training_samples_proven')}, flush=True)


if __name__ == '__main__':
    main()
