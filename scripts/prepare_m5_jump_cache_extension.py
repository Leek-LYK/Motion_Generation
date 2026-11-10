#!/usr/bin/env python3
"""Materialize an append-only physical cache and frozen-stat candidate data views."""
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_m5_length_research import read, write, sha, now

BASE = ROOT / 'runs/m5_physical_20261103'
OUTPUT = ROOT / 'data/processed/g1_model_inputs_physical_v2_m5_jump_ext35_20261104'
VIEW = ROOT / 'runs/m5_amplitude_data_20261104'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def main():
    if OUTPUT.exists() or VIEW.exists():
        raise FileExistsError('Preserve existing cache/view preparation; inspect it before continuing')
    original_view = BASE / 'data/READY.json'; old_view = read(original_view); parent = Path(old_view['input_dir'])
    old_ready = read(parent / 'READY'); old_config = read(parent / 'config.json')
    assert old_ready['passed'] and old_ready['fingerprint'] == old_config['fingerprint'] == old_view['input_fingerprint']
    assert sha(parent / 'features.npy') == old_ready['feature_sha256']
    assert sha(parent / 'samples.parquet') == old_ready['samples_sha256']
    shard = BASE / 'data_candidates/medium_high_jump_20261104'; verification = read(shard / 'verification.json')
    assert verification['passed']; additions = read(shard / 'qualified_events.json'); assert len(additions) == 35
    caps = read(shard / 'caption_capped_additions.json')
    assert read(shard / 'CANDIDATE_SHARD_COMPLETE.json')['physical_adapter_sha256'] == old_config['physical_code_sha256']
    old_features = np.load(parent / 'features.npy', mmap_mode='r', allow_pickle=False)
    old_samples = pd.read_parquet(parent / 'samples.parquet')
    train_meta = pd.read_parquet(ROOT / 'data/processed/g1_model_inputs_v1/train.parquet').set_index('sample_id').to_dict('index')
    assert old_features.dtype == np.float32 and old_features.shape == (706336, 417)
    assert len(old_samples) == 7452
    old_ids = set(old_samples.sample_id); assert not old_ids & {r['sample_id'] for r in additions}
    added_frames = sum(r['num_frames'] for r in additions); total_frames = len(old_features) + added_frames
    OUTPUT.mkdir(parents=True); VIEW.mkdir(parents=True)
    features = np.lib.format.open_memmap(OUTPUT / 'features.npy', mode='w+', dtype=np.float32, shape=(total_frames, 417))
    old_payload_hash = hashlib.sha256()
    for start in range(0, len(old_features), 16384):
        part = np.asarray(old_features[start:start + 16384]); features[start:start + len(part)] = part
        old_payload_hash.update(part.tobytes(order='C'))
        if start % (16384 * 8) == 0:
            write(VIEW / 'preparation_progress.json', {'phase': 'copy_original_frames', 'frames': start + len(part),
                  'target_frames': total_frames, 'updated_at_utc': now()})
            print('Copied original frame prefix through', start + len(part), flush=True)
    cursor = len(old_features); extra_rows = []; appended = []
    for item in sorted(additions, key=lambda r: r['sample_id']):
        for file, expected in item['prepared_files_sha256'].items():
            assert sha(shard / file) == expected
        raw_path = shard / 'events' / item['sample_id'] / 'raw_features.npy'; raw = np.load(raw_path, allow_pickle=False)
        assert raw.shape == (item['num_frames'], 417) and raw.dtype == np.float32 and np.isfinite(raw).all()
        features[cursor:cursor + len(raw)] = raw
        source = train_meta[item['sample_id']]
        row = {column: source.get(column) for column in old_samples.columns}
        row.update(sample_id=item['sample_id'], feature_path='features.npy', frame_offset=cursor,
                   physical_frame_offset=cursor, physical_stats_train=False,
                   first_heading_angle=item['physical_features']['first_heading_angle'],
                   origin_x=item['physical_features']['origin_x'], origin_z=item['physical_features']['origin_z'])
        extra_rows.append(row)
        appended.append({'sample_id': item['sample_id'], 'start': cursor, 'frames': len(raw),
                         'raw_file': str(raw_path), 'raw_sha256': sha(raw_path)})
        cursor += len(raw)
    assert cursor == total_frames; features.flush(); del features
    new_features = np.load(OUTPUT / 'features.npy', mmap_mode='r', allow_pickle=False); new_prefix_hash = hashlib.sha256()
    for start in range(0, len(old_features), 16384):
        end = min(start + 16384, len(old_features))
        part = np.asarray(new_features[start:end]); old_part = np.asarray(old_features[start:end])
        assert part.tobytes(order='C') == old_part.tobytes(order='C')
        new_prefix_hash.update(part.tobytes(order='C'))
    assert new_prefix_hash.hexdigest() == old_payload_hash.hexdigest()
    for extra in appended:
        assert np.array_equal(new_features[extra['start']:extra['start'] + extra['frames']], np.load(extra['raw_file'], allow_pickle=False))
    extra_frame = pd.DataFrame(extra_rows, columns=old_samples.columns).astype(old_samples.dtypes.to_dict())
    samples = pd.concat([old_samples, extra_frame], ignore_index=True)
    pd.testing.assert_frame_equal(samples.iloc[:len(old_samples)].reset_index(drop=True), old_samples.reset_index(drop=True), check_exact=True)
    assert samples.sample_id.is_unique and len(samples) == 7487
    samples.to_parquet(OUTPUT / 'samples.parquet', index=False)
    reread = pd.read_parquet(OUTPUT / 'samples.parquet')
    pd.testing.assert_frame_equal(reread.iloc[:len(old_samples)].reset_index(drop=True), old_samples.reset_index(drop=True), check_exact=True)
    # Cache-default statistics retain their original provenance. The model uses
    # the separate qualified M5 statistics below, exactly as before extension.
    shutil.copytree(parent / 'stats', OUTPUT / 'stats')
    shutil.copytree(BASE / 'stats/motion', VIEW / 'stats/motion')
    for relative, expected in old_ready['stats_sha256'].items():
        assert sha(OUTPUT / relative) == expected
    for relative, expected in old_view['stats_sha256'].items():
        assert sha(VIEW / relative) == sha(BASE / relative) == expected
    feature_sha = sha(OUTPUT / 'features.npy'); samples_sha = sha(OUTPUT / 'samples.parquet')
    descriptor = {'schema': 'm5-physical-append-cache-content-v1', 'parent_fingerprint': old_ready['fingerprint'],
        'parent_feature_sha256': old_ready['feature_sha256'], 'feature_sha256': feature_sha,
        'samples_sha256': samples_sha, 'physical_code_sha256': old_config['physical_code_sha256'],
        'text_config_fingerprint': old_config['text_config_fingerprint'], 'fps': 30, 'feature_dim': 417,
        'cache_default_stats_sha256': old_ready['stats_sha256'],
        'appended_raw_files': {r['sample_id']: r['raw_sha256'] for r in appended}}
    cache_fp = fingerprint(descriptor)
    write(OUTPUT / 'content_descriptor.json', descriptor)
    config = dict(old_config, schema='m5-physical-inputs-v2-append-v1', fingerprint=cache_fp,
                  parent_fingerprint=old_ready['fingerprint'], frames=total_frames, total_events=7487,
                  unique_train_events=old_ready['unique_train_events'] + 35,
                  statistics_policy='Parent cache-default statistics copied unchanged; appended train events are not used to refit statistics. M5 models use separately frozen qualified model statistics.',
                  builder_sha256=sha(Path(__file__)), old_cache_modified=False)
    write(OUTPUT / 'config.json', config)
    write(OUTPUT / 'READY', {'schema': 'm5-append-verified-inputs-ready-v1', 'passed': True, 'fingerprint': cache_fp,
          'parent_fingerprint': old_ready['fingerprint'], 'feature_sha256': feature_sha, 'samples_sha256': samples_sha,
          'stats_sha256': old_ready['stats_sha256'], 'original_events': 7452, 'appended_train_events': 35,
          'frames': total_frames, 'original_frame_prefix_bitwise_equal': True,
          'new_source_quality_proof_sha256': sha(shard / 'verification.json'),
          'scope': 'Append-only input-cache preparation; not model acceptance or training.',
          'final_audit_model_scores_read': False, 'M5_complete': False})
    manifests = {name: read(BASE / 'data' / file) for name, file in (
        ('short_control', 'D_short_3500.json'), ('mixed_small_control', 'D_mixed_small_3500.json'),
        ('mixed_large_control', 'D_mixed_large_4500.json'))}
    by_id = {r['sample_id']: r for r in additions}
    def slim(item):
        return {k: item[k] for k in ('sample_id', 'text_id', 'text', 'split', 'motion_id', 'source_sha256',
            'source_group_id', 'validation_group_id', 'take_name', 'actor_uid', 'start_frame', 'end_frame_exclusive', 'num_frames')} | {
            'group': 'jump', 'length_bin': 'medium', 'semantic_tags': ['jump']}
    small_ids = caps['views']['mixed_small']['added_ids']; large_ids = caps['views']['mixed_large']['added_ids']
    common_ids = [s for s in large_ids if s in set(small_ids)]; assert len(common_ids) == 6
    source_small = manifests['mixed_small_control']; source_large = manifests['mixed_large_control']
    manifests['mixed_small_plus19'] = dict(source_small, train=source_small['train'] + [slim(by_id[s]) for s in small_ids])
    manifests['paired_small_plus6'] = dict(source_small, train=source_small['train'] + [slim(by_id[s]) for s in common_ids])
    manifests['paired_large_plus6'] = dict(source_large, train=source_large['train'] + [slim(by_id[s]) for s in common_ids])
    hashes = {}; counts = {}
    for name, manifest in manifests.items():
        manifest = dict(manifest, physical_cache_fingerprint=cache_fp,
                        cache_extension_parent_fingerprint=old_ready['fingerprint'],
                        normalization='Frozen original qualified M5 model statistics; no new fitting.')
        assert len({r['sample_id'] for r in manifest['train']}) == len(manifest['train'])
        assert max(Counter(r['text_id'] for r in manifest['train']).values()) <= 20
        assert {r['sample_id'] for r in manifest['train']} <= set(samples.sample_id)
        assert manifest['val'] == source_small['val'] and manifest['monitor'] == source_small['monitor']
        path = VIEW / 'data' / (name + '.json'); write(path, manifest); hashes[name] = sha(path)
        counts[name] = {'train': len(manifest['train']), 'val': len(manifest['val']), 'monitor': len(manifest['monitor'])}
        manifests[name] = manifest
    small_set = {r['sample_id'] for r in manifests['paired_small_plus6']['train']}
    large_set = {r['sample_id'] for r in manifests['paired_large_plus6']['train']}
    assert small_set < large_set and len(large_set - small_set) == 961
    write(VIEW / 'data/development_flags.json', read(BASE / 'data/development_flags.json'))
    shutil.copyfile(BASE / 'first_wave_assessment.json', VIEW / 'first_wave_assessment.json')
    view_descriptor = {'schema': 'm5-append-compatible-training-view-content-v1', 'input_fingerprint': cache_fp,
                       'parent_view_fingerprint': old_view['fingerprint'], 'manifests_sha256': hashes,
                       'stats_sha256': old_view['stats_sha256'], 'statistics_policy': 'Frozen parent qualified5993 model statistics; additions excluded from fitting.'}
    new_view_fp = fingerprint(view_descriptor)
    write(VIEW / 'data/view_content_descriptor.json', view_descriptor)
    new_view = {'schema': 'm5-append-compatible-training-view-v1', 'passed': True, 'fingerprint': new_view_fp,
        'input_dir': str(OUTPUT), 'input_fingerprint': cache_fp, 'parent_training_view_fingerprint': old_view['fingerprint'],
        'stats_sha256': old_view['stats_sha256'], 'unique_stats_events': old_view['unique_stats_events'],
        'statistics_frames': old_view['statistics_frames'], 'statistics_policy': view_descriptor['statistics_policy'],
        'ground_filter': old_view['ground_filter'], 'unsupported_context_pattern': old_view['unsupported_context_pattern'],
        'manifests_sha256': hashes, 'counts': counts, 'development_events_removed': 0,
        'locked_audit_model_scores_read': False, 'original_E0_E1_replaced': False, 'model_training_started': False}
    write(VIEW / 'data/READY.json', new_view)
    files = [parent / 'config.json', parent / 'READY', parent / 'samples.parquet', OUTPUT / 'config.json', OUTPUT / 'READY',
             OUTPUT / 'samples.parquet', original_view, VIEW / 'data/READY.json', OUTPUT / 'content_descriptor.json',
             VIEW / 'data/view_content_descriptor.json', shard / 'verification.json', shard / 'CANDIDATE_SHARD_COMPLETE.json',
             shard / 'caption_capped_additions.json', ROOT / 'training/g1_physical.py', ROOT / 'training/m5_physical_losses.py',
             ROOT / 'training/m5_physical_data.py', Path(__file__).resolve()]
    files += [BASE / p for p in old_view['stats_sha256']] + [VIEW / p for p in old_view['stats_sha256']]
    files += [VIEW / 'data' / (n + '.json') for n in hashes]
    proof = {'schema': 'm5-append-compatible-cache-proof-v1', 'passed': True, 'verified_at_utc': now(),
        'source_cache': str(parent), 'source_cache_fingerprint': old_ready['fingerprint'],
        'source_cache_feature_sha256': old_ready['feature_sha256'], 'source_training_view': str(original_view),
        'source_training_view_fingerprint': old_view['fingerprint'], 'source_model_stats_dir': str(BASE / 'stats/motion'),
        'target_cache': str(OUTPUT), 'target_cache_fingerprint': cache_fp, 'target_cache_feature_sha256': feature_sha,
        'target_training_view': str(VIEW / 'data/READY.json'), 'target_training_view_fingerprint': new_view_fp,
        'target_model_stats_dir': str(VIEW / 'stats/motion'), 'model_stats_sha256': old_view['stats_sha256'],
        'old_frame_count': len(old_features), 'new_frame_count': total_frames, 'old_event_count': len(old_samples), 'new_event_count': len(samples),
        'old_prefix_raw_sha256': old_payload_hash.hexdigest(), 'new_prefix_raw_sha256': new_prefix_hash.hexdigest(),
        'original_frame_prefix_bitwise_equal': True, 'original_metadata_prefix_exactly_equal': True,
        'appended_events': appended, 'frozen_model_normalization_unchanged': True,
        'paired_common6_small_is_strict_subset_of_large': True, 'paired_expansion_new_events': 961,
        'files_sha256': {str(p): sha(p) for p in files},
        'source_final_model_scores_read': False, 'new_optimizer_updates': 0, 'M5_complete': False}
    write(VIEW / 'cache_compatibility_proof.json', proof)
    assert sha(parent / 'features.npy') == old_ready['feature_sha256']
    write(VIEW / 'preparation_progress.json', {'phase': 'complete', 'frames': total_frames, 'events': len(samples),
          'updated_at_utc': now(), 'model_training_started': False})
    print({'cache_fingerprint': cache_fp, 'training_view_fingerprint': new_view_fp,
           'events': len(samples), 'appended_events': 35, 'views': counts}, flush=True)


if __name__ == '__main__':
    main()
