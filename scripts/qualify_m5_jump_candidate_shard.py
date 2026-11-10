#!/usr/bin/env python3
"""Qualify unused high-amplitude train motions and build an isolated raw shard.

The shard is not a registered model-input cache or training view. Existing inputs,
normalization statistics, selectors and running experiments are never modified.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import mujoco
from scripts import cache_m5_physical_inputs as builder
from scripts.run_m5_length_research import read, write, sha, now


def main():
    base = ROOT / 'runs/m5_physical_20261103'; source = base / 'diagnostics/amplitude_coverage/source_candidates.json'
    output = base / 'data_candidates/medium_high_jump_20261104'
    if output.exists():
        raise FileExistsError('Preserve the existing candidate shard')
    screen = read(source); assert screen['new_source_motion_unique_candidates'] == 35
    view = read(base / 'data/READY.json'); output.mkdir(parents=True)
    builder.initialize(); converter, _, _ = builder._worker
    model = mujoco.MjModel.from_xml_path(str(converter.xml_path)); c = converter.constants
    limited = model.jnt_limited[c['joint_ids']].astype(bool); bounds = c['limits']
    qualified = []; rejected = []; all_records = []
    with torch.inference_mode():
        for event in screen['candidates']:
            path = ROOT / event['npz_path']
            with np.load(path, allow_pickle=False) as archive:
                assert str(archive['source_sha256']) == event['source_sha256']
                qpos = archive['qpos_mujoco'][event['start_frame']:event['end_frame_exclusive']].astype(np.float64)
            assert qpos.shape == (event['num_frames'], 36) and np.isfinite(qpos).all()
            quaternion_error = float(abs(np.linalg.norm(qpos[:, 3:7], axis=-1) - 1.).max())
            violations = np.maximum(0., np.maximum(bounds[:, 0] - qpos[:, 7:], qpos[:, 7:] - bounds[:, 1]))[:, limited]
            maximum_limit_violation = float(violations.max(initial=0.))
            motion = converter.qpos_to_motion_dict(qpos); bottom = converter.foot_bottom(motion).numpy().min(-1)
            maximum_depth = float(max(0., -bottom.min())); deep_fraction = float((bottom < -.05).mean())
            reasons = []
            if quaternion_error > 1e-5:
                reasons.append('nonunit_root_quaternion')
            if maximum_limit_violation > 1e-6:
                reasons.append('native_joint_limit_violation')
            if maximum_depth > .10:
                reasons.append('reference_foot_penetration_over_10cm')
            if deep_fraction > .10:
                reasons.append('reference_over5cm_for_over10pct_frames')
            record = dict(event, native_body_quality_qualified=not reasons,
                source_node_npz_sha256=sha(path), root_quaternion_norm_error=quaternion_error,
                native_max_limit_violation_rad=maximum_limit_violation,
                reference_max_penetration_m=maximum_depth, reference_deep_frame_fraction=deep_fraction,
                rejected_reasons=reasons)
            if reasons:
                rejected.append(record)
            else:
                # Reuse the exact frozen physical-v2 builder, not legacy body targets.
                converted = builder.convert(event); raw = converted.pop('raw'); local = converted.pop('local_root')
                assert converted['root_block_error'] <= 1e-5 and raw.shape == (event['num_frames'], 417)
                assert abs(converted['reference_max_penetration_m'] - maximum_depth) < 1e-6
                amplitude = float(np.quantile(raw[:, 1], .9) - np.quantile(raw[:, 1], .1))
                assert abs(amplitude - event['root_height_p90_p10_m']) <= 2e-5 and amplitude >= .2
                folder = output / 'events' / event['sample_id']; folder.mkdir(parents=True)
                np.save(folder / 'raw_features.npy', raw, allow_pickle=False)
                np.save(folder / 'local_root.npy', local, allow_pickle=False)
                np.save(folder / 'native_qpos.npy', qpos, allow_pickle=False)
                record.update(physical_features=converted, raw_feature_shape=list(raw.shape),
                    prepared_files_sha256={str(p.relative_to(output)): sha(p) for p in folder.iterdir() if p.is_file()})
                qualified.append(record)
            all_records.append(record)
            print('Candidate native quality', len(all_records), '/', len(screen['candidates']), 'accepted', not reasons, flush=True)
    write(output / 'qualified_events.json', qualified); write(output / 'rejected_events.json', rejected)
    result = {'schema': 'm5-isolated-native-jump-candidate-shard-v1', 'prepared_at_utc': now(),
        'source_screen_sha256': sha(source), 'screened_events': len(screen['candidates']),
        'native_quality_qualified_events': len(qualified), 'rejected_events': len(rejected),
        'all_events': all_records, 'qualified_events_sha256': sha(output / 'qualified_events.json'),
        'rejected_events_sha256': sha(output / 'rejected_events.json'),
        'physical_adapter_sha256': sha(ROOT / 'training/g1_physical.py'),
        'physical_builder_sha256': sha(ROOT / 'scripts/cache_m5_physical_inputs.py'), 'source_sha256': sha(Path(__file__)),
        'ground_filter': view['ground_filter'], 'hinge_limit_tolerance_rad': 1e-6,
        'parent_physical_cache_fingerprint': view['input_fingerprint'],
        'legacy_body_features_reused': False, 'normalization_statistics_fitted_or_modified': False,
        'registered_training_view': False, 'usable_as_current_training_cache': False,
        'pending': 'An explicit versioned cache extension/compatibility proof and a frozen matched sampling/data comparison are required before model training. The original E0/E1 matrix remains required.',
        'training_inputs_or_sampling_modified': False, 'new_optimizer_updates': 0, 'new_diffusion_samples': 0,
        'seed3102_quality_scores_read': False, 'final_audit_model_scores_read': False, 'M5_complete': False}
    write(output / 'CANDIDATE_SHARD_COMPLETE.json', result)
    print({'native_quality_qualified': len(qualified), 'rejected': len(rejected), 'registered_training_view': False}, flush=True)


if __name__ == '__main__':
    main()
