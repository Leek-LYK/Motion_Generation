#!/usr/bin/env python3
"""Independent geometry replay of the isolated train candidate shard."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import mujoco
from kimodo.motion_rep import KimodoMotionRep
from training.g1_physical import PhysicalG1Converter, PhysicalG1BoundedDecoder
from training.m5_physical_losses import geometry
from scripts.run_m5_length_research import read, write, sha, now


def main():
    base = ROOT / 'runs/m5_physical_20261103'; shard = base / 'data_candidates/medium_high_jump_20261104'
    output = shard / 'verification.json'
    if output.exists():
        raise FileExistsError('Preserve completed shard verification')
    complete = read(shard / 'CANDIDATE_SHARD_COMPLETE.json'); rows = read(shard / 'qualified_events.json')
    assert sha(shard / 'qualified_events.json') == complete['qualified_events_sha256']
    assert sha(shard / 'rejected_events.json') == complete['rejected_events_sha256']
    assert sha(ROOT / 'training/g1_physical.py') == complete['physical_adapter_sha256']
    assert sha(ROOT / 'scripts/cache_m5_physical_inputs.py') == complete['physical_builder_sha256']
    torch.set_num_threads(4); converter = PhysicalG1Converter(); decoder = PhysicalG1BoundedDecoder(converter.skeleton)
    rep = KimodoMotionRep(converter.skeleton, 30, stats_path=str(base / 'stats/motion'))
    model = mujoco.MjModel.from_xml_path(str(converter.xml_path)); data = mujoco.MjData(model); c = converter.constants
    mean = np.concatenate([np.load(base / 'stats/motion' / group / 'mean.npy') for group in ('global_root', 'body')]).astype(np.float32)
    std = np.concatenate([np.load(base / 'stats/motion' / group / 'std.npy') for group in ('global_root', 'body')]).astype(np.float32)
    scale = np.sqrt(std**2 + np.float32(1e-5)); records = []
    with torch.inference_mode():
        for event in rows:
            for path, expected in event['prepared_files_sha256'].items():
                assert sha(shard / path) == expected
            folder = shard / 'events' / event['sample_id']; raw = np.load(folder / 'raw_features.npy')
            qpos = np.load(folder / 'native_qpos.npy'); n = len(raw)
            normalized = torch.from_numpy((raw - mean) / scale)[None]
            reconstructed_raw, fk, motion, fallback = geometry(rep, normalized)
            motion = {k: value[0] for k, value in motion.items()}
            self_error = float(torch.linalg.vector_norm(fk[0] - reconstructed_raw[0, :, 5:107].reshape(n, 34, 3), dim=-1).max())
            positions = raw[:, 5:107].reshape(n, 34, 3).astype(np.float64)
            positions[..., 0] += raw[:, None, 0]; positions[..., 2] += raw[:, None, 2]
            origin = np.array([event['physical_features']['origin_x'], 0., event['physical_features']['origin_z']])
            compiled_error = 0.
            for frame in sorted({0, n // 2, n - 1}):
                data.qpos[:] = qpos[frame]; mujoco.mj_forward(model, data)
                anchors = np.stack([c['C'].T @ data.xanchor[jid] for jid in c['joint_ids']]) - origin
                compiled_error = max(compiled_error, float(np.linalg.norm(anchors - positions[frame, c['indices']], axis=-1).max()))
            bounded, info = decoder.decode(motion)
            bounded_motion = converter.qpos_to_motion_dict(bounded)
            projection_error = float(np.linalg.norm(bounded_motion['posed_joints'].numpy() - positions, axis=-1).max())
            assert self_error <= 2e-5 and compiled_error <= 2e-5 and projection_error <= 2e-5
            assert not fallback.any() and info['bounded']['max_limit_violation_rad'] <= 1e-6
            records.append({'sample_id': event['sample_id'], 'frames': n, 'max_fk_self_error_m': self_error,
                            'max_compiled_mjcf_anchor_error_m': compiled_error, 'max_bounded_node_projection_error_m': projection_error,
                            'rotation_fallback': False})
            print('Verified new native candidate', len(records), '/', len(rows), flush=True)
    result = {'schema': 'm5-isolated-jump-candidate-verification-v1', 'passed': True, 'verified_at_utc': now(),
        'qualified_events': len(rows), 'records': records,
        'all_frames_fk_self_and_bounded_projection_checked': True,
        'compiled_mjcf_frames_per_event': 'first/middle/last',
        'compiled_mjcf_native_joint_anchors_per_frame': 29,
        'max_fk_self_error_m': max(r['max_fk_self_error_m'] for r in records),
        'max_compiled_mjcf_anchor_error_m': max(r['max_compiled_mjcf_anchor_error_m'] for r in records),
        'max_bounded_node_projection_error_m': max(r['max_bounded_node_projection_error_m'] for r in records),
        'shard_descriptor_sha256': sha(shard / 'CANDIDATE_SHARD_COMPLETE.json'), 'source_sha256': sha(Path(__file__)),
        'normalization_statistics_changed': False, 'current_training_cache_or_sampler_changed': False,
        'registered_training_view': False, 'cache_extension_compatibility_bridge_implemented': False,
        'model_instantiated': False, 'new_optimizer_updates': 0, 'new_diffusion_samples': 0,
        'seed3102_quality_scores_read': False, 'final_audit_model_scores_read': False, 'M5_complete': False}
    write(output, result); print({k: result[k] for k in ('passed', 'qualified_events', 'max_fk_self_error_m',
        'max_compiled_mjcf_anchor_error_m', 'max_bounded_node_projection_error_m')}, flush=True)


if __name__ == '__main__':
    main()
