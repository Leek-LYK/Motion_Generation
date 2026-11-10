#!/usr/bin/env python3
"""Read-only numerical replay of the paper's smooth-root representation on train clips."""
import hashlib
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from kimodo.motion_rep import KimodoMotionRep
from kimodo.motion_rep.smooth_root import get_smooth_root_pos
from kimodo.motion_rep.feature_utils import compute_heading_angle, compute_vel_xyz
from kimodo.geometry import matrix_to_cont6d
from training.g1_physical import PhysicalG1Converter
from scripts.run_m5_length_research import read, write, sha, now


def main():
    out = ROOT / 'runs/m5_physical_20261103/verification/paper_engineering_audit_20261104/root_representation.json'
    if out.exists():
        raise FileExistsError('Preserve the root replay evidence')
    base = ROOT / 'runs/m5_physical_20261103'; manifest_path = base / 'data/D_mixed_small_3500.json'
    manifest = read(manifest_path); view = read(base / 'data/READY.json'); inputs = Path(view['input_dir'])
    rows = pd.read_parquet(inputs / 'samples.parquet').set_index('sample_id').to_dict('index')
    features = np.load(inputs / 'features.npy', mmap_mode='r', allow_pickle=False)
    selected = []
    for group in ('walk', 'jog', 'jump', 'raise_arms', 'turn', 'squat'):
        for label, low, high in (('short', 60, 90), ('medium', 91, 120), ('long', 121, 180)):
            pool = [e for e in manifest['train'] if e['group'] == group and low <= e['num_frames'] <= high]
            assert pool
            event = min(pool, key=lambda e: hashlib.sha256((e['sample_id'] + ':paper-root-replay-v1').encode()).hexdigest())
            selected.append((event, label))
    torch.set_num_threads(4); converter = PhysicalG1Converter(); rep = KimodoMotionRep(converter.skeleton, 30, stats_path=None)
    records = []
    with torch.inference_mode():
        for event, length_bin in selected:
            row = rows[event['sample_id']]; path = ROOT / row['npz_path']; path.resolve().relative_to(ROOT / 'data/processed/g1_30hz_nodes_v1')
            with np.load(path, allow_pickle=False) as source:
                assert str(source['source_sha256']) == row['source_sha256']
                qpos = source['qpos_mujoco'][row['start_frame']:row['end_frame_exclusive']].astype(np.float64)
            length = len(qpos); assert length == event['num_frames']
            motion = converter.qpos_to_motion_dict(qpos); root = motion['root_positions'].clone()
            origin = root[0, [0, 2]].clone(); root[:, [0, 2]] -= origin
            global_rot, world_positions, _ = converter.skeleton.fk(motion['local_rot_mats'], root)
            official_smooth = get_smooth_root_pos(root[None])[0]
            start = int(row['frame_offset']); cached = torch.from_numpy(np.array(features[start:start + length], copy=True)).double()
            smooth_error = float(abs(cached[:, :3] - official_smooth).max())
            y_error = float(abs(cached[:, 1] - root[:, 1]).max())
            residual = cached[:, 5:107].reshape(length, 34, 3)
            restored = residual.clone(); restored[..., 0] += cached[:, None, 0]; restored[..., 2] += cached[:, None, 2]
            positions_error = float(abs(restored - world_positions).max())
            residual_y_error = float(abs(residual[..., 1] - world_positions[..., 1]).max())
            velocity = compute_vel_xyz(world_positions[None], 30, lengths=torch.tensor([length]))[0]
            velocity_error = float(abs(cached[:, 311:413].reshape(length, 34, 3) - velocity).max())
            global6d_error = float(abs(cached[:, 107:311] - matrix_to_cont6d(global_rot).reshape(length, 204)).max())
            heading = compute_heading_angle(world_positions[None], converter.skeleton)[0]
            heading_error = float(abs(cached[:, 3:5] - torch.stack((heading.cos(), heading.sin()), -1)).max())
            # A per-frame heading canonicalization would change these world-axis residuals.
            expected_residual = world_positions.clone()
            expected_residual[..., 0] -= cached[:, None, 0]; expected_residual[..., 2] -= cached[:, None, 2]
            residual_error = float(abs(residual - expected_residual).max())
            original_accel = (root[2:, [0, 2]] - 2 * root[1:-1, [0, 2]] + root[:-2, [0, 2]]) * 30**2
            smoothed_accel = (cached[2:, [0, 2]] - 2 * cached[1:-1, [0, 2]] + cached[:-2, [0, 2]]) * 30**2
            local_root = rep.global_root_to_local_root(cached[None, :, :5].float(), normalized=False,
                                                       lengths=torch.tensor([length]))
            assert local_root.shape == (1, length, 4) and torch.isfinite(local_root).all()
            record = {'sample_id': event['sample_id'], 'group': event['group'], 'length_bin': length_bin, 'frames': length,
                'smooth_root_replay_max_abs_m': smooth_error, 'vertical_height_max_abs_m': y_error,
                'global_positions_reconstruction_max_abs_m': positions_error,
                'global_height_body_features_max_abs_m': residual_y_error,
                'world_axis_residual_position_max_abs_m': residual_error,
                'global_velocity_replay_max_abs_m_s': velocity_error, 'global_rotation6d_replay_max_abs': global6d_error,
                'hip_heading_replay_max_abs': heading_error,
                'pelvis_to_smooth_root_xz_max_component_m': float(abs(root[:, [0, 2]] - cached[:, [0, 2]]).max()),
                'pelvis_sway_residual_xz_rms_m': float((root[:, [0, 2]] - cached[:, [0, 2]]).square().mean().sqrt()),
                'pelvis_xz_acceleration_rms_m_s2': float(original_accel.square().mean().sqrt()),
                'smooth_root_xz_acceleration_rms_m_s2': float(smoothed_accel.square().mean().sqrt()),
                'first_smoothed_root_xz_m': cached[0, [0, 2]].tolist(),
                'source_npz_sha256': sha(path), 'training_motion_source_sha256': row['source_sha256']}
            record['passed'] = max(smooth_error, y_error, positions_error, residual_y_error, residual_error,
                                    velocity_error, global6d_error, heading_error) <= 2e-5
            records.append(record)
            print('Root representation replay', event['group'], length_bin, length, 'passed', record['passed'], flush=True)
    package = ROOT.parent / 'kimodo/source/kimodo'
    result = {'schema': 'm5-paper-root-representation-replay-v1', 'verified_at_utc': now(),
        'passed': all(r['passed'] for r in records), 'samples': len(records), 'split': 'train only',
        'selection': 'One deterministic train event per class and short/medium/long bin; no model score used.',
        'scope': 'Numerical replay of 18 train events plus source-code provenance; not an exhaustive re-read of every cache frame.',
        'records': records, 'tolerance': 2e-5,
        'official_smoother': 'ADMM/multigrid acceleration smoothing; get_smooth_root_pos uses 0.06m margins, preserves vertical height.',
        'paper_margin_parameter_disclosed': False,
        'smoother_and_rep_source_sha256': {str(p.relative_to(ROOT.parent)): sha(p) for p in
            (package / 'motion_rep/smooth_root.py', package / 'motion_rep/reps/kimodo_motionrep.py',
             package / 'motion_rep/reps/base.py', ROOT / 'data_pipeline/motion_features.py', ROOT / 'scripts/cache_m5_physical_inputs.py')},
        'source_sha256': sha(Path(__file__)), 'manifest_sha256': sha(manifest_path),
        'physical_input_fingerprint': view['input_fingerprint'], 'training_view_fingerprint': view['fingerprint'],
        'input_or_model_files_modified': False, 'new_optimizer_updates': 0, 'new_diffusion_samples': 0,
        'seed3102_quality_scores_read': False, 'final_audit_model_scores_read': False, 'M5_complete': False}
    write(out, result); print({'passed': result['passed'], 'samples': 18, 'output': str(out)}, flush=True)


if __name__ == '__main__':
    main()
