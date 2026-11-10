#!/usr/bin/env python3
"""Trace rated-pilot temporal jumps through saved 6D poses and native projection."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from training.g1_physical import PhysicalG1Converter
from training.m5_physical_losses import stable_cont6d
from training.m5_physical_data import PhysicalSubsetDataset
from kimodo.skeleton.transforms import global_rots_to_local_rots
from scripts.run_m5_length_research import read, write, sha, now


def rotation_changes(values):
    return np.degrees(Rotation.from_matrix(np.swapaxes(values[:-1], -1, -2) @ values[1:]).magnitude())


def twist_fit(local, constants):
    r = np.swapaxes(constants['rest'], -1, -2) @ local[:, constants['indices']]
    axes = constants['axes']
    a = np.trace(r, axis1=-2, axis2=-1) - (axes * (r @ axes[..., None]).squeeze(-1)).sum(-1)
    skew = np.stack((r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]), -1)
    b = (skew * axes).sum(-1)
    return np.sqrt(a * a + b * b)


def sixd_condition(raw):
    value = raw[:, 107:311].reshape(len(raw), 34, 6); a = value[..., :3]; b = value[..., 3:]
    norm = np.linalg.norm(a, axis=-1); unit = a / np.maximum(norm[..., None], 1e-30)
    perpendicular = b - (unit * b).sum(-1, keepdims=True) * unit
    return norm, np.linalg.norm(perpendicular, axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve existing projection diagnosis')
    private = args.private_package.resolve(); complete = read(private / 'COMPLETE.json')
    for relative, expected in complete['private_sha256'].items(): assert sha(private / relative) == expected
    manifest = read(private / 'manifest.json'); mapping = read(private / 'mapping.json')
    converter = PhysicalG1Converter(); c = converter.constants; C = c['C']; rows = []; inputs = {}
    sk = converter.skeleton
    for artifact in read(private / 'artifacts.json'):
        trial = artifact['trial_id']; side = artifact['side']; small = mapping[trial]['small_side'] == side
        model = manifest['models'][0] if small else manifest['tiny']; source = Path(artifact['source']); stem = source.with_suffix('')
        q = np.loadtxt(private / artifact['private_qpos_copy'], delimiter=','); assert sha(source) == artifact['source_qpos_sha256']
        unbounded_path = stem.with_name(stem.name + '_unbounded.csv'); unbounded = np.loadtxt(unbounded_path, delimiter=',')
        motion_path = stem.with_suffix('.npz'); feature_path = stem.with_name(stem.name + '_features.npy')
        for file in (source, unbounded_path, motion_path, feature_path): inputs[str(file)] = sha(file)
        stats = Path(model['run']) / 'stats/motion'
        for relative, expected in model['identity']['stats_sha256'].items(): assert sha(Path(model['run']) / relative) == expected
        mean = np.concatenate([np.load(stats / part / 'mean.npy') for part in ('global_root', 'body')]).astype(np.float32)
        std = np.concatenate([np.load(stats / part / 'std.npy') for part in ('global_root', 'body')]).astype(np.float32)
        features = np.load(feature_path); raw = features * np.sqrt(std * std + np.float32(1e-5)) + mean
        with np.load(motion_path) as archive:
            local = archive['local_rot_mats']; global_r = archive['global_rot_mats']; saved_heading = archive['global_root_heading']
        torch_r, fallback = stable_cont6d(torch.from_numpy(raw[:, 107:311].reshape(len(raw), 34, 6)))
        assert np.max(abs(torch_r.numpy() - global_r)) < 3e-6
        assert np.max(abs(saved_heading - raw[:, 3:5])) < 1e-6
        native_root = C @ global_r[:, sk.root_idx] @ C.T
        q_root = Rotation.from_quat(q[:, 3:7], scalar_first=True).as_matrix()
        raw_root = Rotation.from_quat(unbounded[:, 3:7], scalar_first=True).as_matrix()
        root_matrix_difference = float(abs(native_root - q_root).max())
        assert root_matrix_difference < 2e-6 and np.max(abs(q_root - raw_root)) < 2e-6
        assert np.max(abs(q[:, :3] - unbounded[:, :3])) < 1e-10
        root_changes = rotation_changes(q_root); t = int(np.argmax(root_changes))
        heading_angle = np.arctan2(raw[:, 4], raw[:, 3]); heading_delta = np.arctan2(np.sin(np.diff(heading_angle)), np.cos(np.diff(heading_angle)))
        first_norm, second_norm = sixd_condition(raw)
        hinge_delta = np.diff(q[:, 7:], axis=0); shortest = abs(np.arctan2(np.sin(hinge_delta), np.cos(hinge_delta)))
        frame, hinge = np.unravel_index(int(shortest.argmax()), shortest.shape); node = int(c['indices'][hinge])
        fitted = twist_fit(local, c)
        local_delta = Rotation.from_matrix(local[frame, node].T @ local[frame + 1, node]).magnitude()
        unbounded_delta = unbounded[frame + 1, hinge + 7] - unbounded[frame, hinge + 7]
        raw_shortest = np.arctan2(np.sin(unbounded_delta), np.cos(unbounded_delta))
        root_heading_norm = np.linalg.norm(raw[:, 3:5], axis=-1)
        rows.append({'trial_id': trial, 'model': 'small' if small else 'tiny', 'group': mapping[trial]['group'],
            'root_bounded_vs_saved_matrix_max_error': root_matrix_difference,
            'root_clipping_changes_rotation': False, 'root_worst_interval_0based': [t, t + 1],
            'max_root_rotation_per_frame_deg': float(root_changes[t]),
            'root_sixd_first_norm_at_worst_interval': first_norm[t:t + 2, sk.root_idx].tolist(),
            'root_sixd_second_perpendicular_norm_at_worst_interval': second_norm[t:t + 2, sk.root_idx].tolist(),
            'all_nodes_sixd_first_norm_min': float(first_norm.min()),
            'all_nodes_sixd_second_perpendicular_norm_min': float(second_norm.min()),
            'sixd_exact_fallback_fraction': float(fallback.float().mean()),
            'predicted_heading_vector_norm_min': float(root_heading_norm.min()),
            'max_Root_heading_change_per_frame_deg': float(np.degrees(abs(heading_delta).max())),
            'Root_heading_change_at_body_root_worst_interval_deg': float(np.degrees(heading_delta[t])),
            'worst_bounded_hinge_interval_0based': [int(frame), int(frame + 1)], 'worst_hinge_index_0based': int(hinge),
            'worst_hinge_skeleton_node': sk.bone_names[node] if hasattr(sk, 'bone_names') else next(k for k, v in sk.bone_index.items() if v == node),
            'max_bounded_hinge_shortest_change_deg': float(np.degrees(shortest[frame, hinge])),
            'same_interval_unbounded_hinge_shortest_change_deg': float(np.degrees(abs(raw_shortest))),
            'same_interval_saved_local_rotation_change_deg': float(np.degrees(local_delta)),
            'same_interval_native_twist_fit_vector_norm': fitted[frame:frame + 2, hinge].tolist(),
            'native_twist_fit_vector_norm_min_all_hinges_frames': float(fitted.min())})
    # Reference conditioning uses only twelve previously frozen training events.
    fixture = read(ROOT / 'runs/m5_geometry_20261104/verification/long_sequence_contract/subset.json')
    base = ROOT / 'runs/m5_physical_20261103'; view = read(base / 'data/READY.json')
    dataset = PhysicalSubsetDataset(fixture['train'], view['input_dir'], base / 'stats/motion')
    reference = []
    for i, entry in enumerate(fixture['train']):
        raw = dataset[i]['x0'].numpy() * dataset.scale + dataset.mean; first, second = sixd_condition(raw)
        global_r, fallback = stable_cont6d(torch.from_numpy(raw[:, 107:311].reshape(len(raw), 34, 6)))
        local = global_rots_to_local_rots(global_r, sk).numpy(); fitted = twist_fit(local, c)
        reference.append({'sample_id': entry['sample_id'], 'split': entry['split'], 'frames': len(raw),
            'sixd_first_norm_min': float(first.min()), 'sixd_second_perpendicular_norm_min': float(second.min()),
            'twist_fit_vector_norm_min': float(fitted.min()), 'sixd_fallback_count': int(fallback.sum())})
    assert len(reference) == 12 and all(r['split'] == 'train' for r in reference)
    result = {'schema': 'm5-rated-pilot-projection-temporal-diagnosis-v1', 'created_at_utc': now(),
        'package_id': manifest['package_id'], 'rows': rows, 'train_only_reference_conditioning': reference,
        'definitions': {'native_twist_fit_vector_norm': 'sqrt(a^2+b^2) for native atan2 angle fitting;2 for an exact legal hinge rotation,approaching0 indicates an ill-conditioned off-axis projection.',
            'root_sixd_norms': 'Norm of the first raw6D column and the orthogonal component of the second;unit target columns have norms1. No new numerical cutoff is used.',
            'Root_heading_vs_body_root': 'Root network predicted cosine/sine heading and Body predicted global6D root rotation are separately reported; they encode related directions through distinct outputs.'},
        'all24_root_quaternions_agree_with_saved_unbounded_and_6d': True,
        'input_sha256': inputs, 'source_sha256': sha(Path(__file__)),
        'source_files_are_original_retained_outputs': True, 'trajectory_modified': False,
        'new_optimizer_updates': 0, 'new_diffusion_samples': 0, 'new_acceptance_thresholds': False,
        'seed3102_quality_scores_read': False, 'final_audit_model_scores_read': False, 'M5_complete': False}
    write(args.output, result); print({'clips': len(rows), 'train_references': len(reference), 'output': str(args.output)})


if __name__ == '__main__': main()
