#!/usr/bin/env python3
"""Check temporal formulas and independently sweep legal hinge projections."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from training.g1_physical import PhysicalG1Converter
from scripts.diagnose_m5_rated_pilot_temporal import describe
from scripts.run_m5_length_research import read, write, sha, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feedback-directory', type=Path, required=True)
    parser.add_argument('--private-package', type=Path, required=True); args = parser.parse_args()
    out = args.feedback_directory.resolve(); result_path = out / 'kinematic_verification.json'
    if result_path.exists(): raise FileExistsError('Preserve existing verification')
    temporal = read(out / 'temporal_diagnostics.json'); projection = read(out / 'projection_diagnostics.json')
    private = args.private_package.resolve(); artifacts = read(private / 'artifacts.json')
    by_key = {(r['trial_id'], r['side']): r for r in artifacts}; mapping = read(private / 'mapping.json')
    converter = PhysicalG1Converter(); c = converter.constants
    # Controlled reference: 0,10,20 degrees of native root yaw, no hinge motion.
    q = np.tile(c['qpos0'], (3, 1)); q[:, 2] = .8
    q[:, 3:7] = Rotation.from_euler('z', [0, 10, 20], degrees=True).as_quat(scalar_first=True)
    fixture = describe(q, converter)
    assert abs(fixture['max_root_rotation_per_frame_deg'] - 10.) < 1e-10
    assert abs(fixture['max_projected_root_heading_per_frame_deg'] - 10.) < 1e-10
    assert abs(fixture['net_projected_root_heading_deg'] - 20.) < 1e-10
    assert fixture['max_shortest_hinge_rotation_per_frame_deg'] == 0.
    sweeps = []; points = 20001
    for row in projection['rows']:
        trial = row['trial_id']; side = mapping[trial]['small_side']
        if row['model'] == 'tiny': side = 'B' if side == 'A' else 'A'
        artifact = by_key[(trial, side)]; source = Path(artifact['source']); stem = source.with_suffix('')
        bounded = np.loadtxt(source, delimiter=','); raw = np.loadtxt(stem.with_name(stem.name + '_unbounded.csv'), delimiter=',')
        with np.load(stem.with_suffix('.npz')) as archive: local = archive['local_rot_mats']
        hinge = row['worst_hinge_index_0based']; node = int(c['indices'][hinge]); start, end = row['worst_bounded_hinge_interval_0based']
        low, high = c['limits'][hinge]; theta = np.linspace(low, high, points)
        matrices = Rotation.from_rotvec(theta[:, None] * c['axes'][hinge]).as_matrix()
        checks = []
        for frame in (start, end):
            target = c['rest'][hinge].T @ local[frame, node]
            chosen = Rotation.from_rotvec(c['axes'][hinge] * bounded[frame, hinge + 7]).as_matrix()
            chosen_error = np.linalg.norm(chosen - target)
            sweep_error = np.linalg.norm(matrices - target[None], axis=(-2, -1))
            # Full3x3 Frobenius computation, independent of the scalar atan2/trace optimizer.
            assert chosen_error <= sweep_error.min() + 2e-6
            assert low - 1e-8 <= bounded[frame, hinge + 7] <= high + 1e-8
            checks.append({'frame_0based': frame, 'chosen_angle_deg': float(np.degrees(bounded[frame, hinge + 7])),
                'unbounded_angle_deg': float(np.degrees(raw[frame, hinge + 7])),
                'chosen_matrix_error': float(chosen_error), 'dense_sweep_best_matrix_error': float(sweep_error.min()),
                'selected_is_legal_optimum': True})
        sweeps.append({'trial_id': trial, 'model': row['model'], 'hinge': row['worst_hinge_skeleton_node'],
            'limits_deg': np.degrees([low, high]).tolist(),
            'bounded_shortest_change_deg': row['max_bounded_hinge_shortest_change_deg'],
            'unbounded_shortest_change_deg': row['same_interval_unbounded_hinge_shortest_change_deg'],
            'local_rotation_change_deg': row['same_interval_saved_local_rotation_change_deg'], 'endpoints': checks})
    assert len(sweeps) == 24
    for path, expected in {**projection['input_sha256'], **temporal['input_qpos_sha256']}.items(): assert sha(path) == expected
    assert sha(ROOT / 'scripts/diagnose_m5_rated_pilot_temporal.py') == temporal['source_sha256']
    assert sha(ROOT / 'scripts/diagnose_m5_rated_pilot_projection.py') == projection['source_sha256']
    record = {'passed': True, 'verified_at_utc': now(), 'root_yaw_10deg_per_frame_fixture_passed': True,
        'clips_checked': 24, 'worst_hinge_endpoint_sweeps': 48, 'legal_angles_per_sweep': points,
        'all_selected_hinge_rotations_are_legal_optima': True, 'hinge_sweeps': sweeps,
        'all24_root_rotations_agree_with_original_unbounded_output': projection['all24_root_quaternions_agree_with_saved_unbounded_and_6d'],
        'source_file_hashes_unchanged': True, 'temporal_report_sha256': sha(out / 'temporal_diagnostics.json'),
        'projection_report_sha256': sha(out / 'projection_diagnostics.json'),
        'source_sha256': sha(Path(__file__)), 'bounded_decoder_sha256': sha(ROOT / 'training/g1_qpos.py'),
        'physical_adapter_sha256': sha(ROOT / 'training/g1_physical.py'), 'trajectory_modified': False,
        'human_scores_changed': False, 'new_optimizer_updates': 0, 'new_diffusion_samples': 0, 'M5_complete': False}
    write(result_path, record)
    print({'passed': True, 'sweeps': 48})
    for row in sweeps:
        if row['bounded_shortest_change_deg'] > 179.: print(row)


if __name__ == '__main__': main()
