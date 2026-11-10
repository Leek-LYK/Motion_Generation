#!/usr/bin/env python3
"""Descriptive kinematics on exact human-rated pilot qpos; no sampling or repair."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from training.g1_physical import PhysicalG1Converter
from scripts.run_m5_length_research import read, write, sha, now


def describe(q, converter):
    """All temporal derivatives use the original 30Hz frames, without filtering."""
    assert q.ndim == 2 and q.shape[1] == 36 and np.isfinite(q).all()
    assert np.allclose(np.linalg.norm(q[:, 3:7], axis=-1), 1., atol=2e-6, rtol=0)
    rotation = Rotation.from_quat(q[:, 3:7], scalar_first=True)
    root_delta = (rotation[:-1].inv() * rotation[1:]).magnitude()
    forward = rotation.as_matrix()[:, :2, 0]
    valid_heading = np.linalg.norm(forward, axis=-1) > 1e-6
    heading = np.unwrap(np.arctan2(forward[:, 1], forward[:, 0]))
    yaw_delta = np.diff(heading)
    native_delta = np.diff(q[:, 7:], axis=0)
    short_delta = np.arctan2(np.sin(native_delta), np.cos(native_delta))
    motion = converter.qpos_to_motion_dict(q.astype(np.float32))
    bottoms = converter.foot_bottom(motion).numpy()
    positions = motion['posed_joints'].numpy(); matrices = motion['global_rot_mats'].numpy()
    centers = np.stack([positions[:, f['node']] + (matrices[:, f['node']] @ f['offset'][None, :, None]).squeeze(-1)
                        for f in converter.constants['foot']], axis=1)
    speed = np.linalg.norm(np.diff(centers[..., [0, 2]], axis=0), axis=-1) * 30
    # A geometric proxy, not a measured contact/friction state. Both interval
    # endpoints must lie near the floor; deep penetration is reported separately.
    near = (bottoms[:-1] <= .03) & (bottoms[1:] <= .03)
    shallow = near & (bottoms[:-1] >= -.01) & (bottoms[1:] >= -.01)
    arms = {}
    for side in ('left', 'right'):
        shoulder = converter.skeleton.bone_index[side + '_shoulder_pitch_skel']
        wrist = converter.skeleton.bone_index[side + '_wrist_roll_skel']
        vector = positions[:, wrist] - positions[:, shoulder]
        elevation = np.degrees(np.arccos(np.clip(-vector[:, 1] / np.linalg.norm(vector, axis=-1), -1., 1.)))
        arms[side] = {'p90_world_elevation_from_down_deg': float(np.quantile(elevation, .9)),
                      'max_world_elevation_from_down_deg': float(elevation.max()),
                      'wrist_height_range_m': float(np.ptp(positions[:, wrist, 1]))}
    def speed_summary(mask):
        values = speed[mask]
        return {'intervals': int(mask.sum()), 'total_foot_intervals': int(mask.size),
                'mean_m_s': float(values.mean()) if len(values) else None,
                'p95_m_s': float(np.quantile(values, .95)) if len(values) else None}
    return {'frames': len(q), 'max_root_rotation_per_frame_deg': float(np.degrees(root_delta.max())),
        'p95_root_rotation_per_frame_deg': float(np.degrees(np.quantile(root_delta, .95))),
        'max_projected_root_heading_per_frame_deg': float(np.degrees(abs(yaw_delta).max())) if valid_heading.all() else None,
        'net_projected_root_heading_deg': float(np.degrees(heading[-1] - heading[0])) if valid_heading.all() else None,
        'max_shortest_hinge_rotation_per_frame_deg': float(np.degrees(abs(short_delta).max())),
        'p95_frame_max_shortest_hinge_rotation_deg': float(np.degrees(np.quantile(abs(short_delta).max(-1), .95))),
        'hinge_coordinate_jumps_over_pi': int((abs(native_delta) > np.pi).sum()),
        'root_horizontal_path_m': float(np.linalg.norm(np.diff(q[:, :2], axis=0), axis=-1).sum()),
        'root_height_range_m': float(np.ptp(q[:, 2])),
        'max_bounded_foot_penetration_m': float(max(0., -bottoms.min())),
        'bounded_deep_penetration_frame_fraction': float((bottoms.min(-1) < -.05).mean()),
        'near_floor_collision_center_speed': speed_summary(near),
        'shallow_near_floor_collision_center_speed': speed_summary(shallow), 'arms': arms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-package', type=Path, required=True)
    parser.add_argument('--feedback-analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve retained-trajectory diagnostic evidence')
    private = args.private_package.resolve(); feedback = read(args.feedback_analysis)
    assert feedback['actual_reviewers'] == 1 and feedback['ratings_are_actual_human_feedback']
    complete = read(private / 'COMPLETE.json'); mapping = read(private / 'mapping.json')
    for relative, expected in complete['private_sha256'].items(): assert sha(private / relative) == expected
    converter = PhysicalG1Converter(); rows = []; inputs = {}; trial_to_rows = {}
    for artifact in read(private / 'artifacts.json'):
        source = private / artifact['private_qpos_copy']; digest = sha(source)
        assert digest == artifact['source_qpos_sha256']; inputs[str(source)] = digest
        q = np.loadtxt(source, delimiter=','); assert len(q) == artifact['frames']
        trial = artifact['trial_id']; model = 'small' if mapping[trial]['small_side'] == artifact['side'] else 'tiny'
        row = {'trial_id': trial, 'side': artifact['side'], 'model': model, 'group': mapping[trial]['group'],
               'qpos_sha256': digest, 'checkpoint_sha256': artifact['checkpoint_sha256'], **describe(q, converter)}
        rows.append(row); trial_to_rows.setdefault(trial, {})[model] = row
    assert len(rows) == 24 and len(trial_to_rows) == 12 and all(set(v) == {'small', 'tiny'} for v in trial_to_rows.values())
    linked = [{**rating, 'kinematics': trial_to_rows[rating['trial_id']]} for rating in feedback['decoded_rows']]
    result = {'schema': 'm5-rated-pilot-retained-kinematics-v1', 'created_at_utc': now(),
        'package_id': feedback['package_id'], 'models': feedback['models'], 'clips': len(rows), 'fps': 30,
        'scope': 'Exact bounded qpos shown in the40k pilot,24 clips/12 prompts/one sample seed; descriptive diagnosis only.',
        'definitions': {'root_rotation': 'Quaternion geodesic change per original frame, degrees.',
            'heading': 'Unwrapped projected native root forward-axis yaw; distinct from a whole-motion turn-completion rating.',
            'hinge_rotation': 'Shortest equivalent angle between adjacent native hinge poses; raw coordinate branch jumps reported separately.',
            'near_floor_speed': 'Planar collision-capsule center speed when capsule bottom<=3cm at both interval ends. Shallow subset also requires bottom>=-1cm. Geometry proxy, not measured contact, friction or slip.',
            'arm_elevation': 'World shoulder-pitch anchor to wrist-roll anchor angle from downward vertical; root/body lean can affect it.'},
        'linked_ratings_and_kinematics': linked, 'input_qpos_sha256': inputs,
        'feedback_analysis_sha256': sha(args.feedback_analysis), 'source_sha256': sha(Path(__file__)),
        'physical_adapter_sha256': sha(ROOT / 'training/g1_physical.py'),
        'trajectory_filtering_or_modification': False, 'new_optimizer_updates': 0, 'new_diffusion_samples': 0,
        'automatic_human_rating_changes': False, 'new_acceptance_thresholds': False,
        'seed3102_quality_scores_read': False, 'final_audit_model_scores_read': False, 'M5_complete': False}
    write(args.output, result)
    print({'clips': len(rows), 'output': str(args.output), 'new_optimizer_updates': 0})


if __name__ == '__main__':
    main()
