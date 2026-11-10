#!/usr/bin/env python3
"""CPU-only output-gradient diagnosis on retained development projection tails.

Illustrative angular reweighting and per-frame top-three reductions are not
training losses or calibrated coefficients. No model is instantiated or updated.
"""
import argparse
import math
import sys
from pathlib import Path
from statistics import mean
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from kimodo.motion_rep import KimodoMotionRep
from kimodo.skeleton.registry import build_skeleton
from training.m5_physical_losses import geometry
from training.g1_legality_loss import NativeLegalityLoss
from training.g1_physical import physical_constants, hinge_matrices
from scripts.run_m5_length_research import read, write, sha, now


def angular_squared(energy):
    # Both torch.where branches remain finite. Clipping is explicitly reported;
    # this illustrative transform is not approved as a production objective.
    bounded = energy.clamp(1e-5, 4. - 1e-5)
    angle = torch.acos((1. - .5 * bounded).clamp(-1., 1.))
    return torch.where(energy < 1e-5, energy, angle.square())


def controlled_sweep(legality, skeleton):
    c = physical_constants(); indices = c['indices']
    joint = next(i for i, node in enumerate(indices) if skeleton.bone_order_names[int(node)] == 'left_hip_roll_skel')
    axis = torch.from_numpy(c['axes'][joint]).double()
    basis = torch.eye(3, dtype=torch.float64)[axis.abs().argmin()]
    perpendicular = basis - axis * (axis * basis).sum(); perpendicular /= perpendicular.norm()
    records = []
    for degrees in (5., 30., 90., 150., 170., 175., 179.):
        phi = torch.tensor(math.radians(degrees), dtype=torch.float64, requires_grad=True)
        local = torch.eye(3, dtype=torch.float64).expand(1, 1, 34, 3, 3).clone()
        local[..., indices, :, :] = torch.from_numpy(c['rest'])
        local[..., int(indices[joint]), :, :] = torch.from_numpy(c['rest'][joint]) @ hinge_matrices(perpendicular[None], phi[None])[0]
        energy = legality.per_joint(local)[0][0, 0, joint]
        angular = angular_squared(energy)
        old_grad = torch.autograd.grad(energy, phi, retain_graph=True)[0]
        new_grad = torch.autograd.grad(angular, phi)[0]
        assert abs(float(energy) - 2. * (1. - math.cos(float(phi)))) < 1e-8
        assert abs(float(old_grad) - 2. * math.sin(float(phi))) < 1e-8
        assert abs(float(new_grad) - 2. * float(phi)) < 1e-7
        records.append({'off_axis_degrees': degrees, 'frob_energy': float(energy.detach()),
                        'current_dloss_dangle': float(old_grad), 'illustrative_angular_dloss_dangle': float(new_grad),
                        'unscaled_gradient_ratio': float(new_grad / old_grad)})
    return records


def gradient_summary(gradient, tail_frames, worst_frame):
    six = gradient[0, :, 107:311].reshape(len(tail_frames), 34, 6)
    per_frame = six.square().sum((-1, -2)); total = per_frame.sum()
    return {'rotation6d_output_gradient_l2': float(total.sqrt()),
            'squared_gradient_fraction_on_gt120deg_frames': float(per_frame[tail_frames].sum() / total.clamp_min(1e-30)),
            'squared_gradient_fraction_on_projection_worst_frame': float(per_frame[worst_frame] / total.clamp_min(1e-30)),
            'nonrotation_feature_gradient_max_abs': float(torch.cat((gradient[..., :107], gradient[..., 311:]), -1).abs().max()),
            'all_finite': bool(torch.isfinite(gradient).all())}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--top-cases', type=int, default=3); a = p.parse_args()
    if a.top_cases != 3:
        raise ValueError('This diagnosis predeclares the three largest projection cases per run')
    output = ROOT / 'runs/m5_geometry_20261104/diagnostics/legality_gradient_tails_050000.json'
    if output.exists():
        raise FileExistsError('Preserve the existing diagnostic')
    torch.set_num_threads(4); skeleton = build_skeleton(34); legality = NativeLegalityLoss('cpu')
    sweep = controlled_sweep(legality, skeleton); records = []
    for name in ('G1_consistency', 'G2_legality', 'G4_all'):
        run = ROOT / 'runs/m5_geometry_20261104' / name
        done = read(run / 'COMPLETED.json'); cfg = read(run / 'config.json')
        folder = run / 'evaluations/generation_full_050000'; report = read(folder / 'report.json')
        assert done['step'] == 50000 and report['identity'] == done['identity']
        assert sha(run / 'checkpoints/step_050000.pt') == done['terminal_checkpoint_sha256']
        for relative, expected in done['identity']['stats_sha256'].items():
            assert sha(run / relative) == expected
        rep = KimodoMotionRep(skeleton, fps=30, stats_path=str(run / 'stats/motion'))
        rows = sorted(read(folder / 'per_sample.json'), key=lambda r: r['projection_node_max_m'], reverse=True)[:3]
        for rank, row in enumerate(rows, 1):
            stem = folder / row['artifact_prefix']; feature_path = stem.with_name(stem.name + '_features.npy')
            feature_sha = sha(feature_path)
            features = torch.from_numpy(np.load(feature_path).copy())[None].requires_grad_(True)
            raw, _, motion, fallback = geometry(rep, features)
            retained = np.load(stem.with_suffix('.npz'))
            reconstruction_error = float(abs(motion['local_rot_mats'][0].detach().numpy() - retained['local_rot_mats']).max())
            assert reconstruction_error < 1e-5
            values, _ = legality.per_joint(motion['local_rot_mats'])
            angle_values = angular_squared(values)
            objectives = {'current_frobenius_mean': values.mean(), 'illustrative_angular_mean': angle_values.mean(),
                          'illustrative_angular_frame_top3': angle_values.topk(3, dim=-1).values.mean()}
            gradients = {key: torch.autograd.grad(value, features, retain_graph=True)[0]
                         for key, value in objectives.items()}
            bounded = np.loadtxt(stem.with_suffix('.csv'), delimiter=',')
            from training.g1_physical import PhysicalG1Converter
            with torch.inference_mode():
                bounded_positions = PhysicalG1Converter(skeleton).qpos_to_motion_dict(bounded)['posed_joints'].numpy()
            difference = np.linalg.norm(bounded_positions - retained['posed_joints'], axis=-1)
            worst_frame, worst_node = np.unravel_index(difference.argmax(), difference.shape)
            assert abs(float(difference.max()) - row['projection_node_max_m']) < 1e-5
            degrees = torch.acos((1. - .5 * values.detach()).clamp(-1., 1.)) * (180 / math.pi)
            tail_frames = (degrees[0] > 120.).any(-1)
            summaries = {key: gradient_summary(g, tail_frames, worst_frame) for key, g in gradients.items()}
            assert all(value['all_finite'] and value['rotation6d_output_gradient_l2'] > 0 for value in summaries.values())
            first, second = raw.detach()[0, :, 107:311].reshape(row['frames'], 34, 6).split(3, dim=-1)
            first_norm = first.norm(dim=-1); direction = first / first_norm[..., None].clamp_min(1e-6)
            residual_norm = (second - direction * (direction * second).sum(-1, keepdim=True)).norm(dim=-1)
            derivative_input = values.detach().clone().requires_grad_(True)
            derivative = torch.autograd.grad(angular_squared(derivative_input).sum(), derivative_input)[0]
            worst_joint = int(degrees[0, worst_frame].argmax())
            vjp_comparison = {}
            baseline = gradients['current_frobenius_mean'].flatten()
            for key in ('illustrative_angular_mean', 'illustrative_angular_frame_top3'):
                other = gradients[key].flatten()
                vjp_comparison[key] = {'cosine_vs_current_output_gradient': float(torch.nn.functional.cosine_similarity(baseline, other, dim=0)),
                    'unscaled_output_gradient_norm_ratio': float(other.norm() / baseline.norm()),
                    'inverse_ratio_for_output_norm_only_not_parameter_calibration': float(baseline.norm() / other.norm())}
            records.append({'run': name, 'rank_by_projection_max': rank,
                'checkpoint_sha256': done['terminal_checkpoint_sha256'], 'report_sha256': sha(folder / 'report.json'),
                'case': {k: row[k] for k in ('sample_id', 'group', 'seed', 'condition', 'frames', 'artifact_prefix')},
                'active_training_legality_weight': cfg.get('legality_weight', 0.),
                'maximum_projection_error_m': row['projection_node_max_m'], 'projection_worst_frame': int(worst_frame),
                'projection_worst_node': skeleton.bone_order_names[worst_node],
                'worst_frame_max_bounded_rotation_correction_degrees': float(degrees[0, worst_frame].max()),
                'worst_frame_largest_angular_multiplier': float(derivative[0, worst_frame, worst_joint]),
                'rotations_gt120deg_fraction': float((degrees > 120.).float().mean()),
                'angle_transform_upper_clipped_fraction': float((values.detach() >= 4. - 1e-5).float().mean()),
                'rot6d_fallback_fraction': float(fallback.float().mean()),
                'rot6d_min_first_column_norm': float(first_norm.min()),
                'rot6d_min_orthogonal_residual_norm': float(residual_norm.min()),
                'loss_values_unweighted': {k: float(v.detach()) for k, v in objectives.items()},
                'output_gradient_summaries': summaries, 'direction_comparison': vjp_comparison,
                'reconstructed_local_rotation_max_abs_difference': reconstruction_error,
                'retained_features_sha256': feature_sha, 'retained_motion_sha256': sha(stem.with_suffix('.npz'))})
            assert sha(feature_path) == feature_sha
            print('Diagnosed', name, rank, 'projection', row['projection_node_max_m'], 'm; correction',
                  round(float(degrees[0, worst_frame].max()), 2), 'deg', flush=True)
    write(output, {'schema': 'm5-native-legality-tail-gradient-diagnosis-v1', 'completed_at_utc': now(),
        'controlled_off_axis_rotation_sweep': sweep, 'records': records,
        'source_sha256': sha(Path(__file__)),
        'loss_source_sha256': sha(ROOT / 'training/g1_legality_loss.py'),
        'scope': 'Nine predeclared retained worst development cases, CPU output-space gradients only. Not a train-distribution calibration or proof of improved network training.',
        'illustrative_transform': 'acos(clamp(1-energy/2))^2 with an explicit small-energy linear branch and upper energy clip 4-1e-5; not a production objective.',
        'training_coefficients_selected': False, 'model_instantiated': False, 'new_optimizer_updates': 0,
        'new_diffusion_samples': 0, 'retained_features_or_qpos_modified': False,
        'seed3102_quality_scores_read': False, 'final_audit_model_scores_read': False, 'M5_complete': False})
    print('Saved diagnostic:', output, flush=True)


if __name__ == '__main__':
    main()
