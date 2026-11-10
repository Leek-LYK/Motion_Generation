#!/usr/bin/env python3
"""Independent geometry checks and full-frame diagnostics of retained M4 probes."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation
from kimodo.skeleton.registry import build_skeleton
from training.g1_qpos import G1BoundedDecoder, nearest_angles, DECODER_VERSION
from training.m3 import atomic_json, sha
from training.m4_readiness import movement


def main():
    skeleton = build_skeleton(34)
    decoder = G1BoundedDecoder(skeleton)
    rng = np.random.default_rng(20261103)
    q = np.zeros((64, 36), dtype=np.float32)
    q[:, :3] = rng.normal(size=(64, 3))*.1
    q[:, 2] += .8
    q[:, 3:7] = Rotation.random(64, random_state=rng).as_quat(scalar_first=True)
    for j, address in enumerate(decoder.addresses):
        lo, hi = decoder.bounds[j] if decoder.limited[j] else (-2, 2)
        q[:, address] = rng.uniform(lo+1e-3, hi-1e-3, 64)
    motion = decoder.converter.qpos_to_motion_dict(q, source_fps=30)
    recovered, info = decoder.decode(motion)
    angle_error = float(abs(recovered[:, 7:]-q[:, 7:]).max())
    assert angle_error < 2e-5, angle_error
    d0, d1 = mujoco.MjData(decoder.model), mujoco.MjData(decoder.model)
    fk_error = 0.0
    for before, after in zip(q, recovered):
        d0.qpos[:] = before; d1.qpos[:] = after
        mujoco.mj_forward(decoder.model, d0); mujoco.mj_forward(decoder.model, d1)
        fk_error = max(fk_error, float(abs(d0.xpos-d1.xpos).max()))
    assert fk_error < 1e-5, fk_error
    crossing = np.linspace(2.8, 3.5, 100)
    r = Rotation.from_rotvec(np.c_[crossing, np.zeros((100, 2))]).as_matrix()[:, None]
    unwrapped = nearest_angles(r, [[1, 0, 0]], [[0, 0]], [False])[:, 0]
    assert abs(unwrapped-crossing).max() < 1e-12
    # Independent dense feasible-angle search, including strongly off-axis poses.
    random_r = Rotation.random(32*29, random_state=rng).as_matrix().reshape(32, 29, 3, 3)
    projected = nearest_angles(random_r, decoder.axes, decoder.bounds, decoder.limited)
    worst_grid_advantage = -np.inf
    for j, axis in enumerate(decoder.axes.astype(float)):
        axis /= np.linalg.norm(axis)
        lo, hi = decoder.bounds[j] if decoder.limited[j] else (-np.pi, np.pi)
        grid = Rotation.from_rotvec(np.linspace(lo, hi, 2001)[:, None]*axis).as_matrix()
        picked = Rotation.from_rotvec(projected[:, j, None]*axis).as_matrix()
        scores = np.einsum('gik,tik->tg', grid, random_r[:, j])
        selected = np.einsum('tik,tik->t', picked, random_r[:, j])
        worst_grid_advantage = max(worst_grid_advantage, float((scores.max(1)-selected).max()))
    assert worst_grid_advantage < 1e-10, worst_grid_advantage
    research = ROOT/'runs/m4_research_20261103'
    output = research/'diagnostics/bounded_decoder_v1'
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for run in ('R3_D_high_noise', 'R9_D_high_noise_seed20261104'):
        rows = []
        for probe in ('controlled_three_seeds_cfg2', 'all_unseen_jump_controls_cfg2'):
            directory = research/run/probe
            if not directory.exists():
                continue
            for source in sorted(directory.glob('seed_*/*.npz')):
                with np.load(source, allow_pickle=False) as archive:
                    motion = {k: torch.from_numpy(archive[k].copy()) for k in archive.files}
                bounded, diagnostics = decoder.decode(motion)
                legacy = np.loadtxt(source.with_suffix('.csv'), delimiter=',')
                assert np.allclose(bounded[:, :3], legacy[:, :3], atol=0, rtol=0)
                rebuilt = decoder.converter.qpos_to_motion_dict(bounded.astype(np.float32), source_fps=30)
                distance = torch.linalg.vector_norm(rebuilt['posed_joints']-motion['posed_joints'], dim=-1)
                group = next(g for g in ('raise_arms', 'walk', 'jog', 'jump') if g in source.stem)
                target = output/run/probe/source.parent.name/source.with_suffix('.csv').name
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(target, bounded, delimiter=',')
                rows.append({'source': str(source.relative_to(ROOT)), 'bounded_qpos': str(target.relative_to(ROOT)),
                             'source_sha256': sha(source), **diagnostics,
                             'mean_node_difference_m': float(distance.mean()),
                             'max_node_difference_m': float(distance.max()),
                             'raw_amplitude': movement(motion, skeleton, group),
                             'bounded_amplitude': movement(rebuilt, skeleton, group)})
        reports[run] = {'clips': len(rows), 'rows': rows,
                        'max_limit_violation_rad': max(r['bounded']['max_limit_violation_rad'] for r in rows),
                        'max_joint_speed_rad_s': max(r['bounded']['max_joint_speed_rad_s'] for r in rows),
                        'median_clip_p99_joint_speed_rad_s': float(np.median([r['bounded']['p99_joint_speed_rad_s'] for r in rows]))}
    report = {'decoder': DECODER_VERSION, 'decoder_sha256': sha(ROOT/'training/g1_qpos.py'),
              'verification_script_sha256': sha(__file__), 'xml_sha256': sha(decoder.converter.xml_path),
              'roundtrip_max_hinge_error_rad': angle_error, 'roundtrip_mujoco_body_position_error_m': fk_error,
              'cross_pi_unlimited_joint_check': True, 'dense_grid_max_objective_advantage': worst_grid_advantage,
              'checks_passed': True, 'experiments': reports,
              'note': 'Postprocessing is explicit. Raw model features and legacy qpos are retained. No temporal/root/contact correction.'}
    atomic_json(output/'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'experiments'}, indent=2))
    for run, value in reports.items():
        print(run, {k: v for k, v in value.items() if k != 'rows'})


if __name__ == '__main__':
    main()
