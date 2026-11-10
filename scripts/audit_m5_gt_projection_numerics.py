#!/usr/bin/env python3
"""Isolate precision effects in native-hinge projection of frozen training GT."""
import argparse
import hashlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from kimodo.motion_rep import KimodoMotionRep
from kimodo.skeleton.registry import build_skeleton
from training.m3 import batch_to
from training.m4 import rotate_batch
from training.g1_legality_loss import NativeLegalityLoss
from training.g1_physical import physical_constants, hinge_matrices
from training.g1_qpos import nearest_angles
from training.m5_physical_data import PhysicalSubsetDataset
from training.m5_physical_losses import geometry
from scripts.run_m5_length_research import read, write, sha, now


def main(output):
    if (output / 'report.json').exists(): raise FileExistsError('Preserve numeric audit')
    torch.set_num_threads(4); base = ROOT / 'runs/m5_physical_20261103'
    original = read(base / 'verification/paper_engineering_audit_20261104/root_representation.json')
    manifest = read(base / 'data/D_mixed_small_3500.json'); by_id = {r['sample_id']: r for r in manifest['train']}
    selected = [by_id[r['sample_id']] for r in original['records']]; assert len(selected) == 18
    view = read(base / 'data/READY.json'); dataset = PhysicalSubsetDataset(selected, view['input_dir'], base / 'stats/motion', cache_size=18)
    c = physical_constants(); constants_hashes = {p: sha(ROOT / p) for p in ('training/g1_physical.py', 'training/g1_legality_loss.py', 'training/g1_qpos.py', 'training/m4.py')}
    records = []; global_worst = None
    for device_name in ('cpu', 'cuda:0'):
        device = torch.device(device_name); skeleton = build_skeleton(34)
        rep = KimodoMotionRep(skeleton, fps=30, stats_path=str(base / 'stats/motion'))
        model = SimpleNamespace(motion_rep=rep); legality = NativeLegalityLoss(device)
        for label, minimum, maximum in (('short', 60, 90), ('medium', 91, 120), ('long', 121, 180)):
            ids = [i for i, row in enumerate(selected) if minimum <= row['num_frames'] <= maximum]
            assert len(ids) == 6
            entries = [selected[i] for i in ids]; batch = batch_to(dataset.batch(ids), device)
            angles = [((int(hashlib.sha256((e['sample_id'] + ':tail-probe-yaw-v1').encode()).hexdigest()[:12], 16) / 2**48) * 2 - 1) * math.pi for e in entries]
            batch = rotate_batch(model, batch, torch.tensor(angles, dtype=torch.float32, device=device)); mask = batch['x_pad_mask']
            with torch.no_grad():
                _, _, motion, _ = geometry(rep, batch['x0']); local = motion['local_rot_mats']
                e32, a32 = legality.per_joint(local)
                native32 = legality.native(local); p64, a64 = legality.project_native(native32.double())
                e64_same_native = .5 * (native32.double() - p64).square().sum((-2, -1))
                e64, _ = legality.per_joint(local.double())
            values = e32.masked_fill(~mask[..., None], -1); position = np.unravel_index(int(values.argmax()), values.shape)
            i, frame, hinge = map(int, position); native = native32[i, frame].double().cpu().numpy()
            optimal_np = nearest_angles(native[None], c['axes'], c['limits'], legality.limited.cpu().numpy())[0]
            expected_np = hinge_matrices(torch.from_numpy(c['axes']).double(), torch.from_numpy(optimal_np).double()).numpy()
            np_energy = .5 * np.square(native - expected_np).sum((-2, -1))
            entry = entries[i]
            record = {'device': device_name, 'bin': label, 'valid_joint_frames': int(mask.sum()) * 29,
                'fp32_mean_energy': float(e32[mask].mean()), 'fp32_max_energy': float(e32[mask].max()),
                'fp64_same_native_max_energy': float(e64_same_native[mask].max()),
                'fp64_recomputed_native_max_energy': float(e64[mask].max()),
                'worst_case': {'sample_id': entry['sample_id'], 'text': entry['text'], 'group': entry['group'],
                    'frame_0based': frame, 'hinge_index': hinge, 'node': skeleton.bone_order_names[int(c['indices'][hinge])],
                    'fp32_energy': float(e32[i, frame, hinge]), 'fp64_same_native_energy': float(e64_same_native[i, frame, hinge]),
                    'numpy_float64_energy': float(np_energy[hinge]), 'fp32_angle_rad': float(a32[i, frame, hinge]),
                    'fp64_angle_rad': float(a64[i, frame, hinge]), 'numpy_float64_angle_rad': float(optimal_np[hinge]),
                    'legal_bounds_rad': c['limits'][hinge].tolist()}}
            records.append(record)
            if global_worst is None or record['fp32_max_energy'] > global_worst['fp32_max_energy']:
                global_worst = record
            print(device_name, label, {k: record[k] for k in ('fp32_mean_energy', 'fp32_max_energy', 'fp64_same_native_max_energy')}, flush=True)
    # Analytic rotations inside the worst hinge's legal boundary. The exact
    # unbounded optimum is the generating angle; no model or GTfile is involved.
    worst = global_worst['worst_case']; hinge = worst['hinge_index']; bounds = c['limits'][hinge]
    boundary = 0 if abs(worst['fp32_angle_rad'] - bounds[0]) < abs(worst['fp32_angle_rad'] - bounds[1]) else 1
    sweep = []; legality = NativeLegalityLoss('cpu')
    for delta in (1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3):
        theta = float(bounds[boundary]) + delta * (1 if boundary == 0 else -1)
        angles = torch.zeros(29, dtype=torch.float64); angles[hinge] = theta
        rotation = hinge_matrices(torch.from_numpy(c['axes']).double(), angles)
        p32, a32 = legality.project_native(rotation.float()[None]); p64, a64 = legality.project_native(rotation[None])
        e32 = .5 * (rotation.float() - p32[0]).square().sum((-2, -1)); e64 = .5 * (rotation - p64[0]).square().sum((-2, -1))
        sweep.append({'distance_inside_bound_rad': delta, 'true_angle_rad': theta, 'fp32_angle_rad': float(a32[0, hinge]),
            'fp64_angle_rad': float(a64[0, hinge]), 'fp32_energy': float(e32[hinge]), 'fp64_energy': float(e64[hinge]),
            'fp32_selected_bound': bool(abs(float(a32[0, hinge]) - bounds[boundary]) < 2e-7)})
        assert abs(float(a64[0, hinge]) - theta) < 1e-10 and float(e64[hinge]) < 1e-18
    for path, digest in constants_hashes.items(): assert sha(ROOT / path) == digest
    report = {'schema': 'm5-native-projection-numeric-audit-v1', 'completed_at_utc': now(),
        'records': records, 'global_worst': global_worst, 'controlled_boundary_sweep': sweep,
        'all_fp64_GT_max_below_original1e_8': all(r['fp64_same_native_max_energy'] < 1e-8 and r['fp64_recomputed_native_max_energy'] < 1e-8 for r in records),
        'all_fp32_GT_mean_below_existing1e_8': all(r['fp32_mean_energy'] < 1e-8 for r in records),
        'input_contract_sources_unchanged': True, 'bound_source_sha256': constants_hashes,
        'source_sha256': sha(Path(__file__)), 'model_instantiated': False, 'new_optimizer_updates': 0,
        'new_DDIM_samples': 0, 'existing_frozen_loss_or_decoder_modified': False,
        'seed3102_quality_scores_read': False, 'final_audit_model_scores_read': False, 'M5_complete': False}
    write(output / 'report.json', report); print({'saved': str(output / 'report.json')}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output-dir', type=Path, required=True)
    main(p.parse_args().output_dir.resolve())
