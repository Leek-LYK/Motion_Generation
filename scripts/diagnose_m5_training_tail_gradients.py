#!/usr/bin/env python3
"""Train-only raw-model parameter gradients by noise stage; never optimizer.step."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import gc
import hashlib
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from kimodo.model.diffusion import Diffusion
from training.m3 import configure, make_model, batch_to, forward
from training.m4 import rotate_batch
from training.m5_physical_data import PhysicalSubsetDataset
from training.m5_physical_losses import geometry, objective, NativeHingeGeometry
from training.g1_legality_loss import NativeLegalityLoss
from training.m5_tail_prototypes import legality_reductions, temporal_supervision, VERSION
from scripts.run_m5_length_research import read, write, sha, now

TIMESTEPS = (0, 249, 499, 749, 999)


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def norm(gradients, device):
    values = [g.float().square().sum() for g in gradients if g is not None]
    return torch.sqrt(torch.stack(values).sum()) if values else torch.zeros((), device=device)


def cosine(left, right, device):
    a = norm(left, device); b = norm(right, device)
    if float(a) <= 1e-20 or float(b) <= 1e-20: return None
    pieces = [(x * y).sum() for x, y in zip(left, right) if x is not None and y is not None]
    return float(torch.stack(pieces).sum() / (a * b)) if pieces else 0.


def point(model, batch, times, noisy, config, hinge, legality):
    mask = batch['x_pad_mask']; rep = model.motion_rep; device = noisy.device
    root_params = tuple(model.root_model.parameters()); body_params = tuple(model.body_model.parameters())
    parameters = root_params + body_params; split = len(root_params)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        prediction = forward(model, noisy, batch, times)
    prediction = prediction.float()
    with torch.autocast(device_type='cuda', enabled=False):
        active_base, parts = objective(rep, prediction, batch['x0'], mask, config, hinge)
        raw, _, motion, fallback = geometry(rep, prediction)
        truth = rep.unnormalize(batch['x0'])
        energy, _ = legality.per_joint(motion['local_rot_mats'])
        reductions = legality_reductions(energy, mask, 3, .1)
        legacy_mean = legality.loss(motion['local_rot_mats'], mask)
        assert abs(float(reductions['mean'] - legacy_mean)) < 1e-7
        a = torch.autograd.grad(reductions['mean'], prediction, retain_graph=True)[0]
        b = torch.autograd.grad(legacy_mean, prediction, retain_graph=True)[0]
        parity = float(abs(a - b).max()); assert parity < 1e-6
        terms = {'active_raw7_plus_G1': active_base, 'native_mean': reductions['mean'],
            'native_frame_top3': reductions['frame_topk'],
            'native_tail10pct_frame_top3': reductions['frame_tail_topk'],
            'root_heading_velocity_GT': temporal_supervision(raw[..., 3:5], truth[..., 3:5], mask),
            'body_global6d_velocity_GT': temporal_supervision(raw[..., 107:311], truth[..., 107:311], mask)}
    base_grads = torch.autograd.grad(active_base, parameters, allow_unused=True, retain_graph=True)
    base_root = float(norm(base_grads[:split], device)); base_body = float(norm(base_grads[split:], device))
    assert base_root > 0 and base_body > 0
    gradients = {}; severe = (energy.detach() >= 2.).any(-1) & mask
    for name, value in terms.items():
        g = base_grads if name == 'active_raw7_plus_G1' else torch.autograd.grad(value, parameters, allow_unused=True, retain_graph=True)
        assert all(v is None or torch.isfinite(v).all() for v in g)
        root_norm = float(norm(g[:split], device)); body_norm = float(norm(g[split:], device))
        record = {'root_parameter_gradient_l2': root_norm, 'body_parameter_gradient_l2': body_norm,
            'root_gradient_ratio_vs_active_base': root_norm / base_root,
            'body_gradient_ratio_vs_active_base': body_norm / base_body,
            'root_cosine_vs_active_base': cosine(g[:split], base_grads[:split], device),
            'body_cosine_vs_active_base': cosine(g[split:], base_grads[split:], device)}
        if name.startswith('native_'):
            assert root_norm <= 1e-6
            output_gradient = torch.autograd.grad(value, prediction, retain_graph=True)[0]
            square = output_gradient[..., 107:311].square().sum(-1); total = square.sum()
            record.update(rot6d_output_gradient_l2=float(total.sqrt()),
                gradient_energy_fraction_on_worst10pct_frames=float(square[reductions['selected_tail_frames']].sum() / total.clamp_min(1e-30)),
                gradient_energy_fraction_on_gt90deg_correction_frames=float(square[severe].sum() / total.clamp_min(1e-30)))
        if name == 'root_heading_velocity_GT': assert body_norm <= 1e-6
        if name == 'body_global6d_velocity_GT': assert root_norm <= 1e-6
        gradients[name] = record
        if name != 'active_raw7_plus_G1': del g
    valid_energy = energy.detach()[mask]
    return {'losses': {name: float(value.detach()) for name, value in terms.items()},
        'parameter_gradients': gradients, 'legacy_mean_output_gradient_max_abs_difference': parity,
        'geometry': {'native_energy_mean': float(valid_energy.mean()), 'native_energy_max': float(valid_energy.max()),
            'native_correction_gt90deg_fraction': float((valid_energy >= 2.).float().mean()),
            'severe_frame_fraction': float(severe.sum() / mask.sum()),
            'valid_rot6d_fallback_fraction': float((fallback * mask[..., None]).sum() / (mask.sum() * 34)),
            'fk_self_mean_m': float(parts['fk_position_inconsistency_m'])},
        'gradient_into_Root_from_native_terms_zero': True, 'root_temporal_gradient_into_Body_zero': True}


def main(output):
    output = output.resolve(); output.relative_to(ROOT / 'runs')
    if (output / 'protocol.json').exists() or (output / 'report.json').exists(): raise FileExistsError('Preserve prior diagnosis')
    verification = ROOT / 'runs/m5_physical_20261103/verification/tail_prototypes_20261104.json'
    verified = read(verification); assert verified['passed'] and verified['prototype_sha256'] == sha(ROOT / 'training/m5_tail_prototypes.py')
    base = ROOT / 'runs/m5_physical_20261103'; selection_path = base / 'verification/paper_engineering_audit_20261104/root_representation.json'
    selection = read(selection_path); source_manifest = base / 'data/D_mixed_small_3500.json'
    manifest = read(source_manifest); by_id = {r['sample_id']: r for r in manifest['train']}
    selected = [by_id[r['sample_id']] for r in selection['records']]
    assert len(selected) == 18 and all(r['split'] == 'train' for r in selected)
    groups = ('walk', 'jog', 'jump', 'raise_arms', 'turn', 'squat'); bins = ('short', 'medium', 'long')
    bucket = lambda row: 'short' if row['num_frames'] <= 90 else 'medium' if row['num_frames'] <= 120 else 'long'
    batches = {label: [next(r for r in selected if r['group'] == g and bucket(r) == label) for g in groups] for label in bins}
    length_run = ROOT / 'runs/m5_length_research_20261104/L1_mixed_G1'
    sources = {'C40': {'run': base / 'C_small_raw7', 'checkpoint': ROOT / 'runs/m5_curriculum_factorial_20261104/parent_snapshot/checkpoint.pt', 'step': 40000},
               'T3_60': {'run': length_run, 'checkpoint': length_run / 'checkpoints/latest.pt', 'step': 60000}}
    common = read(length_run / 'config.json')
    assert common['consistency_weight'] == .012261994051407363 and all(common.get(k, 0.) == 0 for k in ('legality_weight', 'ground_weight', 'axis_weight'))
    config = dict(common)
    for value in sources.values(): value['sha256'] = sha(value['checkpoint'])
    output.mkdir(parents=True, exist_ok=True)
    protocol = {'schema': 'm5-train-only-tail-noise-gradient-probe-v1', 'frozen_at_utc': now(),
        'models': {k: {f: str(v) if isinstance(v, Path) else v for f, v in value.items()} for k, value in sources.items()},
        'weights': 'raw training model,not EMA', 'selected_train_sample_ids': [r['sample_id'] for r in selected],
        'selection_evidence_sha256': sha(selection_path), 'subset_sha256': sha(source_manifest),
        'batch_size': 6, 'timesteps': list(TIMESTEPS), 'precision': 'BF16 forward/FP32 losses and parameter gradients',
        'common_objective': 'weighted_raw7 plus existingG1consistency; same for both frozen models.',
        'tail_reductions': 'Mean vs per-frame top3 hinges vs within-clip worst10%frame top3; original valid-frame weighting acrossclips preserved.',
        'temporal_prototypes': 'Supervised30Hz finite differences of rawRootcos/sin and Bodyglobal6D; no filtering.',
        'text_dropout': False, 'yaw': 'Fixed per-event hash angle shared acrossmodels/timesteps',
        'noise': 'Fixed per-event/timestep CUDAgenerator shared acrossmodels',
        'prototype_sha256': sha(ROOT / 'training/m5_tail_prototypes.py'), 'source_sha256': sha(Path(__file__)),
        'prototype_verification_sha256': sha(verification), 'training_coefficients_selected': False,
        'new_optimizer_updates': 0, 'new_DDIM_samples': 0, 'M5_complete': False}
    write(output / 'protocol.json', protocol); configure(8119); device = torch.device('cuda:0')
    diffusion = Diffusion(1000).to(device); records = []; ground_truth = []; state_checks = {}
    for label, spec in sources.items():
        state = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False)
        assert state['step'] == spec['step'] and state['config']['seed'] == 3101 and state['config']['network_dropout'] == 0
        assert state['identity']['physical_input_fingerprint'] == read(base / 'data/READY.json')['input_fingerprint']
        for relative, expected in state['identity']['stats_sha256'].items(): assert sha(spec['run'] / relative) == expected
        model = make_model(state['config']['model'], spec['run'] / 'stats/motion').to(device).train()
        model.load_state_dict(state['model']); del state
        assert sum(p.numel() for p in model.parameters()) == 56755617
        assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in model.root_model.parameters()) + sum(p.numel() for p in model.body_model.parameters())
        before = model_digest(model); hinge = NativeHingeGeometry(model.motion_rep.skeleton, device); legality = NativeLegalityLoss(device)
        dataset = PhysicalSubsetDataset(selected, config['input_dir'], spec['run'] / 'stats/motion', cache_size=18)
        indices = {r['sample_id']: i for i, r in enumerate(selected)}
        for bin_name, entries in batches.items():
            samples = dataset.batch([indices[e['sample_id']] for e in entries]); batch = batch_to(samples, device)
            angles = [((int(hashlib.sha256((e['sample_id'] + ':tail-probe-yaw-v1').encode()).hexdigest()[:12], 16) / 2**48) * 2 - 1) * math.pi for e in entries]
            batch = rotate_batch(model, batch, torch.tensor(angles, dtype=torch.float32, device=device))
            mask = batch['x_pad_mask']
            with torch.no_grad():
                truth, _, gt_motion, _ = geometry(model.motion_rep, batch['x0'])
                energy, _ = legality.per_joint(gt_motion['local_rot_mats']); gt_reduced = legality_reductions(energy, mask)
                assert float(energy[mask].max()) < 1e-8
                ground_truth.append({'model': label, 'bin': bin_name, 'native_mean': float(gt_reduced['mean']),
                    'native_frame_top3': float(gt_reduced['frame_topk']), 'native_tail10pct_frame_top3': float(gt_reduced['frame_tail_topk']),
                    'root_temporal_GT_self': float(temporal_supervision(truth[..., 3:5], truth[..., 3:5], mask)),
                    'body_temporal_GT_self': float(temporal_supervision(truth[..., 107:311], truth[..., 107:311], mask))})
            for timestep in TIMESTEPS:
                noise = torch.zeros_like(batch['x0']); seeds = []
                for i, event in enumerate(samples):
                    seed = int(hashlib.sha256(f"train-tail-probe-v1:{event['sample_id']}:{timestep}".encode()).hexdigest()[:15], 16)
                    rng = torch.Generator(device=device).manual_seed(seed); seeds.append(seed)
                    noise[i, :event['length']] = torch.randn((event['length'], 417), device=device, generator=rng)
                times = torch.full((len(samples),), timestep, device=device, dtype=torch.long)
                row = point(model, batch, times, diffusion.q_sample(batch['x0'], times, noise), config, hinge, legality)
                row.update(model=label, step=spec['step'], bin=bin_name, t=timestep,
                    sample_ids=[e['sample_id'] for e in entries], noise_seeds=seeds,
                    example_only_low_noise_squared_gate=((999 - timestep) / 999)**2)
                records.append(row)
                write(output / 'progress.json', {'phase': 'parameter_gradient_probe', 'model': label, 'bin': bin_name,
                    't': timestep, 'completed_batches': len(records), 'total_batches': 30, 'updated_at_utc': now(), 'new_optimizer_updates': 0})
                print('Gradient probe', label, bin_name, timestep, len(records), '/30', flush=True)
        unchanged = before == model_digest(model) and all(p.grad is None for p in model.parameters())
        assert unchanged and sha(spec['checkpoint']) == spec['sha256']; state_checks[label] = unchanged
        del model, dataset, hinge, legality; gc.collect(); torch.cuda.empty_cache()
    assert len(records) == 30
    summaries = {}
    for label in sources:
        summaries[label] = {}
        for t in TIMESTEPS:
            part = [r for r in records if r['model'] == label and r['t'] == t]
            summary = {}
            for term in part[0]['parameter_gradients']:
                summary[term] = {}
                for field in part[0]['parameter_gradients'][term]:
                    values = [r['parameter_gradients'][term][field] for r in part if r['parameter_gradients'][term][field] is not None]
                    summary[term][field] = float(np.median(values)) if values else None
            summaries[label][str(t)] = summary
    for path, expected in read(ROOT / 'runs/m5_amplitude_regularization_20261104/protocol.json')['protected_sha256'].items(): assert sha(path) == expected
    result = {'schema': 'm5-training-noise-tail-parameter-gradient-diagnosis-v1', 'completed_at_utc': now(),
        'protocol_sha256': sha(output / 'protocol.json'), 'records': records, 'ground_truth_checks': ground_truth,
        'median_gradient_summaries_across_three_length_batches': summaries, 'model_and_checkpoint_unchanged': state_checks,
        'source_sha256': sha(Path(__file__)), 'prototype_sha256': sha(ROOT / 'training/m5_tail_prototypes.py'),
        'new_optimizer_updates': 0, 'new_DDIM_samples': 0, 'denoising_event_forwards': 180,
        'training_coefficients_selected': False, 'formal_coefficients_usable': False,
        'scope': '18frozen train events,6classes x3lengths,5fixed timesteps,two rawmodels. Parameter-gradient diagnosis,not fulltraining calibration,DDIM geometry,modelqualityorproofthatnewlosseshelp.',
        'gate_note': 'Squared low-noise factors are only annotated; no gate was applied and no lambda selected.',
        'current_formal_workers_or_sources_modified': False, 'registered_formal_update_budget_unchanged': 495000,
        'seed3102_quality_scores_read': False, 'final_audit_model_scores_read': False, 'M5_complete': False}
    write(output / 'report.json', result); write(output / 'progress.json', {'phase': 'complete', 'batches': 30,
          'updated_at_utc': now(), 'new_optimizer_updates': 0}); print({'saved': str(output / 'report.json'), 'new_optimizer_updates': 0})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output-dir', type=Path, required=True)
    main(p.parse_args().output_dir)
