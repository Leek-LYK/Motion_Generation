#!/usr/bin/env python3
"""Oracle intervention on internal Root->Body conditioning; no training/generation.

The normal branch is checked against the unchanged released forward function.
The oracle branch substitutes ground-truth local root only at the body input.
It is a denoising diagnostic, not an externally controllable generation mode.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from kimodo.model.diffusion import Diffusion
from training.m3 import configure, make_model, batch_to, forward
from training.m5_physical_data import PhysicalSubsetDataset
from training.m5_physical_losses import geometry
from training.m5_evaluation import amplitude, MINIMUM
from scripts.run_m5_length_research import read, write, sha, now

TIMESTEPS = (0, 249, 499, 749, 999)
GROUPS = ('walk', 'jog', 'jump', 'raise_arms', 'turn', 'squat')


def average(rows):
    keys = ('body_norm4', 'joint_pos_norm', 'joint_rot_norm', 'fk_gt_norm', 'root_global_norm',
            'local_root_velocity_norm', 'local_root_height_norm', 'amplitude_ratio', 'root_height_range_ratio')
    return {key: float(np.mean([r[key] for r in rows if r.get(key) is not None]))
            if any(r.get(key) is not None for r in rows) else None for key in keys}


def main():
    output = ROOT / 'runs/m5_length_research_20261104/diagnostics/root_conditioning_050000.json'
    if output.exists():
        raise FileExistsError('Preserve completed oracle diagnostic')
    configure(3101); device = torch.device('cuda:0'); diffusion = Diffusion(1000).to(device)
    runs = {'short': ROOT / 'runs/m5_geometry_20261104/G1_consistency',
            'mixed': ROOT / 'runs/m5_length_research_20261104/L1_mixed_G1'}
    all_records = []; evidence = {}; parity = {}
    for label, run in runs.items():
        checkpoint = run / 'checkpoints/step_050000.pt'; checksum = sha(checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False); cfg = read(run / 'config.json')
        assert state['step'] == 50000 and cfg == state['config'] and cfg['seed'] == 3101
        for relative, expected in cfg['launch_code_sha256'].items():
            assert sha(ROOT / relative) == expected
        for relative, expected in state['identity']['stats_sha256'].items():
            assert sha(run / relative) == expected
        model = make_model(cfg['model'], run / 'stats/motion').to(device).eval(); model.load_state_dict(state['ema'])
        identity = state['identity']; del state
        manifest = read(run / 'subset.json'); entries = manifest['monitor']; assert len(entries) == 91
        dataset = PhysicalSubsetDataset(entries, cfg['input_dir'], run / 'stats/motion', cache_size=91)
        evidence[label] = {'run': str(run), 'checkpoint_sha256': checksum, 'checkpoint_step': 50000,
                           'identity': identity, 'monitor_sample_ids': [e['sample_id'] for e in entries]}
        rep = model.motion_rep; scale = torch.sqrt(rep.stats.std.to(device)[5:107]**2 + rep.stats.eps).reshape(34, 3)
        maximum_parity_error = 0.; model_records = []
        with torch.inference_mode():
            for t_value in TIMESTEPS:
                for start in range(0, len(dataset), 16):
                    samples = dataset.batch(range(start, min(start + 16, len(dataset))))
                    batch = batch_to(samples, device); target = batch['x0']; mask = batch['x_pad_mask']; lengths = mask.sum(-1)
                    eps = torch.zeros_like(target)
                    for i, event in enumerate(samples):
                        key = int(hashlib.sha256(f"root-conditioning-v1:{event['sample_id']}:{t_value}".encode()).hexdigest()[:15], 16)
                        rng = torch.Generator(device=device).manual_seed(key)
                        eps[i, :event['length']] = torch.randn((event['length'], 417), generator=rng, device=device)
                    times = torch.full((len(samples),), t_value, dtype=torch.long, device=device)
                    noisy = diffusion.q_sample(target, times, eps); zero_mask = torch.zeros_like(noisy)
                    args = (mask, batch['text_feat'], batch['text_feat_pad_mask'], times, batch['first_heading_angle'])
                    root_prediction = model.root_model(torch.cat((noisy, zero_mask), -1), *args)
                    local_prediction = rep.global_root_to_local_root(root_prediction, normalized=True, lengths=lengths)
                    local_truth = rep.global_root_to_local_root(target[..., :5], normalized=True, lengths=lengths)
                    bodies = {}
                    for branch, local in (('predicted_root', local_prediction), ('oracle_root', local_truth)):
                        body_input = torch.cat((local, noisy[..., rep.body_slice], zero_mask), -1)
                        bodies[branch] = model.body_model(body_input, *args)
                    normal = torch.cat((root_prediction, bodies['predicted_root']), -1)
                    if start == 0:
                        stock = forward(model, noisy, batch, times)
                        difference = float(abs(stock - normal).max()); maximum_parity_error = max(maximum_parity_error, difference)
                        assert difference <= 1e-6
                    truth_raw, _, truth_motion, _ = geometry(rep, target)
                    root_raw = rep.global_root_stats.unnormalize(root_prediction)
                    root_errors = F.smooth_l1_loss(root_prediction, target[..., :5], reduction='none').mean(-1)
                    velocity_errors = F.smooth_l1_loss(local_prediction[..., :3], local_truth[..., :3], reduction='none').mean(-1)
                    height_errors = F.smooth_l1_loss(local_prediction[..., 3], local_truth[..., 3], reduction='none')
                    for branch, body in bodies.items():
                        prediction = torch.cat((root_prediction, body), -1)
                        _, fk, motion, _ = geometry(rep, prediction)
                        target_position = truth_raw[..., 5:107].reshape(*target.shape[:2], 34, 3)
                        fk_errors = F.smooth_l1_loss(fk / scale, target_position / scale, reduction='none').mean((-1, -2))
                        component = {}
                        for key, low, high in (('joint_pos_norm', 5, 107), ('joint_rot_norm', 107, 311),
                                               ('velocity_norm', 311, 413), ('contacts_norm', 413, 417)):
                            component[key] = F.smooth_l1_loss(prediction[..., low:high], target[..., low:high], reduction='none').mean(-1)
                        for i, event in enumerate(samples):
                            n = event['length']; group = entries[start + i]['group']
                            pose = {k: v[i, :n] for k, v in motion.items()}; truth_pose = {k: v[i, :n] for k, v in truth_motion.items()}
                            reference_amplitude = amplitude(truth_pose, rep.skeleton, group)
                            predicted_amplitude = amplitude(pose, rep.skeleton, group)
                            original_height = truth_raw[i, :n, 1].cpu().numpy(); predicted_height = root_raw[i, :n, 1].cpu().numpy()
                            height_range = np.quantile(original_height, .9) - np.quantile(original_height, .1)
                            item = {'model': label, 'sample_id': event['sample_id'], 'group': group,
                                'length_stratum': 'short' if n <= 90 else 'extended', 'frames': n, 't': t_value, 'branch': branch,
                                'root_global_norm': float(root_errors[i, :n].mean()),
                                'local_root_velocity_norm': float(velocity_errors[i, :n].mean()),
                                'local_root_height_norm': float(height_errors[i, :n].mean()),
                                'fk_gt_norm': float(fk_errors[i, :n].mean()),
                                'amplitude_ratio': predicted_amplitude / reference_amplitude if reference_amplitude >= MINIMUM[group] else None,
                                'root_height_range_ratio': float((np.quantile(predicted_height, .9) - np.quantile(predicted_height, .1)) / height_range) if height_range >= .02 else None,
                                **{key: float(value[i, :n].mean()) for key, value in component.items()}}
                            item['body_norm4'] = float(np.mean([item[k] for k in component])); model_records.append(item)
                print('Root oracle diagnosis', label, 't', t_value, 'complete', flush=True)
        assert len(model_records) == 91 * 5 * 2 and sha(checkpoint) == checksum
        all_records.extend(model_records); parity[label] = maximum_parity_error
        del model; torch.cuda.empty_cache()
    assert evidence['short']['monitor_sample_ids'] == evidence['mixed']['monitor_sample_ids']
    summaries = {}
    for model in runs:
        summaries[model] = {}
        for stratum in ('short', 'extended'):
            summaries[model][stratum] = {str(t): {branch: {
                group: average([r for r in all_records if r['model'] == model and r['length_stratum'] == stratum and
                                r['t'] == t and r['branch'] == branch and r['group'] == group])
                for group in GROUPS} for branch in ('predicted_root', 'oracle_root')} for t in TIMESTEPS}
    write(output, {'schema': 'm5-root-conditioning-oracle-v1', 'completed_at_utc': now(), 'evidence': evidence,
        'normal_forward_parity_max_abs': parity, 'records': all_records, 'summaries': summaries,
        'timesteps': list(TIMESTEPS), 'monitor_events': 91, 'branches': ['predicted_root', 'oracle_root'],
        'noise_policy': 'Per-event/timestep fixed noise shared across models and oracle intervention; conditional single-forward denoising, no CFG or DDIM generation.',
        'intervention': 'Only the body-stage local-root conditioning is replaced by ground truth. The returned global root block remains the same predicted block in both branches.',
        'scope': 'Internal oracle denoising diagnosis, not free generation, deployed conditioning, a quality gate, or proof that teacher-forcing training improves the model. Oracle inputs may be outside the body model training distribution.',
        'source_sha256': sha(Path(__file__)), 'new_optimizer_updates': 0, 'new_diffusion_samples': 0,
        'model_or_input_files_modified': False, 'seed3102_quality_scores_read': False,
        'final_audit_model_scores_read': False, 'M5_complete': False})
    print('Saved root conditioning oracle evidence:', output, flush=True)


if __name__ == '__main__':
    main()
