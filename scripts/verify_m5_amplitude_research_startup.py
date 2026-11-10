#!/usr/bin/env python3
"""Verify actual formal initial checkpoints and the first100 draws of each runner."""
import argparse
import ast
import hashlib
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.run_m5_amplitude_research import validate, RUNNERS, PARTS
from scripts.run_m5_length_research import read, write, sha, now, process_live
from training.m5 import M5Trainer
from training.m5_amplitude_sampling import MediumJumpResampler


def prefix(path, count=100):
    rows = []
    if not path.exists(): return rows
    with path.open() as stream:
        for line in stream:
            try: rows.append(__import__('json').loads(line))
            except ValueError: break
            if len(rows) == count: break
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--wait', action='store_true'); args = parser.parse_args(); study = args.study_dir.resolve()
    output = study / 'verification/startup.json'
    if output.exists(): raise FileExistsError('Preserve formal startup proof')
    plan = validate(study); processes = read(study / 'processes.json')
    while True:
        pending = [n for n in RUNNERS if len(prefix(study / n / 'sample_trace.jsonl')) < 100]
        if not pending: break
        for name in pending:
            if not process_live(processes[name], study / name): raise RuntimeError('Worker handle lost before100draws: ' + name)
        write(study / 'verification/startup_wait.json', {'phase': 'waiting_for_100_real_updates',
              'pending': pending, 'verified_at_utc': now(), 'new_optimizer_updates': 0})
        if not args.wait: print({'pending': pending}); return
        time.sleep(10)
    methods = ROOT / 'runs/m5_geometry_20261104/verification/verify_inheritance.py'
    assert sha(methods) == 'a271055eff7f072db234a756d6ad06f0da785219dd16061a9efb66650c7df792'
    node = next(n for n in ast.parse(methods.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'fingerprint')
    namespace = {'hashlib': hashlib, 'np': np, 'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(methods), 'exec'), namespace); fingerprint = namespace['fingerprint']
    assert sha(plan['parent_checkpoint']) == plan['parent_sha256']
    parent = torch.load(plan['parent_checkpoint'], map_location='cpu', weights_only=False)
    expected = {k: fingerprint(parent[k]) for k in PARTS}; keys = ('step', 'ema_updates', 'samples_seen', 'text_dropped')
    counters = {k: parent[k] for k in keys}; first_ema_losses = []; records = {}
    class Sampler: pass
    for name in RUNNERS:
        run = study / name; cfg = read(run / 'config.json'); identity = read(run / 'run_config.json')['identity']
        manifest = read(run / 'subset.json'); lineage = read(run / 'lineage.json')
        assert cfg == read(study / 'configs' / (name + '.json')) and sha(run / 'subset.json') == cfg['subset_sha256']
        assert lineage['parent_sha256'] == plan['parent_sha256'] and lineage['optimizer_rng_preserved']
        state = torch.load(run / 'checkpoints/initial_state.pt', map_location='cpu', weights_only=False)
        assert {k: fingerprint(state[k]) for k in PARTS} == expected
        assert {k: state[k] for k in keys} == counters and state['stage_updates'] == 0
        assert state['optimizer_kind'] == 'adamw-fp32-v1' and state['identity'] == identity
        assert all(not v.is_floating_point() or v.dtype == torch.float32 for v in state['model'].values())
        sampler = Sampler(); sampler.config = cfg; sampler.samples = manifest['train']; sampler.pools = {}
        sampler.batch_rng = torch.Generator().set_state(parent['rngs']['batch_rng'].clone())
        for i, row in enumerate(manifest['train']):
            bucket = 0 if row['num_frames'] <= 90 else 1 if row['num_frames'] <= 120 else 2
            sampler.pools.setdefault(bucket, {}).setdefault(row['group'], []).append(i)
        amplitude = None
        if name == 'R3_jump_mix50':
            amplitude = MediumJumpResampler(cfg, manifest['train'], identity, cfg['input_dir'])
            amplitude.restore(state['amplitude_sampler']); assert state['amplitude_sampler_operator'] == 'm5-medium-jump-amplitude-resampling-v1'
        elif name == 'R1_dropout10':
            assert state['dropout_operator'] == 'm5-transformer-attention-feedforward-dropout-v1' and state['transformer_dropout_rate'] == .1
        else:
            assert state['cache_extension_operator'] == 'm5-append-only-frozen-stat-continuation-v1'
            assert state['cache_compatibility_proof_sha256'] == cfg['cache_compatibility_proof_sha256']
            assert identity['physical_input_fingerprint'] == read(Path(cfg['input_dir']) / 'READY')['fingerprint']
        expected_trace = []; selected_medium = high = 0
        for i in range(100):
            indices = M5Trainer.draw_indices(sampler)
            if amplitude:
                indices = amplitude.transform(indices); selected_medium += amplitude.last_counts['eligible']; high += amplitude.last_counts['high']
            expected_trace.append({'step': 40001 + i, 'indices': indices})
        assert prefix(run / 'sample_trace.jsonl') == expected_trace
        if name == 'R1_dropout10':
            assert expected_trace == prefix(Path(plan['reused_control']) / 'sample_trace.jsonl')
        metrics = prefix(run / 'metrics.jsonl', count=10000)
        last = next(r for r in reversed(metrics) if r['step'] > 40000)
        assert all(np.isfinite(last[k]) and last[k] > 0 for k in ('loss', 'root_grad_norm', 'body_grad_norm'))
        assert last['high_noise_replacement_fraction'] > 0 if cfg['high_noise_fraction'] else last['high_noise_replacement_fraction'] == 0
        denoise = read(run / 'evaluations/denoising_040000.json')['ema']; first_ema_losses.append(denoise['macro_loss'])
        records[name] = {'last_verified_update': last['step'], 'initial_checkpoint_sha256': sha(run / 'checkpoints/initial_state.pt'),
            'all_eight_parent_state_parts_identical': True, 'optimizer_rng_counters_preserved': True,
            'first100_actual_draws_match_own_full_sampler_replay': True, 'initial_ema_norm6': denoise['macro_loss'],
            'parameters': read(run / 'run_config.json')['parameters'], 'actual_training_events': len(manifest['train']),
            'first100_medium_jump_draws': selected_medium if amplitude else None,
            'first100_high_jump_draws': high if amplitude else None,
            'configured_dropout': cfg['network_dropout'], 'actual_checkpoint_operator': state.get('dropout_operator', state.get('cache_extension_operator', state.get('amplitude_sampler_operator'))),
            'positive_finite_root_body_gradients': True, 'last_learning_rate': last['learning_rate']}
        del state
        print('Verified actual startup', name, 'through', last['step'], flush=True)
    assert max(first_ema_losses) - min(first_ema_losses) < 1e-8
    baseline_initial = read(Path(plan['reused_control']) / 'evaluations/denoising_040000.json')['ema']['macro_loss']
    assert all(abs(value - baseline_initial) < 1e-8 for value in first_ema_losses)
    validate(study)
    result = {'passed': True, 'verified_at_utc': now(), 'protocol_sha256': sha(study / 'protocol.json'),
        'parent_checkpoint_sha256': plan['parent_sha256'], 'parent_parts_fingerprints': expected,
        'records': records, 'all_initial_ema_denoising_matches_reused_control': True,
        'new_optimizer_updates_by_verifier': 0, 'source_sha256': sha(Path(__file__)),
        'seed3102_quality_scores_read': False, 'final_audit_model_scores_read': False,
        'quality_acceptance': False, 'M5_complete': False}
    write(output, result); print({'passed': True, 'records': list(records)}, flush=True)


if __name__ == '__main__': main()
