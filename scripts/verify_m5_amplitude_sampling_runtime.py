#!/usr/bin/env python3
"""Verify sampling distribution, zero-change parity and actual Small full replay."""
import argparse
import ast
import gc
import hashlib
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from training.m3 import configure
from training.m5_constraints import PhysicalConstraintTrainer
from training.m5_amplitude_sampling import AmplitudeBalancedConstraintTrainer, MediumJumpResampler, VERSION
from training.m5_physical_data import PhysicalSubsetDataset
from scripts.train_m5_amplitude_sampling import CODE
from scripts.run_m5_length_research import read, write, sha, now

PARTS = ('model', 'ema', 'optimizer', 'rngs', 'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng')
COUNTERS = ('step', 'stage_updates', 'ema_updates', 'samples_seen', 'text_dropped')


def main(out):
    out = out.resolve(); out.relative_to(ROOT / 'runs')
    if (out / 'progress.json').exists() or (out / 'report.json').exists(): raise FileExistsError('Preserve prior attempts')
    out.mkdir(parents=True, exist_ok=True); updates = 0
    def progress(phase, **extra):
        write(out / 'progress.json', {'phase': phase, 'successful_optimizer_updates': updates,
                                     'updated_at_utc': now(), **extra}); print(phase, extra, flush=True)
    methods = ROOT / 'runs/m5_geometry_20261104/verification/verify_inheritance.py'
    assert sha(methods) == 'a271055eff7f072db234a756d6ad06f0da785219dd16061a9efb66650c7df792'
    node = next(n for n in ast.parse(methods.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'fingerprint')
    namespace = {'hashlib': hashlib, 'torch': torch, 'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(methods), 'exec'), namespace); fingerprint = namespace['fingerprint']
    study = ROOT / 'runs/m5_curriculum_factorial_20261104'; protocol = read(study / 'protocol.json')
    parent_path = Path(protocol['parent_checkpoint']); before_sha = sha(parent_path); assert before_sha == protocol['parent_sha256']
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    subset_path = ROOT / 'runs/m5_physical_20261103/data/D_mixed_small_3500.json'; manifest = read(subset_path)
    pool_path = ROOT / 'runs/m5_physical_20261103/data_candidates/medium_jump_sampling_20261104/pools.json'
    pool = read(pool_path)
    cfg = dict(read(study / 'configs/T2_cons_uniform.json'), stage='amplitude_sampling_engineering',
        engineering_fixture=True, cache_events=512, amplitude_pool_file=str(pool_path),
        amplitude_pool_sha256=sha(pool_path), medium_jump_amplitude_sampling_enabled=False,
        medium_jump_high_probability=.5, launch_code_sha256={p: sha(ROOT / p) for p in CODE})
    identity = dict(parent['identity'], subset_sha256=sha(subset_path), amplitude_pool_sha256=sha(pool_path),
                    code_sha256=cfg['launch_code_sha256'])
    dataset = PhysicalSubsetDataset(manifest['train'], cfg['input_dir'], cfg['stats_dir'], cache_size=512)
    active_cfg = dict(cfg, medium_jump_amplitude_sampling_enabled=True)
    def light(cls, config):
        # Exercise exactly the real draw_indices methods without constructing a network.
        value = cls.__new__(cls); value.config = config; value.samples = dataset; value.pools = {}
        value.batch_rng = torch.Generator().set_state(parent['rngs']['batch_rng'].clone())
        for i, row in enumerate(dataset.entries):
            bucket = 0 if row['num_frames'] <= 90 else 1 if row['num_frames'] <= 120 else 2
            value.pools.setdefault(bucket, {}).setdefault(row['group'], []).append(i)
        if cls is AmplitudeBalancedConstraintTrainer:
            value.amplitude_sampler = MediumJumpResampler(config, dataset.entries, identity, dataset.directory)
        return value
    control = light(PhysicalConstraintTrainer, cfg); disabled = light(AmplitudeBalancedConstraintTrainer, cfg)
    initial_amp_rng = disabled.amplitude_sampler.rng.get_state().clone()
    for _ in range(100):
        assert control.draw_indices() == disabled.draw_indices()
        assert torch.equal(control.batch_rng.get_state(), disabled.batch_rng.get_state())
    assert torch.equal(initial_amp_rng, disabled.amplitude_sampler.rng.get_state())
    control = light(PhysicalConstraintTrainer, cfg); balanced = light(AmplitudeBalancedConstraintTrainer, active_cfg)
    high = eligible = original_high = untouched = changed = 0
    for _ in range(1000):
        raw = control.draw_indices(); actual = balanced.draw_indices()
        assert torch.equal(control.batch_rng.get_state(), balanced.batch_rng.get_state())
        for before, after in zip(raw, actual):
            if before not in balanced.amplitude_sampler.eligible:
                assert before == after; untouched += 1
            else:
                eligible += 1; original_high += before in balanced.amplitude_sampler.high_set
                high += after in balanced.amplitude_sampler.high_set; changed += before != after
                assert after in balanced.amplitude_sampler.eligible
    assert eligible > 1000 and changed > 0 and abs(high / eligible - .5) < .05
    empirical = {'draws': 1000, 'total_event_draws': 64000, 'eligible_medium_jump_draws': eligible,
        'original_high_draws': original_high, 'balanced_high_draws': high, 'original_high_fraction': original_high / eligible,
        'balanced_high_fraction': high / eligible, 'unchanged_outside_eligible': untouched, 'reassigned_draws': changed,
        'base_batch_rng_states_identical': True, 'disabled_first100_draws_identical': True}
    progress('sampler_distribution_and_unrelated_draws_verified', **empirical)
    del control, disabled, balanced; gc.collect()
    def construct(cls, config):
        configure(config['seed']); trainer = cls(config, dataset, identity, config['stats_dir'])
        # Separate torch.load per branch avoids AdamW CPU step-tensor aliasing.
        trainer.load_state(torch.load(parent_path, map_location='cpu', weights_only=False), continuation=True)
        return trainer
    baseline = construct(PhysicalConstraintTrainer, cfg); old_metrics = baseline.update(); updates += 1; state = baseline.state()
    old_parts = {k: fingerprint(state[k]) for k in PARTS}; old_counts = {k: state[k] for k in COUNTERS}
    del baseline, state; gc.collect(); torch.cuda.empty_cache()
    zero = construct(AmplitudeBalancedConstraintTrainer, cfg); zero_metrics = zero.update(); updates += 1; state = zero.state()
    assert {k: fingerprint(state[k]) for k in PARTS} == old_parts and {k: state[k] for k in COUNTERS} == old_counts
    assert old_metrics == zero_metrics
    del zero, state; gc.collect(); torch.cuda.empty_cache()
    progress('disabled_actual_update_all_eight_parts_and_metrics_identical')
    active = construct(AmplitudeBalancedConstraintTrainer, active_cfg); ema_before = active.ema_updates
    actual_counts = {'eligible': 0, 'high': 0, 'changed': 0}; lengths = set()
    for _ in range(9):
        metrics = active.update(); updates += 1
        assert all(np.isfinite(metrics[k]) and metrics[k] > 0 for k in ('loss', 'root_grad_norm', 'body_grad_norm'))
        for key, value in active.amplitude_sampler.last_counts.items(): actual_counts[key] += value
        lengths.update(dataset.entries[i]['num_frames'] for i in active.last_indices)
        progress('active_small_update', step=active.step)
    assert active.step == 40009 and actual_counts['eligible'] > 0 and actual_counts['changed'] > 0
    active.save(out / 'before_ema_tick.pt'); metrics = active.update(); updates += 1
    assert active.step == 40010 and active.ema_updates == ema_before + 1
    expected = fingerprint(active.state()); indices = list(active.last_indices)
    parameters = sum(p.numel() for p in active.model.parameters())
    del active; gc.collect(); torch.cuda.empty_cache()
    resumed = AmplitudeBalancedConstraintTrainer(active_cfg, dataset, identity, active_cfg['stats_dir'])
    resumed.restore(out / 'before_ema_tick.pt'); replay_metrics = resumed.update(); updates += 1
    assert resumed.last_indices == indices and replay_metrics == metrics and fingerprint(resumed.state()) == expected
    progress('full_state_and_extra_mixture_rng_replay_identical')
    negatives = []
    def reject(label, fn):
        try: fn()
        except ValueError: negatives.append(label)
        else: raise AssertionError('Invalid sampler accepted: ' + label)
    source = torch.load(parent_path, map_location='cpu', weights_only=False)
    reject('old_parent_without_explicit_continuation', lambda: resumed.load_state(source))
    reject('unknown_operator', lambda: resumed.load_state(dict(resumed.state(), amplitude_sampler_operator='wrong')))
    altered = dict(resumed.state(), amplitude_sampler=dict(resumed.amplitude_sampler.state(), high_probability=.8))
    reject('changed_resume_probability', lambda: resumed.load_state(altered))
    altered = dict(resumed.state(), amplitude_sampler=dict(resumed.amplitude_sampler.state(), pool_sha256='0' * 64))
    reject('changed_resume_pool', lambda: resumed.load_state(altered))
    reject('wrong_pool_pin', lambda: MediumJumpResampler(dict(active_cfg, amplitude_pool_sha256='0' * 64), dataset.entries, identity, dataset.directory))
    reject('wrong_train_order', lambda: MediumJumpResampler(active_cfg, list(reversed(dataset.entries)), identity, dataset.directory))
    reject('wrong_subset_identity', lambda: MediumJumpResampler(active_cfg, dataset.entries, dict(identity, subset_sha256='wrong'), dataset.directory))
    reject('non_boolean_activation', lambda: MediumJumpResampler(dict(active_cfg, medium_jump_amplitude_sampling_enabled=1), dataset.entries, identity, dataset.directory))
    assert sha(parent_path) == before_sha
    for path, expected_sha in protocol['protected_sha256'].items(): assert sha(path) == expected_sha, path
    result = {'passed': True, 'verified_at_utc': now(), 'engineering_fixture': True, 'operator': VERSION,
        'parent_checkpoint_sha256': before_sha, 'parameters': parameters, 'batch_size': 64, 'microbatch': 64,
        'precision': 'BF16 forward/FP32 parameters and geometry', 'train_events': len(dataset),
        'train_pool_high': pool['high_count'], 'train_pool_low': pool['low_count'], 'empirical_distribution': empirical,
        'disabled_actual_update_all_eight_parts_and_metrics_identical': True,
        'actual_first9_update_mixture_counts': actual_counts, 'actual_sampled_length_range': [min(lengths), max(lengths)],
        'full_state_replay_start': 40009, 'full_state_replay_end': 40010, 'ema_tick_replayed': True,
        'extra_sampler_rng_and_complete_state_replay_identical': True, 'actual_small_engineering_updates': updates,
        'negative_cases_rejected': negatives, 'original_parent_and_formal_sources_unchanged': True,
        'source_sha256': sha(Path(__file__)), 'launch_source_sha256': cfg['launch_code_sha256'],
        'amplitude_pool_sha256': sha(pool_path), 'formal_amplitude_training_started': False,
        'registered_formal_update_budget_unchanged': 435000, 'quality_evidence': False,
        'seed3102_quality_scores_read': False, 'final_audit_model_scores_read': False, 'M5_complete': False}
    write(out / 'report.json', result); progress('complete'); print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir', type=Path, required=True)
    main(parser.parse_args().output_dir)
