#!/usr/bin/env python3
"""Actual Small update parity, appended-data loading and EMA-boundary full replay."""
import argparse
import ast
import copy
import gc
import hashlib
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from training.m3 import configure
from training.m5_constraints import PhysicalConstraintTrainer
from training.m5_cache_extension import AppendCompatibleConstraintTrainer, validate_cache_compatibility, VERSION
from training.m5_physical_data import PhysicalSubsetDataset
from scripts.train_m5_cache_extension import CODE
from scripts.run_m5_length_research import read, write, sha, now

PARTS = ('model', 'ema', 'optimizer', 'rngs', 'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng')
COUNTERS = ('step', 'stage_updates', 'ema_updates', 'samples_seen', 'text_dropped')


def main(out):
    out = out.resolve(); out.relative_to(ROOT / 'runs')
    if (out / 'report.json').exists() or (out / 'progress.json').exists():
        raise FileExistsError('Preserve engineering attempts; choose a new output directory')
    out.mkdir(parents=True, exist_ok=True); updates = 0
    def progress(phase, **extra):
        write(out / 'progress.json', {'phase': phase, 'successful_optimizer_updates': updates,
              'updated_at_utc': now(), **extra})
        print(phase, extra, flush=True)
    methods = ROOT / 'runs/m5_geometry_20261104/verification/verify_inheritance.py'
    assert sha(methods) == 'a271055eff7f072db234a756d6ad06f0da785219dd16061a9efb66650c7df792'
    node = next(n for n in ast.parse(methods.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'fingerprint')
    namespace = {'hashlib': hashlib, 'np': np, 'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(methods), 'exec'), namespace)
    fingerprint = namespace['fingerprint']
    study = ROOT / 'runs/m5_curriculum_factorial_20261104'; protocol = read(study / 'protocol.json')
    parent_path = Path(protocol['parent_checkpoint']); before_sha = sha(parent_path)
    assert before_sha == protocol['parent_sha256']
    extension = ROOT / 'runs/m5_amplitude_data_20261104'; proof_path = extension / 'cache_compatibility_proof.json'
    proof = read(proof_path); parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    old_manifest = read(ROOT / 'runs/m5_geometry_20261104/verification/long_sequence_contract/subset.json')
    assert len(old_manifest['train']) == 12
    cfg = dict(read(study / 'configs/T2_cons_uniform.json'), stage='cache_extension_engineering',
               engineering_fixture=True, cache_events=64)
    code = {p: sha(ROOT / p) for p in CODE}
    old_identity = dict(parent['identity'], code_sha256=code, subset_sha256=fingerprint(old_manifest))
    new_identity = dict(old_identity, physical_input_fingerprint=proof['target_cache_fingerprint'],
        training_view_fingerprint=proof['target_training_view_fingerprint'],
        cache_compatibility_proof_sha256=sha(proof_path))
    new_cfg = dict(cfg, input_dir=proof['target_cache'], stats_dir=proof['target_model_stats_dir'],
        training_view=proof['target_training_view'], cache_compatibility_proof=str(proof_path),
        cache_compatibility_proof_sha256=sha(proof_path), launch_code_sha256=code)
    old_dataset = PhysicalSubsetDataset(old_manifest['train'], cfg['input_dir'], cfg['stats_dir'], cache_size=12)
    new_old_dataset = PhysicalSubsetDataset(old_manifest['train'], new_cfg['input_dir'], new_cfg['stats_dir'], cache_size=12)
    assert fingerprint(old_dataset.batch(range(12))) == fingerprint(new_old_dataset.batch(range(12)))
    progress('old_sample_values_byte_identical')
    def construct(cls, config, dataset, identity):
        configure(config['seed']); trainer = cls(config, dataset, identity, config['stats_dir'])
        # Each branch owns a separate source deserialization (CPU AdamW step aliases).
        independent = torch.load(parent_path, map_location='cpu', weights_only=False)
        trainer.load_state(independent, continuation=True)
        return trainer
    baseline = construct(PhysicalConstraintTrainer, cfg, old_dataset, old_identity)
    old_metrics = baseline.update(); updates += 1; state = baseline.state()
    old_parts = {k: fingerprint(state[k]) for k in PARTS}; old_counters = {k: state[k] for k in COUNTERS}
    del baseline, state; gc.collect(); torch.cuda.empty_cache()
    versioned = construct(AppendCompatibleConstraintTrainer, new_cfg, new_old_dataset, new_identity)
    new_metrics = versioned.update(); updates += 1; state = versioned.state()
    assert {k: fingerprint(state[k]) for k in PARTS} == old_parts
    assert {k: state[k] for k in COUNTERS} == old_counters and new_metrics == old_metrics
    assert state['identity']['physical_input_fingerprint'] == proof['target_cache_fingerprint']
    assert state['cache_extension_operator'] == VERSION
    del versioned, state; gc.collect(); torch.cuda.empty_cache()
    progress('old_data_actual_update_all_eight_state_parts_and_metrics_identical')
    additions = read(ROOT / 'runs/m5_physical_20261103/data_candidates/medium_high_jump_20261104/qualified_events.json')
    by_id = {r['sample_id']: r for r in additions}
    def entry(event):
        return {k: event[k] for k in ('sample_id', 'text_id', 'text', 'split', 'motion_id', 'num_frames')} | {'group': 'jump'}
    all_new = PhysicalSubsetDataset([entry(r) for r in additions], new_cfg['input_dir'], new_cfg['stats_dir'], cache_size=35)
    for i, event in enumerate(additions):
        value = all_new[i]; assert torch.isfinite(value['x0']).all() and value['length'] == event['num_frames']
        raw = np.load(ROOT / 'runs/m5_physical_20261103/data_candidates/medium_high_jump_20261104/events' / event['sample_id'] / 'raw_features.npy')
        assert np.array_equal(value['x0'].numpy(), (raw - all_new.mean) / all_new.scale)
    plus = read(extension / 'data/mixed_small_plus19.json'); old_ids = {r['sample_id'] for r in read(extension / 'data/mixed_small_control.json')['train']}
    new_entries = [r for r in plus['train'] if r['sample_id'] not in old_ids]; assert len(new_entries) == 19
    mixed_entries = old_manifest['train'] + new_entries
    mixed = PhysicalSubsetDataset(mixed_entries, new_cfg['input_dir'], new_cfg['stats_dir'], cache_size=len(mixed_entries))
    fixture = dict(old_manifest, train=mixed_entries, physical_cache_fingerprint=proof['target_cache_fingerprint'])
    write(out / 'subset.json', fixture)
    mixed_identity = dict(new_identity, subset_sha256=sha(out / 'subset.json'))
    active = construct(AppendCompatibleConstraintTrainer, new_cfg, mixed, mixed_identity)
    initial_ema = active.ema_updates; sampled_new = 0; sampled_lengths = set()
    for _ in range(9):
        metrics = active.update(); updates += 1
        assert all(np.isfinite(metrics[k]) and metrics[k] > 0 for k in ('loss', 'root_grad_norm', 'body_grad_norm'))
        sampled_new += sum(active.samples.entries[i]['sample_id'] in by_id for i in active.last_indices)
        sampled_lengths.update(active.samples.entries[i]['num_frames'] for i in active.last_indices)
        progress('appended_data_real_update', step=active.step)
    assert active.step == 40009 and sampled_new > 0 and max(sampled_lengths) > 120
    active.save(out / 'before_ema_tick.pt')
    metrics = active.update(); updates += 1; assert active.step == 40010 and active.ema_updates == initial_ema + 1
    expected_digest = fingerprint(active.state()); expected_indices = list(active.last_indices)
    parameters = sum(p.numel() for p in active.model.parameters())
    del active; gc.collect(); torch.cuda.empty_cache()
    resumed = AppendCompatibleConstraintTrainer(new_cfg, mixed, mixed_identity, new_cfg['stats_dir'])
    resumed.restore(out / 'before_ema_tick.pt'); replay_metrics = resumed.update(); updates += 1
    assert resumed.last_indices == expected_indices and fingerprint(resumed.state()) == expected_digest and replay_metrics == metrics
    progress('full_state_ema_boundary_replay_byte_identical')
    negatives = []
    def rejected(label, fn):
        try: fn()
        except ValueError: negatives.append(label)
        else: raise AssertionError('Invalid compatibility accepted: ' + label)
    fresh_source = torch.load(parent_path, map_location='cpu', weights_only=False)
    rejected('old_cache_requires_explicit_continuation', lambda: resumed.load_state(fresh_source))
    bad = dict(fresh_source, identity=dict(fresh_source['identity'], stats_sha256={}))
    rejected('source_model_statistics_changed', lambda: resumed.load_state(bad, continuation=True))
    bad = dict(fresh_source, identity=dict(fresh_source['identity'], training_view_fingerprint='wrong-source-view'))
    rejected('source_view_not_proven_parent', lambda: resumed.load_state(bad, continuation=True))
    for label, changes in (('unknown_operator', {'cache_extension_operator': 'unsupported'}),
                            ('resume_proof_hash_changed', {'cache_compatibility_proof_sha256': '0' * 64}),
                            ('resume_origin_changed', {'cache_extension_origin_identity': {}})):
        bad = dict(resumed.state(), **changes)
        rejected(label, lambda bad=bad: resumed.load_state(bad))
    rejected('wrong_proof_pin', lambda: validate_cache_compatibility(dict(new_cfg, cache_compatibility_proof_sha256='0' * 64), mixed_identity, mixed.directory))
    rejected('target_stat_identity_changed', lambda: validate_cache_compatibility(new_cfg, dict(mixed_identity, stats_sha256={}), mixed.directory))
    rejected('wrong_target_cache_directory', lambda: validate_cache_compatibility(new_cfg, mixed_identity, old_dataset.directory))
    bound_file = out / 'invalid_bound_file.txt'; bound_file.write_text('original')
    bad_proof = copy.deepcopy(proof); bad_proof['files_sha256'][str(bound_file)] = sha(bound_file)
    bad_path = out / 'invalid_bound_file_proof.json'; write(bad_path, bad_proof); bound_file.write_text('tampered')
    bad_cfg = dict(new_cfg, cache_compatibility_proof=str(bad_path), cache_compatibility_proof_sha256=sha(bad_path))
    rejected('actual_bound_file_tampering', lambda: validate_cache_compatibility(bad_cfg, mixed_identity, mixed.directory))
    assert sha(parent_path) == before_sha
    for path, expected in protocol['protected_sha256'].items(): assert sha(path) == expected, path
    result = {'passed': True, 'engineering_fixture': True, 'verified_at_utc': now(), 'operator': VERSION,
        'parent_checkpoint_sha256': before_sha, 'parameters': parameters, 'batch_size': 64, 'microbatch': 64,
        'precision': 'BF16 forward / FP32 parameters and geometry', 'old_sample_values_identical': True,
        'old_data_actual_update_all_eight_state_parts_identical': True, 'old_data_metrics_identical': True,
        'independent_source_load_per_branch': True, 'all_appended_events_loaded': len(additions),
        'mixed_fixture_old_events': 12, 'mixed_fixture_new_events': 19,
        'sampled_appended_event_exposures_in_first9_updates': sampled_new,
        'sampled_length_range': [min(sampled_lengths), max(sampled_lengths)],
        'full_state_replay_start': 40009, 'full_state_replay_end': 40010,
        'full_state_replay_byte_identical': True, 'ema_tick_replayed': True,
        'actual_small_engineering_updates': updates, 'negative_cases_rejected': negatives,
        'saved_new_cache_identity_truthful': True, 'current_formal_workers_and_parents_unchanged': True,
        'source_sha256': sha(Path(__file__)), 'launch_source_sha256': code,
        'cache_compatibility_proof_sha256': sha(proof_path), 'formal_data_extension_training_started': False,
        'quality_evidence': False, 'seed3102_quality_scores_read': False, 'final_audit_model_scores_read': False,
        'registered_formal_update_budget_unchanged': 435000, 'M5_complete': False}
    write(out / 'report.json', result); progress('complete'); print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir', type=Path, required=True)
    main(parser.parse_args().output_dir)
