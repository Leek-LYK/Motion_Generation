#!/usr/bin/env python3
"""Register the required second Small seed's independent, fixed 0->40k prefix."""
import argparse
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m5_length_research import read, write, sha, now, process_live

BASE = ROOT / 'runs/m5_physical_20261103'
NAME = 'S1_small_raw7_seed3102'
REGISTRATION = BASE / 'auxiliary_protocols/independent_small_seed3102_prefix'
RUN = BASE / NAME
CONFIG = BASE / 'configs' / (NAME + '.json')


def validate():
    plan = read(REGISTRATION / 'protocol.json')
    for path, expected in plan['protected_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Frozen independent-seed evidence/source changed: ' + path)
    cfg = read(CONFIG)
    source = read(BASE / 'C_small_raw7/config.json')
    changes = {k for k in set(cfg) | set(source) if cfg.get(k) != source.get(k)}
    if changes != {'seed', 'stage'} or cfg['seed'] != 3102 or cfg['max_steps'] != 40000 or cfg['stage_start_step'] != 0:
        raise ValueError('Independent prefix differs outside stage name and training seed')
    return plan


def prepare():
    if REGISTRATION.exists() or RUN.exists() or CONFIG.exists():
        raise FileExistsError('Preserve existing registration/run/configuration')
    source_config = BASE / 'C_small_raw7/config.json'
    source = read(source_config)
    decision = read(BASE / 'diagnostics/optimizer_040000_decision.json')
    if decision['next_default_optimizer'] != 'adamw-fp32-v1':
        raise ValueError('The selected foundation optimizer changed')
    if source['loss_objective'] != 'weighted_raw7' or source['seed'] != 3101:
        raise ValueError('Unexpected first-seed foundation')
    assert source['stage_start_step'] == 0 and source['minimum_steps'] == source['max_steps'] == 40000
    assert all(source.get(k, 0.) == 0. for k in ('axis_weight', 'consistency_weight', 'ground_weight', 'legality_weight'))
    for relative, expected in source['launch_code_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Frozen worker source changed: ' + relative)
    subset = BASE / 'data/D_short_3500.json'
    assert sha(subset) == source['subset_sha256']
    view = read(source['training_view'])
    assert view['passed'] and read(Path(source['input_dir']) / 'READY')['fingerprint'] == view['input_fingerprint']
    for path, expected in view['stats_sha256'].items():
        assert sha(BASE / path) == expected
    assert len(read(subset)['train']) == 3404
    cfg = dict(source, stage=NAME, seed=3102)
    write(CONFIG, cfg)
    protected = [ROOT / p for p in source['launch_code_sha256']]
    protected += [Path(__file__).resolve(), ROOT / 'scripts/run_m5_length_research.py', CONFIG, subset,
                  source_config, Path(source['training_view']), Path(source['input_dir']) / 'READY',
                  BASE / 'diagnostics/optimizer_040000_decision.json', BASE / 'protocol/M5_acceptance_protocol_v2.json',
                  BASE / 'protocol/launch.json', BASE / 'C_small_raw7/checkpoints/initial_state.pt',
                  ROOT / 'runs/m5_length_research_20261104/protocol.json']
    protected += [BASE / p for p in view['stats_sha256']]
    plan = {'schema': 'm5-independent-small-seed-prefix-v1', 'frozen_at_utc': now(), 'run': NAME,
            'seed': 3102, 'source_run': 'C_small_raw7', 'source_seed': 3101,
            'source_initial_checkpoint': str(BASE / 'C_small_raw7/checkpoints/initial_state.pt'),
            'source_initial_model_sha256': read(BASE / 'C_small_raw7/run_config.json')['initial_model_sha256'],
            'configuration': str(CONFIG), 'configuration_sha256': sha(CONFIG), 'subset': str(subset),
            'subset_sha256': sha(subset), 'stage_start_step': 0, 'max_updates': 40000,
            'initialization': 'fresh random initialization; no parent or initial-weight argument',
            'configuration_changes_vs_seed3101': {'seed': [3101, 3102], 'stage': ['C_small_raw7', NAME]},
            'dependency': 'The 0->40k foundation is independent of the ongoing 40->60k length comparison. Post-40k data/constraint course remains dependent and is not launched here.',
            'prefix_reuse_condition': 'Count toward the final required seed3102 repetition only if the selected 0->40k data, statistics, optimizer, objective, noise and augmentation foundation remains exactly matched. Complete 40->60->80 remains required.',
            'seed_selection_policy': 'Seed3102 is fixed before outcomes. Do not replace it with a more favourable seed after evaluation.',
            'registered_physical_update_budget_before_this_prefix': 335000,
            'registered_total_physical_updates': 375000,
            'budget_scope': '40k separately registered required-repetition prefix; no extra variant or automatic post-40k continuation.',
            'protected_sha256': {str(path): sha(path) for path in protected},
            'complete_80k_repetition_proven': False, 'automatic_post40k_continuation': False,
            'final_audit_model_scores_read': False, 'M5_complete': False}
    write(REGISTRATION / 'protocol.json', plan)
    for relative in (*source['launch_code_sha256'], 'scripts/start_m5_independent_seed.py'):
        dest = REGISTRATION / 'source_at_launch' / relative
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / relative, dest)
    validate()
    write(REGISTRATION / 'preflight.json', {'passed': True, 'verified_at_utc': now(),
          'protocol_sha256': sha(REGISTRATION / 'protocol.json'), 'only_seed_and_stage_differ': True,
          'source_data_statistics_and_worker_hashes_verified': True, 'new_optimizer_updates': 0,
          'complete_80k_repetition_proven': False, 'M5_complete': False})
    print('Independent seed3102 prefix registered, fixed 40k budget.', flush=True)


def launch():
    lock = (REGISTRATION / 'launch.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = validate()
    if (REGISTRATION / 'process.json').exists() or RUN.exists():
        raise FileExistsError('One-shot launcher: inspect the existing handle; do not restart')
    free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()[0])
    if free < 24576:
        raise RuntimeError('Insufficient measured GPU memory')
    command = [str(ROOT / '.venv/bin/python'), '-B', '-u', str(ROOT / 'scripts/train_m5_physical.py'),
               '--config', str(CONFIG), '--subset', plan['subset'], '--run-dir', str(RUN)]
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (REGISTRATION / 'training.log').open('ab', buffering=0) as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    ticks = Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    write(REGISTRATION / 'process.json', {'pid': proc.pid, 'start_ticks': ticks, 'command': command,
          'run_dir': str(RUN), 'started_at_utc': now(), 'protocol_sha256': sha(REGISTRATION / 'protocol.json')})
    print('Started seed3102 prefix, PID', proc.pid, flush=True)


def verify_startup():
    import torch
    from training.m3 import configure, make_model
    from training.m5 import M5Trainer, RNGS
    from scripts.train_m5_physical import model_digest
    plan = validate(); record = read(REGISTRATION / 'process.json')
    assert process_live(record, RUN) or (RUN / 'COMPLETED.json').exists()
    cfg = read(RUN / 'config.json'); assert cfg == read(CONFIG)
    registered = read(RUN / 'run_config.json'); lineage = read(RUN / 'lineage.json')
    assert lineage['initialization'] == registered['initialization'] == 'random'
    assert lineage['parent_checkpoint'] is None and lineage['start_step'] == 0
    state = torch.load(RUN / 'checkpoints/initial_state.pt', map_location='cpu', weights_only=False)
    assert state['schema'] == 'kimodo-m5-physical-checkpoint-v2' and state['config'] == cfg
    assert all(state[k] == 0 for k in ('step', 'stage_updates', 'ema_updates', 'samples_seen', 'text_dropped'))
    assert not state['optimizer']['state']
    configure(3102); replay = make_model(cfg['model'], RUN / 'stats/motion')
    reconstructed = model_digest(replay)
    assert reconstructed == registered['initial_model_sha256'] != plan['source_initial_model_sha256']
    assert all(torch.equal(value, state['model'][k]) and torch.equal(value, state['ema'][k]) for k, value in replay.state_dict().items())
    del replay
    other = torch.load(plan['source_initial_checkpoint'], map_location='cpu', weights_only=False)
    assert other['step'] == 0 and other['config']['seed'] == 3101
    rng_seeds = {}
    for i, name in enumerate(RNGS, 1):
        expected = torch.Generator(device='cpu' if name == 'batch_rng' else 'cuda:0').manual_seed(3102 + i).get_state().cpu()
        assert torch.equal(expected, state['rngs'][name].cpu())
        assert not torch.equal(state['rngs'][name].cpu(), other['rngs'][name].cpu())
        rng_seeds[name] = 3102 + i
    subset = read(plan['subset'])
    class Sampler:
        pass
    sampler = Sampler(); sampler.config = cfg; sampler.samples = subset['train']; sampler.pools = {}
    sampler.batch_rng = torch.Generator().set_state(state['rngs']['batch_rng'])
    for i, row in enumerate(subset['train']):
        bucket = 0 if row['num_frames'] <= 90 else 1 if row['num_frames'] <= 120 else 2
        sampler.pools.setdefault(bucket, {}).setdefault(row['group'], []).append(i)
    expected = [{'step': i + 1, 'indices': M5Trainer.draw_indices(sampler)} for i in range(100)]
    with (RUN / 'sample_trace.jsonl').open() as stream:
        actual = [json.loads(next(stream)) for _ in range(100)]
    assert actual == expected
    with (BASE / 'C_small_raw7/sample_trace.jsonl').open() as stream:
        reference = [json.loads(next(stream)) for _ in range(100)]
    assert actual != reference
    rows = (RUN / 'metrics.jsonl').read_text().splitlines(keepends=True)
    last = [json.loads(row) for row in rows if row.endswith('\n')][-1]
    assert last['step'] >= 100 and all(math.isfinite(last[k]) and last[k] > 0 for k in ('loss', 'root_grad_norm', 'body_grad_norm'))
    assert registered['identity']['subset_sha256'] == plan['subset_sha256']
    assert registered['identity']['code_sha256'] == cfg['launch_code_sha256']
    write(REGISTRATION / 'startup_verification.json', {'passed': True, 'verified_at_utc': now(),
          'protocol_sha256': sha(REGISTRATION / 'protocol.json'), 'initial_checkpoint_sha256': sha(RUN / 'checkpoints/initial_state.pt'),
          'initial_model_sha256': reconstructed, 'initial_model_exactly_reconstructed_from_seed3102': True,
          'initial_model_differs_from_seed3101': True, 'empty_initial_optimizer_and_zero_global_counters': True,
          'initial_model_and_ema_equal': True, 'dedicated_rng_seeds': rng_seeds,
          'first_100_draws_match_own_initial_rng': True, 'first_100_draws_differ_from_seed3101': True,
          'verified_step': last['step'], 'new_optimizer_updates_by_verifier': 0,
          'complete_80k_repetition_proven': False, 'M5_complete': False})
    print('Verified genuine random seed3102 initialization and first 100 draws through step', last['step'], flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('prepare', 'launch', 'verify-startup')); a = p.parse_args()
    {'prepare': prepare, 'launch': launch, 'verify-startup': verify_startup}[a.action]()


if __name__ == '__main__':
    main()
