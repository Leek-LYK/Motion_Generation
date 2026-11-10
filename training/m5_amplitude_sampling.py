"""Versioned within-medium-jump resampling; unrelated base draws stay unchanged."""
import hashlib
import json
import math
from pathlib import Path
import torch
from training.m5_constraints import PhysicalConstraintTrainer

VERSION = 'm5-medium-jump-amplitude-resampling-v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''): digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def train_entries_fingerprint(entries):
    fields = ('sample_id', 'text_id', 'text', 'motion_id', 'split', 'num_frames', 'group')
    return fingerprint([{k: row[k] for k in fields} for row in entries])


class MediumJumpResampler:
    def __init__(self, config, entries, identity, input_dir):
        path = Path(config['amplitude_pool_file']); expected = config['amplitude_pool_sha256']
        if sha(path) != expected: raise ValueError('Amplitude pool SHA changed')
        pool = json.loads(path.read_text()); descriptor = pool['descriptor']
        if descriptor.get('schema') != 'm5-medium-jump-amplitude-pools-v1' or fingerprint(descriptor) != pool['fingerprint']:
            raise ValueError('Unsupported or inconsistent amplitude pools')
        for file, digest in pool['files_sha256'].items():
            if sha(file) != digest: raise ValueError('Bound train-pool input changed: ' + file)
        if descriptor['training_entries_fingerprint'] != train_entries_fingerprint(entries): raise ValueError('Ordered train entries differ from the amplitude pools')
        for key in ('physical_input_fingerprint', 'training_view_fingerprint'):
            if identity[key] != descriptor[key]: raise ValueError('Amplitude-pool identity mismatch: ' + key)
        if identity['subset_sha256'] != descriptor['subset_sha256']: raise ValueError('Amplitude-pool subset identity mismatch')
        if Path(input_dir).resolve() != Path(descriptor['input_dir']).resolve(): raise ValueError('Amplitude-pool cache path mismatch')
        if sha(Path(input_dir) / 'features.npy') != descriptor['features_sha256']: raise ValueError('Amplitude-pool source feature bytes changed')
        if descriptor['train_only'] is not True or any(r['split'] != 'train' for r in entries): raise ValueError('Resampling pools must be train-only')
        if type(config.get('medium_jump_amplitude_sampling_enabled')) is not bool: raise ValueError('Explicit bool sampling activation is required')
        probability = config['medium_jump_high_probability']
        if not math.isfinite(probability) or not 0 <= probability <= 1: raise ValueError('Invalid amplitude mixture probability')
        self.enabled = config['medium_jump_amplitude_sampling_enabled']; self.probability = probability
        self.eligible = {i for i, r in enumerate(entries) if r['group'] == 'jump' and 91 <= r['num_frames'] <= 120}
        self.high = descriptor['high_indices']; self.low = descriptor['low_indices']
        if not self.high or not self.low or len(set(self.high + self.low)) != len(self.high + self.low) or set(self.high + self.low) != self.eligible:
            raise ValueError('Amplitude pools are incomplete, overlapping or empty')
        self.high_set = set(self.high); self.pool_sha256 = expected; self.pool_fingerprint = pool['fingerprint']
        self.rng = torch.Generator(device='cpu').manual_seed(config['seed'] + 7919)
        self.last_counts = {'eligible': 0, 'high': 0, 'changed': 0}

    def transform(self, base_indices):
        indices = list(base_indices); positions = [i for i, index in enumerate(indices) if index in self.eligible]
        if self.enabled:
            mixture = torch.rand(len(positions), generator=self.rng)
            for position, high in zip(positions, mixture < self.probability):
                pool = self.high if bool(high) else self.low
                indices[position] = pool[int(torch.randint(len(pool), (1,), generator=self.rng))]
        self.last_counts = {'eligible': len(positions), 'high': sum(indices[i] in self.high_set for i in positions),
                            'changed': sum(a != b for a, b in zip(base_indices, indices))}
        return indices

    def state(self):
        return {'rng': self.rng.get_state(), 'pool_sha256': self.pool_sha256,
                'pool_fingerprint': self.pool_fingerprint, 'enabled': self.enabled, 'high_probability': self.probability}

    def restore(self, state):
        for key, value in self.state().items():
            if key != 'rng' and state.get(key) != value: raise ValueError('Amplitude sampler resume mismatch: ' + key)
        self.rng.set_state(state['rng'])


class AmplitudeBalancedConstraintTrainer(PhysicalConstraintTrainer):
    def __init__(self, config, samples, identity, stats_dir, device='cuda:0'):
        if config.get('sampling') != 'stratified': raise ValueError('Amplitude resampling requires the original stratified base sampler')
        if config.get('network_dropout', 0.) != 0 or config.get('pe_dropout', 0.) != 0: raise ValueError('Dropout composition requires separate verification')
        sampler = MediumJumpResampler(config, samples.entries, identity, samples.directory)
        super().__init__(config, samples, identity, stats_dir, device)
        self.amplitude_sampler = sampler

    def draw_indices(self):
        base = super().draw_indices(); self.last_base_indices = list(base)
        return self.amplitude_sampler.transform(base)

    def update(self):
        metrics = super().update()
        if self.amplitude_sampler.enabled:
            values = self.amplitude_sampler.last_counts
            metrics.update(medium_jump_draws=values['eligible'], medium_jump_high_draws=values['high'],
                           medium_jump_changed_draws=values['changed'],
                           medium_jump_high_fraction=values['high'] / max(1, values['eligible']))
        return metrics

    def state(self):
        return dict(super().state(), amplitude_sampler_operator=VERSION,
                    amplitude_sampler=self.amplitude_sampler.state())

    def load_state(self, state, continuation=False, initialization_only=False):
        if initialization_only: raise ValueError('This version requires full-state continuation')
        operator = state.get('amplitude_sampler_operator')
        if operator not in (None, VERSION): raise ValueError('Unknown amplitude-sampling operator')
        if state.get('dropout_operator') is not None or state.get('cache_extension_operator') is not None:
            raise ValueError('Source has an unsupported composed trainer')
        if state['config'].get('network_dropout', 0.) != 0 or state['config'].get('pe_dropout', 0.) != 0:
            raise ValueError('Unsupported source dropout configuration')
        if operator is None and not continuation: raise ValueError('A base checkpoint requires explicit sampling-stage continuation')
        if operator == VERSION:
            # Validate sampler state before changing any model/optimizer state.
            source = state['amplitude_sampler']; target = self.amplitude_sampler.state()
            if any(source.get(k) != target[k] for k in target if k != 'rng'):
                raise ValueError('Amplitude sampler configuration/pool changed')
        super().load_state(state, continuation=continuation)
        if operator == VERSION: self.amplitude_sampler.restore(state['amplitude_sampler'])
