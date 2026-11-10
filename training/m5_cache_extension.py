"""Explicit, append-only physical-cache continuation with frozen model statistics.

The existing physical/constraint loaders remain strict. This version verifies
the entire cache extension before accepting a declared old-to-new continuation;
saved checkpoints carry their actual new identities and the pinned proof hash.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from training.m5_constraints import PhysicalConstraintTrainer

VERSION = 'm5-append-only-frozen-stat-continuation-v1'
ROOT = Path(__file__).resolve().parents[1]
STAT_KEYS = {f'stats/motion/{part}/{name}.npy' for part in ('global_root', 'body', 'local_root')
             for name in ('mean', 'std')}
INVARIANT_KEYS = ('m2_fingerprint', 'text_config_fingerprint', 'stats_sha256')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def content_fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def validate_cache_compatibility(config, identity, dataset_directory):
    """Re-read bound files, full feature hashes, metadata and every appended frame."""
    path = Path(config['cache_compatibility_proof'])
    expected_sha = config['cache_compatibility_proof_sha256']
    require(len(expected_sha) == 64 and file_sha(path) == expected_sha, 'Cache compatibility proof hash changed')
    proof = read(path)
    require(proof.get('schema') == 'm5-append-compatible-cache-proof-v1' and proof.get('passed') is True,
            'Unsupported or failed append-cache proof')
    require(all(proof.get(k) is True for k in ('original_frame_prefix_bitwise_equal',
                'original_metadata_prefix_exactly_equal', 'frozen_model_normalization_unchanged')),
            'Append-only/frozen-stat invariants were not proven')
    source = Path(proof['source_cache']); target = Path(proof['target_cache'])
    old_view_path = Path(proof['source_training_view']); new_view_path = Path(proof['target_training_view'])
    source_stats = Path(proof['source_model_stats_dir']); target_stats = Path(proof['target_model_stats_dir'])
    for actual, declared in ((config['input_dir'], target), (dataset_directory, target),
                             (config['training_view'], new_view_path), (config['stats_dir'], target_stats)):
        require(Path(actual).resolve() == declared.resolve(), 'Target cache/view/model-stat path mismatch')
    require(set(proof['model_stats_sha256']) == STAT_KEYS, 'Model statistics must include all six original blocks')
    required = {source / 'config.json', source / 'READY', source / 'samples.parquet',
                target / 'config.json', target / 'READY', target / 'samples.parquet',
                old_view_path, new_view_path, target / 'content_descriptor.json',
                new_view_path.parent / 'view_content_descriptor.json',
                ROOT / 'training/g1_physical.py', ROOT / 'training/m5_physical_data.py',
                ROOT / 'training/m5_physical_losses.py'}
    for relative in STAT_KEYS:
        tail = Path(relative).relative_to('stats/motion')
        required.update((source_stats / tail, target_stats / tail))
    require({str(p) for p in required} <= set(proof['files_sha256']), 'Proof omits required cache/stat/source files')
    for name, expected in proof['files_sha256'].items():
        require(file_sha(name) == expected, 'Bound append-cache file changed: ' + name)
    old = read(source / 'config.json'); new = read(target / 'config.json')
    old_ready = read(source / 'READY'); new_ready = read(target / 'READY')
    old_view = read(old_view_path); new_view = read(new_view_path)
    require(old_ready.get('passed') is True and new_ready.get('passed') is True
            and old_view.get('passed') is True and new_view.get('passed') is True, 'Input or view is not ready')
    require(old['fingerprint'] == old_ready['fingerprint'] == old_view['input_fingerprint']
            == proof['source_cache_fingerprint'], 'Source physical-cache identity mismatch')
    require(new['fingerprint'] == new_ready['fingerprint'] == new_view['input_fingerprint']
            == proof['target_cache_fingerprint'] == identity['physical_input_fingerprint'], 'Target physical-cache identity mismatch')
    require(old_view['fingerprint'] == proof['source_training_view_fingerprint']
            and new_view['fingerprint'] == proof['target_training_view_fingerprint']
            == identity['training_view_fingerprint'], 'Training-view identity mismatch')
    require(new['parent_fingerprint'] == old['fingerprint']
            and new_view['parent_training_view_fingerprint'] == old_view['fingerprint'], 'Wrong append parent')
    for key in ('m2_parent_fingerprint', 'feature_dim', 'fps', 'physical_version', 'physical_code_sha256',
                'text_cache', 'text_config_fingerprint', 'normalization_epsilon', 'root_policy'):
        require(old[key] == new[key], 'Representation/text/normalization changed: ' + key)
    require(new['feature_dim'] == 417 and new['fps'] == 30 and new['normalization_epsilon'] == 1e-5,
            'Unsupported physical representation or normalizer')
    require(identity['m2_fingerprint'] == new['m2_parent_fingerprint']
            and identity['text_config_fingerprint'] == new['text_config_fingerprint'], 'M2/text identity mismatch')
    require(file_sha(ROOT / 'training/g1_physical.py') == new['physical_code_sha256'], 'Physical adapter changed')
    require(old_view['stats_sha256'] == new_view['stats_sha256'] == proof['model_stats_sha256']
            == identity['stats_sha256'], 'Qualified model normalization changed')
    for relative, expected in proof['model_stats_sha256'].items():
        tail = Path(relative).relative_to('stats/motion')
        require(file_sha(source_stats / tail) == file_sha(target_stats / tail) == expected, 'Model-stat bytes changed')
    # Cache-default and model statistics have distinct fitting provenance.
    require(old_ready['stats_sha256'] == new_ready['stats_sha256'], 'Cache-default statistics changed')
    for relative, expected in old_ready['stats_sha256'].items():
        require(file_sha(source / relative) == file_sha(target / relative) == expected, 'Cache-default statistic bytes changed')
    descriptor = read(target / 'content_descriptor.json'); view_descriptor = read(new_view_path.parent / 'view_content_descriptor.json')
    require(content_fingerprint(descriptor) == new['fingerprint'], 'Target content fingerprint does not recompute')
    require(content_fingerprint(view_descriptor) == new_view['fingerprint'], 'Target view fingerprint does not recompute')
    for name, expected in new_view['manifests_sha256'].items():
        require(file_sha(new_view_path.parent / (name + '.json')) == expected, 'Candidate manifest changed: ' + name)
    for directory, ready, expected in ((source, old_ready, proof['source_cache_feature_sha256']),
                                        (target, new_ready, proof['target_cache_feature_sha256'])):
        require(file_sha(directory / 'features.npy') == ready['feature_sha256'] == expected, 'Feature-file hash changed')
        require(file_sha(directory / 'samples.parquet') == ready['samples_sha256'], 'Sample-file hash changed')
    old_samples = pd.read_parquet(source / 'samples.parquet'); new_samples = pd.read_parquet(target / 'samples.parquet')
    require(len(old_samples) == proof['old_event_count'] and len(new_samples) == proof['new_event_count']
            and new_samples.sample_id.is_unique, 'Event counts/uniqueness changed')
    try:
        pd.testing.assert_frame_equal(old_samples.reset_index(drop=True),
            new_samples.iloc[:len(old_samples)].reset_index(drop=True), check_exact=True)
    except AssertionError as error:
        raise ValueError('Original sample metadata changed') from error
    a = np.load(source / 'features.npy', mmap_mode='r', allow_pickle=False)
    b = np.load(target / 'features.npy', mmap_mode='r', allow_pickle=False)
    require(a.dtype == b.dtype == np.float32 and a.shape == (proof['old_frame_count'], 417)
            and b.shape == (proof['new_frame_count'], 417), 'Feature shape/dtype changed')
    old_hash = hashlib.sha256(); new_hash = hashlib.sha256()
    for start in range(0, len(a), 16384):
        end = min(start + 16384, len(a)); old_bytes = a[start:end].tobytes(); new_bytes = b[start:end].tobytes()
        require(old_bytes == new_bytes, 'Old feature prefix is not byte-identical')
        old_hash.update(old_bytes); new_hash.update(new_bytes)
    require(old_hash.hexdigest() == new_hash.hexdigest() == proof['old_prefix_raw_sha256']
            == proof['new_prefix_raw_sha256'], 'Raw-prefix digest mismatch')
    appended = proof['appended_events']; extra = new_samples.iloc[len(old_samples):]
    require(len(appended) == len(extra) and set(extra.sample_id) == {r['sample_id'] for r in appended}, 'Appended event set changed')
    require(set(extra.split) == {'train'} and not extra.physical_stats_train.any(), 'Appended events must be train-only and excluded from stat fitting')
    rows = extra.set_index('sample_id').to_dict('index'); cursor = len(a)
    for event in appended:
        row = rows[event['sample_id']]; n = event['frames']
        require(event['start'] == row['frame_offset'] == row['physical_frame_offset'] == cursor
                and n == row['num_frames'], 'Appended offsets/lengths changed')
        require(file_sha(event['raw_file']) == event['raw_sha256'], 'Qualified raw shard changed')
        raw = np.load(event['raw_file'], allow_pickle=False)
        require(raw.dtype == np.float32 and raw.shape == (n, 417) and np.isfinite(raw).all(), 'Malformed appended source features')
        require(raw.tobytes() == b[cursor:cursor + n].tobytes(), 'Appended feature bytes differ from qualified source')
        cursor += n
    require(cursor == len(b), 'Extra unbound frames in new cache')
    require(file_sha(path) == expected_sha, 'Proof changed during verification')
    return proof


def source_origin(proof, identity):
    return {**{k: copy.deepcopy(identity[k]) for k in INVARIANT_KEYS},
            'physical_input_fingerprint': proof['source_cache_fingerprint'],
            'training_view_fingerprint': proof['source_training_view_fingerprint']}


class AppendCompatibleConstraintTrainer(PhysicalConstraintTrainer):
    def __init__(self, config, samples, identity, stats_dir, device='cuda:0'):
        require(config.get('network_dropout', 0.) == config.get('pe_dropout', 0.) == 0.,
                'Dropout requires a separately verified combined trainer version')
        require(Path(stats_dir).resolve() == Path(config['stats_dir']).resolve(), 'Model-stat argument changed')
        proof = validate_cache_compatibility(config, identity, samples.directory)
        self.cache_extension_proof = proof
        self.cache_extension_proof_sha256 = config['cache_compatibility_proof_sha256']
        self.cache_extension_origin = source_origin(proof, identity)
        super().__init__(config, samples, identity, stats_dir, device)

    def state(self):
        result = super().state()
        result.update(cache_extension_operator=VERSION,
            cache_compatibility_proof_sha256=self.cache_extension_proof_sha256,
            cache_extension_origin_identity=copy.deepcopy(self.cache_extension_origin))
        return result

    def load_state(self, state, continuation=False, initialization_only=False):
        require(not initialization_only, 'This version requires full-state continuation, not weight-only initialization')
        operator = state.get('cache_extension_operator'); source_identity = state['identity']
        require(operator in (None, VERSION), 'Unknown cache-extension operator')
        require(state['config'].get('network_dropout', 0.) == state['config'].get('pe_dropout', 0.) == 0.
                and state.get('dropout_operator') is None, 'Source dropout behavior is unsupported by this trainer')
        for key in INVARIANT_KEYS:
            require(source_identity[key] == self.identity[key], 'Checkpoint invariant mismatch: ' + key)
        old_pair = (source_identity['physical_input_fingerprint'], source_identity['training_view_fingerprint'])
        target_pair = (self.identity['physical_input_fingerprint'], self.identity['training_view_fingerprint'])
        if old_pair != target_pair:
            require(continuation and operator is None, 'Changing cache identity requires explicit old-to-new continuation')
            require(old_pair == (self.cache_extension_proof['source_cache_fingerprint'],
                                self.cache_extension_proof['source_training_view_fingerprint']), 'Checkpoint is not the proven append parent')
            for key, proof_key in (('input_dir', 'source_cache'), ('stats_dir', 'source_model_stats_dir'),
                                   ('training_view', 'source_training_view')):
                require(Path(state['config'][key]).resolve() == Path(self.cache_extension_proof[proof_key]).resolve(),
                        'Source checkpoint cache/stat/view configuration mismatch')
            proxy = dict(state, identity=dict(source_identity,
                physical_input_fingerprint=self.identity['physical_input_fingerprint'],
                training_view_fingerprint=self.identity['training_view_fingerprint']))
            # Only this verified boundary maps the in-memory compatibility check.
            # The source file is untouched; state() always records actual target IDs.
            return super().load_state(proxy, continuation=True)
        require(operator == VERSION
                and state.get('cache_compatibility_proof_sha256') == self.cache_extension_proof_sha256
                and state.get('cache_extension_origin_identity') == self.cache_extension_origin,
                'Resume extension operator/proof/origin mismatch')
        return super().load_state(state, continuation=continuation)
