#!/usr/bin/env python3
"""Package exact retained qpos as anonymous two-rater review videos.

Only public files enter the ZIP. Model identities, original qpos, source hashes
and the randomization key are stored in a separate private directory.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import argparse
import csv
import hashlib
import hmac
import json
import secrets
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m5_length_research import read, write, sha, now
GROUPS = ('walk', 'jog', 'jump', 'raise_arms', 'turn', 'squat')
SMALL = {'latent_dim': 512, 'ff_size': 2048, 'num_layers': 8, 'num_heads': 8}
TINY = {'latent_dim': 256, 'ff_size': 1024, 'num_layers': 4, 'num_heads': 4}
FIELDS = ('trial_id', 'reviewer_id', 'preference', 'A_main_correct', 'B_main_correct',
          'A_jump_complete', 'B_jump_complete', 'direction_speed_arms_notes')


def select_events(experiment, phase, fixture):
    protocol_dir = experiment / 'review/protocol'; protocol = read(protocol_dir / 'review_protocol.json')
    if phase == 'authored_development':
        path = protocol_dir / 'authored_development_prompts.json'
        if sha(path) != protocol['authored_prompts_sha256']:
            raise ValueError('Frozen authored prompts changed')
        events = read(path)['events']
        if fixture:
            events = [row for group in GROUPS for row in sorted(
                (e for e in events if e['group'] == group),
                key=lambda r: hashlib.sha256((r['prompt_id'] + ':control-7377').encode()).hexdigest())[:2]]
        return events, protocol['authored_development']['blind_review_seed']
    if fixture:
        raise ValueError('A source-final review cannot be an engineering fixture')
    path = protocol_dir / 'final_audit_prompt_selection.json'
    if sha(path) != protocol['final_selection_sha256']:
        raise ValueError('Frozen final prompt selection changed')
    from training.m5_review_statistics import source_clusters
    selection = read(path); events = selection['events']; clusters = source_clusters(events)
    jumps = read(protocol_dir / 'jump_applicability.json')
    if jumps['source_final_selection_sha256'] != sha(path):
        raise ValueError('Jump applicability belongs to another prompt selection')
    applicable = {r['sample_id']: r['complete_jump_applicable'] for r in jumps['records']}
    if len(set(clusters.values())) != 120:
        raise ValueError('Final review requires the 120 frozen independent source clusters')
    events = [dict(e, prompt_id=e['sample_id'], text_sha256=hashlib.sha256(e['text'].encode()).hexdigest(),
                   cluster_id=clusters[e['sample_id']], complete_jump_applicable=applicable.get(e['sample_id'], False))
              for e in events]
    return events, protocol['final_source_audit']['blind_review_seed']


def load_model(entry, role, experiment, phase, seed, release):
    import torch
    run = Path(entry['run_dir']).resolve(); run.relative_to(ROOT / 'runs')
    checkpoint = Path(entry['checkpoint']).resolve(); checksum = sha(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['schema'] not in ('kimodo-m5-physical-checkpoint-v2', 'kimodo-m5-physical-atan2-checkpoint-v1',
                                'kimodo-m5-physical-constraints-checkpoint-v1'):
        raise ValueError('Physical-v2 model required')
    if state['config']['model'] != (TINY if role == 'tiny' else SMALL):
        raise ValueError('Model role does not match its actual architecture')
    if phase == 'source_final':
        if state['step'] != 80000 or checksum not in release['allowed_checkpoint_sha256']:
            raise ValueError('Final review requires a released fixed 80k model')
    view = read(experiment / 'data/READY.json')
    if state['identity']['training_view_fingerprint'] != view['fingerprint']:
        raise ValueError('Model belongs to another training view')
    for relative, expected in state['identity']['stats_sha256'].items():
        if sha(run / relative) != expected:
            raise ValueError('Model statistics changed')
    folder = Path(entry['report_dir']).resolve(); report = read(folder / 'report.json')
    if report.get('checkpoint_sha256') != checksum or report.get('step') != state['step'] or report.get('weights') != 'ema':
        raise ValueError('Review generation is not bound to this checkpoint EMA')
    if phase == 'authored_development':
        complete = read(folder / 'TEXT_PROBE_COMPLETE.json')
        if complete['report_sha256'] != sha(folder / 'report.json') or report['schema'] != 'm5-physical-text-probe-v2':
            raise ValueError('Incomplete or changed authored text generation')
        cache = read(experiment / 'review/authored_text_cache/COMPLETE.json')
        if report['cache_fingerprint'] != cache['fingerprint'] or state['identity']['text_config_fingerprint'] != cache['parent_encoder_fingerprint']:
            raise ValueError('Text cache identity mismatch')
        if report['DDIM_steps'] != 100 or not report['shared_noise_across_prompts'] or seed not in report['seeds']:
            raise ValueError('Changed authored inference/noise protocol')
    else:
        if (report.get('audit_release_sha256') != release['_sha256'] or report.get('manifest_sha256') != release['manifest_sha256'] or
                report.get('events') != 657 or report.get('seeds') != [23017, 23073, 232026] or report.get('ddim_steps') != 100):
            raise ValueError('Final review requires the complete released 657-event evaluation')
    if report.get('guidance') != 2.:
        raise ValueError('Changed review guidance')
    rows = read(folder / 'per_sample.json')
    selected = [r for r in rows if r.get('text_mode', 'real') == 'real' and r['seed'] == seed and r['condition'] == 'heading0']
    by_id = {r.get('prompt_id', r['sample_id']): r for r in selected}
    if len(by_id) != len(selected):
        raise ValueError('Duplicate review generation keys')
    info = {'run': str(run), 'checkpoint': str(checkpoint), 'checkpoint_sha256': checksum,
            'step': state['step'], 'training_seed': state['config']['seed'], 'identity': state['identity'],
            'config': state['config'], 'report': str(folder / 'report.json'), 'report_sha256': sha(folder / 'report.json'),
            'per_sample_sha256': sha(folder / 'per_sample.json'), 'training_started_at_utc': read(run / 'run_config.json').get('started_at_utc'),
            'training_finished_at_utc': read(run / 'COMPLETED.json').get('completed_at_utc'),
            'lineage': read(run / 'lineage.json'), 'source_noise_key': report.get('noise_key', 'frozen-source-event-key'),
            'text_cache_fingerprint': report.get('cache_fingerprint')}
    del state
    return info, by_id, folder


def csv_template(path, trials):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader()
        writer.writerows({key: t['trial_id'] if key == 'trial_id' else '' for key in FIELDS} for t in trials)


def build(spec, spec_path):
    import numpy as np
    from kimodo.assets import skeleton_asset_path
    from training.video import render
    from training.m5_blind_review_html import review_html
    experiment = Path(spec['experiment']).resolve(); phase = spec['phase']; fixture = bool(spec.get('engineering_fixture'))
    if phase not in ('authored_development', 'source_final'):
        raise ValueError('Unknown review phase')
    release = {}
    if phase == 'source_final':
        if not spec.get('audit_release'):
            raise ValueError('Source-final packaging requires a frozen audit release')
        path = Path(spec['audit_release']); release = read(path); release['_sha256'] = sha(path)
        selection = read(experiment / 'review/protocol/final_audit_prompt_selection.json')
        if (release.get('schema') != 'm5-physical-final-audit-release-v1' or
                release.get('manifest_sha256') != selection['audit_manifest_sha256'] or len(spec['small_models']) != 2):
            raise ValueError('Wrong final release/manifest/model count')
    events, seed = select_events(experiment, phase, fixture)
    count = 2 if fixture else 20
    if len(events) != count * 6 or Counter(e['group'] for e in events) != Counter({g: count for g in GROUPS}):
        raise ValueError('Incomplete balanced prompt coverage')
    if len({e['text_sha256'] for e in events}) != len(events):
        raise ValueError('Repeated prompt text in the predeclared selection')
    loaded = [load_model(entry, 'small', experiment, phase, seed, release) for entry in spec['small_models']]
    tiny = load_model(spec['tiny_model'], 'tiny', experiment, phase, seed, release)
    if phase == 'source_final' and {m[0]['training_seed'] for m in loaded} != {3101, 3102}:
        raise ValueError('Both independent Small training seeds are required')
    if len({m[0]['checkpoint_sha256'] for m in [*loaded, tiny]}) != len(loaded) + 1:
        raise ValueError('Duplicate models in review')
    for model in loaded:
        if model[0]['step'] != tiny[0]['step']:
            raise ValueError('Paired model training budgets must match')
        for field in ('physical_input_fingerprint', 'training_view_fingerprint', 'text_config_fingerprint', 'stats_sha256'):
            if model[0]['identity'][field] != tiny[0]['identity'][field]:
                raise ValueError('Paired model inputs/statistics differ')
        if (model[0]['source_noise_key'], model[0]['text_cache_fingerprint']) != (tiny[0]['source_noise_key'], tiny[0]['text_cache_fingerprint']):
            raise ValueError('Paired text/noise protocol differs')
    nonce = secrets.token_bytes(32); package_id = secrets.token_hex(8)
    output_root = Path(spec['public_root']).resolve(); output_root.relative_to(ROOT / 'exports')
    if fixture and 'verification' not in output_root.relative_to(ROOT / 'exports').parts:
        raise ValueError('Engineering packets must stay under exports/verification')
    public = output_root / package_id
    private_root = Path(spec['private_root']).resolve(); private_root.relative_to(experiment / 'review')
    private = private_root / package_id
    if private.is_relative_to(output_root) or output_root.is_relative_to(private_root):
        raise ValueError('Public and private trees must be separate')
    public.mkdir(parents=True, exist_ok=False); private.mkdir(parents=True, exist_ok=False)
    source_files = ('scripts/package_m5_blind_review.py', 'training/m5_blind_review_html.py',
                    'scripts/analyze_m5_blind_review.py', 'training/video.py', 'training/m5_review_statistics.py')
    for relative in source_files:
        target = private / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / relative, target)
    mapping = {}; trials = []; artifacts = []; xml = skeleton_asset_path('g1skel34', 'xml', 'g1.xml')
    def keyed(value):
        return hmac.new(nonce, value.encode(), hashlib.sha256).hexdigest()
    for small in loaded:
        for group in GROUPS:
            selected = sorted((e for e in events if e['group'] == group), key=lambda e: keyed('side:' + small[0]['checkpoint_sha256'] + ':' + e['prompt_id']))
            for index, event in enumerate(selected):
                prompt_id = event['prompt_id']; trial_id = keyed('trial:' + small[0]['checkpoint_sha256'] + ':' + prompt_id)[:20]
                side = 'A' if index % 2 == 0 else 'B'; pair = {side: small, 'B' if side == 'A' else 'A': tiny}
                pair_rows = [model[1][prompt_id] for model in pair.values()]
                if any(r['text'] != event['text'] or r['group'] != group for r in pair_rows) or pair_rows[0]['frames'] != pair_rows[1]['frames']:
                    raise ValueError('Paired prompt/length changed')
                frames = pair_rows[0]['frames']; applicable = bool(event.get('complete_jump_applicable', group == 'jump'))
                mapping[trial_id] = {'small_run': small[0]['run'], 'small_side': side, 'group': group,
                    'prompt_id': prompt_id, 'text_sha256': event['text_sha256'], 'cluster_id': event['cluster_id'],
                    'complete_jump_applicable': applicable, 'small_checkpoint_sha256': small[0]['checkpoint_sha256'],
                    'tiny_checkpoint_sha256': tiny[0]['checkpoint_sha256']}
                trials.append({'trial_id': trial_id, 'prompt': event['text'], 'video_A': f'videos/{trial_id}/A/replay.mp4',
                               'video_B': f'videos/{trial_id}/B/replay.mp4', 'complete_jump_applicable': applicable})
                for label, model in pair.items():
                    row = model[1][prompt_id]; source = (model[2] / row['artifact_prefix']).with_suffix('.csv').resolve()
                    source.relative_to(model[2]); qpos = np.loadtxt(source, delimiter=',')
                    if qpos.shape != (frames, 36) or not np.isfinite(qpos).all() or not row['valid_qpos']:
                        raise ValueError('Selected case cannot be rendered; do not drop or replace it')
                    video_dir = public / 'videos' / trial_id / label; video_dir.mkdir(parents=True)
                    metadata = {'prompt': event['text'], 'training_run_id': 'Anonymous ' + label,
                                'checkpoint_step': model[0]['step'], 'weights': 'ema', 'seed': seed, 'guidance': 2.,
                                'postprocess_label': 'bounded native hinges; exact retained qpos'}
                    result = render(qpos, video_dir / 'replay.mp4', metadata, xml)
                    copy = private / 'qpos' / trial_id / (label + '.csv'); copy.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, copy)
                    if sha(source) != sha(copy):
                        raise ValueError('Source qpos changed while packaging')
                    artifacts.append({'trial_id': trial_id, 'side': label, 'source': str(source),
                        'checkpoint_sha256': model[0]['checkpoint_sha256'], 'source_qpos_sha256': sha(source),
                        'private_qpos_copy': str(copy.relative_to(private)), 'video': str((video_dir / 'replay.mp4').relative_to(public)),
                        'video_sha256': sha(video_dir / 'replay.mp4'), 'frames': frames, 'render': result,
                        'trajectory_modified': False, 'new_diffusion_samples': 0})
                write(private / 'progress.json', {'completed_trials': len(trials), 'target_trials': len(events) * len(loaded), 'last_trial': trial_id})
                print('Rendered anonymous trial', len(trials), '/', len(events) * len(loaded), flush=True)
    trials.sort(key=lambda t: keyed('order:' + t['trial_id']))
    public_data = {'package_id': package_id, 'engineering_fixture': fixture, 'phase': phase, 'trials': trials}
    write(public / 'trials.json', public_data)
    (public / 'index.html').write_text(review_html(public_data), encoding='utf-8')
    for slot in (1, 2):
        csv_template(public / f'rater_{slot}.csv', trials)
        write(public / f'rater_{slot}_attestation.json', {'reviewer_id': '', 'human': False,
              'independent_review': False, 'model_mapping_not_viewed': False})
    (public / 'README.txt').write_text(
        '打开 index.html，选择评分文件槽，填写评阅者 ID。两位独立人工评阅者分别评分。\n'
        '主动作正确填 1，否则填 0；偏好填 A/B/tie。适用跳跃还需判断起跳、腾空、落地是否完整。\n'
        '记录方向、速度、停步、手臂等失败细节。保存后分别导出 CSV 与评阅声明；草稿和空白不能通过验收。\n'
        '本包为 MuJoCo qpos + mj_forward 运动学回放，30FPS，不含动力学跟踪。\n' +
        ('此包仅验证开发评阅流程，12 条提示不满足最终 120 提示要求。\n' if fixture else '按原协议评阅全部试次，不删除不佳结果。\n'), encoding='utf-8')
    public_hashes = {str(f.relative_to(public)): sha(f) for f in public.rglob('*') if f.is_file()}
    write(public / 'PACKET_COMPLETE.json', {'schema': 'm5-anonymous-review-public-v1', 'package_id': package_id,
          'engineering_fixture': fixture, 'trials': len(trials), 'videos': len(artifacts), 'files_sha256': public_hashes,
          'human_ratings_collected': False, 'M5_complete': False})
    write(private / 'mapping.json', mapping); write(private / 'artifacts.json', artifacts)
    write(private / 'manifest.json', {'schema': 'm5-anonymous-review-private-v1', 'package_id': package_id,
          'created_at_utc': now(), 'phase': phase, 'engineering_fixture': fixture, 'public_directory': str(public),
          'experiment': str(experiment), 'spec_sha256': sha(spec_path), 'randomization_key_hex': nonce.hex(),
          'side_policy': 'Secret-HMAC ordering, balanced A/B within each class and Small model.',
          'review_seed': seed, 'unique_prompts': len(events), 'trials': len(trials), 'videos': len(artifacts),
          'per_class': dict(Counter(e['group'] for e in events)), 'models': [m[0] for m in loaded], 'tiny': tiny[0],
          'robot_xml_sha256': sha(xml), 'new_diffusion_samples_by_packager': 0, 'new_optimizer_updates': 0,
          'source_archive': 'source', 'code_sha256': {p: sha(ROOT / p) for p in source_files},
          'human_gate_passed': False, 'final_audit_model_scores_read': phase == 'source_final', 'M5_complete': False})
    archive = public.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED) as bundle:
        for source in sorted(public.rglob('*')):
            if source.is_file():
                bundle.write(source, str(source.relative_to(public)))
    write(private / 'COMPLETE.json', {'schema': 'm5-anonymous-review-complete-v1', 'public_directory': str(public),
          'public_packet_sha256': sha(public / 'PACKET_COMPLETE.json'), 'zip_path': str(archive), 'zip_sha256': sha(archive),
          'private_sha256': {p: sha(private / p) for p in ('manifest.json', 'mapping.json', 'artifacts.json')}})
    print(json.dumps({'public_directory': str(public), 'private_directory': str(private), 'zip': str(archive),
                      'trials': len(trials), 'videos': len(artifacts), 'human_gate_passed': False}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--spec', type=Path, required=True); a = p.parse_args()
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    build(read(a.spec), a.spec.resolve())


if __name__ == '__main__':
    main()
