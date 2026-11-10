#!/usr/bin/env python3
"""Bind two human rating files to an intact anonymous package; never complete M5."""
import argparse
import csv
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m5_length_research import read, write, sha, now


def analyze(private, rating_paths, attestation_paths, synthetic=False):
    from training.m5_review_statistics import analyze_ratings
    complete = read(private / 'COMPLETE.json')
    for path, expected in complete['private_sha256'].items():
        if sha(private / path) != expected:
            raise ValueError('Private model mapping/provenance changed')
    manifest = read(private / 'manifest.json'); public = Path(complete['public_directory'])
    if sha(public / 'PACKET_COMPLETE.json') != complete['public_packet_sha256']:
        raise ValueError('Public packet changed')
    editable = {'rater_1.csv', 'rater_2.csv', 'rater_1_attestation.json', 'rater_2_attestation.json'}
    for path, expected in read(public / 'PACKET_COMPLETE.json')['files_sha256'].items():
        if path in editable:
            continue
        if sha(public / path) != expected:
            raise ValueError('Delivered public artifact changed: ' + path)
    for artifact in read(private / 'artifacts.json'):
        if sha(private / artifact['private_qpos_copy']) != artifact['source_qpos_sha256']:
            raise ValueError('Archived source qpos changed')
    if rating_paths[0].resolve() == rating_paths[1].resolve():
        raise ValueError('Two separate reviewer files are required')
    rows = []
    for path in rating_paths:
        with path.open(newline='', encoding='utf-8-sig') as stream:
            rows.append(list(csv.DictReader(stream)))
    mapping = read(private / 'mapping.json'); attest = [read(p) for p in attestation_paths]
    if manifest['engineering_fixture']:
        if any(len(v) != len(mapping) or {r['trial_id'] for r in v} != set(mapping) for v in rows):
            raise ValueError('Missing or duplicate pilot rating trials')
        missing = any(not r.get('reviewer_id', '').strip() or r.get('preference') not in ('A', 'B', 'tie') or
                      any(r.get(k) not in ('0', '1') for k in ('A_main_correct', 'B_main_correct')) or
                      (mapping[r['trial_id']]['complete_jump_applicable'] and
                       any(r.get(k) not in ('0', '1') for k in ('A_jump_complete', 'B_jump_complete')))
                      for file in rows for r in file)
        result = {'phase': 'awaiting_human_ratings' if missing else 'engineering_fixture_not_final_semantic_evidence', 'human_gate_passed': False,
                  'reason': 'This pilot does not meet the fixed source-final prompt/model/course scope.'}
    else:
        result = analyze_ratings(mapping, rows, attest, phase=manifest['phase'], synthetic=synthetic)
    eligible = manifest['phase'] == 'source_final' and not manifest['engineering_fixture'] and not synthetic
    return {'analyzed_at_utc': now(), 'package_id': manifest['package_id'], 'review_phase': manifest['phase'],
            'synthetic': synthetic, 'analysis': result, 'human_gate_passed': bool(eligible and result['human_gate_passed']),
            'models': {m['run']: m['checkpoint_sha256'] for m in manifest['models']},
            'tiny_checkpoint_sha256': manifest['tiny']['checkpoint_sha256'],
            'public_packet_sha256': complete['public_packet_sha256'],
            'input_sha256': {str(p.resolve()): sha(p) for p in [*rating_paths, *attestation_paths]},
            'code_sha256': {p: sha(ROOT / p) for p in ('scripts/analyze_m5_blind_review.py', 'training/m5_review_statistics.py')},
            'M5_complete': False}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--private-package', type=Path, required=True)
    p.add_argument('--ratings', type=Path, nargs=2, required=True); p.add_argument('--attestations', type=Path, nargs=2, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--synthetic', action='store_true'); a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve the existing analysis; write a new revision')
    result = analyze(a.private_package.resolve(), a.ratings, a.attestations, a.synthetic)
    write(a.output, result); print(result['analysis']['phase'], '| final human gate:', result['human_gate_passed'])


if __name__ == '__main__':
    main()
