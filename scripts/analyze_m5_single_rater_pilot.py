#!/usr/bin/env python3
"""Archive one actual human's pilot feedback without invoking the two-rater gate."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m5_length_research import read, write, sha, now

GROUPS = ('walk', 'jog', 'jump', 'raise_arms', 'turn', 'squat')
EDITABLE = {'rater_1.csv', 'rater_2.csv', 'rater_1_attestation.json', 'rater_2_attestation.json'}


def analyze(private, ratings, attestation):
    """Return descriptive results only; never duplicate a reviewer or pass M5."""
    complete = read(private / 'COMPLETE.json')
    for relative, expected in complete['private_sha256'].items():
        if sha(private / relative) != expected:
            raise ValueError('Private package changed: ' + relative)
    manifest = read(private / 'manifest.json')
    if manifest['phase'] != 'authored_development' or manifest['engineering_fixture'] is not True:
        raise ValueError('This entry point only accepts development pilot packages')
    if len(manifest['models']) != 1:
        raise ValueError('This pilot analysis requires exactly one Small/Tiny model pair')
    public = Path(complete['public_directory'])
    if sha(public / 'PACKET_COMPLETE.json') != complete['public_packet_sha256']:
        raise ValueError('Public packet identity changed')
    delivered = read(public / 'PACKET_COMPLETE.json')
    for relative, expected in delivered['files_sha256'].items():
        if relative not in EDITABLE and sha(public / relative) != expected:
            raise ValueError('Delivered public artifact changed: ' + relative)
    artifacts = read(private / 'artifacts.json')
    for artifact in artifacts:
        if sha(private / artifact['private_qpos_copy']) != artifact['source_qpos_sha256']:
            raise ValueError('Archived source qpos changed')
    mapping = read(private / 'mapping.json')
    public_trials = {r['trial_id']: r for r in read(public / 'trials.json')['trials']}
    if set(public_trials) != set(mapping):
        raise ValueError('Public/private trial sets differ')
    with ratings.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'trial_id', 'reviewer_id', 'preference', 'A_main_correct', 'B_main_correct',
                    'A_jump_complete', 'B_jump_complete', 'direction_speed_arms_notes'}
        if set(reader.fieldnames or ()) != required or len(reader.fieldnames) != len(required):
            raise ValueError('Rating columns differ from the delivered form')
        rows = list(reader)
    if len(rows) != len(mapping) or {r['trial_id'] for r in rows} != set(mapping):
        raise ValueError('Missing, unknown or duplicate rating trials')
    names = {r['reviewer_id'].strip() for r in rows}
    if len(names) != 1 or '' in names:
        raise ValueError('Exactly one named reviewer is required')
    name = next(iter(names)); attest = read(attestation)
    if attest.get('reviewer_id', '').strip() != name or any(
        attest.get(k) is not True for k in ('human', 'independent_review', 'model_mapping_not_viewed')
    ):
        raise ValueError('A matching human independence/blinding declaration is required')
    decoded = []
    for row in rows:
        trial = mapping[row['trial_id']]; visible = public_trials[row['trial_id']]
        if hashlib.sha256(visible['prompt'].encode()).hexdigest() != trial['text_sha256']:
            raise ValueError('Prompt text does not match the private mapping')
        if row['preference'] not in ('A', 'B', 'tie') or any(
            row[k] not in ('0', '1') for k in ('A_main_correct', 'B_main_correct')
        ):
            raise ValueError('Unfilled or invalid preference/main-action rating')
        jump = bool(trial['complete_jump_applicable'])
        if jump != visible['complete_jump_applicable'] or any(
            row[k] not in (('0', '1') if jump else ('', '0', '1'))
            for k in ('A_jump_complete', 'B_jump_complete')
        ):
            raise ValueError('Unfilled or invalid applicable complete-jump rating')
        small = trial['small_side']; tiny = 'B' if small == 'A' else 'A'
        preference = 'tie' if row['preference'] == 'tie' else 'small' if row['preference'] == small else 'tiny'
        decoded.append({'trial_id': row['trial_id'], 'reviewer_id': name, 'group': trial['group'],
            'prompt': visible['prompt'], 'prompt_id': trial['prompt_id'], 'cluster_id': trial['cluster_id'],
            'small_side': small, 'original_preference': row['preference'], 'preference': preference,
            'small_main_correct': int(row[small + '_main_correct']),
            'tiny_main_correct': int(row[tiny + '_main_correct']), 'complete_jump_applicable': jump,
            'small_jump_complete': int(row[small + '_jump_complete']) if jump else None,
            'tiny_jump_complete': int(row[tiny + '_jump_complete']) if jump else None,
            'notes_verbatim': row['direction_speed_arms_notes']})
    if {r['group'] for r in decoded} != set(GROUPS):
        raise ValueError('Pilot must cover all six classes')
    classes = {}
    for group in GROUPS:
        part = [r for r in decoded if r['group'] == group]; n = len(part)
        classes[group] = {'trials': n, 'unique_prompts': len({r['prompt_id'] for r in part}),
            'small_correct': sum(r['small_main_correct'] for r in part),
            'tiny_correct': sum(r['tiny_main_correct'] for r in part),
            'small_accuracy': sum(r['small_main_correct'] for r in part) / n,
            'tiny_accuracy': sum(r['tiny_main_correct'] for r in part) / n,
            'preferences': {k: sum(r['preference'] == k for r in part) for k in ('small', 'tiny', 'tie')}}
    count = Counter(r['preference'] for r in decoded); jumps = [r for r in decoded if r['complete_jump_applicable']]
    model = manifest['models'][0]; tiny = manifest['tiny']
    return {'schema': 'm5-single-human-development-pilot-analysis-v1', 'analyzed_at_utc': now(),
        'package_id': manifest['package_id'], 'review_phase': manifest['phase'],
        'packet_engineering_fixture': True, 'ratings_are_actual_human_feedback': True,
        'human_identity_evidence': 'User supplied these files as their own feedback; matching self-attestation retained.',
        'actual_reviewers': 1, 'reviewer_id': name,
        'normalized_reviewer_id': unicodedata.normalize('NFKC', name).casefold(),
        'unique_prompts': len({r['prompt_id'] for r in decoded}),
        'declared_authored_prompt_families': len({r['cluster_id'] for r in decoded}), 'trials': len(decoded),
        'main_action': {'small_correct': sum(r['small_main_correct'] for r in decoded),
            'tiny_correct': sum(r['tiny_main_correct'] for r in decoded), 'denominator': len(decoded),
            'small_macro_accuracy': sum(v['small_accuracy'] for v in classes.values()) / len(GROUPS),
            'tiny_macro_accuracy': sum(v['tiny_accuracy'] for v in classes.values()) / len(GROUPS)},
        'preferences': {'small_wins': count['small'], 'tiny_wins': count['tiny'], 'ties': count['tie'],
            'small_tie_half_score': (count['small'] + .5 * count['tie']) / len(decoded),
            'scope': 'Descriptive paired counts from one person on this pilot; no final statistical admission.'},
        'complete_jump': {'applicable_prompts': len(jumps),
            'small_complete': sum(r['small_jump_complete'] for r in jumps),
            'tiny_complete': sum(r['tiny_jump_complete'] for r in jumps)},
        'per_class': classes, 'decoded_rows': decoded,
        'models': {'small': {'run': model['run'], 'step': model['step'], 'checkpoint_sha256': model['checkpoint_sha256']},
                   'tiny': {'run': tiny['run'], 'step': tiny['step'], 'checkpoint_sha256': tiny['checkpoint_sha256']}},
        'integrity': {'private_mapping_passed': True, 'public_files_passed': True,
            'archived_qpos_passed': True, 'archived_source_qpos_files': len(artifacts),
            'paired_trials': len(decoded),
            'public_packet_sha256': complete['public_packet_sha256'],
            'input_sha256': {str(p.resolve()): sha(p) for p in (ratings, attestation)},
            'analyzer_sha256': sha(Path(__file__))},
        'ratings_modified_or_duplicated': False, 'synthetic': False, 'new_optimizer_updates': 0,
        'new_diffusion_samples': 0, 'formal_two_rater_requirement_satisfied': False,
        'formal_120_prompt_requirement_satisfied': False, 'human_gate_passed': False, 'M5_complete': False,
        'interpretation_limit': 'One human on an authored development pilot. Model steps are recorded above. This is real feedback but not final source-final80k/two-rater evidence.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-package', type=Path, required=True)
    parser.add_argument('--ratings', type=Path, required=True)
    parser.add_argument('--attestation', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.relative_to(ROOT / 'runs')
    if output.exists():
        raise FileExistsError('Preserve feedback analysis; choose a new archive revision')
    result = analyze(args.private_package.resolve(), args.ratings, args.attestation)
    output.mkdir(parents=True)
    shutil.copyfile(args.ratings, output / 'rater_1_original.csv')
    shutil.copyfile(args.attestation, output / 'rater_1_original_attestation.json')
    for original, copied in ((args.ratings, output / 'rater_1_original.csv'),
                             (args.attestation, output / 'rater_1_original_attestation.json')):
        if sha(original) != sha(copied):
            raise ValueError('Feedback snapshot differs from the submitted bytes')
    write(output / 'analysis.json', result)
    print(json.dumps({k: result[k] for k in ('package_id', 'actual_reviewers', 'main_action',
          'preferences', 'complete_jump', 'human_gate_passed')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
