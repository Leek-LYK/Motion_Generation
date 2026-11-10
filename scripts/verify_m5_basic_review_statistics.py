#!/usr/bin/env python3
"""Synthetic tests of basic80/optional40 statistics; never actual human evidence."""
import argparse
import copy
import hashlib
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from training.m5_basic_scope import BASIC_GROUPS
from training.m5_basic_review_statistics import analyze_basic_ratings, VERSION
from scripts.run_m5_length_research import write, sha, now


def main(output):
    if output.exists(): raise FileExistsError('Preserve verification')
    mapping = {}; rows = [[], []]; attestations = []
    for rater in ('synthetic-rater-a', 'synthetic-rater-b'):
        attestations.append({'reviewer_id': rater, 'human': True, 'independent_review': True, 'model_mapping_not_viewed': True,
                             'synthetic_fixture': True})
    for model in ('synthetic-small-seed1', 'synthetic-small-seed2'):
        for group in (*BASIC_GROUPS, 'jump', 'squat'):
            for i in range(20):
                prompt = group + ':' + str(i); trial = model + ':' + prompt; side = 'A' if i % 2 == 0 else 'B'
                required = group in BASIC_GROUPS
                mapping[trial] = {'small_run': model, 'small_side': side, 'group': group,
                    'prompt_id': prompt, 'text_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                    'cluster_id': 'synthetic-source:' + prompt, 'human_rating_required': required}
                for j, rater in enumerate(('synthetic-rater-a', 'synthetic-rater-b')):
                    rows[j].append({'trial_id': trial, 'reviewer_id': rater,
                        'preference': side if required else '',
                        'A_main_correct': str(int(side == 'A')) if required else '',
                        'B_main_correct': str(int(side == 'B')) if required else '',
                        'A_jump_complete': '', 'B_jump_complete': '', 'direction_speed_arms_notes': ''})
    result = analyze_basic_ratings(mapping, rows, attestations, synthetic=True)
    assert result['two_small_model_comparisons_present'] and not result['human_gate_passed'] and not result['M5_complete']
    assert all(r['numeric_basic_checks_passed'] and r['paired_basic_preference']['clusters'] == 80
               and r['basic_class_macro_accuracy'] == 1. for r in result['results'].values())
    omitted_flag = analyze_basic_ratings(mapping, rows, attestations)
    assert omitted_flag['synthetic'] and not omitted_flag['human_gate_passed']
    optional_bad = copy.deepcopy(rows)
    for file in optional_bad:
        for row in file:
            if not mapping[row['trial_id']]['human_rating_required']:
                row.update(preference='tie', A_main_correct='0', B_main_correct='0', A_jump_complete='0', B_jump_complete='0')
    bad_result = analyze_basic_ratings(mapping, optional_bad, attestations, synthetic=True)
    assert bad_result['results'] == result['results']
    mandatory_only = [[r for r in file if mapping[r['trial_id']]['human_rating_required']] for file in rows]
    only = analyze_basic_ratings(mapping, mandatory_only, attestations, synthetic=True)
    for model in result['results']:
        assert only['results'][model]['checks'] == result['results'][model]['checks']
        assert only['results'][model]['paired_basic_preference'] == result['results'][model]['paired_basic_preference']
    low_basic = copy.deepcopy(rows)
    for file in low_basic:
        for row in file:
            if mapping[row['trial_id']]['human_rating_required']:
                side = mapping[row['trial_id']]['small_side']; other = 'B' if side == 'A' else 'A'
                row.update(preference=other, A_main_correct=str(int(other == 'A')), B_main_correct=str(int(other == 'B')))
    low = analyze_basic_ratings(mapping, low_basic, attestations, synthetic=True)
    assert not any(v['numeric_basic_checks_passed'] for v in low['results'].values())
    ties = copy.deepcopy(rows)
    for file in ties:
        for row in file:
            if mapping[row['trial_id']]['human_rating_required']: row.update(preference='tie', A_main_correct='1', B_main_correct='1')
    tie_result = analyze_basic_ratings(mapping, ties, attestations, synthetic=True)
    assert all(v['paired_basic_preference']['ci95'] == [.5, .5] and not v['checks']['paired_basic_preference_ci_lower_gt50pct'] for v in tie_result['results'].values())
    rejected = []
    def reject(label, fn):
        try: fn()
        except ValueError: rejected.append(label)
        else: raise AssertionError('Invalid review accepted: ' + label)
    reject('one_reviewer_not_final_two', lambda: analyze_basic_ratings(mapping, [rows[0]], [attestations[0]], synthetic=True))
    reject('missing_required_trial', lambda: analyze_basic_ratings(mapping, [rows[0][1:], rows[1]], attestations, synthetic=True))
    reject('duplicate_trial', lambda: analyze_basic_ratings(mapping, [rows[0] + [rows[0][0]], rows[1]], attestations, synthetic=True))
    duplicated = copy.deepcopy(rows); duplicated_attest = copy.deepcopy(attestations)
    for row in duplicated[1]: row['reviewer_id'] = 'ＳＹＮＴＨＥＴＩＣ－ＲＡＴＥＲ－Ａ'
    duplicated_attest[1]['reviewer_id'] = 'ＳＹＮＴＨＥＴＩＣ－ＲＡＴＥＲ－Ａ'
    reject('normalized_duplicate_reviewer', lambda: analyze_basic_ratings(mapping, duplicated, duplicated_attest, synthetic=True))
    clustered = copy.deepcopy(mapping)
    for trial in clustered.values(): trial['cluster_id'] = trial['group'] + ':one-shared-source'
    reject('one_source_repeated_as20_prompts', lambda: analyze_basic_ratings(clustered, rows, attestations, synthetic=True))
    unfilled = copy.deepcopy(rows); unfilled[0][0]['A_main_correct'] = ''
    assert analyze_basic_ratings(mapping, unfilled, attestations, synthetic=True)['phase'] == 'awaiting_required_basic_ratings'
    unattested = copy.deepcopy(attestations); unattested[0]['independent_review'] = False
    assert analyze_basic_ratings(mapping, rows, unattested, synthetic=True)['phase'] == 'awaiting_basic_reviewer_attestation'
    authored = analyze_basic_ratings(mapping, rows, attestations, phase='authored_development', synthetic=True)
    assert not authored['human_gate_passed']
    clean_attestations = [{k: v for k, v in a.items() if k != 'synthetic_fixture'} for a in attestations]
    authored_only = analyze_basic_ratings(mapping, rows, clean_attestations, phase='authored_development')
    assert not authored_only['synthetic'] and not authored_only['human_gate_passed']
    single_mapping = {k: v for k, v in mapping.items() if v['small_run'] == 'synthetic-small-seed1'}
    single_rows = [[r for r in file if r['trial_id'] in single_mapping] for file in rows]
    single = analyze_basic_ratings(single_mapping, single_rows, attestations, synthetic=True)
    assert not single['two_small_model_comparisons_present'] and not single['human_gate_passed']
    write(output, {'passed': True, 'verified_at_utc': now(), 'operator': VERSION, 'synthetic_only': True,
        'synthetic_mapping_trials': len(mapping), 'models': 2, 'required_basic_prompts_per_model': 80,
        'optional_complex_prompts_per_model': 40, 'fixed_source_clusters_per_model': 80,
        'repeated_reviewers_not_counted_as_independent_sources': True,
        'unfilled_or_omitted_complex_rows_do_not_block': True, 'complex_zero_quality_does_not_change_basic_gate': True,
        'basic_failures_not_compensated_by_complex': True, 'ties_half_do_not_satisfy_strict_preference_ci': True,
        'negative_cases_rejected': rejected, 'synthetic_or_authored_or_one_model_never_final_human_gate': True,
        'known_synthetic_declaration_overrides_omitted_call_flag': True,
        'actual_human_ratings_created_or_modified': False, 'actual_human_raters_added': 0,
        'statistics_source_sha256': sha(ROOT / 'training/m5_basic_review_statistics.py'),
        'source_sha256': sha(Path(__file__)), 'human_gate_passed': False, 'M5_complete': False})
    print({'passed': True, 'synthetic_only': True, 'human_gate_passed': False}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    main(p.parse_args().output.resolve())
