"""Basicv3 two-human review statistics; complex feedback never enters the gate."""
from collections import defaultdict
import unicodedata
import numpy as np
from training.m5_basic_scope import BASIC_GROUPS

VERSION = 'm5-basic-source-human-statistics-v1'


def reviewer_key(value):
    return unicodedata.normalize('NFKC', value.strip()).casefold()


def basic_clustered_preference(records, seed=7373, draws=5000):
    clusters = sorted({r['cluster_id'] for r in records}); index = {name: i for i, name in enumerate(clusters)}
    numerator = np.zeros((len(clusters), len(BASIC_GROUPS))); denominator = np.zeros_like(numerator)
    for row in records:
        i = index[row['cluster_id']]; j = BASIC_GROUPS.index(row['group'])
        numerator[i, j] += row['preference']; denominator[i, j] += 1
    per_class = (denominator > 0).sum(0)
    if (per_class < 2).any(): raise ValueError('At least two independent basic clusters per class are needed for a CI')
    rng = np.random.default_rng(seed); samples = []; obtained = attempted = rejected = 0
    while obtained < draws:
        count = min(1000, draws - obtained)
        weights = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=count)
        d = weights @ denominator; good = (d > 0).all(1)
        attempted += count; rejected += int((~good).sum())
        if good.any():
            part = ((weights[good] @ numerator) / d[good]).mean(1); samples.append(part); obtained += len(part)
        if attempted > 10 * draws: raise ValueError('Insufficient cluster coverage for the four-class macro')
    values = np.concatenate(samples)[:draws]
    return {'mean': float((numerator.sum(0) / denominator.sum(0)).mean()),
        'ci95': np.quantile(values, [.025, .975]).tolist(), 'clusters': len(clusters),
        'per_basic_class_clusters': dict(zip(BASIC_GROUPS, map(int, per_class))), 'draws': draws,
        'seed': seed, 'empty_class_draws_rejected': rejected, 'attempted_draws': attempted,
        'estimand': 'Fourbasicclasses equallyweighted;ties0.5;sourceclusters jointlyresampled,reviewers fixed.'}


def analyze_basic_ratings(mapping, rater_rows, attestations, phase='source_final', synthetic=False):
    if phase not in ('source_final', 'authored_development'): raise ValueError('Unknown review phase')
    if len(rater_rows) != 2 or len(attestations) != 2: raise ValueError('Two separate independent reviewers are required')
    # Known fixture declarations cannot become human evidence if the caller
    # accidentally omits the synthetic argument.
    synthetic = bool(synthetic or any(a.get('synthetic_fixture') is True or a.get('synthetic') is True for a in attestations))
    if any(type(v.get('human_rating_required')) is not bool for v in mapping.values()):
        raise ValueError('Private mapping must explicitly bind required/optional difficulty flags')
    mandatory = {key for key, value in mapping.items() if value['human_rating_required']}
    if not mandatory or any(mapping[key]['group'] not in BASIC_GROUPS for key in mandatory):
        raise ValueError('Required trials must belong to the frozen basic groups')
    decoded = []; extended = []; reviewer_ids = []; disagreements = defaultdict(list)
    for rows, attest in zip(rater_rows, attestations):
        keys = [r['trial_id'] for r in rows]
        if len(set(keys)) != len(keys) or not set(keys) <= set(mapping) or not mandatory <= set(keys):
            raise ValueError('Required basic trials missing,duplicated or unknown')
        names = {reviewer_key(r.get('reviewer_id', '')) for r in rows if r['trial_id'] in mandatory}
        if len(names) != 1 or '' in names:
            return {'phase': 'awaiting_basic_reviewer_id', 'human_gate_passed': False, 'M5_complete': False}
        name = next(iter(names)); reviewer_ids.append(name)
        if reviewer_key(attest.get('reviewer_id', '')) != name or any(
            attest.get(k) is not True for k in ('human', 'independent_review', 'model_mapping_not_viewed')
        ):
            return {'phase': 'awaiting_basic_reviewer_attestation', 'human_gate_passed': False, 'M5_complete': False}
        for row in rows:
            trial = mapping[row['trial_id']]
            if not trial['human_rating_required']:
                extended.append({'trial_id': row['trial_id'], 'reviewer_id': name,
                                 'group': trial['group'], 'original_optional_feedback': dict(row)})
                continue
            if row.get('preference') not in ('A', 'B', 'tie') or any(row.get(k) not in ('0', '1') for k in ('A_main_correct', 'B_main_correct')):
                return {'phase': 'awaiting_required_basic_ratings', 'trial_id': row['trial_id'], 'human_gate_passed': False, 'M5_complete': False}
            if reviewer_key(row.get('reviewer_id', '')) != name: raise ValueError('Inconsistent reviewer ID within basic ratings')
            side = trial['small_side']; other = 'B' if side == 'A' else 'A'
            item = {'trial_id': row['trial_id'], 'run': trial['small_run'], 'group': trial['group'],
                'prompt_id': trial['prompt_id'], 'text_sha256': trial['text_sha256'], 'cluster_id': trial['cluster_id'],
                'reviewer_id': name, 'small_correct': int(row[side + '_main_correct']),
                'tiny_correct': int(row[other + '_main_correct']),
                'preference': .5 if row['preference'] == 'tie' else float(row['preference'] == side),
                'notes_verbatim': row.get('direction_speed_arms_notes', '')}
            decoded.append(item); disagreements[row['trial_id']].append(item)
    if len(set(reviewer_ids)) != 2: raise ValueError('Reviewer IDs must identify two distinct independent humans')
    results = {}
    for run in sorted({r['run'] for r in decoded}):
        part = [r for r in decoded if r['run'] == run]
        if {r['group'] for r in part} != set(BASIC_GROUPS): raise ValueError('All four basic classes are required')
        classes = {g: {'small_accuracy': float(np.mean([r['small_correct'] for r in part if r['group'] == g])),
            'tiny_accuracy': float(np.mean([r['tiny_correct'] for r in part if r['group'] == g])),
            'independent_prompts': len({r['text_sha256'] for r in part if r['group'] == g}),
            'source_clusters': len({r['cluster_id'] for r in part if r['group'] == g})} for g in BASIC_GROUPS}
        macro = float(np.mean([v['small_accuracy'] for v in classes.values()]))
        coverage = all(v['independent_prompts'] >= 20 and v['source_clusters'] >= 20 for v in classes.values())
        independent_clusters = len({r['cluster_id'] for r in part}) >= 80
        preference = basic_clustered_preference(part)
        checks = {'basic80_and20_per_class': coverage, 'independent_basic_source_clusters_ge80': independent_clusters,
            'basic_class_macro_accuracy_ge80pct': macro >= .8,
            'paired_basic_preference_ci_lower_gt50pct': preference['ci95'][0] > .5,
            'no_basic_class_semantic_regression_over5pp': all(v['small_accuracy'] + .05 + 1e-12 >= v['tiny_accuracy'] for v in classes.values())}
        results[run] = {'basic_class_macro_accuracy': macro, 'per_basic_class': classes,
            'paired_basic_preference': preference, 'checks': checks, 'numeric_basic_checks_passed': all(checks.values()),
            'optional_complex_feedback_rows': sum(r['original_optional_feedback']['trial_id'] in mapping and mapping[r['trial_id']]['small_run'] == run for r in extended)}
    if not results: raise ValueError('No Small model comparisons in the bound mapping')
    disagreement = {key: {'small_main': values[0]['small_correct'] != values[1]['small_correct'],
                         'preference': values[0]['preference'] != values[1]['preference']}
                    for key, values in disagreements.items()}
    complete_model_scope = len(results) == 2
    return {'schema': VERSION, 'phase': 'synthetic_verification' if synthetic else 'basic_human_ratings_analyzed',
        'reviewer_ids': reviewer_ids, 'results': results, 'basic_records': decoded,
        'extended_report_only_feedback': extended, 'disagreements': disagreement,
        'complex_quality_enters_gate': False, 'complete_jump_quality_enters_gate': False,
        'two_small_model_comparisons_present': complete_model_scope,
        'synthetic': synthetic, 'source_phase': phase,
        'human_gate_passed': phase == 'source_final' and not synthetic and complete_model_scope and all(v['numeric_basic_checks_passed'] for v in results.values()),
        'scope': 'Humancomponentonly;final80kcourse,numericalgeometryandfull657auditremainseparaterequirements.',
        'M5_complete': False}
