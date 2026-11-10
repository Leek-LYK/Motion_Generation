#!/usr/bin/env python3
"""Materialize the user-authorized basic-motion scope without reading model scores."""
import argparse
from collections import Counter
import hashlib
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from training.m5_basic_scope import classify, BASIC_GROUPS, RULES, FAST, VERSION
from training.m5_data import group_keys
from scripts.prepare_m5_physical_study import OBJECT_PATTERN
from scripts.run_m5_length_research import read, write, sha, now

BASE = ROOT / 'runs/m5_physical_20261103'
DEFAULT = ROOT / 'runs/m5_basic_scope_20261104'


def main(output):
    if output.exists(): raise FileExistsError('Preserve existing scope version')
    development_path = BASE / 'data/D_mixed_small_3500.json'
    audit_path = ROOT / 'runs/m5_20261103/data/locked_audit.json'
    development = read(development_path)['val']; audit = read(audit_path)['events']
    assert len(development) == 300 and len(audit) == 657
    def cohort(rows):
        classifications = {r['sample_id']: classify(r) for r in rows}
        counts = {group: dict(Counter(classifications[r['sample_id']]['difficulty'] for r in rows if r['group'] == group))
                  for group in ('walk', 'jog', 'jump', 'raise_arms', 'turn', 'squat')}
        return {'classifications': classifications, 'counts_by_group': counts,
                'basic_events': sum(v['difficulty'] == 'basic' for v in classifications.values()),
                'extended_events': sum(v['difficulty'] == 'extended' for v in classifications.values())}
    cohorts = {'schema': 'm5-basic-difficulty-cohorts-v3', 'created_at_utc': now(), 'classifier_version': VERSION,
               'development': cohort(development), 'locked_audit': cohort(audit),
               'classification_uses_only_text_and_group': True, 'generated_model_scores_read': False,
               'source_sha256': {str(development_path): sha(development_path), str(audit_path): sha(audit_path),
                                 str(ROOT / 'training/m5_basic_scope.py'): sha(ROOT / 'training/m5_basic_scope.py')},
               'original_data_or_scores_modified': False}
    assert cohorts['development']['basic_events'] + cohorts['development']['extended_events'] == 300
    assert cohorts['locked_audit']['basic_events'] + cohorts['locked_audit']['extended_events'] == 657
    # Preserve120 source prompts in a new selection:80 basic(20/class),40 extension.
    # The old120 selection remains immutable. This step consults original GTquality
    # and source metadata only, never generated scores from the locked audit.
    training_keys = set()
    for path in (BASE / 'data/D_short_3500.json', BASE / 'data/D_mixed_large_4500.json',
                 ROOT / 'runs/m5_amplitude_data_20261104/data/mixed_small_plus19.json'):
        for row in read(path)['train']: training_keys.update(group_keys(row))
    quality = {r['sample_id']: r for r in read(BASE / 'cache_audit.json')['records']}
    used_text = set(); used_source = set(); chosen = []; availability = {}
    all_groups = ('walk', 'jog', 'raise_arms', 'turn', 'jump', 'squat')
    for group in all_groups:
        pool = [r for r in audit if r['group'] == group and (group not in BASIC_GROUPS or classify(r)['difficulty'] == 'basic')
                and not re.search(OBJECT_PATTERN, r['text'], re.I)
                and quality[r['sample_id']]['max_penetration_m'] <= .1
                and quality[r['sample_id']]['deep_frame_fraction'] <= .1]
        availability[group] = {'source_events': len(pool), 'unique_texts': len({r['text_id'] for r in pool})}
        options = sorted(pool, key=lambda r: hashlib.sha256(('basic-v3-source-review-20261104:' + r['sample_id']).encode()).hexdigest())
        selected = []
        def add(row):
            keys = group_keys(row)
            if row['text_id'] in used_text or keys & used_source or keys & training_keys: return False
            used_text.add(row['text_id']); used_source.update(keys); selected.append(row); return True
        for extended in (False, True):
            for row in options:
                if (row['num_frames'] > 90) != extended: continue
                if sum((r['num_frames'] > 90) == extended for r in selected) >= 5: break
                add(row)
        for row in options:
            if len(selected) >= 20: break
            add(row)
        if len(selected) != 20: raise ValueError('Cannot meet new source-prompt quota without model-score selection: ' + group + ':' + str(len(selected)))
        chosen += [dict(r, **classify(r), human_rating_required=group in BASIC_GROUPS) for r in selected]
    assert len(chosen) == 120 and len({r['text_id'] for r in chosen}) == 120
    assert sum(r['human_rating_required'] for r in chosen) == 80
    output.mkdir(parents=True)
    write(output / 'cohorts.json', cohorts)
    write(output / 'review_source_selection_v3.json', {'schema': 'm5-basic-source-review-selection-v3',
        'events': chosen, 'basic_required_prompts': 80, 'extended_report_only_prompts': 40,
        'per_basic_group_required': 20, 'availability': availability, 'model_scores_read': False,
        'audit_released': False, 'selection': 'Fixed salted order, unique text and disjoint source keys, GTquality filters inherited fromv2.'})
    previous = BASE / 'protocol/M5_acceptance_protocol_v2.json'
    protocol = {'schema': 'm5-physical-acceptance-v3-basic', 'authorized_at_utc': now(),
        'user_instruction': '我认为可以降低一下考核标准，跳跃、快速奔跑、低姿态、大幅度舞蹈等动作属于复杂动作，不需要严格要求小型网络能够生成很好的。',
        'scope': 'M5text-to-motion basic-motion quality. Complex-motion quality is reported,not blocking. NoM6,dynamicsorcontrollertraining.',
        'previous_strict_protocol': str(previous), 'previous_strict_protocol_sha256': sha(previous),
        'previous_results_are_not_relabelled': True, 'historical_v2_classification': 'Retain every old failure and decision; v3 is a user-authorized change of task scope.',
        'basic_groups': list(BASIC_GROUPS), 'difficulty_rules': RULES, 'fast_motion_rule': FAST,
        'classification_inputs': ['text', 'group'], 'classifier_version': VERSION,
        'classifier_sha256': sha(ROOT / 'training/m5_basic_scope.py'), 'cohorts_sha256': sha(output / 'cohorts.json'),
        'review_selection_sha256': sha(output / 'review_source_selection_v3.json'),
        'unchanged_research_requirements': read(previous)['final_requirements']['full_matrix'],
        'complete_repetition': read(previous)['final_requirements']['complete_repetition'],
        'numerical_validity_all_prompts': 'Finite generation and valid boundedqpos still required for all basic/extended prompts.',
        'basic_quality_gates': {'amplitude': {'reference_minimum': {'walk': .3, 'jog': .3, 'raise_arms': .05, 'turn': .2},
            'minimum_generated_reference_ratio': .35, 'noncollapsed_fraction_min': .75, 'median_ratio_range': [.5, 2.],
            'headings': ['native', 'heading0'], 'length_strata': ['short', 'extended'], 'missing_required_group_or_stratum_passes': False},
            'ground': {'p95_clip_max_penetration_m_max': .05, 'absolute_max_penetration_m_max': .1, 'mean_deep_frame_fraction_max': .01},
            'projection': {'per_class_mean_clip_p95_m_max': .03, 'absolute_max_node_error_m_max': .1, 'pooled_node_p95_m_max': .03},
            'continuity': {'per_class_reference_calibrated_rotation_exceed_fraction_max': .001},
            'relative_geometry': 'On the same frozen basiccohort,meanFKself-inconsistency and poolednodeprojectionP95 each improve>=30%vs matchedTiny; otherkeymetrics no>10%regression withv2nearzerotolerances.',
            'human': {'independent_raters': 2, 'basic_prompts': 80, 'per_basic_group': 20, 'basic_class_macro_accuracy_min': .8,
                      'paired_basic_preference_cluster_bootstrap_ci95_lower_gt': .5, 'hidden_model_identities': True}},
        'extended_quality': {'jump': 'report_only', 'fast_locomotion': 'report_only', 'low_posture': 'report_only',
            'dance_acrobatics': 'report_only', 'large_rotation': 'report_only', 'complete_jump_fraction': 'report_only',
            'complex_quality_can_block_M5': False, 'complex_gain_can_compensate_for_basic_regression': False},
        'audit': {'all657_events_still_generated': True, 'basic_hard_quality_scope': 'Frozen384event basiccohort; all other events remain in extended reports.',
                  'generation_seeds': [23017, 23073, 232026], 'model_score_selection_allowed': False},
        'human_review_scope': '120sourceprompts kept:80 basicmandatoryratings plus40extendedreport-only. Existing one-human12prompt40kpilot is developmentfeedback,not a final review.',
        'existing_trials': 'Complete the already registered fixed budgets/configs. Keep theirv2assessors/results and add separatev3sidecarassessment. No newcomplex-only repair experiments are required forM5admission.',
        'references_and_archives': read(previous)['final_requirements']['references'] + ' ' + read(previous)['final_requirements']['artifacts'],
        'current_registered_formal_update_budget_unchanged': 495000, 'new_optimizer_updates': 0,
        'new_DDIM_samples': 0, 'final_audit_model_scores_read': False, 'M5_complete': False}
    write(output / 'M5_acceptance_protocol_v3_basic.json', protocol)
    write(output / 'scope_verification.json', {'passed': True, 'verified_at_utc': now(),
        'development_basic': cohorts['development']['basic_events'], 'development_extended': cohorts['development']['extended_events'],
        'audit_basic': cohorts['locked_audit']['basic_events'], 'audit_extended': cohorts['locked_audit']['extended_events'],
        'source_review_prompts': 120, 'required_basic_review_prompts': 80, 'unique_source_texts': len(used_text),
        'no_train_source_overlap': True, 'legacy_protocol_unchanged': sha(previous) == protocol['previous_strict_protocol_sha256'],
        'generated_scores_read_for_difficulty': False, 'source_sha256': sha(Path(__file__)), 'M5_complete': False})
    print({'output': str(output), 'development_basic': cohorts['development']['basic_events'],
           'audit_basic': cohorts['locked_audit']['basic_events'], 'human_basic_prompts': 80}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output-dir', type=Path, default=DEFAULT)
    main(p.parse_args().output_dir.resolve())
