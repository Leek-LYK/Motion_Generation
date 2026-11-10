#!/usr/bin/env python3
"""Add user-authorized basicv3 assessments to frozen ongoing trials,without training."""
import argparse
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_m5_length_research import read, write, sha, now, process_live
from scripts.run_m5_amplitude_research import validate
from scripts.assess_m5_basic_scope_v3 import assess, compare, POLICY_FILE, SCOPE


def main(study, wait):
    plan = validate(study); processes = read(study / 'processes.json')
    policy_sha = sha(POLICY_FILE); policy = read(POLICY_FILE); cohorts = read(SCOPE / 'cohorts.json')
    cohort_sha = sha(SCOPE / 'cohorts.json'); code_sha = sha(ROOT / 'scripts/assess_m5_basic_scope_v3.py')
    output = SCOPE / 'current_trials_assessments'
    if (output / 'decision_v3.json').exists(): raise FileExistsError('Preserve existing v3 sidecar decision')
    while True:
        pending = [n for n in plan['first_wave'] if not (study / n / 'COMPLETED.json').exists()]
        if not pending: break
        missing = [n for n in pending if not process_live(processes[n], study / n)]
        write(SCOPE / 'current_trial_observer_status.json', {'phase': 'requires_inspection' if missing else 'waiting_for_fixed60k_completion',
            'pending': pending, 'missing_live_handles': missing, 'checked_at_utc': now(),
            'policy_sha256': policy_sha, 'training_changed': False, 'M5_complete': False})
        if missing: raise RuntimeError('Original workers terminal/missing before completion: ' + str(missing))
        if not wait: print({'pending': pending}); return
        time.sleep(30)
    validate(study)
    assert sha(POLICY_FILE) == policy_sha and sha(SCOPE / 'cohorts.json') == cohort_sha
    assert sha(ROOT / 'scripts/assess_m5_basic_scope_v3.py') == code_sha
    control = read(SCOPE / 'assessments_revision2/T3_cons_high/report.json')
    old_t2 = read(SCOPE / 'assessments_revision2/T2_cons_uniform/report.json')
    assert control['policy_sha256'] == old_t2['policy_sha256'] == policy_sha
    comparisons = {'T2_cons_uniform': compare(old_t2, control)}
    scores = {'T2_cons_uniform': old_t2['basic_metrics']['geometry_score_m']}
    checkpoints = {'T2_cons_uniform': old_t2['checkpoint_sha256']}
    for name in plan['first_wave']:
        report = assess(name, study / name, output / name, policy, cohorts)
        comparisons[name] = compare(report, control); scores[name] = report['basic_metrics']['geometry_score_m']
        checkpoints[name] = report['checkpoint_sha256']
    qualified = [name for name, value in comparisons.items() if value['research_guards_passed_v3']]
    selected = min(qualified, key=lambda n: scores[n]) if qualified else None
    write(output / 'decision_v3.json', {'schema': 'm5-basic-current-trials-sidecar-decision-v3',
        'assessed_at_utc': now(), 'policy_sha256': policy_sha, 'cohorts_sha256': cohort_sha,
        'original_trial_protocol_sha256': sha(study / 'protocol.json'), 'comparisons_vs_T3_basic': comparisons,
        'basic_geometry_scores': scores, 'checkpoint_sha256': checkpoints,
        'provisional_basic_recipe': selected, 'comparison_includes_previously_qualified_T2': True,
        'complex_quality_can_block_selection': False, 'original_v2_decisions_modified': False,
        'matched_short_control_still_required': True, 'final_recipe_frozen': False,
        'automatic_training_or_E_launch': False, 'new_optimizer_updates': 0, 'new_DDIM_samples': 0,
        'source_sha256': sha(Path(__file__)), 'assessment_source_sha256': code_sha,
        'seed3102_quality_scores_read': False, 'final_audit_model_scores_read': False, 'M5_complete': False})
    write(SCOPE / 'current_trial_observer_status.json', {'phase': 'basic_v3_assessment_complete',
          'assessed_at_utc': now(), 'provisional_basic_recipe': selected, 'M5_complete': False})
    print({'v3_basic_candidate': selected, 'M5_complete': False}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--study-dir', type=Path, required=True)
    p.add_argument('--wait', action='store_true'); a = p.parse_args(); main(a.study_dir.resolve(), a.wait)
