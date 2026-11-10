"""Fixed-endpoint length-course comparison; no automatic M5 or E admission."""
from training.m5_geometry_study import evidence
from training.m5_protocol import core_passed, no_core_regression, final_geometry_passed


def compare(mixed, short):
    a, b = mixed['metrics'], short['metrics']
    report, reference = mixed['report'], short['report']
    short_ok, short_reasons = no_core_regression(report, reference, stratum='short')
    full_ok, full_reasons = no_core_regression(report, reference)
    class_reasons = []
    for heading, values in report['conditions'].items():
        for group in ('turn', 'squat'):
            x = values['groups'].get(group, {}).get('noncollapsed_fraction')
            y = reference['conditions'][heading]['groups'].get(group, {}).get('noncollapsed_fraction')
            if x is None or y is None or x + .05 + 1e-12 < y:
                class_reasons.append(heading + '/' + group)
    geometry = {
        key: a[key] <= 1.1 * b[key] + tolerance
        for key, tolerance in {
            'mean_fk_inconsistency_m': 1e-4,
            'pooled_projection_node_p95_m': 1e-4,
            'mean_projection_node_p95_m': 1e-4,
            'max_penetration_m': 1e-4,
            'max_projection_node_error_m': 1e-4,
            'mean_penetration_over_5cm_fraction': 1e-4,
            'mean_rotation_threshold_exceed_fraction': 1e-5,
        }.items()
    }
    checks = {
        'all_finite': report['all_finite'],
        'valid_qpos': report['all_valid_qpos'],
        'short_core_both_headings': all(
            v['length_strata']['short']['core_amplitude_passed'] for v in report['conditions'].values()),
        'extended_core_both_headings': all(
            v['length_strata']['extended']['core_amplitude_passed'] for v in report['conditions'].values()),
        'full_core_both_headings': core_passed(report),
        'all_six_amplitude_both_headings': all(
            len(v['groups']) == 6 and all(g['amplitude_passed'] for g in v['groups'].values())
            for v in report['conditions'].values()),
        'short_core_no_regression_over_5pp': short_ok,
        'full_core_no_regression_over_5pp': full_ok,
        'turn_squat_no_regression_over_5pp': not class_reasons,
        'common_norm6_guard': a['norm6_validation'] <= 1.05 * b['norm6_validation'],
        'supervised_fk_mean_guard': a['supervised_fk_validation'] <= 1.1 * b['supervised_fk_validation'],
        'supervised_fk_any_t_guard': all(
            mixed['denoising']['geometry_by_timestep'][str(t)]['fk'] <=
            1.25 * short['denoising']['geometry_by_timestep'][str(t)]['fk']
            for t in (0, 249, 499, 749, 999)),
        'key_geometry_no_regression_over_10pct': all(geometry.values()),
    }
    return {
        'checks': checks, 'failed_checks': [k for k, v in checks.items() if not v],
        'length_direction_supported': all(checks.values()),
        'geometry_regression_checks': geometry,
        'short_regression_reasons': short_reasons, 'full_regression_reasons': full_reasons,
        'turn_squat_regression_reasons': class_reasons,
        'metrics': {'short': b, 'mixed': a},
        'amplitude_by_stratum': {
            label: {h: {s: {'core_amplitude_passed': values['core_amplitude_passed'],
                            'groups': {g: {k: row[k] for k in ('noncollapsed_fraction', 'median_amplitude_ratio', 'amplitude_passed')}
                                       for g, row in values['groups'].items()}}
                        for s, values in condition['length_strata'].items()}
                    for h, condition in v['report']['conditions'].items()}
            for label, v in (('short', short), ('mixed', mixed))},
        'mixed_relative_changes': {k: (a[k] / b[k] - 1) if b[k] else None for k in a},
        'full_final_absolute_geometry': {
            label: bool(final_geometry_passed(v['report']) and v['report']['pooled_projection_node_p95_m'] <= .03)
            for label, v in (('short', short), ('mixed', mixed))},
        'scope': 'Fixed 60k development comparison of available event/length distributions; not a pure length-only intervention.',
        'M5_complete': False, 'automatic_E_launch': False,
        'final_audit_model_scores_read': False,
    }
