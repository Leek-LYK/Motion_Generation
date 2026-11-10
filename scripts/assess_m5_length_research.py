#!/usr/bin/env python3
"""Observe the registered workers and assess their fixed 60k endpoints once."""
import argparse
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m5_length_research import NAMES, now, read, write, sha, process_live, validate


def assess(study):
    from training.m5_length_study import evidence, compare
    plan = validate(study)
    entries = {name: evidence(study / name, 60000) for name in NAMES}
    short, mixed = (entries[n] for n in NAMES)
    for value in entries.values():
        report = value['report']
        if not (report['events'] == 300 and report['generated_clips'] == 1800 and
                report['seeds'] == [0, 42, 1234] and report['headings'] == [None, 0.] and
                report['ddim_steps'] == 100 and report['guidance'] == 2.):
            raise ValueError('Incomplete or changed fixed endpoint evaluation')
    keys = ('m2_fingerprint', 'physical_input_fingerprint', 'training_view_fingerprint',
            'text_config_fingerprint', 'stats_sha256', 'code_sha256')
    if any(short['report']['identity'][k] != mixed['report']['identity'][k] for k in keys):
        raise ValueError('Evaluation identities differ outside the intended training subset')
    result = compare(mixed, short)
    result.update(assessed_at_utc=now(), protocol_sha256=sha(study / 'protocol.json'),
                  checkpoints={n: v['checkpoint_sha256'] for n, v in entries.items()},
                  actual_new_updates=sum(read(study / n / 'COMPLETED.json')['updates_this_stage'] for n in NAMES),
                  reused_short_prefix_updates=10000)
    if result['actual_new_updates'] != plan['maximum_new_updates']:
        raise ValueError('Completed updates differ from registered budget')
    if (study / 'research_decision.json').exists():
        previous = read(study / 'research_decision.json')
        if any(previous[k] != result[k] for k in result if k != 'assessed_at_utc'):
            raise ValueError('Existing fixed-endpoint decision changed; preserve and inspect it')
        print('Fixed endpoint already assessed; original decision preserved.', flush=True)
        return
    write(study / 'research_decision.json', result)
    write(study / 'study_status.json', {'phase': 'research_budget_complete', 'completed': list(NAMES),
          'active': {}, 'updated_at_utc': now(), 'length_direction_supported': result['length_direction_supported'],
          'M5_complete': False, 'automatic_E_launch': False})
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--study-dir', type=Path, required=True); p.add_argument('--wait', action='store_true')
    args = p.parse_args(); study = args.study_dir.resolve(); study.relative_to(ROOT / 'runs'); validate(study)
    records = read(study / 'processes.json')
    while True:
        complete = [n for n in NAMES if (study / n / 'COMPLETED.json').exists()]
        active = {n: records[n]['pid'] for n in NAMES if n not in complete and process_live(records[n], study / n)}
        missing = [n for n in NAMES if n not in complete and n not in active]
        if len(complete) == len(NAMES):
            assess(study); return
        write(study / 'study_status.json', {'phase': 'requires_inspection' if missing else 'training_or_evaluating',
              'active': active, 'completed': complete, 'missing_worker_handles': missing,
              'updated_at_utc': now(), 'M5_complete': False, 'automatic_E_launch': False})
        if missing:
            raise RuntimeError('Worker handle missing before completion; no automatic restart: ' + ', '.join(missing))
        if not args.wait:
            print(json.dumps({'active': active, 'completed': complete, 'status': 'specific_worker_handles_confirmed_live'})); return
        time.sleep(30)


if __name__ == '__main__':
    main()
