#!/usr/bin/env python3
"""Matched basic-geometry G4 short/mixed curriculum after failed uniform length guards."""
import argparse
import ast
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'runs/m5_physical_20261103'
GEOMETRY = ROOT / 'runs/m5_geometry_20261104'
DEFAULT = ROOT / 'runs/m5_basic_geometry_curriculum_20261104'
NAMES = ('P0_short_joint_control', 'P1_mixed_joint')
PARTS = ('model', 'ema', 'optimizer', 'rngs', 'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng')
ORCHESTRATION = ('scripts/run_m5_basic_geometry_curriculum.py', 'scripts/assess_m5_basic_scope_v3.py', 'training/m5_basic_scope.py')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def process_live(record, run):
    try:
        args = Path(f"/proc/{record['pid']}/cmdline").read_bytes().split(b'\0')
        start = Path(f"/proc/{record['pid']}/stat").read_text().rsplit(')', 1)[1].split()[19]
        return (start == record['start_ticks'] and b'--run-dir' in args and
                Path(os.fsdecode(args[args.index(b'--run-dir') + 1])).resolve() == Path(run).resolve())
    except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
        return False


def validate(study):
    plan = read(study / 'protocol.json')
    for path, expected in plan['protected_sha256'].items():
        if sha(Path(path)) != expected:
            raise ValueError('Frozen study evidence/source changed: ' + path)
    for name in NAMES:
        entry = plan['runs'][name]
        for key in ('config', 'subset', 'parent'):
            if sha(entry[key]) != entry[key + '_sha256']:
                raise ValueError(name + ': frozen ' + key + ' changed')
        cfg = read(entry['config'])
        if not (cfg['training_stop_step'] == cfg['minimum_steps'] == cfg['max_steps'] == 60000 and
                cfg['lr_schedule_origin_step'] == 40000 and cfg['lr_schedule_end_step'] == 60000):
            raise ValueError('Wrong fixed endpoint/LR origin')
    return plan



def prepare(study):
    if study.exists(): raise FileExistsError('Preserve the existing basic geometry study')
    scope = ROOT / 'runs/m5_basic_scope_20261104'
    failure_path = ROOT / 'runs/m5_basic_curriculum_20261104/research_decision.json'; failure = read(failure_path)
    assert not failure['research_guards_passed_v3'] and not failure['checks']['basic_amplitude_no_regression_over5pp']
    assert not failure['basic_geometry_regression_checks']['max_penetration_m'] and not failure['basic_geometry_regression_checks']['mean_penetration_over_5cm_fraction']
    basis_path = scope / 'geometry50_assessments/geometry50_decision_v3.json'; basis = read(basis_path)
    assert basis['comparisons_vs_G0']['G4_all']['research_guards_passed_v3']
    short_source = GEOMETRY / 'G4_all'; prefix = read(short_source / 'COMPLETED.json'); cfg0 = read(short_source / 'config.json')
    common = read(GEOMETRY / 'protocol.json')['parent_sha256']; assert common == read(BASE / 'C_small_raw7/COMPLETED.json')['terminal_checkpoint_sha256']
    assert read(short_source / 'lineage.json')['parent_sha256'] == common
    assert cfg0['lr_schedule_origin_step'] == 40000 and cfg0['lr_schedule_end_step'] == cfg0['max_steps'] == 60000
    assert cfg0['warmup_steps'] == 500 and cfg0['training_stop_step'] == prefix['step'] == 50000
    coefficients = {k: cfg0.get(k, 0.) for k in ('consistency_weight','axis_weight','ground_weight','legality_weight')}
    assert coefficients == {'consistency_weight':.012261994051407363,'axis_weight':0.,'ground_weight':.18434478553265232,'legality_weight':.017114662545252495}
    assert cfg0['high_noise_fraction'] == .5 and cfg0['network_dropout'] == 0
    for file, expected in cfg0['launch_code_sha256'].items(): assert sha(ROOT / file) == expected
    trace_path = short_source / 'sample_trace.jsonl'; trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [r['step'] for r in trace] == list(range(40001,50001))
    metrics_path = short_source / 'metrics.jsonl'; metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert [r['step'] for r in metrics] == list(range(40050,50001,50))
    for row in metrics:
        local=row['step']-1-40000
        expected=2e-5*(local+1)/500 if local<500 else 2e-6+(2e-5-2e-6)*.5*(1+math.cos(math.pi*(local-500)/19500))
        assert abs(row['learning_rate']-expected)<1e-15
    manifests=[BASE/'data/D_short_3500.json',BASE/'data/D_mixed_small_3500.json'];data=[read(p) for p in manifests]
    assert len(data[0]['train'])==3404 and len(data[1]['train'])==3366
    assert data[0]['val']==data[1]['val'] and data[0]['monitor']==data[1]['monitor']
    policy_info=read(scope/'CURRENT_POLICY.json');policy=Path(policy_info['policy']);assert sha(policy)==policy_info['policy_sha256']
    view_path=Path(cfg0['training_view']);view=read(view_path)
    study.mkdir(parents=True);activation={'schema':'m5-constraint-activation-v1','passed':True,'engineering_fixture':False,'physical_input_fingerprint':view['input_fingerprint'],'training_view_fingerprint':view['fingerprint'],'runs':{}};runs={}
    parents=[short_source/'checkpoints/latest.pt',ROOT/'runs/m5_curriculum_factorial_20261104/parent_snapshot/checkpoint.pt']
    hashes=[prefix['terminal_checkpoint_sha256'],common]
    for i,name in enumerate(NAMES):
        assert sha(parents[i])==hashes[i]
        snapshot=study/'parent_snapshots'/('G4_step050000.pt' if i==0 else 'C_step040000.pt');snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(parents[i],snapshot);assert sha(snapshot)==hashes[i]
        subset=study/'data'/manifests[i].name;subset.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(manifests[i],subset)
        cfg=dict(cfg0,stage=name,stage_start_step=50000 if i==0 else 40000,training_stop_step=60000,minimum_steps=60000,constraint_activation_protocol=str(study/'activation.json'),subset_sha256=sha(subset),train_samples=len(data[i]['train']))
        diff={k for k in set(cfg)|set(cfg0) if cfg.get(k)!=cfg0.get(k)}
        assert diff <= {'stage','stage_start_step','training_stop_step','minimum_steps','constraint_activation_protocol','subset_sha256','train_samples'}
        config=study/'configs'/(name+'.json');write(config,cfg);activation['runs'][name]={'parent_sha256':hashes[i],'coefficients':coefficients}
        runs[name]={'config':str(config),'config_sha256':sha(config),'subset':str(subset),'subset_sha256':sha(subset),'parent':str(snapshot),'parent_sha256':hashes[i],'source_start_step':cfg['stage_start_step'],'target_step':60000,'new_updates':10000 if i==0 else 20000,'reused_prefix_updates':10000 if i==0 else 0}
    write(study/'activation.json',activation)
    files=[ROOT/f for f in (*cfg0['launch_code_sha256'],*ORCHESTRATION)]
    files += [failure_path,policy,scope/'cohorts.json',basis_path,scope/'geometry50_assessments/verification.json',GEOMETRY/'calibration/report.json',short_source/'config.json',short_source/'COMPLETED.json',short_source/'lineage.json',trace_path,metrics_path,GEOMETRY/'verification/startup_inheritance.json',GEOMETRY/'verification/verify_inheritance.py',BASE/'verification/curriculum_data_readiness.json',GEOMETRY/'verification/long_sequence_contract/report.json',view_path,Path(cfg0['input_dir'])/'READY',Path(cfg0['input_dir'])/'samples.parquet',Path(cfg0['input_dir'])/'features.npy',study/'activation.json']
    files += [BASE/f for f in view['stats_sha256']]
    ledger=read(BASE/'M5_requirement_ledger.json');assert ledger['registered_total_physical_update_budget']==515000
    plan={'schema':'m5-basic-geometry-matched-curriculum-v1','frozen_at_utc':now(),'first_wave':list(NAMES),'runs':runs,'target_step':60000,'maximum_new_updates':30000,'nominal_pair_updates':40000,'reused_prefix_updates':10000,'registered_total_physical_updates_before':515000,'registered_total_physical_updates':545000,'common_40k_parent_sha256':common,'protected_sha256':{str(p):sha(p) for p in files},'max_parallel':2,'engineering_fixture':False,'code_sha256':cfg0['launch_code_sha256'],'policy':str(policy),'policy_sha256':sha(policy),'scope':'Repairbasicarmretentionandground/posegeometryusingthealreadyimplementedG4meanobjective;notcomplexmotionqualityrepair.','prior_uniform_pair_failure':failure['failed_checks'],'G4_basic50_guards_passed':True,'new_loss_prototypes_adopted':False,'prefix_reuse':'ShortC40->G4_50reused,continues50->60withLRorigin40andno50kwarmuprestart;MixedstartsC40->60withsameG4/highnoisealgorithm.','data_interpretation':'Short3404vsmixed3366availableeventcurriculum,notidenticaleventscroppedtodifferentlengths.','automatic_E_launch':False,'final_recipe_frozen':False,'seed3102_quality_scores_used':False,'final_audit_model_scores_read':False,'M5_complete':False}
    write(study/'protocol.json',plan)
    for file in files:
        if file.suffix=='.py' and file.is_relative_to(ROOT):
            dest=study/'source_at_launch'/file.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
    validate(study)
    write(study/'verification/preflight.json',{'passed':True,'verified_at_utc':now(),'protocol_sha256':sha(study/'protocol.json'),'prefix10000steps_and200LRrecords_checked':True,'same_ordered_val_and_monitor':True,'same_G4_mean_loss_and_high_noise_algorithm':True,'lr_warmup_restarts_at50k':False,'new_optimizer_updates':0,'M5_complete':False})
    print({'prepared':str(study),'new_updates':30000},flush=True)


def launch(study):
    lock = (study / 'launch.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = validate(study)
    if (study / 'processes.json').exists():
        raise FileExistsError('Launch is one-shot; inspect existing handles instead of restarting')
    free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()[0])
    if free < 24576:
        raise RuntimeError('Insufficient measured GPU memory for the registered pair')
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    records = {}
    for name in NAMES:
        entry = plan['runs'][name]; run = study / name
        if run.exists():
            raise FileExistsError('Do not overwrite a run: ' + str(run))
        command = [str(ROOT / '.venv/bin/python'), '-B', '-u', str(ROOT / 'scripts/train_m5_constraints.py'),
                   '--config', entry['config'], '--subset', entry['subset'], '--run-dir', str(run), '--parent', entry['parent']]
        with (study / (name + '.log')).open('ab', buffering=0) as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ticks = Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        records[name] = {'pid': proc.pid, 'start_ticks': ticks, 'command': command,
                         'started_at_utc': now(), 'run_dir': str(run)}
        write(study / 'processes.json', records)
    write(study / 'study_status.json', {'phase': 'training', 'active': {n: v['pid'] for n, v in records.items()},
          'completed': [], 'maximum_additional_updates': 30000, 'updated_at_utc': now(), 'M5_complete': False})
    command = [str(ROOT / '.venv/bin/python'), '-B', '-u', str(Path(__file__).resolve()), 'assess',
               '--study-dir', str(study), '--wait']
    with (study / 'assessment_waiter.log').open('ab', buffering=0) as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(study / 'assessment_process.json', {'pid': proc.pid, 'command': command, 'started_at_utc': now()})
    print(json.dumps({'workers': records, 'assessment_waiter_pid': proc.pid}, indent=2), flush=True)


def verify_startup(study):
    import numpy as np
    import torch
    from training.m5 import M5Trainer
    plan = validate(study); records = read(study / 'processes.json')
    source = GEOMETRY / 'verification/verify_inheritance.py'
    assert sha(source) == 'a271055eff7f072db234a756d6ad06f0da785219dd16061a9efb66650c7df792'
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'fingerprint')
    namespace = {'hashlib': hashlib, 'np': np, 'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    fingerprint = namespace['fingerprint']; result = {}
    for name in NAMES:
        entry = plan['runs'][name]; run = study / name; cfg = read(entry['config'])
        assert process_live(records[name], run) or (run / 'COMPLETED.json').exists()
        parent = torch.load(entry['parent'], map_location='cpu', weights_only=False)
        initial = torch.load(run / 'checkpoints/initial_state.pt', map_location='cpu', weights_only=False)
        before = {k: fingerprint(parent[k]) for k in PARTS}; after = {k: fingerprint(initial[k]) for k in PARTS}
        assert before == after
        for k in ('step', 'ema_updates', 'samples_seen', 'text_dropped'):
            assert initial[k] == parent[k]
        assert initial['stage_updates'] == 0 and initial['optimizer_kind'] == 'adamw-fp32-v1'
        assert parent['step'] == entry['source_start_step']
        subset = read(entry['subset'])
        class Sampler:
            pass
        sampler = Sampler(); sampler.config = cfg; sampler.samples = subset['train']; sampler.pools = {}
        sampler.batch_rng = torch.Generator().set_state(parent['rngs']['batch_rng'])
        for i, row in enumerate(subset['train']):
            bucket = 0 if row['num_frames'] <= 90 else 1 if row['num_frames'] <= 120 else 2
            sampler.pools.setdefault(bucket, {}).setdefault(row['group'], []).append(i)
        expected = [{'step': parent['step'] + i + 1, 'indices': M5Trainer.draw_indices(sampler)} for i in range(100)]
        with (run / 'sample_trace.jsonl').open() as stream:
            actual = [json.loads(next(stream)) for _ in range(100)]
        assert actual == expected
        lineage = read(run / 'lineage.json'); assert lineage['parent_sha256'] == entry['parent_sha256'] and lineage['optimizer_rng_preserved']
        lines = (run / 'metrics.jsonl').read_text().splitlines(keepends=True)
        values = [json.loads(line) for line in lines if line.endswith('\n')]
        latest = values[-1]; assert latest['step'] >= entry['source_start_step'] + 100
        assert all(math.isfinite(latest[k]) and latest[k] > 0 for k in ('loss', 'root_grad_norm', 'body_grad_norm'))
        initial_ema = read(run / f"evaluations/denoising_{parent['step']:06d}.json")['ema']['macro_loss']
        reference_file = GEOMETRY / 'G4_all/evaluations' / f"denoising_{parent['step']:06d}.json"
        reference_ema = read(reference_file)['ema']['macro_loss']; assert abs(initial_ema - reference_ema) <= 1e-8
        local = latest['step'] - 1 - cfg['lr_schedule_origin_step']; warm = cfg['warmup_steps']
        lr = (cfg['learning_rate'] * (local + 1) / warm if local < warm else
              cfg['minimum_learning_rate'] + (cfg['learning_rate'] - cfg['minimum_learning_rate']) * .5 *
              (1 + math.cos(math.pi * (local - warm) / (60000 - 40000 - warm))))
        assert abs(latest['learning_rate'] - lr) <= 1e-15
        result[name] = {'parent_sha256': entry['parent_sha256'], 'parts_fingerprints': before,
                        'global_counters_preserved': True, 'first_100_draws_replayed': True,
                        'initial_ema_max_absolute_difference': abs(initial_ema - reference_ema),
                        'verified_step': latest['step'], 'verified_learning_rate': lr,
                        'initial_state_sha256': sha(run / 'checkpoints/initial_state.pt')}
        del parent, initial
    earlier = read(GEOMETRY / 'verification/startup_inheritance.json')
    assert earlier['passed'] and result['P1_mixed_joint']['parts_fingerprints'] == earlier['parent_parts_fingerprints']
    write(study / 'verification/startup_inheritance.json', {'passed': True, 'verified_at_utc': now(),
          'protocol_sha256': sha(study / 'protocol.json'), 'runs': result,
          'conceptual_common_40k_parent_sha256': plan['common_40k_parent_sha256'],
          'local_initial_steps_differ_because_short_prefix_is_reused': True,
          'new_optimizer_updates_by_verifier': 0, 'M5_complete': False})
    print(json.dumps({'passed': True, 'steps': {n: v['verified_step'] for n, v in result.items()}}, indent=2), flush=True)



def assess(study, wait=False):
    import time
    plan=validate(study);processes=read(study/'processes.json')
    while True:
        done=[n for n in NAMES if (study/n/'COMPLETED.json').exists()]
        missing=[n for n in NAMES if n not in done and not process_live(processes[n],study/n)]
        write(study/'study_status.json',{'phase':'requires_inspection' if missing else 'training_or_evaluating','completed':done,'active':{n:processes[n]['pid'] for n in NAMES if n not in done and n not in missing},'missing_live_handles':missing,'updated_at_utc':now(),'M5_complete':False})
        if missing:raise RuntimeError('Workerterminalbeforecompletion;noautomaticrestart:'+str(missing))
        if len(done)==2:break
        if not wait:print({'completed':done});return
        time.sleep(30)
    from scripts.assess_m5_basic_scope_v3 import assess as basic,compare
    scope=ROOT/'runs/m5_basic_scope_20261104';policy=read(plan['policy']);cohorts=read(scope/'cohorts.json')
    reports={n:basic(n,study/n,study/'basic_v3'/n,policy,cohorts) for n in NAMES}
    result=compare(reports[NAMES[1]],reports[NAMES[0]])
    result.update(schema='m5-basic-geometry-matched-curriculum-decision-v1',assessed_at_utc=now(),protocol_sha256=sha(study/'protocol.json'),policy_sha256=plan['policy_sha256'],actual_new_updates=sum(read(study/n/'COMPLETED.json')['updates_this_stage'] for n in NAMES),reused_short_prefix_updates=10000,automatic_E_launch=False,final_recipe_frozen=False,original_uniform_failure_preserved=True,M5_complete=False)
    assert result['actual_new_updates']==30000
    write(study/'research_decision.json',result);write(study/'study_status.json',{'phase':'basic_geometry_pair_budget_complete','completed':list(NAMES),'active':{},'updated_at_utc':now(),'M5_complete':False})
    print({'matched_basic_guards_passed':result['research_guards_passed_v3'],'failed':result['failed_checks']},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','launch','verify-startup','assess'));parser.add_argument('--study-dir',type=Path,default=DEFAULT);parser.add_argument('--wait',action='store_true');args=parser.parse_args();study=args.study_dir.resolve();study.relative_to(ROOT/'runs')
    if args.action=='assess':assess(study,args.wait)
    else:{'prepare':prepare,'launch':launch,'verify-startup':verify_startup}[args.action](study)


if __name__=='__main__':main()
