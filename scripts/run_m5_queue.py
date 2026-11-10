#!/usr/bin/env python3
"""Sequential M5 main matrix with durable decisions and explicit failed/pending gates."""
import os,json,fcntl,signal,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.m3 import atomic_json,sha
from training.m5_protocol import configuration,TINY,SMALL,numerical_ok,core_passed,no_core_regression,fk_mean,final_geometry_passed


class GateFailure(Exception):pass


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,required=True);args=parser.parse_args()
    root=args.experiment.resolve();lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=json.loads((root/'protocol/launch.json').read_text());data=json.loads((root/'data/READY.json').read_text())
    for name,checksum in protocol['code_sha256'].items():
        if sha(ROOT/name)!=checksum:raise ValueError(f'Frozen code changed: {name}')
    for filename,checksum in protocol['data_manifest_sha256'].items():
        if sha(root/'data'/filename)!=checksum:raise ValueError(f'Frozen data changed: {filename}')
    micro=protocol['microbatch'];stop=[False];child=[None];completed=[];started=time.monotonic()
    def status(phase,**details):
        atomic_json(root/'queue_status.json',{'phase':phase,'pid':os.getpid(),'updated_at_utc':datetime.now(timezone.utc).isoformat(),
          'elapsed_s_this_process':time.monotonic()-started,'completed_runs':completed,'protocol_sha256':sha(root/'protocol/launch.json'),**details})
    def halt(*_):
        stop[0]=True
        if child[0] is not None and child[0].poll() is None:child[0].send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    def execute(command,log,phase,**details):
        if stop[0]:raise InterruptedError('Queue stopped')
        with log.open('ab',buffering=0) as handle:
            child[0]=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
            status(phase,child_pid=child[0].pid,command=command,log=str(log),**details)
            print(phase,details,child[0].pid,flush=True);code=child[0].wait()
        if stop[0]:raise InterruptedError('Queue stopped')
        if code:raise RuntimeError(f'Child exited {code}: {log}')
    def train(name,manifest,cfg,parent=None,initial=None):
        cfg=dict(cfg,train_samples=len(json.loads(manifest.read_text())['train']),metric_thresholds=protocol['rotation_thresholds_rad'])
        run=root/name;config_path=root/'configs'/f'{name}.json'
        if config_path.exists():
            if json.loads(config_path.read_text())!=cfg:raise ValueError(f'Configuration drift: {name}')
        else:atomic_json(config_path,cfg)
        command=[sys.executable,'-B','-u',str(ROOT/'scripts/train_m5.py'),'--config',str(config_path),'--subset',str(manifest),'--run-dir',str(run)]
        if (run/'run_config.json').exists():command+=['--resume']
        elif parent:command+=['--parent',str(parent)]
        elif initial:command+=['--initial-weights',str(initial)]
        marker=run/'COMPLETED.json'
        if not marker.exists():execute(command,root/f'{name}.log','training',run=name)
        completion=json.loads(marker.read_text())
        if json.loads((run/'config.json').read_text())!=cfg or completion['identity']['subset_sha256']!=sha(manifest):raise ValueError('Completed configuration/data drift')
        if completion['step']!=cfg['max_steps']:raise ValueError('Incomplete stage budget')
        if sha(run/'checkpoints/latest.pt')!=completion['terminal_checkpoint_sha256']:raise ValueError('Terminal checkpoint mutated')
        report=json.loads((run/'evaluations'/f'generation_full_{cfg["max_steps"]:06d}'/'report.json').read_text())
        if not numerical_ok(report):raise GateFailure(f'{name}: nonfinite or invalid exports')
        probe=run/'controlled_terminal/report.json'
        if not probe.exists():
            execute([sys.executable,'-B','-u',str(ROOT/'scripts/probe_m5_text.py'),'--run-dir',str(run)],root/f'{name}_text_controls.log','text_controls',run=name)
        elif json.loads(probe.read_text())['checkpoint_sha256']!=completion['terminal_checkpoint_sha256']:raise ValueError('Text probe weights changed')
        completed.append(name);return run,report
    def decision(name,value):
        path=root/'decisions'/f'{name}.json'
        if path.exists():
            if json.loads(path.read_text())!=value:raise ValueError(f'Decision drift: {name}')
        else:atomic_json(path,value)
    short=root/'data'/data['manifests']['short'];mixed=root/'data'/data['manifests']['mixed_small'];large=root/'data'/data['manifests']['mixed_large']
    anchor=root/'data/D_anchor_1024.json';parent=ROOT/'runs/m4_research_20261103/R3_D_high_noise/checkpoints/latest.pt'
    try:
        if sha(parent)!=protocol['R3_parent_sha256']:raise ValueError('Frozen R3 parent changed')
        status('starting_main_matrix')
        a0,r0=train('A0_tiny_control',anchor,configuration('A0',start=20000,end=30000,loss='norm6',anchor=True),parent)
        a1,r1=train('A1_tiny_fk',anchor,configuration('A1',start=20000,end=30000,anchor=True),parent)
        def fk_gate(candidate):
            good,reasons=no_core_regression(candidate,r0)
            return core_passed(candidate) and good and fk_mean(candidate)<=.9*fk_mean(r0),reasons
        passed,reasons=fk_gate(r1);weight=1.
        if not passed:
            a1,r1=train('A1b_tiny_fk025',anchor,configuration('A1b',start=20000,end=30000,anchor=True,fk_weight=.25),parent)
            passed,reasons=fk_gate(r1);weight=.25
        decision('A_fk',{'passed':passed,'chosen_run':a1.name,'fk_weight':weight,'control_fk_m':fk_mean(r0),'candidate_fk_m':fk_mean(r1),
                         'regression_reasons':reasons,'semantic_review':'not asserted by this numerical gate'})
        if not passed:raise GateFailure('A FK diagnostic did not improve geometry with preserved core generation; dependent B-F stages not launched')
        b0,tiny_report=train('B0_tiny_short',short,configuration('B0',model=TINY,microbatch=micro,fk_weight=weight))
        b1,small_report=train('B1_small_short',short,configuration('B1',model=SMALL,microbatch=micro,fk_weight=weight))
        c,raw_report=train('C_small_raw7',short,configuration('C',model=SMALL,microbatch=micro,loss='weighted_raw7',fk_weight=weight),initial=b1/'checkpoints/initial_state.pt')
        candidates=[]
        for run,report,loss in [(b1,small_report,'norm7'),(c,raw_report,'weighted_raw7')]:
            good,_=no_core_regression(report,tiny_report)
            if core_passed(report) and good:candidates.append((report['geometry_score_m'],run,report,loss))
        if not candidates:raise GateFailure('B1/C have no eligible Small candidate; numerical/cross-class generation gate not satisfied')
        _,chosen,chosen_report,loss=min(candidates,key=lambda x:(x[0],x[1].name))
        decision('C_loss',{'chosen_run':chosen.name,'loss_objective':loss,'fk_weight':weight,'rule':'core generation + no core regression vs matched Tiny, then lowest declared geometry score; provisional until semantic review'})
        d0,short_control=train('D0_short_control',short,configuration('D0',model=SMALL,start=40000,end=60000,loss=loss,microbatch=micro,fk_weight=weight),chosen/'checkpoints/step_040000.pt')
        d1,long_report=train('D1_mixed_curriculum',mixed,configuration('D1',model=SMALL,start=40000,end=60000,loss=loss,microbatch=micro,fk_weight=weight),chosen/'checkpoints/step_040000.pt')
        good,reasons=no_core_regression(long_report,short_control,stratum='short')
        long_ok=all(v['length_strata']['extended']['core_amplitude_passed'] for v in long_report['conditions'].values())
        if not(core_passed(long_report) and good and long_ok):raise GateFailure(f'D length curriculum failed generation/regression gate: {reasons}')
        decision('D_length',{'passed':True,'chosen_run':d1.name,'short_regression_reasons':reasons})
        e0,small_data_report=train('E0_data_control',mixed,configuration('E0',model=SMALL,start=60000,end=80000,loss=loss,microbatch=micro,fk_weight=weight),d1/'checkpoints/step_060000.pt')
        e1,large_data_report=train('E1_data_expand',large,configuration('E1',model=SMALL,start=60000,end=80000,loss=loss,microbatch=micro,fk_weight=weight),d1/'checkpoints/step_060000.pt')
        good,reasons=no_core_regression(large_data_report,small_data_report)
        gain=large_data_report['geometry_score_m']<small_data_report['geometry_score_m']
        decision('E_data',{'passed':bool(core_passed(large_data_report) and good and gain),'geometry_score_control_m':small_data_report['geometry_score_m'],
          'geometry_score_expand_m':large_data_report['geometry_score_m'],'regression_reasons':reasons,'actual_training_events':data['actual_counts']})
        if not(core_passed(large_data_report) and good and gain):raise GateFailure('E expansion did not demonstrate improvement; do not silently advance or extend budget')
        if loss=='norm7':tiny_parent=b0
        else:tiny_parent,_=train('F0_tiny_short_matched',short,configuration('F0_short',model=TINY,loss=loss,microbatch=micro,fk_weight=weight))
        ft1,_=train('F0_tiny_mixed',mixed,configuration('F0_mixed',model=TINY,start=40000,end=60000,loss=loss,microbatch=micro,fk_weight=weight),tiny_parent/'checkpoints/step_040000.pt')
        ft2,matched_report=train('F0_tiny_expanded',large,configuration('F0_expanded',model=TINY,start=60000,end=80000,loss=loss,microbatch=micro,fk_weight=weight),ft1/'checkpoints/step_060000.pt')
        fs0,_=train('F1_small_short_seed3102',short,configuration('F1_short',model=SMALL,seed=3102,loss=loss,microbatch=micro,fk_weight=weight))
        fs1,_=train('F1_small_mixed_seed3102',mixed,configuration('F1_mixed',model=SMALL,seed=3102,start=40000,end=60000,loss=loss,microbatch=micro,fk_weight=weight),fs0/'checkpoints/step_040000.pt')
        fs2,repeat_report=train('F1_small_expanded_seed3102',large,configuration('F1_expanded',model=SMALL,seed=3102,start=60000,end=80000,loss=loss,microbatch=micro,fk_weight=weight),fs1/'checkpoints/step_060000.pt')
        decision('F_repetition',{'primary_run':e1.name,'repeat_run':fs2.name,'matched_tiny':ft2.name,'complete_curriculum_repeated':True,
                               'primary_final_geometry_passed':final_geometry_passed(large_data_report),'repeat_final_geometry_passed':final_geometry_passed(repeat_report)})
        # Stage G is an implemented separate runner: preparation/official baseline,
        # then a hard semantic-review gate before any locked-audit score is read.
        command=[sys.executable,'-B','-u',str(ROOT/'scripts/finalize_m5.py'),'--experiment',str(root)]
        execute(command,root/'G_finalization.log','G_development_review',run='G')
        final=json.loads((root/'G_status.json').read_text());status(final['phase'],G=final)
    except InterruptedError:status('stopped')
    except GateFailure as error:
        status('quality_gate_failed',reason=str(error),M5_complete=False);print(str(error),flush=True)
    except Exception as error:
        status('failed',error=repr(error),M5_complete=False);raise


if __name__=='__main__':main()
