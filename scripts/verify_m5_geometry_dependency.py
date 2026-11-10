#!/usr/bin/env python3
"""Control-flow fixtures for the dependency waiter; no model/optimizer work."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.await_m5_geometry_study import atomic,probe,run_once,sha,verify_hashes


def rejected(function,exception):
    try:function()
    except exception:return True
    raise AssertionError('Expected '+exception.__name__)


def main():
    out=ROOT/'runs/m5_physical_20261103/verification/geometry_dependency_waiter'
    if out.exists():raise FileExistsError('Preserve existing dependency fixtures')
    source=out/'fake_metadata_only_source';(source/'protocol').mkdir(parents=True)
    names=['fixture_A','fixture_B','fixture_C','fixture_D'];launch={'jobs':{name:{} for name in names}}
    atomic(source/'protocol/launch.json',launch);checksum=sha(source/'protocol/launch.json')
    state={'phase':'training_first_wave','pid':500,'active':{},'waiting':[]}
    alive=set()
    for i,name in enumerate(names,1):
        (source/name).mkdir();pid=500+i;state['active'][name]=pid;alive.add(pid)
        atomic(source/name/'status.json',{'pid':pid,'phase':'training','engineering_fixture':True})
    atomic(source/'study_status.json',state)
    is_live=lambda pid,kind,path:pid==500 or pid in alive
    waiting=probe(source,checksum,is_live);assert not waiting['ready'] and len(waiting['active'])==4
    alive.remove(501);assert rejected(lambda:probe(source,checksum,is_live),RuntimeError);alive.add(501)
    state['waiting']=[names[0]];state['active'].pop(names[0]);(source/names[0]/'status.json').unlink();atomic(source/'study_status.json',state)
    assert probe(source,checksum,is_live)['pending_launch']==[names[0]]
    for i,name in enumerate(names,1):atomic(source/name/'COMPLETED.json',{'step':40000,'terminal_checkpoint_sha256':f'{i:064x}','engineering_fixture':True})
    assert probe(source,checksum,is_live)['phase']=='waiting_original_assessment'
    assert rejected(lambda:probe(source,checksum,lambda *_:False),RuntimeError)
    assessment={'phase':'first_wave_assessed','protocol_sha256':checksum,'metrics':{name:{'checkpoint_sha256':f'{i:064x}'} for i,name in enumerate(names,1)}}
    atomic(source/'first_wave_assessment.json',assessment);assert probe(source,checksum,lambda *_:False)['ready']
    assessment['metrics'][names[0]]['checkpoint_sha256']='0'*64;atomic(source/'first_wave_assessment.json',assessment)
    assert rejected(lambda:probe(source,checksum,is_live),ValueError)
    dummy=out/'dummy_source.txt';dummy.write_text('fixture');relative=str(dummy.relative_to(ROOT));hashes={relative:sha(dummy)}
    verify_hashes(hashes);dummy.write_text('changed');assert rejected(lambda:verify_hashes(hashes),ValueError)
    positive=out/'harmless_positive';positive.mkdir();marker=positive/'invocations.txt'
    command=[sys.executable,'-B','-c','from pathlib import Path; import sys; Path(sys.argv[1]).open("a").write("fixture only\\n")',str(marker)]
    run_once(positive,command,dict(os.environ));assert marker.read_text().splitlines()==['fixture only']
    assert rejected(lambda:run_once(positive,command,dict(os.environ)),FileExistsError)
    failed=out/'harmless_failed';failed.mkdir();bad=[sys.executable,'-B','-c','raise SystemExit(7)']
    assert rejected(lambda:run_once(failed,bad,dict(os.environ)),RuntimeError)
    assert rejected(lambda:run_once(failed,bad,dict(os.environ)),FileExistsError)
    stopped=out/'harmless_stopped';stopped.mkdir()
    assert rejected(lambda:run_once(stopped,command,dict(os.environ),[True]),InterruptedError)
    assert not (stopped/'activation_attempt.json').exists()
    report={'passed':True,'scope':'Metadata/control-flow and harmless Python subprocess fixtures only; never an actual model endpoint or training-quality proof.',
      'new_model_or_optimizer_updates':0,'checks':{'unfinished_live_dependencies_wait':True,'missing_unfinished_handle_rejected':True,
      'not_yet_launched_job_waits_for_live_controller':True,'completed_workers_wait_for_real_assessment':True,'missing_assessment_and_controller_rejected':True,
      'assessment_terminal_sha_mismatch_rejected':True,'source_drift_rejected':True,'single_preparation_attempt':True,'failed_attempt_not_retried':True,'preparation_stop_before_spawn':True},
      'code_sha256':{f:sha(ROOT/f) for f in ['scripts/await_m5_geometry_study.py','scripts/verify_m5_geometry_dependency.py']},'M5_complete':False}
    atomic(out/'report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
