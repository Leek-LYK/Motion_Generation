#!/usr/bin/env python3
"""Release the frozen geometry preparer once real first-wave dependencies finish.

This process never resumes a missing training worker or retries a failed launch.
The existing preparer remains responsible for full endpoint, calibration and
activation checks. Waiting uses verified process arguments, not lock files.
"""
import argparse,fcntl,hashlib,json,os,shlex,signal,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text()) if path.exists() else {}


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def atomic(path,value):
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)


def live(pid,kind,target):
    if not pid:return False
    result=subprocess.run(['ps','-p',str(pid),'-o','args='],capture_output=True,text=True)
    script,flag={'worker':('scripts/train_m5_physical.py','--run-dir'),'controller':('scripts/run_m5_physical_study.py','--experiment')}[kind]
    try:
        args=shlex.split(result.stdout)
        return str(ROOT/script) in args and flag in args and Path(args[args.index(flag)+1]).resolve()==target.resolve()
    except (ValueError,IndexError):return False


def probe(experiment,protocol_sha256,is_live=live):
    path=experiment/'protocol/launch.json'
    if sha(path)!=protocol_sha256:raise ValueError('Original first-wave protocol changed')
    launch=read(path);status=read(experiment/'study_status.json');completed={};active={};pending=[]
    if status.get('phase') in ['failed','stopped']:raise RuntimeError('Original controller stopped or failed; inspect it before releasing a new study')
    controller_live=is_live(status.get('pid'),'controller',experiment)
    for name in launch['jobs']:
        run=experiment/name;done=read(run/'COMPLETED.json')
        if done:
            if done['step']!=40000:raise ValueError('Original endpoint budget differs: '+name)
            completed[name]=done;continue
        state=read(run/'status.json')
        if state.get('phase') in ['failed','stopped']:raise RuntimeError('Original worker stopped or failed: '+name)
        pid=state.get('pid') or status.get('active',{}).get(name)
        if is_live(pid,'worker',run):active[name]=pid;continue
        if controller_live and name in status.get('waiting',[]):pending.append(name);continue
        raise RuntimeError('Unfinished worker has no verified live handle: '+name)
    summary={'completed':list(completed),'active':active,'pending_launch':pending,'source_controller_live':controller_live}
    if len(completed)!=len(launch['jobs']):return dict(summary,phase='waiting_original_endpoints',ready=False)
    assessment=read(experiment/'first_wave_assessment.json')
    if not assessment:
        if not controller_live:raise RuntimeError('All workers completed but assessment/controller is missing; inspect before release')
        return dict(summary,phase='waiting_original_assessment',ready=False)
    if assessment.get('phase')!='first_wave_assessed' or assessment.get('protocol_sha256')!=protocol_sha256:raise ValueError('First-wave assessment has another identity')
    if set(assessment['metrics'])!=set(completed) or any(assessment['metrics'][name]['checkpoint_sha256']!=done['terminal_checkpoint_sha256'] for name,done in completed.items()):raise ValueError('Assessment/terminal markers disagree')
    return dict(summary,phase='dependencies_ready',ready=True)


def verify_hashes(files):
    for filename,expected in files.items():
        if sha(ROOT/filename)!=expected:raise ValueError('Registered source/evidence changed: '+filename)


def run_once(directory,command,environment,stop=None):
    # Claim before spawning: a process interruption cannot silently cause a
    # second preparation/calibration attempt on the next watcher invocation.
    if stop and stop[0]:raise InterruptedError('Stopped before preparation')
    with (directory/'activation_attempt.json').open('x') as stream:
        json.dump({'command':command,'claimed_at_utc':datetime.now(timezone.utc).isoformat()},stream,indent=2)
        stream.flush();os.fsync(stream.fileno())
    with (directory/'preparation.log').open('ab',buffering=0) as log:
        if stop and stop[0]:raise InterruptedError('Stopped before preparation process was spawned')
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
        atomic(directory/'preparer_process.json',{'pid':process.pid,'command':command})
        while process.poll() is None:
            if stop and stop[0]:
                os.killpg(process.pid,signal.SIGTERM);process.wait();raise InterruptedError('Preparation stopped; preserve its partial evidence')
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
    if process.returncode:raise RuntimeError('Geometry preparation failed; inspect preparation.log before any retry')
    return process.pid


def main():
    p=argparse.ArgumentParser();p.add_argument('--registration',type=Path,required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    directory=a.registration.resolve();directory.relative_to(ROOT/'runs');protocol_path=directory/'protocol.json';registration_sha256=sha(protocol_path)
    protocol=read(protocol_path);experiment=Path(protocol['experiment']);target=Path(protocol['study_dir'])
    experiment.resolve().relative_to(ROOT/'runs');target.resolve().relative_to(ROOT/'runs')
    verify_hashes(protocol['source_and_evidence_sha256'])
    if a.check_only:
        print(json.dumps(probe(experiment,protocol['original_launch_sha256']),indent=2),flush=True);return
    lock=(directory/'waiter.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);stop=[False]
    def halt(*_):stop[0]=True
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,halt)
    def status(phase,**values):
        atomic(directory/'status.json',{'phase':phase,'pid':os.getpid(),'updated_at_utc':datetime.now(timezone.utc).isoformat(),
          'registration_sha256':registration_sha256,'M5_complete':False,**values})
    try:
        if (directory/'activation_attempt.json').exists():raise FileExistsError('A preparation attempt is already recorded; inspect it, do not duplicate it')
        while not stop[0]:
            if sha(protocol_path)!=registration_sha256:raise ValueError('Dependency registration changed while waiting')
            verify_hashes(protocol['source_and_evidence_sha256']);state=probe(experiment,protocol['original_launch_sha256'])
            status(state['phase'],dependencies=state)
            if state['ready']:
                if (target/'activation.json').exists() or (target/'controller_process.json').exists():raise FileExistsError('Target study already prepared/registered; inspect its existing handle')
                command=[protocol['python_executable'],'-B','-u',str(ROOT/'scripts/prepare_m5_geometry_study.py'),'--experiment',str(experiment),'--study-dir',str(target),'--launch']
                environment=dict(os.environ)
                environment.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                status('preparing_geometry_study',command=command)
                run_once(directory,command,environment,stop)
                registered=read(target/'controller_process.json')
                if not registered.get('pid'):raise RuntimeError('Preparer exited without a registered geometry controller')
                status('geometry_controller_registered',controller=registered,study_protocol_sha256=sha(target/'protocol.json'));return
            # Short sleeps keep signal response bounded without polling GPU work.
            for _ in range(protocol.get('poll_seconds',30)):
                if stop[0]:break
                time.sleep(1)
        status('stopped_before_release')
    except InterruptedError:status('stopped_during_preparation')
    except Exception as error:status('requires_inspection',error=repr(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
