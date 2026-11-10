#!/usr/bin/env python3
"""Read-only status of the frozen first wave and separately registered M5 branches."""
import argparse,json,shlex,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text()) if path.exists() else {}


def live(pid,flag,target):
    if not pid:return False
    result=subprocess.run(['ps','-p',str(pid),'-o','args='],capture_output=True,text=True)
    try:
        args=shlex.split(result.stdout);return flag in args and Path(args[args.index(flag)+1]).resolve()==target.resolve()
    except (ValueError,IndexError):return False


def study_models(directory,plan,state):
    names=list(plan.get('first_wave',[]));active=state.get('active',{})
    for name in plan.get('conditional_combinations',{}):
        if name in active or name in state.get('completed',[]) or (directory/name/'run_config.json').exists():names.append(name)
    rows=[]
    for name in names:
        run=directory/name;status=read(run/'status.json');config=read(run/'config.json') or read(directory/'configs'/(name+'.json'));done=read(run/'COMPLETED.json')
        pid=status.get('pid') or active.get(name)
        process_live=live(pid,'--run-dir',run);start=config.get('stage_start_step',0);step=status.get('step',start)
        row={'run':name,'pid':pid,'process_live':process_live,'phase':status.get('phase','initializing' if process_live else 'not_launched'),
          'step':step,'stage_start_step':start,'updates_this_stage':max(0,step-start),
          'target_step':config.get('training_stop_step',config.get('max_steps')),'budget_complete':bool(done),'quality_accepted':False}
        if status.get('evaluation'):row['generation_progress']=read(Path(status['evaluation'])/'progress.json')
        if status.get('error'):row['error']=status['error']
        if not row['process_live'] and not done and (run/'run_config.json').exists():row['needs_inspection']=True
        rows.append(row)
    return rows


def dependent_studies(root):
    details=[]
    registration=root/'runtime/geometry_activation';registered=read(registration/'protocol.json')
    if registered:
        directory=Path(registered['study_dir']);waiting=read(registration/'status.json');waiter=read(registration/'process.json')
        controller=read(directory/'controller_process.json');state=read(directory/'study_status.json');plan=read(directory/'protocol.json');preparer=read(registration/'preparer_process.json')
        detail={'study':directory.name,'path':str(directory),'phase':state.get('phase',waiting.get('phase','registered')),
          'waiter_pid':waiter.get('pid'),'waiter_live':live(waiter.get('pid'),'--registration',registration),
          'preparer_pid':preparer.get('pid'),'preparer_live':live(preparer.get('pid'),'--study-dir',directory),
          'controller_pid':controller.get('pid'),'controller_live':live(controller.get('pid'),'--study-dir',directory),
          'maximum_additional_updates':registered['maximum_additional_new_updates'],'models':study_models(directory,plan,state) if plan else [],'M5_complete':False}
        if waiting.get('error'):detail['error']=waiting['error']
        if waiting.get('phase')=='requires_inspection' or waiting.get('phase','').startswith('waiting_') and not detail['waiter_live']:detail['needs_inspection']=True
        details.append(detail)
    for item in read(root/'runtime/dependent_studies.json').get('studies',[]):
        directory=Path(item['study_dir'])
        if any(row['path']==str(directory) for row in details):continue
        plan=read(directory/'protocol.json');state=read(directory/'study_status.json');waiter=read(directory/'assessment_process.json')
        detail={'study':directory.name,'path':str(directory),'phase':state.get('phase','registered'),
          'role':item.get('role'),'launch_mode':'independent_workers',
          'waiter_pid':waiter.get('pid'),'waiter_live':live(waiter.get('pid'),'--study-dir',directory),
          'controller_pid':None,'controller_live':False,'maximum_additional_updates':plan.get('maximum_new_updates'),
          'models':study_models(directory,plan,state),'M5_complete':False}
        if state.get('phase')=='requires_inspection':detail['needs_inspection']=True
        details.append(detail)
    return details


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,default=ROOT/'runs/m5_physical_20261103');p.add_argument('--json',action='store_true');a=p.parse_args();root=a.experiment.resolve()
    main=read(root/'protocol/launch.json');jobs=[(name,'first_wave',{}) for name in main.get('jobs',{})]
    for folder in sorted((root/'auxiliary_protocols').glob('*')):
        protocol=read(folder/'protocol.json');process=read(folder/'process.json')
        if protocol.get('run'):jobs.append((protocol['run'],folder.name,process))
    rows=[]
    for name,role,process in jobs:
        run=root/name;status=read(run/'status.json');config=read(run/'config.json') or read(root/'configs'/(name+'.json'));done=read(run/'COMPLETED.json')
        pid=status.get('pid',process.get('pid'));row={'run':name,'role':role,'pid':pid,'process_live':live(pid,'--run-dir',run),'phase':status.get('phase','initializing'),
          'step':status.get('step',0),'target_step':config.get('training_stop_step',config.get('max_steps')),'budget_complete':bool(done),'quality_accepted':False}
        evaluation=Path(status['evaluation']) if status.get('evaluation') else None
        if evaluation:row['generation_progress']=read(evaluation/'progress.json')
        if status.get('error'):row['error']=status['error']
        if not row['process_live'] and not done:row['needs_inspection']=True
        rows.append(row)
    baselines=[]
    for name,process in read(root/'baselines/processes.json').items():
        folder=root/'baselines'/name;baselines.append({'reference':name,'process_live':live(process.get('pid'),'--output',folder),
          'complete':(folder/'REFERENCE_COMPLETE.json').exists(),'progress':read(folder/'progress.json')})
    result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'first_wave_phase':read(root/'study_status.json').get('phase'),
      'models':rows,'dependent_studies':dependent_studies(root),'baselines':baselines,'M5_complete':False,'note':'Budget completion is not quality acceptance. This command neither restarts nor mutates jobs.'}
    if a.json:print(json.dumps(result,ensure_ascii=False,indent=2));return
    print(result['checked_at_utc'],'| first wave:',result['first_wave_phase'])
    for row in rows:
        print(f"{row['run']:<25} {row['step']:>6}/{str(row['target_step']):<6} {row['phase']:<23} live={row['process_live']} complete={row['budget_complete']}")
    for study in result['dependent_studies']:
        print('dependent study',study['study'],study['phase'],'waiter_live='+str(study['waiter_live']),'controller_live='+str(study['controller_live']))
        for row in study['models']:print(f"  {row['run']:<25} {row['step']:>6}/{str(row['target_step']):<6} {row['phase']:<23} live={row['process_live']} complete={row['budget_complete']}")
    for row in baselines:print('reference',row['reference'],'live='+str(row['process_live']),'complete='+str(row['complete']))
    print('M5 completion: unproven; TensorBoard http://localhost:6012')


if __name__=='__main__':main()
