#!/usr/bin/env python3
"""M5-G: frozen development baselines, independent blind ratings, then locked audit.

No human scores are fabricated. A pending review is a real stage status, not a
successful experiment. Calling again after both completed rating CSVs resumes G.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,hashlib,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from kimodo.assets import skeleton_asset_path
from training.m3 import atomic_json,sha
from training.m5_protocol import final_geometry_passed
from training.video import render


def read(path):return json.loads(Path(path).read_text())


def rating_summary(folder,mapping):
    ratings=[];reviewers=set()
    for filename in ['rater_1.csv','rater_2.csv']:
        path=folder/filename
        with path.open(newline='') as handle:rows=list(csv.DictReader(handle))
        if len(rows)!=len(mapping) or {r['trial_id'] for r in rows}!=set(mapping):raise ValueError('Rating trial IDs differ')
        if any(not r['reviewer_id'].strip() or r['preference'] not in ('A','B','tie') or r['A_main_correct'] not in ('0','1') or r['B_main_correct'] not in ('0','1') for r in rows):return None
        names={r['reviewer_id'].strip() for r in rows}
        if len(names)!=1:raise ValueError('One independent reviewer per file')
        reviewers.update(names)
        for row in rows:
            trial=mapping[row['trial_id']];side=trial['small_side'];other='B' if side=='A' else 'A'
            if trial['group']=='jump' and row[f'{side}_jump_complete'] not in ('0','1'):return None
            ratings.append({'trial_id':row['trial_id'],'run':trial['small_run'],'group':trial['group'],'sample_id':trial['sample_id'],
              'small_correct':int(row[f'{side}_main_correct']),'tiny_correct':int(row[f'{other}_main_correct']),
              'small_jump_complete':int(row[f'{side}_jump_complete']) if trial['group']=='jump' else None,
              'preference':.5 if row['preference']=='tie' else float(row['preference']==side)})
    if len(reviewers)!=2:raise ValueError('Two distinct independent reviewer IDs are required')
    results={}
    for run in sorted({r['run'] for r in ratings}):
        chosen=[r for r in ratings if r['run']==run];groups=sorted({r['group'] for r in chosen})
        macro=float(np.mean([np.mean([r['small_correct'] for r in chosen if r['group']==g]) for g in groups]))
        by_event={s:np.mean([r['preference'] for r in chosen if r['sample_id']==s]) for s in {r['sample_id'] for r in chosen}}
        values=np.array(list(by_event.values()));rng=np.random.default_rng(7301)
        bootstrap=values[rng.integers(len(values),size=(2000,len(values)))].mean(1)
        ci=np.quantile(bootstrap,[.025,.975]);jump=float(np.mean([r['small_jump_complete'] for r in chosen if r['group']=='jump']))
        results[run]={'main_action_macro_accuracy':macro,'paired_preference_mean':float(values.mean()),'source_event_bootstrap_95_ci':ci.tolist(),
          'complete_jump_rating_fraction':jump,'independent_source_events':len(values),'passed':macro>=.8 and ci[0]>.5 and jump>=.8}
    return {'reviewer_ids':sorted(reviewers),'runs':results,'passed':all(v['passed'] for v in results.values()),
            'rating_file_sha256':{name:sha(folder/name) for name in ['rater_1.csv','rater_2.csv']}}


def prepare_blind(root,models,tiny,phase='development',seed=42):
    folder=root/f'review/{phase}_blind';folder.mkdir(parents=True,exist_ok=True);mapping_path=root/f'review/private_{phase}_model_mapping.json'
    if mapping_path.exists():return folder,read(mapping_path)
    tiny_rows=read(Path(tiny['report']).parent/'per_sample.json');tiny_by={r['sample_id']:r for r in tiny_rows if r['condition']=='heading0' and r['seed']==seed}
    mapping={};blank=[]
    for entry in models:
        rows=read(Path(entry['report']).parent/'per_sample.json')
        rows=[r for r in rows if r['condition']=='heading0' and r['seed']==seed]
        for group in ('walk','jog','jump','raise_arms','turn','squat'):
            selected=sorted([r for r in rows if r['group']==group],key=lambda r:hashlib.sha256((r['sample_id']+':blind-7301').encode()).hexdigest())[:20]
            if len(selected)<20:raise ValueError('Blind review requires twenty events per class')
            for row in selected:
                trial=hashlib.sha256((entry['run']+':'+row['sample_id']+':blind-v1').encode()).hexdigest()[:16]
                side='A' if int(trial,16)%2==0 else 'B';pair={side:(entry,row),('B' if side=='A' else 'A'):(tiny,tiny_by[row['sample_id']])}
                for label,(source,event) in pair.items():
                    qpath=Path(source['report']).parent/(event['artifact_prefix']+'.csv');q=np.loadtxt(qpath,delimiter=',')
                    out=folder/trial/label;out.mkdir(parents=True,exist_ok=True)
                    if not (out/'replay.mp4').exists():
                        metadata={'prompt':row['text'],'training_run_id':'Anonymous '+label,'checkpoint_step':0,'weights':'ema','seed':seed,
                                  'guidance':2,'postprocess_label':'bounded hinge projection'}
                        render(q,out/'replay.mp4',metadata,skeleton_asset_path('g1skel34','xml','g1.xml'))
                mapping[trial]={'small_run':entry['run'],'small_side':side,'group':group,'sample_id':row['sample_id'],
                               'small_checkpoint_sha256':entry['checkpoint_sha256'],'tiny_checkpoint_sha256':tiny['checkpoint_sha256']}
                blank.append({'trial_id':trial,'prompt':row['text'],'video_A':f'{trial}/A/replay.mp4','video_B':f'{trial}/B/replay.mp4',
                  'reviewer_id':'','preference':'','A_main_correct':'','B_main_correct':'','A_jump_complete':'','B_jump_complete':'',
                  'direction_speed_arms_notes':''})
        print('Prepared blind review',entry['run'],flush=True)
    blank=sorted(blank,key=lambda r:r['trial_id'])
    for name in ('rater_1.csv','rater_2.csv'):
        with (folder/name).open('x',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(blank[0]));writer.writeheader();writer.writerows(blank)
    (folder/'README.txt').write_text('Two independent human reviewers fill separate CSVs. Preference: A/B/tie. Main action: 0/1. For jump prompts also rate complete takeoff/flight/landing 0/1. Record direction, stop and arm failures in notes. Do not open the private mapping before finishing. These are bounded kinematic replays, not dynamic tracking.\n')
    atomic_json(mapping_path,mapping);return folder,mapping


def prepare_failure_review(root,models):
    folder=root/'review/audit_failures';folder.mkdir(parents=True,exist_ok=True);cases=[]
    for model in models:
        rows=read(Path(model['report']).parent/'per_sample.json')
        selected={}
        for group in ('walk','jog','jump','raise_arms','turn','squat'):
            part=[r for r in rows if r['group']==group]
            for metric in ('max_penetration_m','max_joint_speed_rad_s','projection_node_max_m'):
                row=max(part,key=lambda r:r[metric]);selected[(row['sample_id'],row['seed'],row['condition'])]=row
            row=min(part,key=lambda r:r['amplitude_ratio']);selected[(row['sample_id'],row['seed'],row['condition'])]=row
            row=min(part,key=lambda r:hashlib.sha256((r['sample_id']+str(r['seed'])+':random-failure-review').encode()).hexdigest())
            selected[(row['sample_id'],row['seed'],row['condition'])]=row
        for key,row in selected.items():
            case=hashlib.sha256((model['run']+str(key)).encode()).hexdigest()[:16];out=folder/case;out.mkdir(exist_ok=True)
            qpath=Path(model['report']).parent/(row['artifact_prefix']+'.csv');q=np.loadtxt(qpath,delimiter=',')
            if not (out/'replay.mp4').exists():
                metadata={'prompt':row['text'],'training_run_id':model['run'],'checkpoint_step':0,'weights':'ema','seed':row['seed'],
                          'guidance':2,'postprocess_label':'bounded hinge projection'}
                render(q,out/'replay.mp4',metadata,skeleton_asset_path('g1skel34','xml','g1.xml'))
            cases.append({'case_id':case,'run':model['run'],'sample_id':row['sample_id'],'video':str(out/'replay.mp4'),
                          'video_sha256':sha(out/'replay.mp4'),'reviewed':False,'notes':''})
    template=folder/'review_template.json'
    if not template.exists():atomic_json(template,{'reviewer_id':'','scope':'Kinematic artifacts only; no dynamic tracking claim','cases':cases})
    result=folder/'review_completed.json'
    if not result.exists():return False
    completed=read(result)
    if not completed.get('reviewer_id') or {r['case_id'] for r in completed.get('cases',[])}!={r['case_id'] for r in cases}:raise ValueError('Failure review case set differs')
    expected={r['case_id']:r for r in cases}
    return all(r.get('reviewed') is True and r.get('video_sha256')==expected[r['case_id']]['video_sha256'] for r in completed['cases'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);args=p.parse_args();root=args.experiment.resolve()
    def status(phase,**details):atomic_json(root/'G_status.json',{'phase':phase,'updated_at_utc':datetime.now(timezone.utc).isoformat(),'M5_complete':False,**details})
    def evaluate(checkpoint,manifest,out,seeds,release=None,official=False):
        if (out/'report.json').exists():
            report=read(out/'report.json')
            if report['checkpoint_sha256']!=sha(checkpoint) or report['manifest_sha256']!=sha(manifest):raise ValueError('Cached evaluation changed')
            return report
        command=[sys.executable,'-B','-u',str(ROOT/'scripts/evaluate_m5.py'),'--checkpoint',str(checkpoint),'--manifest',str(manifest),'--output',str(out),'--seeds',*map(str,seeds),
                 '--metrics-protocol',str(root/'protocol/launch.json')]
        if release:command+=['--audit-release',str(release)]
        if official:command+=['--official']
        subprocess.run(command,cwd=ROOT,check=True);return read(out/'report.json')
    decision=read(root/'decisions/F_repetition.json');models=[]
    for role,key in [('primary','primary_run'),('repeat','repeat_run'),('tiny','matched_tiny')]:
        run=root/decision[key];best=read(run/'best_generation.json');checkpoint=run/'checkpoints/best_generation.pt'
        if sha(checkpoint)!=best['checkpoint_sha256']:raise ValueError('Selected weights changed')
        models.append({'role':role,'run':run.name,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'report':best['report']})
    data=read(root/'data/READY.json');manifest=root/'data'/data['manifests']['mixed_large']
    status('development_baselines')
    legacy=ROOT/'runs/m4_research_20261103/R3_D_high_noise/checkpoints/best_val_ema.pt'
    official=ROOT.parent/'kimodo/checkpoints/Kimodo-G1-SEED-v1/model.safetensors'
    evaluate(legacy,manifest,root/'baselines/R3_development',[0,42,1234])
    evaluate(official,manifest,root/'baselines/official_development',[0,42,1234],official=True)
    status('preparing_blind_review');folder,mapping=prepare_blind(root,models[:2],models[2])
    summary=rating_summary(folder,mapping)
    if summary is None:status('awaiting_independent_human_semantic_review',review_directory=str(folder));return
    atomic_json(root/'review/semantic_summary.json',summary)
    if not summary['passed']:status('development_semantic_gate_failed',summary=summary);return
    reports=[read(m['report']) for m in models]
    if not all(final_geometry_passed(r) for r in reports[:2]):status('development_geometry_gate_failed');return
    # Two named improvements were frozen at launch; do not select favourable metrics after seeing results.
    for candidate in reports[:2]:
        for field in ['mean_fk_inconsistency_m','p95_clip_max_penetration_m']:
            a=np.mean([v[field] for v in candidate['groups'].values()]);b=np.mean([v[field] for v in reports[2]['groups'].values()])
            if a>b*.7:status('development_relative_improvement_gate_failed',metric=field,candidate=float(a),tiny=float(b));return
    release_path=root/'audit_release.json';audit=root/'data/locked_audit.json'
    if not release_path.exists():
        release={'schema':'m5-final-audit-release-v1','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
          'allowed_checkpoint_sha256':[m['checkpoint_sha256'] for m in models]+[sha(legacy),sha(official)],
          'manifest_sha256':sha(audit),'seeds':[23017,23073,232026],'guidance':2,'headings':[None,0],
          'evaluator_sha256':sha(ROOT/'scripts/evaluate_m5.py'),'metrics_sha256':sha(ROOT/'training/m5_evaluation.py'),
          'protocol_sha256':sha(root/'protocol/launch.json'),'semantic_summary_sha256':sha(root/'review/semantic_summary.json'),'models':models}
        atomic_json(release_path,release)
    status('final_locked_audit');results={}
    for entry in models:
        results[entry['role']]=evaluate(Path(entry['checkpoint']),audit,root/'final_audit'/entry['role'],[23017,23073,232026],release_path)
    results['R3']=evaluate(legacy,audit,root/'final_audit/R3',[23017,23073,232026],release_path)
    results['official']=evaluate(official,audit,root/'final_audit/official',[23017,23073,232026],release_path,official=True)
    numeric=all(final_geometry_passed(results[role]) for role in ('primary','repeat'))
    # Independent audit semantic/failure review remains explicit even if numerics pass.
    audit_models=[dict(m,report=str(root/'final_audit'/m['role']/'report.json')) for m in models]
    audit_review=None;failures_reviewed=False
    if numeric:
        status('preparing_final_audit_review')
        audit_folder,audit_mapping=prepare_blind(root,audit_models[:2],audit_models[2],phase='audit',seed=23017)
        audit_review=rating_summary(audit_folder,audit_mapping)
        if audit_review is not None:atomic_json(root/'review/audit_semantic_summary.json',audit_review)
        failures_reviewed=prepare_failure_review(root,audit_models[:2])
    relative=True
    for role in ('primary','repeat'):
        for field in ['mean_fk_inconsistency_m','p95_clip_max_penetration_m']:
            a=np.mean([v[field] for v in results[role]['groups'].values()]);b=np.mean([v[field] for v in results['tiny']['groups'].values()])
            relative=relative and a<=.7*b
    complete=bool(numeric and relative and audit_review and audit_review['passed'] and failures_reviewed)
    final={'phase':'final_audit_requires_semantic_and_failure_review' if numeric else 'final_audit_numeric_gate_failed',
      'M5_complete':complete,'M6_started':False,'audit_consumed':True,'numeric_gates_passed':numeric,
      'relative_improvement_passed':bool(relative),'audit_semantic_review':audit_review,'failures_reviewed':failures_reviewed,
      'audit_release_sha256':sha(release_path),'results':{k:{'report':str(root/'final_audit'/k/'report.json'),'sha256':sha(root/'final_audit'/k/'report.json')} for k in results},
      'remaining':'Review independent audit semantics and retained worst/random failures; do not claim M5 success from numeric gates alone.'}
    if complete:final.update(phase='M5_complete_M6_not_started',remaining=None)
    elif numeric and not relative:final['phase']='final_audit_relative_improvement_failed'
    elif audit_review and not audit_review['passed']:final['phase']='final_audit_semantic_gate_failed'
    atomic_json(root/'M5_exit_decision.json',final);status(final['phase'],decision=str(root/'M5_exit_decision.json'),M5_complete=complete)


if __name__=='__main__':main()
