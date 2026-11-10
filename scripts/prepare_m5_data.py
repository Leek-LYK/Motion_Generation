#!/usr/bin/env python3
"""Inventory and freeze source-separated M5 datasets; never modify M2."""
import argparse,json,sys
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from training.m3 import INPUTS,atomic_json,sha,validate_inputs
from training.m5_data import GROUPS,PATTERNS,EXCLUDE,classify,rank,group_keys


def inventory():
    meta=pd.read_parquet(ROOT/'data/processed/g1_text_motion_v1/motions.parquet',
      columns=['motion_id','source_group_id','validation_group_id','take_name','actor_uid','is_mirror'])
    frames={};summary={}
    for split in ('train','val','test_repetition','test_content'):
        frame=pd.read_parquet(INPUTS/f'{split}.parquet').merge(meta,on='motion_id',validate='many_to_one')
        frame=frame[frame.num_frames.between(60,180)&~frame.is_mirror].copy()
        labels={t:classify(t) for t in frame.text.unique()}
        frame['group']=frame.text.map(lambda t:labels[t][0]);frame['semantic_tags']=frame.text.map(lambda t:labels[t][1])
        frame=frame[frame.group.notna()];frame['length_bin']=frame.num_frames.map(lambda n:'short' if n<=90 else 'medium' if n<=120 else 'long')
        frames[split]=frame
        summary[split]={g:{b:{'events':len(q),'source_groups':q.source_group_id.nunique(),'captions':q.text_id.nunique()}
          for b in ('short','medium','long') for q in [frame[(frame.group==g)&(frame.length_bin==b)]]} for g in GROUPS}
    return frames,summary


def slim(row):
    keys=('sample_id','split','text_id','text','motion_id','source_sha256','source_group_id','validation_group_id',
          'take_name','actor_uid','group','length_bin','semantic_tags','start_frame','end_frame_exclusive','num_frames')
    return {k:row[k] for k in keys}


def pick(rows,count,used=None,caption_counts=None,cap=20,anchors=()):
    used=set() if used is None else used;caption_counts=Counter() if caption_counts is None else caption_counts
    buckets=defaultdict(list)
    for r in sorted(rows,key=lambda r:rank(r['sample_id'])):buckets[r['text_id']].append(r)
    chosen=[]
    def add(r):
        # validation_group_id is a cross-split leakage key, not a license to
        # collapse distinct performances inside the same training split.
        keys={k for k in group_keys(r) if k[0]!='validation_group_id'}
        if keys&used or caption_counts[r['text_id']]>=cap:return False
        chosen.append(r);used.update(keys);caption_counts[r['text_id']]+=1;return True
    for r in anchors:
        if not add(r):raise ValueError('Invalid anchor overlap')
    while len(chosen)<count:
        before=len(chosen)
        for text in sorted(buckets,key=rank):
            while buckets[text]:
                r=buckets[text].pop()
                if add(r):break
            if len(chosen)==count:break
        if len(chosen)==before:break
    return chosen


def allocate(total,weights):
    values={k:int(total*w) for k,w in weights.items()}
    for k in sorted(weights,key=lambda k:-(total*weights[k]-values[k]))[:total-sum(values.values())]:values[k]+=1
    return values


def training_selection(frame,count,mixed,anchors=()):
    quotas=allocate(count,{'walk':.25,'jog':.25,'jump':.15,'raise_arms':.10,'turn':.20,'squat':.05})
    cells={}
    for g,n in quotas.items():
        bins=allocate(n,{'short':.4,'medium':.3,'long':.3}) if mixed else {'short':n}
        for b,amount in bins.items():cells[g,b]=(frame[(frame.group==g)&(frame.length_bin==b)].to_dict('records'),amount)
    selected=[];used=set();texts=Counter()
    for r in anchors:
        keys={k for k in group_keys(r) if k[0]!='validation_group_id'}
        if keys&used:raise ValueError('Duplicate source anchors')
        used.update(keys);texts[r['text_id']]+=1
    for (g,b),(rows,n) in sorted(cells.items(),key=lambda kv:len(kv[1][0])/max(kv[1][1],1)):
        fixed=[r for r in anchors if r['group']==g and r['length_bin']==b]
        if len(fixed)>n:raise ValueError('Too many cell anchors')
        chosen=fixed+pick(rows,n-len(fixed),used,texts)
        if len(chosen)!=n:raise ValueError(f'{g}/{b} {len(chosen)}/{n}')
        selected.extend(chosen)
    return sorted(selected,key=lambda r:(GROUPS.index(r['group']),rank(r['sample_id'])))


def freeze(experiment,ready,frames,summary):
    out=experiment/'data'
    if (out/'READY.json').exists():raise FileExistsError('M5 data already frozen')
    prior=ROOT/'runs/m4_research_20261103';old_audit=json.loads((prior/'locked_audit.json').read_text())['test_repetition']
    consumed_ids={r['sample_id'] for r in old_audit};registry=[]
    for path in sorted((ROOT/'runs').glob('m[34]*/**/report.json')):
        report=json.loads(path.read_text())
        if isinstance(report,dict) and report.get('source_split','').startswith('test'):
            details=path.parent/'per_sample.json'
            if details.exists():consumed_ids.update(r['sample_id'] for r in json.loads(details.read_text()))
            registry.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
    consumed_keys=set().union(*(group_keys(r) for r in old_audit))
    for f in frames.values():
        for row in f[f.sample_id.isin(consumed_ids)].to_dict('records'):consumed_keys.update(group_keys(row))
    anchor=json.loads((ROOT/'runs/m4_20261102/data/BCD.json').read_text())
    old_unseen={r['text_id'] for r in anchor['val'] if r['cohort']=='unseen_text_same_class'}
    # Caption holdouts chosen using metadata only, minimizing scarce training loss.
    train_counts=Counter(frames['train'].text_id);anchor_texts={r['text_id'] for r in anchor['train']}
    heldout=set(old_unseen)
    for g in GROUPS:
        candidates=frames['val'][frames['val'].group==g]
        ids=set(candidates.text_id)-anchor_texts
        if ids:heldout.add(min(ids,key=lambda t:(train_counts[t],rank(t))))
    val=[];used=set();vc=Counter()
    for g in GROUPS:
        for label in ('short','extended'):
            f=frames['val'];f=f[(f.group==g)&((f.length_bin=='short') if label=='short' else (f.length_bin!='short'))]
            rows=f.to_dict('records');anchors=pick([r for r in rows if r['text_id'] in heldout],min(4,len(rows)))
            # Some anchors may share a performance with the previously selected cell.
            anchors=[r for r in anchors if not group_keys(r)&used]
            val+=pick(rows,32,used,vc,cap=16,anchors=anchors)
    valkeys=set().union(*(group_keys(r) for r in val))
    train=frames['train'];train=train[~train.text_id.isin(heldout)].copy()
    train=train[[not bool(group_keys(r)&valkeys) for r in train.to_dict('records')]]
    failures={};large=None
    for n in range(30000,999,-500):
        try:large=training_selection(train,n,True);break
        except ValueError as e:failures[f'mixed_{n}']=str(e)
    if large is None:raise ValueError('No useful source-diverse mixed training set')
    short=None;small=None
    for n in range(min(10000,len(large))//500*500,999,-500):
        try:
            small=training_selection(pd.DataFrame(large),n,True)
            short=training_selection(train,n,False,anchors=[r for r in small if r['length_bin']=='short']);break
        except ValueError as e:failures[f'short_{n}']=str(e);short=None
    if short is None or small is None:
        atomic_json(out/'selection_failure.json',{'failures':failures,'largest_mixed':len(large),'post_holdout_counts':{str(k):int(v) for k,v in train.groupby(['group','length_bin']).size().items()}})
        raise ValueError('Cannot form matched short/mixed data; see selection_failure.json')
    all_train_keys=set().union(*(group_keys(r) for r in short+large))
    audit=[];blocked=set(valkeys)|all_train_keys|consumed_keys;used=set();ac=Counter()
    for split in ('test_repetition','test_content'):
        for g in GROUPS:
            for label in ('short','extended'):
                f=frames[split];f=f[(f.group==g)&((f.length_bin=='short') if label=='short' else (f.length_bin!='short'))]
                rows=[r for r in f.to_dict('records') if r['sample_id'] not in consumed_ids and not group_keys(r)&blocked]
                audit+=pick(rows,32,used,ac,cap=32)
    if not all(any(r['group']==g for r in audit) for g in GROUPS):raise ValueError('Missing audit class')
    train_texts={r['text_id'] for r in short+large}
    for r in val:r['cohort']='unseen_text' if r['text_id'] not in train_texts else 'seen_text_new_motion'
    for r in audit:r['cohort']='locked_audit';r['caption_seen_in_any_m5_training']=r['text_id'] in train_texts
    validation=[dict(slim(r),cohort=r['cohort']) for r in val]
    monitor=[]
    for g in GROUPS:
        for label in ('short','extended'):
            subset=[r for r in validation if r['group']==g and (r['length_bin']=='short')==(label=='short')]
            monitor+=sorted(subset,key=lambda r:rank(r['sample_id'],'monitor'))[:8]
    common={'schema':'kimodo-m5-subset-v1','m2_fingerprint':ready['fingerprint'],'val':validation,'monitor':monitor,
      'human_semantic_review':'not a human review; reference agent review recorded separately',
      'heldout_text_ids':sorted(heldout),'selection_script_sha256':sha(__file__),'data_code_sha256':sha(ROOT/'training/m5_data.py')}
    manifest_paths={}
    for label,rows in [('short',short),('mixed_small',small),('mixed_large',large)]:
        name=f'D_{label}_{len(rows)}.json';path=out/name
        atomic_json(path,dict(common,train=[slim(r) for r in rows],actual_train_events=len(rows)))
        manifest_paths[label]=name
    anchor_copy=dict(anchor,train=[dict(r,split='train') for r in anchor['train']],val=[dict(r,split='val') for r in anchor['val']])
    anchor_copy['monitor']=anchor_copy['val'];atomic_json(out/'D_anchor_1024.json',anchor_copy)
    atomic_json(out/'locked_audit.json',{'schema':'m5-locked-audit-v1','events':[dict(slim(r),cohort='locked_audit',caption_seen_in_any_m5_training=r['caption_seen_in_any_m5_training']) for r in audit],
       'consumed_test_events_excluded':sorted(consumed_ids),'source_reports':registry,'model_scores_read':False})
    group_sets={label:set().union(*(group_keys(r) for r in rows)) for label,rows in [('train',short+large),('val',val),('audit',audit)]}
    for a,b in [('train','val'),('train','audit'),('val','audit')]:assert not group_sets[a]&group_sets[b],(a,b)
    assert not set(r['text_id'] for r in short+large)&heldout
    assert set(r['sample_id'] for r in small)<=set(r['sample_id'] for r in large)
    report={'passed':True,'m2_fingerprint':ready['fingerprint'],'manifests':manifest_paths,
      'actual_counts':{'short':len(short),'mixed_small':len(small),'mixed_large':len(large),'val':len(val),'monitor':len(monitor),'locked_audit':len(audit)},
      'requested_counts':{'short':10000,'mixed_small':10000,'mixed_large':30000,'val':384,'locked_audit':768},
      'adjustment_reason':'Six-class non-mirrored source-distinct inventory cannot meet planned arm/squat quotas; use smaller honestly named paired manifests. No duplicated filling.',
      'failed_larger_quota_attempts':failures,'groups':{label:dict(Counter(r['group'] for r in rows)) for label,rows in [('short',short),('mixed_small',small),('mixed_large',large),('val',val),('audit',audit)]},
      'source_group_overlaps':0,'heldout_caption_overlap':0,'one_event_per_source_group':True,'per_train_caption_cap':20,
      'manifest_sha256':{p.name:sha(p) for p in out.glob('*.json') if p.name!='READY.json'}}
    atomic_json(out/'READY.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['failed_larger_quota_attempts','manifest_sha256']},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--inventory-only',action='store_true');args=p.parse_args()
    args.experiment.mkdir(parents=True,exist_ok=True);ready=validate_inputs();frames,summary=inventory()
    report={'m2_fingerprint':ready['fingerprint'],'groups':list(GROUPS),'patterns':PATTERNS,'exclude_pattern':EXCLUDE,
      'counts':summary,'script_sha256':sha(__file__),'data_code_sha256':sha(ROOT/'training/m5_data.py')}
    atomic_json(args.experiment/'data/inventory.json',report)
    if args.inventory_only:print(json.dumps(summary,indent=2),flush=True)
    else:freeze(args.experiment,ready,frames,summary)


if __name__=='__main__':main()
