"""Two-rater semantic analysis with declared clusters, never fabricated ratings."""
import hashlib,unicodedata
from collections import defaultdict
import numpy as np
from training.m5_data import GROUPS,group_keys


def source_clusters(events):
    """Connect shared valid source, motion and validation-family keys; ignore missing keys."""
    parent=list(range(len(events)));owner={}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,event in enumerate(events):
        for key in group_keys(event):
            if key in owner:parent[find(i)]=find(owner[key])
            else:owner[key]=i
    members=defaultdict(list)
    for i,event in enumerate(events):members[find(i)].append(event['sample_id'])
    labels={key:hashlib.sha256(':'.join(sorted(ids)).encode()).hexdigest()[:20] for key,ids in members.items()}
    return {event['sample_id']:labels[find(i)] for i,event in enumerate(events)}


def clustered_preference(records,seed=7373,draws=5000):
    """One global source-cluster draw preserves cross-class/rater/seed dependence."""
    clusters=sorted({r['cluster_id'] for r in records});index={c:i for i,c in enumerate(clusters)}
    numerator=np.zeros((len(clusters),6));denominator=np.zeros_like(numerator)
    for row in records:
        i=index[row['cluster_id']];j=GROUPS.index(row['group']);numerator[i,j]+=row['preference'];denominator[i,j]+=1
    per_class_clusters=(denominator>0).sum(0)
    if (per_class_clusters<2).any():raise ValueError('At least two independent declared clusters per class are required')
    rng=np.random.default_rng(seed);samples=[];attempts=0;rejected=0
    while sum(len(x) for x in samples)<draws:
        count=min(1000,draws-sum(len(x) for x in samples));weights=rng.multinomial(len(clusters),np.full(len(clusters),1/len(clusters)),size=count)
        d=weights@denominator;good=(d>0).all(1);attempts+=count;rejected+=int((~good).sum())
        if good.any():samples.append(((weights[good]@numerator)/d[good]).mean(1))
        if attempts>draws*10:raise ValueError('Insufficient cluster coverage for the predeclared six-class estimand')
    values=np.concatenate(samples)[:draws]
    return {'mean':float((numerator.sum(0)/denominator.sum(0)).mean()),'ci95':np.quantile(values,[.025,.975]).tolist(),
      'clusters':len(clusters),'per_class_clusters':dict(zip(GROUPS,map(int,per_class_clusters))),'draws':draws,'seed':seed,
      'empty_class_draws_rejected':rejected,'attempted_draws':attempts,'estimand':'equal class macro preference; ties=0.5; reviewers fixed, not resampled'}


def analyze_ratings(mapping,rater_rows,attestations,phase='source_final',synthetic=False):
    if phase not in ('source_final','authored_development'):raise ValueError('Unknown review phase')
    if len(rater_rows)!=2 or len(attestations)!=2:raise ValueError('Exactly two independent reviewers are required')
    records=[];ids=[];disagreements=defaultdict(list)
    for rows,attest in zip(rater_rows,attestations):
        if len(rows)!=len(mapping) or {r['trial_id'] for r in rows}!=set(mapping):raise ValueError('Rating trials are missing or duplicated')
        names={r.get('reviewer_id','').strip() for r in rows}
        if names=={''}:return {'phase':'awaiting_human_ratings','human_gate_passed':False,'M5_complete':False}
        if len(names)!=1 or '' in names:raise ValueError('Each file must represent one reviewer')
        name=next(iter(names));ids.append(unicodedata.normalize('NFKC',name).casefold())
        if attest.get('reviewer_id','').strip()!=name or any(attest.get(k) is not True for k in ['human','independent_review','model_mapping_not_viewed']):
            return {'phase':'awaiting_reviewer_attestation','human_gate_passed':False,'M5_complete':False}
        for row in rows:
            trial=mapping[row['trial_id']];side=trial['small_side'];other='B' if side=='A' else 'A'
            required=[row.get('A_main_correct'),row.get('B_main_correct')]
            if row.get('preference') not in ('A','B','tie') or any(x not in ('0','1') for x in required):
                return {'phase':'awaiting_human_ratings','human_gate_passed':False,'M5_complete':False}
            jump=bool(trial.get('complete_jump_applicable'))
            if jump and any(row.get(k) not in ('0','1') for k in ['A_jump_complete','B_jump_complete']):
                return {'phase':'awaiting_human_ratings','human_gate_passed':False,'M5_complete':False}
            values={'trial_id':row['trial_id'],'run':trial['small_run'],'group':trial['group'],'prompt_id':trial['prompt_id'],
              'text_sha256':trial['text_sha256'],'cluster_id':trial['cluster_id'],'small_correct':int(row[side+'_main_correct']),
              'tiny_correct':int(row[other+'_main_correct']),'preference':.5 if row['preference']=='tie' else float(row['preference']==side),
              'jump_complete':int(row[side+'_jump_complete']) if jump else None}
            records.append(values);disagreements[row['trial_id']].append(values)
    if len(set(ids))!=2:raise ValueError('Reviewer IDs must represent two distinct independent humans')
    results={}
    for run in sorted({r['run'] for r in records}):
        part=[r for r in records if r['run']==run]
        if {r['group'] for r in part}!=set(GROUPS):raise ValueError('All six classes must be present')
        per_class={g:{'small_accuracy':float(np.mean([r['small_correct'] for r in part if r['group']==g])),
          'tiny_accuracy':float(np.mean([r['tiny_correct'] for r in part if r['group']==g])),
          'unique_prompts':len({r['text_sha256'] for r in part if r['group']==g})} for g in GROUPS}
        accuracy=float(np.mean([v['small_accuracy'] for v in per_class.values()]));pref=clustered_preference(part)
        jumps=[r['jump_complete'] for r in part if r['jump_complete'] is not None];jump=float(np.mean(jumps)) if jumps else None
        unique=len({r['text_sha256'] for r in part});coverage=unique>=120 and all(v['unique_prompts']>=20 for v in per_class.values())
        no_regression=all(per_class[g]['small_accuracy']+.05+1e-12>=per_class[g]['tiny_accuracy'] for g in GROUPS[:4])
        checks={'120_distinct_prompts':coverage,'six_class_macro_accuracy_ge80pct':accuracy>=.8,'paired_preference_ci_lower_gt50pct':pref['ci95'][0]>.5,
          'complete_jump_ge80pct_on_applicable_subset':jump is not None and jump>=.8,'no_core_semantic_regression_over5pp':no_regression}
        results[run]={'main_action_macro_accuracy':accuracy,'per_class':per_class,'paired_preference':pref,'complete_jump_fraction':jump,
          'applicable_jump_prompts':len({r['prompt_id'] for r in part if r['jump_complete'] is not None}),
          'checks':checks,'numeric_semantic_checks_passed':all(checks.values())}
    disagreement={trial:{'main':r[0]['small_correct']!=r[1]['small_correct'],'preference':r[0]['preference']!=r[1]['preference'],
      'jump':r[0]['jump_complete']!=r[1]['jump_complete']} for trial,r in disagreements.items()}
    return {'phase':'synthetic_verification' if synthetic else 'human_ratings_analyzed','reviewer_ids':ids,'results':results,
      'clustering':'source connected components' if phase=='source_final' else 'predeclared authored prompt families',
      'disagreement_policy':'Average the two independent judgments in the primary analysis; retain all disagreements without selecting favourable ratings.',
      'disagreements':disagreement,'synthetic':synthetic,'human_gate_passed':not synthetic and all(r['numeric_semantic_checks_passed'] for r in results.values()),'M5_complete':False}
