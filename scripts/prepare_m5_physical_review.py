#!/usr/bin/env python3
"""Pre-register novel development prompts and 120 unique final-audit prompts."""
import argparse,hashlib,json,re,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from training.m3 import atomic_json,sha,INPUTS
from training.m5_data import group_keys
from training.m5_review_prompts import PROMPTS
from scripts.prepare_m5_physical_study import OBJECT_PATTERN


def rank(text,salt='physical-review-7371'):return hashlib.sha256((salt+':'+text).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();root=a.experiment.resolve();out=root/'review/protocol'
    if (out/'review_protocol.json').exists():raise FileExistsError('Review protocol already frozen')
    main=json.loads((root/'data/D_short_3500.json').read_text());view=json.loads((root/'data/READY.json').read_text())
    texts=pd.read_parquet(INPUTS/'texts.parquet');by_text={r['text']:r for r in texts.to_dict('records')}
    training_texts=set();training_keys=set()
    for name in ['D_short_3500','D_mixed_large_4500']:
        rows=json.loads((root/'data'/(name+'.json')).read_text())['train'];training_texts.update(r['text'] for r in rows)
        training_keys.update(set().union(*(group_keys(r) for r in rows)))
    authored=[]
    for group,items in PROMPTS.items():
        assert len(items)==20 and len(set(items))==20
        for i,text in enumerate(items):
            if text in training_texts:raise ValueError('Authored prompt already occurs in current training: '+text)
            cached=by_text.get(text);prompt_hash=hashlib.sha256(text.encode()).hexdigest()
            authored.append({'prompt_id':prompt_hash[:24],'text_sha256':prompt_hash,'text':text,'group':group,'frames':90,'heading':0.,
              'cluster_id':group+':authored_family_'+str(i//4),'cached_text_row':None if cached is None else int(cached['text_row']),
              'provenance':'agent-authored before this prompt battery is evaluated','ground_truth_motion':None})
    assert len(authored)==120 and len({r['text'] for r in authored})==120
    audit_path=ROOT/'runs/m5_20261103/data/locked_audit.json';audit=json.loads(audit_path.read_text())['events']
    quality={r['sample_id']:r for r in json.loads((root/'cache_audit.json').read_text())['records']};selected=[];unavailable={};used_text=set();used_source=set()
    for group in PROMPTS:
        pool=[r for r in audit if r['group']==group and not re.search(OBJECT_PATTERN,r['text'],re.I)
          and quality[r['sample_id']]['max_penetration_m']<=.10 and quality[r['sample_id']]['deep_frame_fraction']<=.10]
        options=sorted(pool,key=lambda r:rank(r['sample_id']));chosen=[]
        def add(row):
            source=group_keys(row)
            if row['text_id'] in used_text or source&used_source:return False
            assert not source&training_keys
            used_text.add(row['text_id']);used_source.update(source);chosen.append(row);return True
        for extended in [False,True]:
            for row in options:
                if (row['num_frames']>90)!=extended:continue
                if sum((r['num_frames']>90)==extended for r in chosen)>=5:break
                add(row)
        for row in options:
            if len(chosen)>=20:break
            add(row)
        if len(chosen)<20:unavailable[group]={'selected':len(chosen),'unique_available_texts':len({r['text_id'] for r in pool})}
        selected.extend(chosen)
    if unavailable:raise ValueError('Cannot satisfy final independent-prompt quota: '+json.dumps(unavailable))
    assert len(selected)==120 and len({r['text_id'] for r in selected})==120
    census={g:{'development_events':len([r for r in main['val'] if r['group']==g]),'development_unique_texts':len({r['text_id'] for r in main['val'] if r['group']==g})} for g in PROMPTS}
    atomic_json(out/'authored_development_prompts.json',{'schema':'m5-authored-review-prompts-v1','events':authored,'human_ratings':'not collected','labels':'intended prompt actions; no source-motion correctness claim'})
    atomic_json(out/'final_audit_prompt_selection.json',{'schema':'m5-final-semantic-selection-v2','events':selected,'audit_manifest_sha256':sha(audit_path),
      'model_scores_read':False,'audit_released':False,'selection_scope':'Metadata and existing source quality only; generation remains release-gated.'})
    protocol={'schema':'m5-physical-review-protocol-v2','training_view_fingerprint':view['fingerprint'],'development_census':census,
      'authored_development':{'prompts':120,'per_class':20,'frames':90,'heading':0.,'seeds':[0,42,1234],'blind_review_seed':42,
        'purpose':'Additional novel-text development battery; does not replace source-grounded final audit.','clustering':'Declared semantic prompt families; 30 families, not 120 independent source motions.'},
      'final_source_audit':{'prompts':120,'distinct_text_ids':120,'per_class':20,'blind_review_seed':23017,'geometry_seeds':[23017,23073,232026],
        'clustering':'Connected source/motion/valid validation-group keys; repeated seeds and reviewers remain in the same cluster.',
        'geometry_scope':'All 657 locked events remain required, not just these semantic-review prompts.'},
      'human_review':{'independent_raters':2,'six_class_macro_accuracy_min':.8,'paired_preference_clustered_95_ci_lower_gt':.5,
        'complete_jump_fraction_min':.8,'model_names_hidden':True,'agent_scores_can_satisfy_human_gate':False},
      'authored_prompts_sha256':sha(out/'authored_development_prompts.json'),'final_selection_sha256':sha(out/'final_audit_prompt_selection.json'),
      'prompt_code_sha256':sha(ROOT/'training/m5_review_prompts.py'),'preparation_script_sha256':sha(__file__),
      'model_scores_used_for_selection':False,'human_ratings_collected':False,'final_audit_model_scores_read':False,'M5_complete':False}
    atomic_json(out/'review_protocol.json',protocol);print(json.dumps(protocol,indent=2),flush=True)


if __name__=='__main__':main()
