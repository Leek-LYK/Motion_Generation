#!/usr/bin/env python3
"""Synthetic numerical fixtures only; no human rating files are written or credited."""
import copy,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.m3 import atomic_json,sha
from training.m5_data import GROUPS
from training.m5_review_statistics import source_clusters,clustered_preference,analyze_ratings


def main():
    events=[{'sample_id':'a','motion_id':'a','source_group_id':'x','validation_group_id':None},
      {'sample_id':'b','motion_id':'b','source_group_id':'y','validation_group_id':None},
      {'sample_id':'c','motion_id':'c','source_group_id':'z','validation_group_id':'v'},
      {'sample_id':'d','motion_id':'d','source_group_id':'q','validation_group_id':'v'}]
    clusters=source_clusters(events);assert clusters['a']!=clusters['b'] and clusters['c']==clusters['d']
    mapping={};rows=[]
    for group in GROUPS:
        for i in range(20):
            key=group+str(i);mapping[key]={'small_run':'synthetic_model','small_side':'A' if i%2==0 else 'B','group':group,'prompt_id':key,'text_sha256':key,
              'cluster_id':key,'complete_jump_applicable':group=='jump'}
            side=mapping[key]['small_side'];rows.append({'trial_id':key,'reviewer_id':'synthetic_rater_1','preference':side,'A_main_correct':'1','B_main_correct':'1',
              'A_jump_complete':'1' if group=='jump' else '', 'B_jump_complete':'1' if group=='jump' else ''})
    rows2=copy.deepcopy(rows)
    for r in rows2:r['reviewer_id']='synthetic_rater_2'
    attest=[{'reviewer_id':name,'human':True,'independent_review':True,'model_mapping_not_viewed':True} for name in ['synthetic_rater_1','synthetic_rater_2']]
    positive=analyze_ratings(mapping,[rows,rows2],attest,synthetic=True);assert positive['results']['synthetic_model']['numeric_semantic_checks_passed']
    assert not positive['human_gate_passed'] and not positive['M5_complete']
    negative=copy.deepcopy([rows,rows2])
    for rr in negative:
        for row in rr:row['preference']='B' if mapping[row['trial_id']]['small_side']=='A' else 'A'
    assert not analyze_ratings(mapping,negative,attest,synthetic=True)['results']['synthetic_model']['checks']['paired_preference_ci_lower_gt50pct']
    blank=copy.deepcopy(rows)
    for row in blank:row['reviewer_id']=''
    assert analyze_ratings(mapping,[blank,rows2],attest,synthetic=True)['phase']=='awaiting_human_ratings'
    basic=[{'group':g,'cluster_id':g+str(i),'preference':float(i%3>0)} for g in GROUPS for i in range(10)]
    a=clustered_preference(basic);b=clustered_preference(basic*8);assert a['ci95']==b['ci95'] and a['clusters']==b['clusters']
    result={'passed':True,'fixtures':'synthetic, excluded from all human acceptance','checks':{
      'missing_group_keys_do_not_merge_sources':True,'shared_validation_family_stays_clustered':True,'positive_and_negative_metric_paths':True,
      'blank_forms_remain_pending':True,'repeated_seeds_or_judgments_do_not_create_independent_clusters':True,'synthetic_ratings_cannot_pass_human_gate':True},
      'script_sha256':sha(__file__),'statistics_code_sha256':sha(ROOT/'training/m5_review_statistics.py')}
    out=ROOT/'runs/m5_physical_20261103/verification/review_statistics.json';atomic_json(out,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
