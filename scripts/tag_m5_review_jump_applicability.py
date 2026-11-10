#!/usr/bin/env python3
"""Freeze complete-jump applicability from source references before model scoring."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from kimodo.geometry import cont6d_to_matrix
from training.g1_physical import PhysicalG1Converter
from training.m3 import atomic_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();root=a.experiment.resolve();out=root/'review/protocol/jump_applicability.json'
    if out.exists():raise FileExistsError('Jump applicability already frozen')
    selection=root/'review/protocol/final_audit_prompt_selection.json';events=json.loads(selection.read_text())['events'];view=json.loads((root/'data/READY.json').read_text())
    source=Path(view['input_dir']);by_id={r['sample_id']:r for r in pd.read_parquet(source/'samples.parquet').to_dict('records')};features=np.load(source/'features.npy',mmap_mode='r')
    converter=PhysicalG1Converter();records=[]
    with torch.inference_mode():
        for event in events:
            if event['group']!='jump':continue
            r=by_id[event['sample_id']];n=int(r['num_frames']);start=int(r['frame_offset']);x=torch.from_numpy(np.array(features[start:start+n],copy=True))
            P=x[:,5:107].reshape(n,34,3)+torch.stack((x[:,0],torch.zeros(n),x[:,2]),-1)[:,None]
            G=cont6d_to_matrix(x[:,107:311].reshape(n,34,6));bottom=converter.foot_bottom({'posed_joints':P,'global_rot_mats':G}).numpy().min(-1)
            flight=bottom>.015;longest=current=0
            for f in flight:current=current+1 if f else 0;longest=max(longest,current)
            start_grounded=bool(bottom[:5].min()<=.015);end_grounded=bool(bottom[-5:].min()<=.015)
            records.append({'sample_id':event['sample_id'],'text_id':event['text_id'],'text':event['text'],'frames':n,'start_grounded':start_grounded,
              'end_grounded':end_grounded,'longest_two_foot_flight_frames':longest,'complete_jump_applicable':bool(start_grounded and end_grounded and longest>=2)})
    result={'schema':'m5-complete-jump-applicability-v2','source_final_selection_sha256':sha(selection),'physical_input_fingerprint':view['input_fingerprint'],
      'reference_rule':'Both-foot capsule bottoms >15mm for >=2 consecutive frames, with a near-ground frame in the first and last 5 source frames.',
      'source_jump_events':len(records),'applicable_source_jump_events':sum(r['complete_jump_applicable'] for r in records),'records':records,
      'authored_development_jump_prompts_applicable':20,'authored_reason':'All authored jump instructions ask for landing/completion; no source-reference claim.',
      'rating_rule':'Human complete-jump >=80% is evaluated only on pre-tagged applicable prompts. Geometric flight is an eligibility aid, not a human rating.',
      'model_outputs_used':False,'human_ratings_collected':False,'script_sha256':sha(__file__)}
    atomic_json(out,result);print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':main()
