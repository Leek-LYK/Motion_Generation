#!/usr/bin/env python3
"""Physical-v2 novel-prompt generation and matched-noise real/zero/permuted controls."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from training.m3 import atomic_json,configure,make_model,batch_to,sha
from training.m5_data import GROUPS
from training.m5_evaluation import sample_keyed,amplitude
from training.m5_physical_losses import geometry
from training.m5_physical_evaluation import PhysicalGeometryMetrics
from training.video_archive import freeze_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--checkpoint',default='latest.pt');p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['generation','controls'],default='controls')
    p.add_argument('--seeds',type=int,nargs='+',default=[0,42,1234]);p.add_argument('--steps',type=int,default=100);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--engineering-fixture',action='store_true');a=p.parse_args();root=a.experiment.resolve();run=a.run_dir.resolve();out=a.output.resolve()
    if (out/'TEXT_PROBE_COMPLETE.json').exists() or (out/'report.json').exists():raise FileExistsError('Text probe already exists')
    if ('verification' in run.relative_to(ROOT/'runs').parts or a.steps!=100 or a.seeds!=[0,42,1234]) and not a.engineering_fixture:
        raise ValueError('Altered budgets or verification checkpoints require --engineering-fixture')
    cache=root/'review/authored_text_cache';complete=json.loads((cache/'COMPLETE.json').read_text());rows=json.loads((cache/'rows.json').read_text())
    if not complete['passed'] or sha(cache/'embeddings.npy')!=complete['embeddings_sha256'] or sha(cache/'rows.json')!=complete['rows_sha256']:raise ValueError('Frozen review embeddings changed')
    by_group={g:sorted([r for r in rows if r['group']==g],key=lambda r:hashlib.sha256((r['prompt_id']+':control-7377').encode()).hexdigest()) for g in GROUPS}
    count=2 if a.mode=='controls' else 20;rows=[by_group[g][i] for i in range(count) for g in GROUPS]
    source=run/'checkpoints'/a.checkpoint;checksum=sha(source);snapshot=freeze_model(run,source,checksum);state=torch.load(snapshot,map_location='cpu',weights_only=False)
    if state.get('schema') not in ('kimodo-m5-physical-checkpoint-v2','kimodo-m5-physical-atan2-checkpoint-v1','kimodo-m5-physical-constraints-checkpoint-v1'):raise ValueError('Physical-v2 input checkpoint required')
    if state['identity']['text_config_fingerprint']!=complete['parent_encoder_fingerprint']:raise ValueError('Text encoder identity mismatch')
    for path,expected in state['identity']['stats_sha256'].items():
        if sha(run/path)!=expected:raise ValueError('Run statistics changed')
    configure(state['config']['seed']);model=make_model(state['config']['model'],run/'stats/motion').cuda().eval();model.load_state_dict(state['ema'])
    vectors=np.load(cache/'embeddings.npy',mmap_mode='r');events=[]
    for row in rows:
        events.append({'x0':torch.zeros(90,417),'text_feat':torch.from_numpy(np.array(vectors[row['text_row']],copy=True)),
          'length':90,'first_heading_angle':torch.tensor(0.),'origin_xz':torch.zeros(2),'sample_id':'physical-review-shared-noise-v1','text':row['text'],'split':'review'})
    output_modes=['real','zero','shuffled'] if a.mode=='controls' else ['real'];batch_size=len(events) if a.mode=='controls' else a.batch_size
    out.mkdir(parents=True,exist_ok=True);metric=PhysicalGeometryMetrics(state['config'].get('metric_thresholds'));records=[];checks=[]
    with torch.inference_mode():
        for seed in a.seeds:
            for start in range(0,len(events),batch_size):
                part=events[start:start+batch_size];batch=batch_to(part,'cuda:0');outputs={mode:sample_keyed(model,batch,part,seed,a.steps,2.,text_mode=mode) for mode in output_modes}
                if a.mode=='controls':
                    z=float(abs(outputs['zero']-outputs['zero'][:1]).max());q=float(abs(outputs['shuffled']-outputs['real'].roll(1,0)).max())
                    assert z<=1e-4 and q<=1e-4,(z,q)
                    checks.append({'seed':seed,'zero_text_across_rows_max_difference':z,'shuffled_vs_real_permutation_max_difference':q,
                      'real_vs_zero_normalized_feature_rms':float((outputs['real']-outputs['zero']).square().mean().sqrt())})
                for mode,features in outputs.items():
                    folder=out/f'seed_{seed}'/mode;folder.mkdir(parents=True,exist_ok=True)
                    for i,event in enumerate(part):
                        row=rows[start+i];raw,fk,motion,fallback=geometry(model.motion_rep,features[i:i+1]);motion={k:v[0] for k,v in motion.items()}
                        info,q,unbounded=metric.measure(motion,row['group']);stem=folder/row['prompt_id']
                        np.savez_compressed(stem.with_suffix('.npz'),**{k:v.cpu().numpy() for k,v in motion.items()});np.save(stem.with_name(stem.name+'_features.npy'),features[i].cpu().numpy())
                        np.savetxt(stem.with_suffix('.csv'),q,delimiter=',');np.savetxt(stem.with_name(stem.name+'_unbounded.csv'),unbounded,delimiter=',')
                        condition_text=None if mode=='zero' else row['text'] if mode=='real' else rows[(start+i-1)%len(rows)]['text']
                        records.append({'sample_id':row['prompt_id'],'prompt_id':row['prompt_id'],'text_sha256':row['text_sha256'],'group':row['group'],'cluster_id':row['cluster_id'],
                          'cohort':'authored_novel_text','text':row['text'],'condition_text':condition_text,'text_mode':mode,'seed':seed,'condition':'heading0','heading':0.,'frames':90,
                          'artifact_prefix':str(stem.relative_to(out)),'generated_amplitude':amplitude(motion,model.motion_rep.skeleton,row['group']),
                          'fk_inconsistency_m':float(torch.linalg.vector_norm(fk-raw[...,5:107].reshape(1,90,34,3),dim=-1).mean()),
                          'rotation_fallback_fraction':float(fallback.float().mean()),**info})
                atomic_json(out/'progress.json',{'checkpoint_step':state['step'],'seed':seed,'completed_cases':len(records),'target_cases':len(events)*len(a.seeds)*len(output_modes)})
            print('physical novel text probe',a.mode,'seed',seed,'complete',flush=True)
    report={'schema':'m5-physical-text-probe-v2','checkpoint_sha256':checksum,'checkpoint_snapshot':str(snapshot),'step':state['step'],'weights':'ema',
      'mode':a.mode,'prompts':len(rows),'generated_clips':len(records),'frames':90,'heading':0.,'guidance':2.,'DDIM_steps':a.steps,'seeds':a.seeds,
      'shared_noise_across_prompts':True,'noise_key':'physical-review-shared-noise-v1','cache_fingerprint':complete['fingerprint'],'text_controls':checks,
      'engineering_fixture':a.engineering_fixture,'human_semantic_correctness':'Not evaluated; text sensitivity and finite qpos do not prove prompt correctness.',
      'source_sha256':{f:sha(ROOT/f) for f in ['scripts/probe_m5_physical_text.py','training/m5_physical_losses.py','training/m5_physical_evaluation.py']},'M5_complete':False}
    atomic_json(out/'per_sample.json',records);atomic_json(out/'report.json',report);atomic_json(out/'TEXT_PROBE_COMPLETE.json',{'checkpoint_sha256':checksum,'report_sha256':sha(out/'report.json'),'engineering_fixture':a.engineering_fixture})
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
