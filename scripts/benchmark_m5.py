#!/usr/bin/env python3
"""Bounded M5 update throughput profile; these weights are discarded."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,gc,json,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from training.m3 import configure,atomic_json,sha,validate_inputs
from training.m5 import M5Trainer
from training.m5_data import SubsetDataset


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--updates',type=int,default=200);args=p.parse_args()
    data=json.loads((args.experiment/'data/READY.json').read_text());manifest=json.loads((args.experiment/'data'/data['manifests']['mixed_large']).read_text())
    source=json.loads((ROOT/'runs/m4_research_20261103/R3_D_high_noise/config.json').read_text());ready=validate_inputs();results=[]
    for size,frames,micro in [('tiny',90,16),('small',90,16),('small',180,16),('small',180,32),('small',180,64)]:
        rows=sorted([r for r in manifest['train'] if r['num_frames']<=frames],key=lambda r:-r['num_frames'])[:64]
        samples=SubsetDataset(rows,cache_size=len(rows))
        cfg=dict(source,model={'latent_dim':256,'ff_size':1024,'num_layers':4,'num_heads':4} if size=='tiny' else {'latent_dim':512,'ff_size':2048,'num_layers':8,'num_heads':8},
          seed=3101,stage='benchmark',batch_size=64,microbatch=micro,precision='bfloat16',loss_objective='norm7',fk_weight=1.,
          stage_start_step=0,max_steps=args.updates+10,sampling='legacy',scheduler='constant',learning_rate=2e-5,high_noise_after=0,legacy_exact_update=False)
        configure(3101);torch.cuda.reset_peak_memory_stats();start_free,total_memory=torch.cuda.mem_get_info()
        trainer=M5Trainer(cfg,samples,{'m2_fingerprint':ready['fingerprint'],'purpose':'noncompetitive profile'})
        for _ in range(10):trainer.update()
        times=[];valid=[]
        for i in range(args.updates):
            torch.cuda.synchronize();start=time.perf_counter();row=trainer.update();torch.cuda.synchronize()
            times.append(time.perf_counter()-start);valid.append(row['valid_frames'])
            if (i+1)%50==0:
                atomic_json(args.experiment/'benchmark/status.json',{'model':size,'max_frames':frames,'microbatch':micro,'updates':i+1,'target':args.updates})
                print(size,frames,micro,i+1,round(statistics.median(times),4),flush=True)
        result={'model':size,'frames':frames,'observed_max_frames':max(r['num_frames'] for r in rows),'microbatch':micro,'effective_batch':64,
          'parameters':sum(p.numel() for p in trainer.model.parameters()),'median_seconds_per_update':statistics.median(times),
          'mean_seconds_per_update':statistics.mean(times),'p95_seconds_per_update':sorted(times)[int(.95*(len(times)-1))],
          'valid_frames_per_second':sum(valid)/sum(times),'samples_per_second':args.updates*64/sum(times),
          'peak_cuda_gib':torch.cuda.max_memory_allocated()/1024**3,'free_cuda_gib_before':start_free/1024**3,'measured_updates':args.updates,'warmup_updates':10,
          'final_loss':row['loss'],'shared_gpu_measurement':True}
        results.append(result);atomic_json(args.experiment/'benchmark/partial_results.json',results)
        del trainer,samples;gc.collect();torch.cuda.empty_cache()
    choices=[r for r in results if r['model']=='small' and r['frames']==180 and r['peak_cuda_gib']<.8*r['free_cuda_gib_before']]
    if not choices:raise RuntimeError('No configuration leaves safe memory headroom')
    chosen=min(choices,key=lambda r:r['median_seconds_per_update'])
    report={'passed':True,'results':results,'recommended_microbatch':chosen['microbatch'],'effective_batch':64,
      'total_noncompetitive_updates':sum(r['measured_updates']+r['warmup_updates'] for r in results),
      'code_sha256':{f:sha(ROOT/f) for f in ['training/m5.py','training/m5_losses.py','scripts/benchmark_m5.py']},
      'note':'Measured with another GPU task present; do not extrapolate M4 Tiny throughput to Small.'}
    atomic_json(args.experiment/'benchmark/report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
