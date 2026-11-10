#!/usr/bin/env python3
"""Serial Tiny/Small capacity and throughput profile with real short/long inputs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,gc,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from training.m3 import configure,atomic_json,sha
from training.m5_physical import PhysicalTrainer
from training.m5_physical_data import PhysicalSubsetDataset


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--steps',type=int,default=200)
    p.add_argument('--microbatch',type=int,default=16);p.add_argument('--report-name',default='report.json');a=p.parse_args();root=a.experiment.resolve()
    view=json.loads((root/'data/READY.json').read_text());manifest=json.loads((root/'data/D_mixed_large_4500.json').read_text());results=[]
    configs={'tiny':{'latent_dim':256,'ff_size':1024,'num_layers':4,'num_heads':4},'small':{'latent_dim':512,'ff_size':2048,'num_layers':8,'num_heads':8}}
    for name,architecture in configs.items():
        for length in (90,180):
            entries=sorted([r for r in manifest['train'] if r['num_frames']<=length],key=lambda r:r['num_frames'],reverse=True)[:16]
            samples=PhysicalSubsetDataset(entries,view['input_dir'],root/'stats/motion',cache_size=len(entries))
            cfg={'model':architecture,'seed':3101,'batch_size':64,'microbatch':a.microbatch,'learning_rate':2e-5,'minimum_learning_rate':2e-6,'scheduler':'constant',
              'stage_start_step':0,'max_steps':1000,'minimum_steps':1000,'grad_clip':1.,'ema_every':10,'ema_decay':.995,'precision':'bfloat16',
              'text_dropout':.1,'heading_augmentation':True,'sampling':'legacy','high_noise_fraction':.5,'high_noise_start':800,'high_noise_after':0,'loss_objective':'norm7'}
            configure(3101);trainer=PhysicalTrainer(cfg,samples,{'benchmark':True},root/'stats/motion');torch.cuda.reset_peak_memory_stats()
            for _ in range(10):trainer.update()
            torch.cuda.synchronize();begin=time.monotonic();frames=0;losses=[]
            for i in range(a.steps):
                result=trainer.update();frames+=result['valid_frames'];losses.append(result['loss'])
                if (i+1)%50==0:print(f'profile {name} <= {length} frames: {i+1}/{a.steps}',flush=True)
            torch.cuda.synchronize();elapsed=time.monotonic()-begin
            record={'model':name,'parameters':sum(p.numel() for p in trainer.model.parameters()),'max_frames':max(r['num_frames'] for r in entries),
              'batch_size':64,'microbatch':a.microbatch,'precision':'bfloat16','steps':a.steps,'seconds_per_update':elapsed/a.steps,
              'valid_frames_per_second':frames/elapsed,'peak_cuda_gib':torch.cuda.max_memory_allocated()/2**30,'last_loss':losses[-1],
              'all_losses_finite':bool(np.isfinite(losses).all())};results.append(record);print(json.dumps(record),flush=True)
            del trainer;gc.collect();torch.cuda.empty_cache()
    atomic_json(root/'benchmark'/a.report_name,{'passed':True,'results':results,'training_view_fingerprint':view['fingerprint'],
      'verification_updates':4*(10+a.steps),'script_sha256':sha(__file__),'formal_model_scores':False})


if __name__=='__main__':main()
