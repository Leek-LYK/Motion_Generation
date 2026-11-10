#!/usr/bin/env python3
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from training.m3 import atomic_json,configure,sha
from training.m4 import M4Trainer,load_data
from training.m4_research import ResearchTrainer
from verify_m3 import max_difference


def main():
    output=ROOT/'runs/m4_research_20261103/variant_verification';output.mkdir(exist_ok=False)
    config=json.loads((ROOT/'configs/m4_D.json').read_text());config.update(high_noise_fraction=0.0,loss_mode='smooth_l1',ema_every=2)
    manifest=json.loads((ROOT/'runs/m4_20261102/data/BCD.json').read_text());all_samples=load_data(manifest,'train')
    samples=[all_samples[i] for i in [0,1,256,257,512,513,768,769]];identity={'verification':True}
    configure(config['seed']);baseline=M4Trainer(config,samples,identity);baseline.update()
    configure(config['seed']);research=ResearchTrainer(config,samples,identity);research.update()
    zero=max_difference(baseline.model.state_dict(),research.model.state_dict());assert zero==0.0,zero
    config=dict(config,high_noise_fraction=1.0,high_noise_start=800)
    configure(config['seed']);research=ResearchTrainer(config,samples,identity)
    for _ in range(3):last=research.update()
    assert last['t_ge800_fraction']==1.0
    research.save(output/'step3.pt')
    for _ in range(3):research.update()
    expected={k:v.cpu().clone() for k,v in research.model.state_dict().items()}
    opt=copy.deepcopy(research.optimizer.state_dict());rng=research.high_noise_rng.get_state()
    restored=ResearchTrainer(config,samples,identity);restored.restore(output/'step3.pt')
    for _ in range(3):restored.update()
    resume=max_difference(expected,restored.model.state_dict());assert resume==0.0,resume
    assert max_difference(opt,restored.optimizer.state_dict())==0 and torch.equal(rng,restored.high_noise_rng.get_state())
    config=dict(config,high_noise_fraction=0.0,loss_mode='mse');configure(config['seed'])
    mse=ResearchTrainer(config,samples,identity);values=mse.update();assert torch.isfinite(torch.tensor(values['loss']))
    result={'passed':True,'zero_high_noise_matches_original_update_max_abs':zero,'all_high_noise_timesteps_ge800':True,
            'high_noise_resume_max_abs':resume,'optimizer_and_extra_rng_resume_exact':True,'mse_update_finite':True,
            'research_source_sha256':sha(ROOT/'training/m4_research.py'),'runner_sha256':sha(ROOT/'scripts/train_m4_research.py')}
    atomic_json(output/'report.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
