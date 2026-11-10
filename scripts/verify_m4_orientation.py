#!/usr/bin/env python3
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from training.m3 import configure,atomic_json,sha
from training.m4 import M4Trainer,load_data,evaluate_full
from training.m4_orientation import OrientationTrainer,balanced_validation
from verify_m3 import max_difference


def main():
    root=ROOT/'runs/m4_research_20261103';out=root/'orientation_verification';out.mkdir(exist_ok=False)
    manifest=json.loads((ROOT/'runs/m4_20261102/data/BCD.json').read_text());train=load_data(manifest,'train');val=load_data(manifest,'val')
    samples=[train[i] for i in [0,1,256,257,512,513,768,769]]
    base=json.loads((ROOT/'configs/m4_B.json').read_text());base['ema_every']=2
    configure(base['seed']);original=M4Trainer(base,samples,{'test':True});original.update()
    zero=dict(base,heading_augmentation=True,heading_augmentation_probability=0.0)
    configure(zero['seed']);test=OrientationTrainer(zero,samples,{'test':True});test.update()
    error=max_difference(original.model.state_dict(),test.model.state_dict());assert error==0,error
    config=dict(base,heading_augmentation=True,heading_augmentation_probability=.5)
    configure(config['seed']);trainer=OrientationTrainer(config,samples,{'test':True})
    for _ in range(3):trainer.update()
    trainer.save(out/'step3.pt')
    for _ in range(3):last=trainer.update()
    expected={k:v.cpu().clone() for k,v in trainer.model.state_dict().items()};opt=copy.deepcopy(trainer.optimizer.state_dict())
    resumed=OrientationTrainer(config,samples,{'test':True});resumed.restore(out/'step3.pt')
    for _ in range(3):resumed.update()
    resume=max_difference(expected,resumed.model.state_dict());assert resume==0,resume
    assert max_difference(opt,resumed.optimizer.state_dict())==0
    assert torch.equal(trainer.heading_rng.get_state(),resumed.heading_rng.get_state())
    native=evaluate_full(resumed.ema,val,'cuda:0');rotated=evaluate_full(resumed.ema,val,'cuda:0',heading_offset=torch.pi/2)
    combined=balanced_validation(native,rotated)
    assert combined['events']==128 and combined['unique_events']==64 and len(combined['per_sample'])==128
    assert combined['macro_loss']==(native['macro_loss']+rotated['macro_loss'])/2
    report={'passed':True,'zero_probability_matches_B_update':error,'resume_max_abs':resume,
        'optimizer_and_heading_rng_resume_exact':True,'yaw_augmentation_observed':last['heading_abs_delta_mean']>0,
        'unique_validation_events':64,'validation_views':128,
        'module_sha256':sha(ROOT/'training/m4_orientation.py'),'runner_sha256':sha(ROOT/'scripts/train_m4_orientation.py')}
    atomic_json(out/'report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
