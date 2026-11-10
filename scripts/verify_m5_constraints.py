#!/usr/bin/env python3
"""Zero-addition equivalence, optimizer lineage, nonzero-loss replay and LR origin."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,gc,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from training.m3 import configure,atomic_json,sha
from training.m5 import M5Trainer
from training.m5_physical import PhysicalTrainer
from training.m5_atan2 import Atan2PhysicalTrainer
from training.adam_atan2 import VERSION as ATAN2_VERSION
from training.m5_constraints import PhysicalConstraintTrainer
from training.m5_physical_data import PhysicalSubsetDataset
from scripts.verify_m5_physical import exact


def compare(a,b):
    for key in ['model','ema','optimizer','rngs','torch_rng','cuda_rng','step','samples_seen','stage_updates','text_dropped']:
        assert exact(a[key],b[key]),key


def main():
    root=ROOT/'runs/m5_physical_20261103';out=root/'verification/constraint_trainer';out.mkdir(parents=True,exist_ok=True)
    view=json.loads((root/'data/READY.json').read_text());m=json.loads((root/'data/D_short_3500.json').read_text());groups=['walk','jog','jump','raise_arms','turn','squat']
    entries=[next(r for r in m['train'] if r['group']==g) for g in groups];ds=PhysicalSubsetDataset(entries,view['input_dir'],root/'stats/motion',cache_size=6)
    cfg={'model':{'latent_dim':64,'ff_size':128,'num_layers':1,'num_heads':4},'seed':7431,'batch_size':6,'microbatch':2,'learning_rate':2e-5,'minimum_learning_rate':2e-6,
      'scheduler':'constant','warmup_steps':2,'stage_start_step':0,'max_steps':12,'minimum_steps':12,'early_stop_relative_delta':.001,'grad_clip':1.,
      'ema_every':1,'ema_decay':.995,'precision':'bfloat16','text_dropout':.1,'heading_augmentation':True,'sampling':'stratified',
      'high_noise_fraction':.5,'high_noise_start':800,'high_noise_after':0,'loss_objective':'weighted_raw7',
      'axis_weight':0.,'consistency_weight':0.,'ground_weight':0.,'legality_weight':0.}
    identity={'physical_input_fingerprint':view['input_fingerprint'],'training_view_fingerprint':view['fingerprint'],'text_config_fingerprint':'fixture','scope':'constraint numerical fixture'}
    checks={}
    for kind in ['adamw','atan2']:
        config=dict(cfg)
        Base=PhysicalTrainer
        if kind=='atan2':config.update(optimizer_name=ATAN2_VERSION,optimizer_betas=[.9,.999],optimizer_a=1.,optimizer_b=1.,optimizer_weight_decay=0.);Base=Atan2PhysicalTrainer
        configure(7431);source=Base(config,ds,identity,root/'stats/motion');source.update();source.update();parent=copy.deepcopy(source.state());del source
        continuation=dict(config,stage_start_step=2)
        control=Base(continuation,ds,identity,root/'stats/motion');control.load_state(copy.deepcopy(parent),continuation=True);control.update();expected=copy.deepcopy(control.state());del control
        candidate=PhysicalConstraintTrainer(continuation,ds,identity,root/'stats/motion');candidate.load_state(copy.deepcopy(parent),continuation=True);candidate.update();compare(candidate.state(),expected)
        checks[kind+'_zero_additions_full_update_exact']=True
        bad=copy.deepcopy(parent);bad['schema']='kimodo-m5-physical-checkpoint-v2' if kind=='atan2' else 'kimodo-m5-physical-atan2-checkpoint-v1'
        rejected=False
        try:candidate.load_state(bad,continuation=True)
        except ValueError:rejected=True
        assert rejected;checks[kind+'_wrong_optimizer_lineage_rejected']=True
        del candidate,expected,parent;gc.collect();torch.cuda.empty_cache()
    config=dict(cfg,consistency_weight=.01,ground_weight=.1,legality_weight=.02)
    configure(7431);candidate=PhysicalConstraintTrainer(config,ds,identity,root/'stats/motion');candidate.update();candidate.save(out/'resume_fixture.pt');candidate.update();candidate.update();expected=copy.deepcopy(candidate.state());del candidate
    configure(7431);restored=PhysicalConstraintTrainer(config,ds,identity,root/'stats/motion');restored.restore(out/'resume_fixture.pt');restored.update();restored.update();compare(restored.state(),expected)
    checks['nonzero_constraints_bf16_accumulated_resume_exact']=True
    original=dict(cfg,scheduler='cosine',warmup_steps=1000,max_steps=40000,stage_start_step=0)
    continued=dict(original,stage_start_step=5000,lr_schedule_origin_step=0)
    for step in [5000,5001,10000,19999,20000,40000]:
        left=M5Trainer.current_lr(SimpleNamespace(config=original,step=step));right=PhysicalConstraintTrainer.current_lr(SimpleNamespace(config=continued,step=step));assert left==right,(step,left,right)
    checks['continued_original_lr_schedule_exact']=True
    report={'passed':True,'checks':checks,'real_training_experiment':False,'scope':'Numerical fixtures only. No live code or datasets changed; no coefficients selected.',
      'code_sha256':{p:sha(ROOT/p) for p in ['training/m5_constraints.py','training/m5_constraint_losses.py','training/g1_legality_loss.py','scripts/verify_m5_constraints.py']}}
    atomic_json(out/'report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
