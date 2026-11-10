#!/usr/bin/env python3
"""Freeze repaired-input M5 first wave and preserve the complete M5 acceptance scope."""
import argparse,json,platform,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,mujoco
from training.m3 import atomic_json,sha,validate_inputs
from scripts.train_m5_physical import CODE


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();root=a.experiment.resolve();target=root/'protocol/launch.json'
    if target.exists():raise FileExistsError('Launch protocol already frozen')
    old=validate_inputs();view=json.loads((root/'data/READY.json').read_text());proofs={}
    for name in ['physical_contract_audit.json','cache_audit.json','training_target_limits_audit.json','reference_review.json','verification/report.json','verification/parallel_pilot_report.json','benchmark/micro64.json']:
        path=root/name;value=json.loads(path.read_text());assert value['passed'],name;proofs[name]=sha(path)
    verification=json.loads((root/'verification/report.json').read_text())
    for path,expected in verification['code_sha256'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Verified source changed: '+path)
    pilot=json.loads((root/'verification/parallel_pilot_report.json').read_text())
    if pilot['runner_sha256']!=sha(ROOT/'scripts/train_m5_physical.py'):raise ValueError('Runner changed after pilot')
    all_code=tuple(CODE)+('scripts/assess_m5_physical_first_wave.py','scripts/run_m5_physical_study.py','scripts/freeze_m5_physical_launch.py',
      'scripts/prepare_m5_physical_study.py','scripts/verify_m5_physical.py','scripts/run_m5_physical_pilot.py','scripts/review_m5_physical_references.py',
      'scripts/audit_m5_physical_training_targets.py','scripts/benchmark_m5_physical.py')
    code={f:sha(ROOT/f) for f in all_code};training_code={f:code[f] for f in CODE}
    references=json.loads((root/'reference_review.json').read_text());subset=root/'data/D_short_3500.json';m=json.loads(subset.read_text())
    jobs={};hashes={};jobspec=[('P0_tiny_norm6','tiny','norm6'),('B0_tiny_norm7','tiny','norm7'),('B1_small_norm7','small','norm7'),('C_small_raw7','small','weighted_raw7')]
    for name,size,mode in jobspec:
        cfg={'stage':name,'model':{'latent_dim':256,'ff_size':1024,'num_layers':4,'num_heads':4} if size=='tiny' else {'latent_dim':512,'ff_size':2048,'num_layers':8,'num_heads':8},
          'seed':3101,'batch_size':64,'microbatch':64,'learning_rate':2e-5,'minimum_learning_rate':2e-6,'scheduler':'cosine','warmup_steps':1000,
          'stage_start_step':0,'max_steps':40000,'minimum_steps':40000,'early_stop_relative_delta':.001,'grad_clip':1.,'ema_every':10,'ema_decay':.995,
          'precision':'bfloat16','network_dropout':0.,'text_dropout':.1,'heading_augmentation':True,'sampling':'stratified','cache_events':len(m['train']),
          'high_noise_fraction':.5,'high_noise_start':800,'high_noise_after':5000,'loss_objective':mode,'fk_weight':0. if mode=='norm6' else 1.,
          'axis_weight':0.,'consistency_weight':0.,'ground_weight':0.,'postprocess_training':False,
          'eval_every':1000,'generation_every':2000,'full_generation_every':10000,'checkpoint_every':5000,'log_every':50,'eval_batch_size':16,'ddim_steps':100,'full_seeds':[0,42,1234],
          'metric_thresholds':references['metric_thresholds'],'input_dir':view['input_dir'],'stats_dir':str(root/'stats/motion'),'training_view':str(root/'data/READY.json'),
          'subset_sha256':sha(subset),'train_samples':len(m['train']),'launch_code_sha256':training_code,
          'normalization':'qualified unique training union, physical-v2; population std and eps=1e-5',
          'objective':'From-scratch M5 '+mode+' on independently aligned physical G1 features',
          'time_sampling':'uniform until global update 5000, then 50% replacement by uniform t=800..999'}
        config=root/'configs'/(name+'.json');atomic_json(config,cfg);hashes[name]=sha(config)
        jobs[name]={'model':size,'objective':mode,'new_updates':40000,'initial_weights':str(root/'B1_small_norm7/checkpoints/initial_state.pt') if name=='C_small_raw7' else None}
    acceptance={'schema':'m5-physical-acceptance-v2','scope':'Complete M5 only; no dynamics, controller training, or automatic M6 entry.',
      'legacy_protocol':'runs/m5_20261103/protocol/launch.json','legacy_gate_code':'training/m5_protocol.py','legacy_failures_preserved':True,
      'reason_for_new_version':'Independent compiled-MJCF evidence invalidates legacy feature/body-coordinate targets. Correct the representation and restart; no gate is relabelled as passed.',
      'engineering_proofs':proofs,'physical_input_fingerprint':view['input_fingerprint'],'training_view_fingerprint':view['fingerprint'],
      'model':'Tiny 9,214,881 vs Small 56,755,617 parameters; both denoisers from random initialization, frozen cached text embeddings.',
      'old_A_gate_status':'Failed and retained. P0/B0 is a new matched from-scratch repair diagnostic; report original >=10% consistency improvement explicitly.',
      'first_wave':'4 independent runs x 40k. Same sample/noise RNG streams. B1/C share identical step-zero weights; only objective differs.',
      'dependencies':'Assess all four fixed endpoints before further D/E. If the repair or core gate fails, bounded M5 research precedes dependent curriculum.',
      'final_requirements':{'full_matrix':'B/C loss/capacity, matched D0/D1 length, matched E0/E1 data; retain negative results.',
        'complete_repetition':'Two independent Small training seeds 3101/3102 repeat selected complete 40k+20k+20k course; final Tiny exactly matches the selected recipe/course.',
        'core_generation':'Both Small seeds pass original four-class amplitude checks at native and heading=0; turn/squat separately reported.',
        'semantics':'At least 120 independent balanced prompts, two independent human raters, hidden model identities; six-class macro correctness >=80%.',
        'preference':'Matched Tiny vs Small paired preference, source-group cluster-bootstrap 95% CI lower bound >50%; agent scores are not human ratings.',
        'relative_geometry':'Two named metrics: mean FK self-inconsistency and pooled node projection P95 each improve >=30% vs matched Tiny; other key metrics no >10% regression with predeclared near-zero tolerance.',
        'ground':'Every class: P95 of per-clip maximum penetration <=0.05m, absolute maximum <=0.10m, mean >0.05m frame fraction <=0.01.',
        'projection':'Every class mean per-clip node projection P95 <=0.03m, absolute node maximum <=0.10m; additionally pooled node P95 <=0.03m.',
        'continuity':'Every class reference-calibrated rotation exceedance fraction <=0.001; raw/bounded qpos velocities and branches retained.',
        'jump':'Semantically complete jump subset >=80% identifiable takeoff/flight/landing; geometry proxy alone is insufficient.',
        'fresh_audit':'Pre-registered independent source groups and new generation seeds 23017/23073/232026; no audit model-score selection.',
        'references':'Report frozen R3 and official baseline versions with appropriate own statistics; disclose unknown official-pretraining overlap.',
        'artifacts':'Checkpoint/code/input identities, complete metrics, worst/random/text-swap videos, archive lineage and failure review.'},
      'final_audit_model_scores_read':False,'M5_complete':False}
    atomic_json(root/'protocol/M5_acceptance_protocol_v2.json',acceptance)
    protocol={'schema':'m5-physical-launch-v2','frozen_at_utc':datetime.now(timezone.utc).isoformat(),'jobs':jobs,'config_sha256':hashes,
      'launch_order':['B1_small_norm7','P0_tiny_norm6','B0_tiny_norm7','C_small_raw7'],'max_parallel':4,'minimum_free_memory_mib':20000,
      'code_sha256':code,'subset':str(subset),'subset_sha256':sha(subset),'training_view_fingerprint':view['fingerprint'],'new_updates_total':160000,
      'engineering_proofs':proofs,'acceptance_sha256':sha(root/'protocol/M5_acceptance_protocol_v2.json'),
      'main_curriculum_auto_start':False,'M5_complete':False,'no_official_denoiser_weights_loaded':True}
    for path in all_code:
        destination=root/'source_at_launch'/path;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,destination)
    atomic_json(root/'environment.json',{'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'mujoco':mujoco.__version__,
      'platform':platform.platform(),'tf32':False,'mha_fastpath':False,'deterministic_algorithms':True,'OMP_NUM_THREADS':4,'CUBLAS_WORKSPACE_CONFIG':':4096:8'})
    atomic_json(target,protocol);print(json.dumps({'protocol':str(target),'sha256':sha(target),'new_updates':160000,'jobs':jobs,'M5_complete':False},indent=2),flush=True)


if __name__=='__main__':main()
