#!/usr/bin/env python3
"""Four real-architecture CLI pilots; numerical fixtures are excluded from ranking."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.m3 import atomic_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();root=a.experiment.resolve();out=root/'verification/parallel_pilot';out.mkdir(parents=True,exist_ok=True)
    m=json.loads((root/'data/D_short_3500.json').read_text());groups=('walk','jog','jump','raise_arms','turn','squat')
    m['train']=[next(r for r in m['train'] if r['group']==g) for g in groups];m['val']=[next(r for r in m['val'] if r['group']==g) for g in groups];m['monitor']=m['val']
    subset=out/'subset.json';atomic_json(subset,m);view=json.loads((root/'data/READY.json').read_text());configs={};processes={};logs=[]
    for name,size,mode in [('B1','small','norm7'),('P0','tiny','norm6'),('B0','tiny','norm7'),('C','small','weighted_raw7')]:
        architecture={'latent_dim':256,'ff_size':1024,'num_layers':4,'num_heads':4} if size=='tiny' else {'latent_dim':512,'ff_size':2048,'num_layers':8,'num_heads':8}
        cfg={'stage':'pilot_'+name,'model':architecture,'seed':3101,'batch_size':64,'microbatch':64,'learning_rate':2e-5,'minimum_learning_rate':2e-6,
          'scheduler':'warmup_cosine','warmup_steps':2,'stage_start_step':0,'max_steps':16,'minimum_steps':16,'early_stop_relative_delta':.001,
          'grad_clip':1.,'ema_every':10,'ema_decay':.995,'precision':'bfloat16','text_dropout':.1,'heading_augmentation':True,'sampling':'stratified',
          'high_noise_fraction':.5,'high_noise_start':800,'high_noise_after':0,'loss_objective':mode,'fk_weight':1.,'axis_weight':0.,'consistency_weight':0.,'ground_weight':0.,
          'eval_every':16,'generation_every':16,'full_generation_every':16,'checkpoint_every':8,'log_every':4,'full_seeds':[42],'eval_batch_size':8,'ddim_steps':10,
          'input_dir':view['input_dir'],'stats_dir':str(root/'stats/motion'),'training_view':str(root/'data/READY.json'),'subset_sha256':sha(subset)}
        config=out/(name+'.json');atomic_json(config,cfg);configs[name]=config
    try:
        for name in ['B1','P0','B0','C']:
            command=[sys.executable,'-B','-u',str(ROOT/'scripts/train_m5_physical.py'),'--config',str(configs[name]),'--subset',str(subset),'--run-dir',str(out/name)]
            if name=='C':
                initial=out/'B1/checkpoints/initial_state.pt';deadline=time.monotonic()+120
                while not initial.exists():
                    if processes['B1'].poll() is not None or time.monotonic()>deadline:raise RuntimeError('Pilot B1 initial checkpoint unavailable')
                    time.sleep(1)
                command+=['--initial-weights',str(initial)]
            log=(out/(name+'.log')).open('w');logs.append(log);processes[name]=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            print('pilot launched',name,processes[name].pid,flush=True)
        atomic_json(out/'processes.json',{k:v.pid for k,v in processes.items()})
        while any(p.poll() is None for p in processes.values()):
            failed={k:p.returncode for k,p in processes.items() if p.poll() is not None and p.returncode!=0}
            if failed:raise RuntimeError('Pilot failed: '+str(failed))
            time.sleep(2)
        traces={name:sha(out/name/'sample_trace.jsonl') for name in processes};assert len(set(traces.values()))==1
        initial={name:json.loads((out/name/'run_config.json').read_text())['initial_model_sha256'] for name in processes}
        assert initial['P0']==initial['B0'] and initial['B1']==initial['C']
        for name in processes:
            done=json.loads((out/name/'COMPLETED.json').read_text());assert done['step']==16
            report=json.loads((out/name/'evaluations/generation_full_000016/report.json').read_text());assert report['all_finite'] and report['all_valid_qpos']
        result={'passed':True,'updates':64,'processes':{k:p.pid for k,p in processes.items()},'sample_trace_sha256':traces,'initial_model_sha256':initial,
          'small_shared_initial_checkpoint':True,'all_terminal_generation_finite_and_valid':True,'formal_model_scores':False,
          'script_sha256':sha(__file__),'runner_sha256':sha(ROOT/'scripts/train_m5_physical.py')}
        atomic_json(root/'verification/parallel_pilot_report.json',result);print(json.dumps(result,indent=2),flush=True)
    finally:
        for p in processes.values():
            if p.poll() is None:p.terminate()
        for p in processes.values():
            if p.poll() is None:p.wait(timeout=120)
        for f in logs:f.close()


if __name__=='__main__':main()
