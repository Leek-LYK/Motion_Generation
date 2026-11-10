#!/usr/bin/env python3
"""Encode pre-registered novel prompts in a separate frozen cache; preserve M2."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',TEXT_ENCODER_MODE='local')
import argparse,fcntl,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from training.m3 import atomic_json,sha,INPUTS
from kimodo.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();root=a.experiment.resolve()
    selection=root/'review/protocol/authored_development_prompts.json';protocol=root/'review/protocol/review_protocol.json';rows=json.loads(selection.read_text())['events']
    frozen=json.loads(protocol.read_text());assert frozen['authored_prompts_sha256']==sha(selection)
    out=root/'review/authored_text_cache';out.mkdir(parents=True,exist_ok=True);lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'COMPLETE.json').exists():raise FileExistsError('Review embeddings already complete')
    parent=json.loads((INPUTS/'text_encoder/config.json').read_text());model_root=ROOT.parent/'kimodo/text_encoder'
    # Reproduce the ORIGINAL encoder recipe, not the denoiser-training CUDA flags.
    # Strict equality against old vectors remains mandatory below.
    torch.set_num_threads(4)
    from transformers.utils.logging import disable_progress_bar
    disable_progress_bar()
    def status(phase,**extra):atomic_json(out/'status.json',{'phase':phase,'pid':os.getpid(),'total':len(rows),**extra})
    status('verifying_frozen_encoder_files')
    for name,expected in parent['model_file_sha256'].items():
        if sha(model_root/name)!=expected:raise ValueError('Encoder file changed: '+name)
    for name,expected in parent['kimodo_source_sha256'].items():
        if sha(ROOT.parent/'kimodo'/name)!=expected:raise ValueError('Frozen Kimodo source changed: '+name)
    os.environ['TEXT_ENCODERS_DIR']=str(model_root);os.environ['TEXT_ENCODER_DEVICE']='cuda:0';status('loading_frozen_encoder')
    encoder=LLM2VecEncoder('McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp','McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised','bfloat16',4096,device='cuda:0')
    assert not any(p.requires_grad for p in encoder.model.parameters())
    reference=torch.load(ROOT.parent/'kimodo/experiments/llm2vec_adapter_audit/embeddings_candidate.pt',map_location='cpu',weights_only=True)
    actual,_=encoder(reference['prompts']);expected=reference['enabled'].cpu();actual=actual[:,0].detach().cpu()
    baseline={'exact_equal':bool(torch.equal(actual,expected)),'max_abs_error':float(abs(actual-expected).max()),'mean_abs_error':float(abs(actual-expected).mean()),
      'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
      'mha_fastpath':torch.backends.mha.get_fastpath_enabled(),'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}
    atomic_json(out/'baseline_check.json',baseline)
    if not baseline['exact_equal']:raise ValueError('Frozen text encoder no longer reproduces the verified baseline: '+json.dumps(baseline))
    old=np.load(INPUTS/'text_encoder/embeddings.npy',mmap_mode='r');texts=pd.read_parquet(INPUTS/'texts.parquet')
    old_checks=[]
    for index in [0,len(texts)//2,len(texts)-1]:
        r=texts.iloc[index];encoded,_=encoder(str(r.text));target=np.asarray(old[int(r.text_row)]);value=encoded.detach().cpu().numpy()
        if not np.array_equal(value,target):raise ValueError('Existing text-cache row differs: '+str(r.text_row))
        old_checks.append({'text_row':int(r.text_row),'exact_equal':True})
    status('encoding',completed=0)
    result=np.empty((len(rows),1,4096),dtype=np.float32);max_tokens=0
    for i,row in enumerate(rows):
        n=len(encoder.model.tokenizer(row['text'],add_special_tokens=False,truncation=False)['input_ids']);max_tokens=max(max_tokens,n)
        if n>400:raise ValueError('Review prompt would be truncated')
        value,_=encoder(row['text']);result[i]=value.detach().cpu().numpy()
        if (i+1)%20==0:status('encoding',completed=i+1);print('review embeddings',i+1,'/',len(rows),flush=True)
    for i in [0,59,119]:
        value,_=encoder(rows[i]['text']);assert np.array_equal(value.detach().cpu().numpy(),result[i])
    assert np.isfinite(result).all();np.save(out/'embeddings.npy',result)
    atomic_json(out/'rows.json',[dict(row,text_row=i) for i,row in enumerate(rows)])
    proof={'passed':True,'parent_encoder_fingerprint':parent['fingerprint'],'parent_encoder_baseline_exact':True,'existing_M2_row_checks':old_checks,
      'shape':list(result.shape),'storage_dtype':'float32','base_dtype':'bfloat16','internal_batch_size':1,'max_tokens':max_tokens,'repeated_rows_exact':True,
      'selection_sha256':sha(selection),'review_protocol_sha256':sha(protocol),'embeddings_sha256':sha(out/'embeddings.npy'),'rows_sha256':sha(out/'rows.json'),
      'script_sha256':sha(__file__),'compute_configuration':baseline,'training_cache_modified':False,'human_ratings_collected':False,'formal_training_updates':0}
    proof['fingerprint']=hashlib.sha256(json.dumps(proof,sort_keys=True).encode()).hexdigest();atomic_json(out/'COMPLETE.json',proof);status('complete',completed=len(rows))
    print(json.dumps(proof,indent=2),flush=True)


if __name__=='__main__':main()
