#!/usr/bin/env python3
"""Standalone M5-compatible evaluation of frozen M4/M5 checkpoints and audit releases."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,io,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from training.m3 import atomic_json,configure,make_model,sha,validate_inputs
from training.m5_data import SubsetDataset
from training.m5_evaluation import evaluate_generation


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[0,42,1234])
    p.add_argument('--audit-release',type=Path);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--steps',type=int,default=100)
    p.add_argument('--metrics-protocol',type=Path,required=True)
    p.add_argument('--official',action='store_true');args=p.parse_args()
    if (args.output/'report.json').exists():raise FileExistsError(args.output)
    ready=validate_inputs();manifest=json.loads(args.manifest.read_text());payload=args.checkpoint.read_bytes();checksum=hashlib.sha256(payload).hexdigest()
    metrics_protocol=json.loads(args.metrics_protocol.read_text())
    if 'events' in manifest:
        if not args.audit_release:raise ValueError('Audit release is required')
        release=json.loads(args.audit_release.read_text())
        if (checksum not in release['allowed_checkpoint_sha256'] or sha(args.manifest)!=release['manifest_sha256'] or args.seeds!=release['seeds']
            or sha(__file__)!=release['evaluator_sha256'] or sha(ROOT/'training/m5_evaluation.py')!=release['metrics_sha256'] or args.steps!=100
            or sha(args.metrics_protocol)!=release['protocol_sha256']):
            raise ValueError('Audit release mismatch')
        events=manifest['events']
    else:events=manifest['val']
    reference_rep=None
    if args.official:
        configure(3101);model=make_model({'latent_dim':1024,'ff_size':2048,'num_layers':16,'num_heads':8},args.checkpoint.parent/'stats/motion').cuda().eval()
        model.load_ckpt(str(args.checkpoint));state={'step':0}
        reference_model=make_model({'latent_dim':64,'ff_size':128,'num_layers':1,'num_heads':4}).cuda();reference_rep=reference_model.motion_rep
    else:
        state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
        if state['identity']['m2_fingerprint']!=ready['fingerprint']:raise ValueError('M2 mismatch')
        configure(state['config']['seed']);model=make_model(state['config']['model']).cuda().eval();model.load_state_dict(state['ema'])
    del payload
    dataset=SubsetDataset(events,cache_size=len(events))
    result=evaluate_generation(model,dataset,args.output,state['step'],seeds=args.seeds,batch_size=args.batch_size,steps=args.steps,retain=True,reference_rep=reference_rep,
      metric_thresholds=metrics_protocol['rotation_thresholds_rad'])
    result.update(checkpoint_sha256=checksum,manifest_sha256=sha(args.manifest),evaluator_sha256=sha(__file__),metrics_sha256=sha(ROOT/'training/m5_evaluation.py'),official_reference=args.official,
      metrics_protocol_sha256=sha(args.metrics_protocol))
    atomic_json(args.output/'report.json',result);print(json.dumps({k:v for k,v in result.items() if k!='conditions'},indent=2),flush=True)


if __name__=='__main__':main()
