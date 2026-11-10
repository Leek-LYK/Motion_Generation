#!/usr/bin/env python3
"""Release the existing, never-scored audit once, after development evidence review."""
import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
RESEARCH=ROOT/'runs/m4_research_20261103'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def main():
    release=RESEARCH/'audit_release.json'
    if release.exists():raise FileExistsError('Audit already released; never overwrite its frozen configuration')
    review=read(RESEARCH/'development_review.json')
    assert review['four_classes_recognizable'] and review['failure_cases_inspected']
    assert review['scope']=='M5 engineering entry; not arbitrary-caption accuracy or dynamic tracking'
    verify=read(RESEARCH/'diagnostics/bounded_decoder_v1/report.json')
    assert verify['checks_passed'] and verify['decoder_sha256']==digest(ROOT/'training/g1_qpos.py')
    assert read(RESEARCH/'diagnostics/assessor_decoder_equivalence.json')['passed']
    candidates={
      'R3_D_high_noise':['cfg_readiness/cfg2/report.json','readiness_heading0_cfg2/report.json'],
      'R9_D_high_noise_seed20261104':['readiness_cfg2/report.json','readiness_heading0_cfg2/report.json']}
    sha_protocol=digest(RESEARCH/'M5_entry_protocol.json');hashes=[];evidence={}
    assert sha_protocol=='690a505eac3136745034fbd0933b8c21744be2c6e87451a96b490b74206cc695'
    for name,reports in candidates.items():
        run=RESEARCH/name;checkpoint=digest(run/'checkpoints/best_val_ema.pt');hashes.append(checkpoint)
        for filename in reports:
            path=run/filename;result=read(path)
            assert result['amplitude_gate_passed'] and result['all_finite']
            assert result['checkpoint_sha256']==checkpoint and result['protocol_sha256']==sha_protocol
            assert result['guidance']==2 and result['events']==64 and result['seeds']==[0,42,1234]
            evidence[str(path.relative_to(ROOT))]=digest(path)
    configs=[read(RESEARCH/n/'config.json') for n in candidates]
    assert configs[0]['seed']!=configs[1]['seed']
    assert {k:v for k,v in configs[0].items() if k not in ['seed','stage']}=={k:v for k,v in configs[1].items() if k not in ['seed','stage']}
    audit=read(RESEARCH/'locked_audit.json')['test_repetition']
    subset=read(RESEARCH/'R3_D_high_noise/subset.json')
    assert len(audit)==64 and len({a['sample_id'] for a in audit})==64
    for key in ['source_group_id','motion_id','sample_id']:
        a={x[key] for x in audit};b={x[key] for split in ['train','val'] for x in subset[split]}
        assert not a & b,(key,a & b)
    value={'schema':'m4-final-audit-release-v1','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
      'allowed_checkpoint_sha256':hashes,'runs':list(candidates),'guidance':2.0,
      'sampling_seeds':[17,73,2026],'conditioning_headings':[None,0.0],
      'batch_size':16,'ddim_steps':100,'weights':'ema','decoder':'bounded-v1',
      'audit_manifest_sha256':digest(RESEARCH/'locked_audit.json'),'protocol_sha256':sha_protocol,
      'assessor_sha256':digest(ROOT/'scripts/assess_m4_readiness.py'),
      'decoder_sha256':digest(ROOT/'training/g1_qpos.py'),'metric_sha256':digest(ROOT/'training/m4_readiness.py'),
      'development_review_sha256':digest(RESEARCH/'development_review.json'),
      'development_evidence_sha256':evidence,'raw_motion_metrics_required':True,'retain_all_samples':True,
      'decision':'Evaluate both frozen training seeds and both headings. No threshold or configuration selection from these audit scores.',
      'if_failed':'Audit becomes consumed evidence. Any further model tuning requires a new source-distinct unused final audit.',
      'not_claimed':['paper parity','all captions correct','continuous joint trajectories','no foot sliding','dynamic tracking']}
    with release.open('x') as file:json.dump(value,file,indent=2);file.write('\n')
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
