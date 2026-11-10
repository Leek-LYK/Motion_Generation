#!/usr/bin/env python3
"""Verify all saved audit artifacts and assemble numerical evidence, without final judgment."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import mujoco
from kimodo.assets import skeleton_asset_path
from data_pipeline.numeric_common import atomic_json


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def main():
    root=ROOT/'runs/m4_research_20261103';release=read(root/'audit_release.json')
    assert read(root/'audit_execution_status.json')['phase']=='audit_complete_requires_final_review'
    protocol=read(root/'M5_entry_protocol.json');audit=read(root/'locked_audit.json')['test_repetition']
    subset=read(root/'R3_D_high_noise/subset.json');train_texts={x['text_id'] for x in subset['train']}
    records=[];artifact_hashes={};total=0;worst_legacy=0;worst_speed=0
    model=mujoco.MjModel.from_xml_path(str(skeleton_asset_path('g1skel34','xml','g1.xml')))
    ids=np.flatnonzero(model.jnt_limited);addresses=model.jnt_qposadr[ids];bounds=model.jnt_range[ids]
    for run in release['runs']:
        ckpt=sha(root/run/'checkpoints/best_val_ema.pt');assert ckpt in release['allowed_checkpoint_sha256']
        for condition,heading in [('native',None),('heading0',0.0)]:
            folder=root/run/'locked_audit_v1'/condition;report=read(folder/'report.json');rows=read(folder/'per_sample.json')
            assert report['checkpoint_sha256']==ckpt and report['conditioning_heading']==heading
            assert report['seeds']==release['sampling_seeds'] and report['guidance']==release['guidance']
            assert report['source_manifest_sha256']==release['audit_manifest_sha256'] and report['protocol_sha256']==release['protocol_sha256']
            assert report['script_sha256']==release['assessor_sha256'] and report['metric_code_sha256']==release['metric_sha256']
            assert report['amplitude_uses_raw_unprojected_motion'] and report['decoder_sha256']==release['decoder_sha256']
            expected={(r['sample_id'],s) for r in audit for s in release['sampling_seeds']}
            assert {(r['sample_id'],r['seed']) for r in rows}==expected and len(rows)==len(expected)
            bounded_groups={}
            for row in rows:
                stem=folder/row['artifact_prefix'];q=np.loadtxt(stem.with_suffix('.csv'),delimiter=',')
                raw=np.loadtxt(stem.with_name(stem.name+'_legacy.csv'),delimiter=',')
                with np.load(stem.with_suffix('.npz'),allow_pickle=False) as archive:
                    assert all(np.isfinite(archive[k]).all() for k in archive.files)
                    assert len(archive['root_positions'])==len(q)
                assert q.shape==tuple(row['qpos_shape']) and q.shape[1]==36 and np.isfinite(q).all()
                assert np.max(abs(np.linalg.norm(q[:,3:7],axis=1)-1))<1e-4
                assert np.array_equal(q[:,:3],raw[:,:3])
                violation=np.maximum(0,np.maximum(bounds[:,0]-q[:,addresses],q[:,addresses]-bounds[:,1]))
                assert violation.max(initial=0)<=1e-6
                for p in [stem.with_suffix('.csv'),stem.with_suffix('.npz'),stem.with_name(stem.name+'_legacy.csv')]:
                    artifact_hashes[str(p.relative_to(root))]=sha(p)
            for group in protocol['reference_minimum']:
                eligible=[r for r in rows if r['group']==group and r['reference_amplitude']>=protocol['reference_minimum'][group]]
                ratios=np.asarray([r['bounded_amplitude']/r['reference_amplitude'] for r in eligible])
                bounded_groups[group]={'median_ratio':float(np.median(ratios)),
                  'noncollapsed_fraction':float(np.mean(ratios>=protocol['noncollapse_ratio_min'])),
                  'diagnostic_only':True}
            total+=len(rows);worst_legacy=max(worst_legacy,report['max_legacy_limit_violation_rad'])
            worst_speed=max(worst_speed,report['max_exported_joint_speed_rad_s'])
            records.append({'run':run,'condition':condition,'checkpoint_sha256':ckpt,'events':report['events'],
              'generated_clips':len(rows),'groups_raw':report['groups'],'bounded_groups_diagnostic':bounded_groups,
              'amplitude_gate_passed':report['amplitude_gate_passed'],'all_exported_qpos_valid':report['all_exported_qpos_valid'],
              'report_sha256':sha(folder/'report.json'),'per_sample_sha256':sha(folder/'per_sample.json')})
    assert total==768
    value={'audit_release_sha256':sha(root/'audit_release.json'),'protocol_sha256':release['protocol_sha256'],
      'generated_clips':total,'independent_source_events':len(audit),'sampling_seeds':release['sampling_seeds'],
      'seen_training_caption_events':sum(r['text_id'] in train_texts for r in audit),
      'unseen_training_caption_events':sum(r['text_id'] not in train_texts for r in audit),
      'all_saved_artifacts_verified':True,'all_four_conditions_passed':all(r['amplitude_gate_passed'] and r['all_exported_qpos_valid'] for r in records),
      'max_legacy_limit_violation_rad':worst_legacy,'max_bounded_joint_speed_rad_s':worst_speed,
      'conditions':records,'artifact_sha256':artifact_hashes,'collector_sha256':sha(__file__),
      'M5_ready':False,'remaining':'Final review of retained audit failures and scope; numerical success is not final decision.'}
    atomic_json(root/'audit_evidence.json',value)
    print(json.dumps({k:v for k,v in value.items() if k not in ['artifact_sha256','conditions']},indent=2),flush=True)
    for r in records:print(r['run'],r['condition'],r['groups_raw'],flush=True)


if __name__=='__main__':main()
