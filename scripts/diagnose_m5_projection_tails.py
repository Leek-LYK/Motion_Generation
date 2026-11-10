#!/usr/bin/env python3
"""Decompose retained endpoint projection tails without sampling or modifying qpos."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,mujoco
from training.m3 import atomic_json,sha
from training.g1_physical import PhysicalG1Converter,hinge_matrices


def main():
    p=argparse.ArgumentParser();p.add_argument('--study-dir',type=Path,required=True);p.add_argument('--step',type=int,default=50000);p.add_argument('--top-cases',type=int,default=1)
    a=p.parse_args();root=a.study_dir.resolve();out=root/'diagnostics'/f'projection_tails_{a.step:06d}.json'
    if out.exists():raise FileExistsError('Tail diagnosis already archived')
    plan=json.loads((root/'protocol.json').read_text());converter=PhysicalG1Converter();c=converter.constants;sk=converter.skeleton
    model=mujoco.MjModel.from_xml_path(str(converter.xml_path));data=mujoco.MjData(model);records=[]
    axes=torch.as_tensor(c['axes'],dtype=torch.float64);by_node={int(node):i for i,node in enumerate(c['indices'])}
    for name in plan['first_wave']:
        run=root/name;done=json.loads((run/'COMPLETED.json').read_text());folder=run/'evaluations'/f'generation_full_{a.step:06d}';report=json.loads((folder/'report.json').read_text())
        assert done['step']==a.step and sha(run/'checkpoints/latest.pt')==done['terminal_checkpoint_sha256'] and report['identity']==done['identity']
        rows=sorted(json.loads((folder/'per_sample.json').read_text()),key=lambda row:row['projection_node_max_m'],reverse=True)[:a.top_cases]
        mean=np.concatenate([np.load(run/'stats/motion'/g/'mean.npy') for g in ['global_root','body']]);std=np.concatenate([np.load(run/'stats/motion'/g/'std.npy') for g in ['global_root','body']]);scale=np.sqrt(std**2+np.float32(1e-5))
        for row in rows:
            stem=folder/row['artifact_prefix'];motion=np.load(stem.with_suffix('.npz'));bounded=np.loadtxt(stem.with_suffix('.csv'),delimiter=',');unbounded=np.loadtxt(stem.with_name(stem.name+'_unbounded.csv'),delimiter=',')
            with torch.inference_mode():
                rebuilt=converter.qpos_to_motion_dict(bounded)['posed_joints'].numpy();axis_only=converter.qpos_to_motion_dict(unbounded)['posed_joints'].numpy()
            raw=motion['posed_joints'];errors=np.linalg.norm(rebuilt-raw,axis=-1);frame,node=np.unravel_index(errors.argmax(),errors.shape)
            assert abs(errors.max()-row['projection_node_max_m'])<1e-5
            compiled=[]
            for q in bounded:
                data.qpos[:]=q;mujoco.mj_forward(model,data);compiled.append(np.stack([c['C'].T@data.xanchor[j] for j in c['joint_ids']]))
            compiled=np.asarray(compiled);agreement=float(np.linalg.norm(rebuilt[:,c['indices']]-compiled,axis=-1).max());assert agreement<2e-6
            chain=[];current=int(node)
            while current>=0 and current not in chain:
                chain.append(current);current=int(sk.joint_parents[current])
            local=motion['local_rot_mats'][frame,c['indices']].astype(np.float64);native=np.swapaxes(c['rest'],-1,-2)@local
            target=hinge_matrices(axes,torch.from_numpy(bounded[frame,7:])).numpy();relative=np.swapaxes(native,-1,-2)@target
            degrees=np.degrees(np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1)));joints=[]
            for index in np.argsort(degrees)[::-1][:8]:
                bone=int(c['indices'][index]);axis=c['axes'][index]
                joints.append({'joint':sk.bone_order_names[bone],'in_worst_node_chain':bone in chain,'bounded_rotation_correction_degrees':float(degrees[index]),
                  'axis_energy':float(.5*np.sum((native[index]@axis-axis)**2)),'unbounded_angle_rad':float(unbounded[frame,7+index]),'bounded_angle_rad':float(bounded[frame,7+index]),'limits_rad':c['limits'][index].tolist()})
            features=np.load(stem.with_name(stem.name+'_features.npy'))*scale+mean;six=features[frame,107:311].reshape(34,6);first=six[:,:3];second=six[:,3:]
            norms=np.linalg.norm(first,axis=-1);unit=first/np.maximum(norms[:,None],1e-6);residual=np.linalg.norm(second-unit*np.sum(unit*second,axis=-1)[:,None],axis=-1)
            records.append({'run':name,'checkpoint_sha256':done['terminal_checkpoint_sha256'],'report_sha256':sha(folder/'report.json'),
              'case':{k:row[k] for k in ['sample_id','group','text','seed','condition','frames','artifact_prefix']},'maximum_projection_error_m':float(errors.max()),
              'worst_frame':int(frame),'worst_time_s':frame/30,'worst_node':sk.bone_order_names[node],
              'worst_node_chain':[sk.bone_order_names[x] for x in chain],
              'unbounded_axis_projection_max_m':float(np.linalg.norm(axis_only-raw,axis=-1).max()),
              'additional_bound_vs_unbounded_max_m':float(np.linalg.norm(rebuilt-axis_only,axis=-1).max()),
              'root_qpos_bound_vs_unbound_max_difference':float(abs(bounded[:,:7]-unbounded[:,:7]).max()),
              'converter_vs_compiled_mjcf_joint_center_max_m':agreement,'worst_frame_max_rotation_correction_degrees':float(degrees.max()),
              'worst_frame_rot6d_min_first_column_norm':float(norms.min()),'worst_frame_rot6d_min_orthogonal_residual_norm':float(residual.min()),'largest_joint_corrections':joints,
              'retained_artifacts_sha256':{'motion':sha(stem.with_suffix('.npz')),'bounded_qpos':sha(stem.with_suffix('.csv')),'unbounded_qpos':sha(stem.with_name(stem.name+'_unbounded.csv'))}})
            print('projection tail',name,sk.bone_order_names[node],int(frame),'max',errors.max(),'axis-only',np.linalg.norm(axis_only-raw,axis=-1).max(),'bounds',np.linalg.norm(rebuilt-axis_only,axis=-1).max(),flush=True)
    result={'passed':True,'scope':'Completed development endpoints, independent compiled-MJCF replay of retained worst projection cases. Diagnostic only, no new generation or model updates.',
      'step':a.step,'top_cases_per_run':a.top_cases,'records':records,'source_sha256':sha(Path(__file__)),'physical_adapter_sha256':sha(ROOT/'training/g1_physical.py'),
      'model_or_qpos_modified':False,'new_optimizer_updates':0,'M5_complete':False}
    atomic_json(out,result);print(json.dumps({'output':str(out),'cases':len(records),'new_optimizer_updates':0},indent=2),flush=True)


if __name__=='__main__':main()
