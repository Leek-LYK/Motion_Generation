#!/usr/bin/env python3
"""Compiled-limit projection equivalence and independent finite differences."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scipy.spatial.transform import Rotation
from training.g1_legality_loss import NativeLegalityLoss,VERSION
from training.g1_physical import PhysicalG1Converter,hinge_matrices
from training.g1_qpos import nearest_angles
from training.m5_physical_losses import NativeHingeGeometry,geometry
from training.m5_physical_data import PhysicalSubsetDataset
from training.m3 import atomic_json,sha,make_model,batch_to,configure


def main():
    root=ROOT/'runs/m5_physical_20261103';out=root/'verification/legality_loss';out.mkdir(parents=True,exist_ok=True);configure(7421)
    converter=PhysicalG1Converter();loss=NativeLegalityLoss();c=converter.constants;rng=np.random.default_rng(7421)
    q=np.tile(c['qpos0'],(64,1));q[:,2]=.8;q[:,3:7]=[1,0,0,0]
    for j in range(29):q[:,7+j]=rng.uniform(c['limits'][j,0]+1e-4,c['limits'][j,1]-1e-4,64)
    motion=converter.qpos_to_motion_dict(q);oracle,_=loss.per_joint(motion['local_rot_mats']);oracle_max=float(oracle.max());assert oracle_max<1e-20
    # A pure native-axis rotation can be outside its legal interval: axis energy
    # remains zero, while bounded-manifold distance correctly becomes positive.
    limited=np.flatnonzero((c['limits'][:,1]<np.pi-.25)&(c['limits'][:,1]-c['limits'][:,0]<2*np.pi-.2));j=int(limited[0])
    bad=q[:1].copy();bad[0,7+j]=c['limits'][j,1]+.2;illegal=converter.qpos_to_motion_dict(bad)
    energy=NativeHingeGeometry(converter.skeleton).defects(illegal['local_rot_mats'].float());distance,_=loss.per_joint(illegal['local_rot_mats'])
    assert float(energy.max())<1e-10 and float(distance.max())>.01
    # Independent existing bounded-angle solver on random SO(3) predictions.
    native=torch.from_numpy(Rotation.random(64*29,random_state=rng).as_matrix().reshape(64,29,3,3));projected,angles=loss.project_native(native)
    expected=nearest_angles(native.numpy(),c['axes'],c['limits'],loss.limited.numpy());other=hinge_matrices(torch.from_numpy(c['axes']),torch.from_numpy(expected))
    error=float(abs(projected-other).max());assert error<2e-8,error
    # Finite difference recomputes the argmin, rather than differentiating the
    # same stopped target; it checks the actual minimum-distance function.
    point=native[:1].clone().requires_grad_(True);target,_=loss.project_native(point);value=.5*(point-target).square().sum();value.backward();gradient=point.grad.clone();step=1e-5;errors=[]
    for index in [(0,0,0,0),(0,0,1,2),(0,5,2,1),(0,11,1,0),(0,20,2,2)]:
        def evaluate(sign):
            x=point.detach().clone();x[index]+=sign*step;nearest,_=loss.project_native(x);return float(.5*(x-nearest).square().sum())
        numeric=(evaluate(1)-evaluate(-1))/(2*step);errors.append(abs(numeric-float(gradient[index])))
    assert max(errors)<2e-7,max(errors)
    view=json.loads((root/'data/READY.json').read_text());m=json.loads((root/'data/D_short_3500.json').read_text());entries=[m['val'][0],m['val'][-1]]
    samples=PhysicalSubsetDataset(entries,view['input_dir'],root/'stats/motion');batch=batch_to(samples.batch([0,1]),'cuda:0');rep=make_model({'latent_dim':64,'ff_size':128,'num_layers':1,'num_heads':4},root/'stats/motion').cuda().motion_rep
    prediction=batch['x0'].clone();prediction[...,107:311]+=.05*torch.randn_like(prediction[...,107:311]);prediction[~batch['x_pad_mask']]=11.;prediction.requires_grad_(True)
    _,_,decoded,_=geometry(rep,prediction);gpu=NativeLegalityLoss('cuda:0');value=gpu.loss(decoded['local_rot_mats'],batch['x_pad_mask']);value.backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad[...,107:311].norm()>0 and (prediction.grad[~batch['x_pad_mask']]==0).all()
    report={'passed':True,'version':VERSION,'checks':{'legal_pose_oracle_max':oracle_max,'pure_axis_out_of_limits_detected':True,
      'independent_bounded_rotation_max_difference':error,'envelope_finite_difference_max_error':max(errors),
      'actual_6d_prediction_rotation_gradients_finite_nonzero':True,'padding_gradients_zero':True},
      'optimizer_updates':0,'active_training_loss_changed':False,'code_sha256':{p:sha(ROOT/p) for p in ['training/g1_legality_loss.py','scripts/verify_g1_legality_loss.py']},
      'note':'Loss prototype only. No coefficients selected and no live experiment changed; projection tie minimizers need not match temporal export tie-breaking.'}
    atomic_json(out/'report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
