"""Expanded denoising diagnostics; generation retains the frozen M5 metric definitions."""
import numpy as np
from training.m5_evaluation import evaluate_generation,evaluate_denoising as original_denoising
from scripts.diagnose_m5_geometry import denoising_details


def evaluate_denoising(model,samples,device,batch_size=16):
    training=model.training
    result=original_denoising(model,samples,device,batch_size)
    records=denoising_details(model,samples,headings=(None,),batch_size=batch_size)
    names=('fk_gt_norm','consistency_norm','consistency_smooth_m','fk_gt_m','fk_self_m','axis_defect','axis_angle_rad')
    result['parts'].update({key:float(np.mean([r[key] for r in records])) for key in names})
    result['geometry_diagnostics']={'per_timestep':{str(t):{key:float(np.mean([r[key] for r in records if r['t']==t])) for key in names}
        for t in (0,249,499,749,999)},'per_group':{g:{key:float(np.mean([r[key] for r in records if r['group']==g])) for key in names}
        for g in sorted({r['group'] for r in records})},'records':records,
        'note':'FK-to-ground-truth supervision and predicted-position/FK self-consistency are distinct metrics.'}
    model.train(training)
    return result
