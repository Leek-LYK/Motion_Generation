"""Amplitude diagnostics and explicit M5-entry evidence requirements (not paper metrics)."""
import numpy as np


def movement(motion, skeleton, group):
    root=motion['root_positions'].detach().cpu().numpy()
    if group in ('walk','jog'):
        xy=root[:,[0,2]]
        smooth=np.stack([np.convolve(np.pad(xy[:,i],(2,2),mode='edge'),np.array([1,2,3,2,1])/9,mode='valid') for i in range(2)],axis=1)
        return float(np.linalg.norm(np.diff(smooth,axis=0),axis=-1).sum())
    if group=='jump':
        return float(np.quantile(root[:,1],.9)-np.quantile(root[:,1],.1))
    positions=motion['posed_joints'].detach().cpu().numpy()
    wrists=[skeleton.bone_index['left_wrist_yaw_skel'],skeleton.bone_index['right_wrist_yaw_skel']]
    root_rot=motion['global_rot_mats'][:,skeleton.root_idx].detach().cpu().numpy()
    local=np.einsum('tji,tkj->tki',root_rot,positions[:,wrists]-root[:,None,:])
    ranges=np.quantile(local,.9,axis=0)-np.quantile(local,.1,axis=0)
    return float(np.linalg.norm(ranges,axis=-1).mean())


def summarize(rows, protocol):
    groups={}
    for group in ('walk','jog','jump','raise_arms'):
        subset=[r for r in rows if r['group']==group and r['reference_amplitude']>=protocol['reference_minimum'][group]]
        ratios=np.array([r['amplitude_ratio'] for r in subset])
        if not len(ratios):
            groups[group]={'passed':False,'eligible':0};continue
        noncollapsed=float(np.mean(ratios>=protocol['noncollapse_ratio_min']))
        median=float(np.median(ratios))
        groups[group]={'eligible':len(subset),'median_amplitude_ratio':median,'noncollapsed_fraction':noncollapsed,
                      'mean_world_mpjpe_m':float(np.mean([r['world_mpjpe_m'] for r in subset])),
                      'passed':noncollapsed>=protocol['required_noncollapsed_fraction'] and protocol['median_ratio_range'][0]<=median<=protocol['median_ratio_range'][1]}
    return {'groups':groups,'amplitude_gate_passed':all(g['passed'] for g in groups.values()),
            'all_finite':all(r['finite'] for r in rows),'M5_ready':False,
            'remaining_evidence':'Independent audit split, repeat training seed and qualitative text-swap review are required separately.'}
