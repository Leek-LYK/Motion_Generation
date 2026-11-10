"""Explicit, bounded nearest-hinge decoding for G1; never modifies motion features.

The frozen upstream converter supplies coordinate transforms and root pose. Its
Euler extraction is retained as the separately archived legacy output. This
decoder maximizes trace(H(theta).T @ R) over each MJCF hinge's legal interval,
which minimizes both SO(3) geodesic distance and Frobenius distance. There is no
temporal smoothing, root correction, IK, contact correction, or dynamics.
"""
import numpy as np
import torch
import mujoco
from kimodo.exports.mujoco import MujocoQposConverter

DECODER_VERSION = "g1-nearest-bounded-hinge-v1"


def nearest_angles(rotations, axes, bounds, limited):
    """Return [T,J] nearest angles. Ties choose the temporally closest optimum."""
    r = np.asarray(rotations, dtype=np.float64)
    u = np.asarray(axes, dtype=np.float64)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    a = np.trace(r, axis1=-2, axis2=-1) - np.einsum('ji,tjik,jk->tj', u, r, u)
    skew = np.stack((r[..., 2, 1]-r[..., 1, 2], r[..., 0, 2]-r[..., 2, 0],
                     r[..., 1, 0]-r[..., 0, 1]), axis=-1)
    b = np.einsum('tji,ji->tj', skew, u)
    theta = np.arctan2(b, a)
    result = np.empty_like(theta)
    for j in range(len(u)):
        previous = 0.0
        if limited[j]:
            lo, hi = bounds[j]
            turns = np.arange(int(np.floor((lo-np.pi)/(2*np.pi))),
                              int(np.ceil((hi+np.pi)/(2*np.pi)))+1)
        for t in range(len(r)):
            if limited[j]:
                interior = theta[t, j] + 2*np.pi*turns
                candidates = np.r_[lo, hi, interior[(interior >= lo) & (interior <= hi)],
                                   np.clip(previous, lo, hi)]
                scores = a[t, j]*np.cos(candidates) + b[t, j]*np.sin(candidates)
                best = candidates[scores >= scores.max()-1e-12]
                angle = best[np.argmin(abs(best-previous))]
            elif np.hypot(a[t, j], b[t, j]) < 1e-12:
                angle = previous
            else:
                angle = theta[t, j] + 2*np.pi*np.round((previous-theta[t, j])/(2*np.pi))
            result[t, j] = previous = angle
    return result


class G1BoundedDecoder:
    def __init__(self, skeleton):
        self.converter = MujocoQposConverter(skeleton)
        self.model = mujoco.MjModel.from_xml_path(str(self.converter.xml_path))
        self.joints = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
        self.addresses = self.model.jnt_qposadr[self.joints]
        self.bounds = self.model.jnt_range[self.joints].copy()
        self.limited = self.model.jnt_limited[self.joints].astype(bool)
        self.indices = self.converter._mujoco_indices_to_kimodo_indices.cpu().numpy()
        self.axes = self.converter._mujoco_joint_axis_values_f2q_space.cpu().numpy()
        self.offsets = self.converter._rot_offsets_f2q.cpu().numpy()[self.indices]
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in self.joints]
        expected = [skeleton.bone_order_names[i].replace('_skel', '_joint') for i in self.indices]
        if names != expected or not np.array_equal(self.addresses, np.arange(7, 36)):
            raise ValueError('G1 XML / skeleton hinge order mismatch')

    def decode(self, motion):
        """Single unbatched motion dict -> float64 [T,36] qpos, diagnostics."""
        local = torch.as_tensor(motion['local_rot_mats']).detach().cpu().float()
        root = torch.as_tensor(motion['root_positions']).detach().cpu().float()
        if local.ndim != 4 or root.shape != (len(local), 3):
            raise ValueError('Expected one motion with [T,J,3,3] rotations')
        if not torch.isfinite(local).all() or not torch.isfinite(root).all():
            raise ValueError('Nonfinite generated motion')
        legacy = self.converter.dict_to_qpos({'local_rot_mats': local[None],
                                             'root_positions': root[None]})[0].astype(np.float64)
        r = self.offsets[None] @ local.numpy()[:, self.indices]
        angles = nearest_angles(r, self.axes, self.bounds, self.limited)
        qpos = legacy.copy()
        qpos[:, self.addresses] = angles
        for t in range(1, len(qpos)):
            if np.dot(qpos[t-1, 3:7], qpos[t, 3:7]) < 0:
                qpos[t, 3:7] *= -1  # Same orientation, continuous quaternion sign.
        return qpos, {'decoder': DECODER_VERSION, 'postprocess': True,
                      'root_translation_max_change_m': float(abs(qpos[:, :3]-legacy[:, :3]).max()),
                      'raw_vs_decoded_hinge_max_difference_rad': float(abs(angles-legacy[:, 7:]).max()),
                      'legacy': self.quality(legacy), 'bounded': self.quality(qpos)}

    def quality(self, qpos, fps=30):
        qpos = np.asarray(qpos)
        q = qpos[:, self.addresses]
        violations = np.maximum(0, np.maximum(self.bounds[:, 0]-q, q-self.bounds[:, 1]))[:, self.limited]
        speeds = abs(np.diff(q, axis=0))*fps
        return {'finite': bool(np.isfinite(qpos).all()),
                'quaternion_norm_max_error': float(abs(np.linalg.norm(qpos[:, 3:7], axis=1)-1).max()),
                'max_limit_violation_rad': float(violations.max(initial=0)),
                'violating_frame_fraction': float(np.mean(np.any(violations > 1e-6, axis=1))),
                'max_joint_speed_rad_s': float(speeds.max(initial=0)),
                'p99_joint_speed_rad_s': float(np.quantile(speeds, .99)) if speeds.size else 0.0}
