"""Kimodo feature construction on individual aligned segments."""
import numpy as np
import torch
from kimodo.skeleton.registry import build_skeleton
from kimodo.motion_rep import KimodoMotionRep


class SegmentFeatureBuilder:
    def __init__(self):
        self.skeleton = build_skeleton(34)
        self.rep = KimodoMotionRep(self.skeleton, fps=30, stats_path=None)
        if self.rep.motion_rep_dim != 417 or self.rep.local_root_dim != 4:
            raise ValueError("Unexpected feature layout")

    def __call__(self, local_rotations, root_positions):
        rotations = torch.as_tensor(np.asarray(local_rotations), dtype=torch.float64).unsqueeze(0)
        root = torch.as_tensor(np.array(root_positions, copy=True), dtype=torch.float64).unsqueeze(0)
        if len(root[0]) < 2:
            raise ValueError("A segment must contain at least two frames")
        origin_xz = root[0, 0, [0, 2]].clone()
        root[:, :, [0, 2]] -= origin_xz
        lengths = torch.tensor([root.shape[1]], dtype=torch.long)
        with torch.inference_mode():
            raw = self.rep(rotations, root, to_normalize=False, to_canonicalize=False, lengths=lengths)
            raw = raw.to(torch.float32)
            local_root = self.rep.global_root_to_local_root(raw[..., :5], normalized=False, lengths=lengths)
            first_heading = self.rep.get_root_heading_angle(raw)[0, 0]
        if raw.shape != (1, len(root[0]), 417) or not torch.isfinite(raw).all() or not torch.isfinite(local_root).all():
            raise ValueError("Invalid motion features")
        if not torch.all((raw[..., 413:] == 0) | (raw[..., 413:] == 1)):
            raise ValueError("Unexpected contact targets")
        return raw[0].numpy(), local_root[0].numpy(), float(first_heading), origin_xz.numpy()


def moments(array):
    x = np.asarray(array, dtype=np.float64)
    mean = x.mean(axis=0)
    return len(x), mean, ((x - mean) ** 2).sum(axis=0)


def merge_moments(left, right):
    n0, m0, s0 = left
    n1, m1, s1 = right
    if not n0:
        return n1, m1.copy(), s1.copy()
    delta = m1 - m0
    n = n0 + n1
    return n, m0 + delta * (n1 / n), s0 + s1 + delta**2 * (n0 * n1 / n)
