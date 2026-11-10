"""Diagnostic-only masked tail/temporal reductions; no registered training recipe."""
import math
import torch
import torch.nn.functional as F

VERSION = 'm5-masked-tail-temporal-diagnostic-v1'


def legality_reductions(energy, mask, top_hinges=3, frame_fraction=.1):
    """Preserve original frame weighting while concentrating within each clip.

    Native energy is [B,T,J]. Tail frames are selected independently per clip;
    their average is weighted by that clip's original valid-frame count.
    """
    if energy.ndim != 3 or mask.shape != energy.shape[:2] or mask.dtype != torch.bool:
        raise ValueError('Expected energy[B,T,J] and boolean mask[B,T]')
    if not 1 <= top_hinges <= energy.shape[-1] or not 0 < frame_fraction <= 1:
        raise ValueError('Invalid tail reduction parameters')
    counts = mask.sum(-1)
    if (counts <= 0).any() or not torch.isfinite(energy[mask]).all():
        raise ValueError('Each clip needs finite valid energy and positive length')
    values = torch.where(mask[..., None], energy, torch.zeros_like(energy))
    total = counts.sum()
    mean = values.sum() / (total * energy.shape[-1])
    frame = values.topk(top_hinges, dim=-1).values.mean(-1)
    frame_topk = (frame * mask).sum() / total
    tail_mask = torch.zeros_like(mask); tails = []
    for i in range(len(energy)):
        ids = mask[i].nonzero(as_tuple=False).flatten()
        k = max(1, math.ceil(len(ids) * frame_fraction))
        selected = frame[i, ids].topk(k).indices
        tail_mask[i, ids[selected]] = True
        tails.append(frame[i, ids[selected]].mean() * counts[i])
    return {'mean': mean, 'frame_topk': frame_topk, 'frame_tail_topk': torch.stack(tails).sum() / total,
            'selected_tail_frames': tail_mask}


def temporal_supervision(prediction, target, mask, fps=30.):
    """SmoothL1 on supervised finite differences; never filters output motion."""
    if prediction.ndim != 3 or prediction.shape != target.shape or mask.shape != prediction.shape[:2] or mask.dtype != torch.bool:
        raise ValueError('Expected matching[B,T,D] features and boolean[B,T] mask')
    if not math.isfinite(fps) or fps <= 0: raise ValueError('Invalid frame rate')
    pairs = mask[:, 1:] & mask[:, :-1]
    if not pairs.any(): raise ValueError('Temporal supervision requires a valid adjacent pair')
    predicted_delta = (prediction[:, 1:] - prediction[:, :-1]) * fps
    target_delta = (target[:, 1:] - target[:, :-1]) * fps
    errors = F.smooth_l1_loss(predicted_delta, target_delta, reduction='none')
    return torch.where(pairs[..., None], errors, torch.zeros_like(errors)).sum() / (pairs.sum() * prediction.shape[-1])
