#!/usr/bin/env python3
"""Controlled masking, frame weighting and double-precision gradient checks."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import torch
from training.m5_tail_prototypes import legality_reductions, temporal_supervision, VERSION
from scripts.run_m5_length_research import write, sha, now


def main(output):
    if output.exists(): raise FileExistsError('Preserve existing prototype verification')
    torch.set_num_threads(4)
    mask = torch.arange(12)[None] < torch.tensor([12, 6])[:, None]
    energy = torch.zeros(2, 12, 29, dtype=torch.float64); energy[0, 3, 2] = .8
    energy.requires_grad_(True); values = legality_reductions(energy, mask)
    expected = {'mean': .8 / (18 * 29), 'frame_topk': .8 / (18 * 3), 'frame_tail_topk': (12 * .8 / (3 * 2)) / 18}
    for key, value in expected.items(): assert abs(float(values[key]) - value) < 1e-14
    padded = torch.cat((energy.detach(), torch.full((2, 5, 29), 1e6, dtype=torch.float64)), 1).requires_grad_(True)
    padded_mask = torch.cat((mask, torch.zeros(2, 5, dtype=torch.bool)), 1); padded_values = legality_reductions(padded, padded_mask)
    for key in expected:
        assert abs(float(values[key] - padded_values[key])) < 1e-14
        a = torch.autograd.grad(values[key], energy, retain_graph=True)[0]
        b = torch.autograd.grad(padded_values[key], padded, retain_graph=True)[0]
        assert torch.equal(a, b[:, :12]) and b[:, 12:].abs().max() == 0 and b[:, :12][~mask].abs().max() == 0
    duplicated = legality_reductions(energy.repeat(2, 1, 1), mask.repeat(2, 1))
    for key in expected: assert abs(float(values[key] - duplicated[key])) < 1e-14
    truth = torch.arange(3, dtype=torch.float64)[None, :, None] / 30
    prediction = torch.zeros_like(truth); full = torch.ones(1, 3, dtype=torch.bool)
    assert abs(float(temporal_supervision(prediction, truth, full)) - .5) < 1e-14
    assert temporal_supervision(truth, truth, full) == 0
    rng = torch.Generator().manual_seed(8119)
    predicted = torch.rand(2, 12, 2, dtype=torch.float64, generator=rng).requires_grad_(True)
    target = torch.rand(2, 12, 2, dtype=torch.float64, generator=rng)
    loss = temporal_supervision(predicted, target, mask)
    padded_prediction = torch.cat((predicted.detach(), torch.full((2, 5, 2), -1e6, dtype=torch.float64)), 1).requires_grad_(True)
    padded_target = torch.cat((target, torch.full((2, 5, 2), 1e6, dtype=torch.float64)), 1)
    padded_loss = temporal_supervision(padded_prediction, padded_target, padded_mask)
    assert abs(float(loss - padded_loss)) < 1e-12
    gradient = torch.autograd.grad(loss, predicted)[0]; padded_gradient = torch.autograd.grad(padded_loss, padded_prediction)[0]
    assert torch.equal(gradient, padded_gradient[:, :12]) and padded_gradient[:, 12:].abs().max() == 0
    angles = torch.deg2rad(torch.tensor([178., 179., -179.], dtype=torch.float64))
    heading = torch.stack((angles.cos(), angles.sin()), -1)[None]
    assert (heading[:, 2] - heading[:, 1]).norm() < .035 and temporal_supervision(heading, heading, full) == 0
    small = (torch.rand(2, 4, 5, generator=rng, dtype=torch.float64) + .2).requires_grad_(True)
    small_mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
    assert torch.autograd.gradcheck(lambda e: torch.stack([legality_reductions(e, small_mask, 3, .5)[k] for k in expected]),
                                    (small,), eps=1e-6, atol=1e-5, rtol=1e-3)
    small_p = torch.rand(2, 4, 2, generator=rng, dtype=torch.float64, requires_grad=True)
    small_t = torch.rand(2, 4, 2, generator=rng, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda p: temporal_supervision(p, small_t, small_mask), (small_p,), eps=1e-6, atol=1e-5, rtol=1e-3)
    rejected = []
    for label, fn in (('empty_valid_clip', lambda: legality_reductions(small, torch.zeros_like(small_mask))),
                      ('invalid_topk', lambda: legality_reductions(small, small_mask, 6)),
                      ('no_valid_pairs', lambda: temporal_supervision(small_p, small_t, torch.zeros_like(small_mask)))):
        try: fn()
        except ValueError: rejected.append(label)
        else: raise AssertionError(label + ' accepted')
    write(output, {'passed': True, 'verified_at_utc': now(), 'operator': VERSION,
        'controlled_single_outlier_values': expected, 'original_frame_weighting_preserved': True,
        'padding_loss_and_gradient_invariance': True, 'duplicated_batch_invariance': True,
        'supervised_velocity_units_checked': True, 'heading_wrap_example_checked': True,
        'tail_and_temporal_double_precision_gradcheck': True, 'negative_cases_rejected': rejected,
        'prototype_sha256': sha(ROOT / 'training/m5_tail_prototypes.py'), 'source_sha256': sha(Path(__file__)),
        'new_optimizer_updates': 0, 'formal_training_implementation_approved': False, 'M5_complete': False})
    print({'passed': True, 'output': str(output)}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    main(p.parse_args().output)
