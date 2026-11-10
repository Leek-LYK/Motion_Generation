#!/usr/bin/env python3
"""Real CUDA updates, checkpoint continuation, mask and sampler-isolation checks."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from training.m3 import (Trainer, atomic_json, batch_to, configure, export_samples, load_samples,
                         losses, sample_ddim, sha, validate_inputs)


def max_difference(left, right):
    if isinstance(left, torch.Tensor):
        return float((left.cpu() - right.cpu()).abs().max()) if left.numel() else 0.0
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        return max([max_difference(left[k], right[k]) for k in left] or [0])
    if isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        return max([max_difference(a, b) for a, b in zip(left, right)] or [0])
    assert left == right
    return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    ready = validate_inputs()
    config = json.loads((ROOT / "configs/m3_tiny.json").read_text())
    # Exercise EMA updates within the six-step continuation test.
    config["ema_every"] = 2
    subset = json.loads(args.subset.read_text())
    samples = load_samples(subset, "train")[:4]
    identity = {"m2": ready["fingerprint"], "subset": sha(args.subset), "code": sha(ROOT / "training/m3.py")}
    configure(config["seed"])
    trainer = Trainer(config, samples, identity)
    initial = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    for _ in range(3):
        trainer.update()
    path = args.output / "resume_step3.pt"
    trainer.save(path)
    for _ in range(3):
        trainer.update()
    expected = {"model": {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()},
                "ema": {k: v.cpu().clone() for k, v in trainer.ema.state_dict().items()},
                "optimizer": trainer.optimizer.state_dict(), "batch_rng": trainer.batch_rng.get_state(),
                "noise_rng": trainer.noise_rng.get_state()}
    changed = {stage: any(not torch.equal(initial[k], v) for k, v in trainer.model.state_dict().items()
                          if k.startswith(stage + ".")) for stage in ("root_model", "body_model")}
    assert all(changed.values())
    resumed = Trainer(config, samples, identity)
    resumed.restore(path)
    for _ in range(3):
        resumed.update()
    actual = {"model": resumed.model.state_dict(), "ema": resumed.ema.state_dict(),
              "optimizer": resumed.optimizer.state_dict(), "batch_rng": resumed.batch_rng.get_state(),
              "noise_rng": resumed.noise_rng.get_state()}
    differences = {k: max_difference(expected[k], actual[k]) for k in expected}
    assert max(differences.values()) == 0.0, differences
    batch = batch_to(samples[:2], "cuda:0")
    before = resumed.diffusion.alphas_cumprod.clone()
    noise_state, batch_state = resumed.noise_rng.get_state(), resumed.batch_rng.get_state()
    evaluation = resumed.evaluate(samples[:2])
    a = sample_ddim(resumed.model, batch, 12345, steps=10)
    b = sample_ddim(resumed.model, batch, 12345, steps=10)
    assert torch.equal(a, b)
    assert torch.equal(before, resumed.diffusion.alphas_cumprod)
    assert torch.equal(noise_state, resumed.noise_rng.get_state())
    assert torch.equal(batch_state, resumed.batch_rng.get_state())
    pred, target = batch["x0"].clone(), batch["x0"].clone()
    target[~batch["x_pad_mask"]] = 1000
    pred[~batch["x_pad_mask"]] = -1000
    masked, _ = losses(pred, target, batch["x_pad_mask"])
    assert float(masked) == 0
    output_metrics = export_samples(resumed.model, samples[:1], args.output / "sampling", 12345, steps=10)
    import mujoco
    from kimodo.assets import skeleton_asset_path
    import numpy as np
    robot = mujoco.MjModel.from_xml_path(str(skeleton_asset_path("g1skel34", "xml", "g1.xml")))
    motion = np.loadtxt(next((args.output / "sampling").glob("*.csv")), delimiter=",")
    data = mujoco.MjData(robot)
    for qpos in motion:
        assert len(qpos) == robot.nq == 36
        data.qpos[:] = qpos
        mujoco.mj_forward(robot, data)
        assert np.isfinite(data.xpos).all()
    report = {"passed": True, "parameters": sum(p.numel() for p in resumed.model.parameters()),
              "real_cuda_steps": 6, "both_stages_parameters_changed": changed,
              "continuous_vs_resume_max_abs": differences, "same_seed_ddim_exact": True,
              "sampling_preserves_train_schedule_and_rng": True, "padding_loss_invariance": True,
              "mujoco_qpos_forward_all_frames_finite": True, "sampling_metrics_untrained": output_metrics,
              "fixed_noise_evaluation": evaluation, "human_semantic_review": "pending"}
    atomic_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
