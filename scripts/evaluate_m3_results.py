#!/usr/bin/env python3
"""Post-training diagnostics on the fixed small M3 train/validation subsets."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from training.m3 import (Trainer, atomic_json, batch_to, configure, forward, load_samples,
                         losses, sample_ddim, sha, validate_inputs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    ready = validate_inputs()
    run = args.run_dir.resolve()
    run_config = json.loads((run / "run_config.json").read_text())
    subset = json.loads((run / "subset.json").read_text())
    train, val = load_samples(subset, "train"), load_samples(subset, "val")
    configure(run_config["config"]["seed"])
    trainer = Trainer(run_config["config"], train, run_config["identity"])
    checkpoint = run / "checkpoints/latest.pt"
    trainer.restore(checkpoint)
    events = EventAccumulator(str(run / "tensorboard"), size_guidance={"scalars": 0})
    events.Reload()
    curves, summary = {}, {}
    for tag in ("fixed_train/loss", "fixed_train_ema/loss", "heldout_val/loss", "heldout_val/t999",
                "fixed_train/t999", "generation_raw/world_mpjpe_m", "generation_ema/world_mpjpe_m"):
        points = events.Scalars(tag)
        best = min(points, key=lambda x: x.value)
        curves[tag] = [{"step": x.step, "value": x.value, "wall_time": x.wall_time} for x in points]
        summary[tag] = {"initial": points[0].value, "final": points[-1].value,
                        "best": best.value, "best_step": best.step}
    atomic_json(args.output / "curves.json", curves)
    fixed, generation = {}, []
    with torch.inference_mode():
        for weights, model in (("raw", trainer.model), ("ema", trainer.ema)):
            fixed[weights] = {"all_train32": trainer.evaluate(train, model), "heldout5": trainer.evaluate(val, model)}
            for split, samples in (("train", train), ("val", val)):
                batch = batch_to(samples, "cuda:0")
                for seed in (0, 42, 1234):
                    predicted = sample_ddim(model, batch, seed, steps=100)
                    for i, sample in enumerate(samples):
                        n = sample["length"]
                        pred = model.motion_rep.inverse(predicted[i:i+1, :n], is_normalized=True)
                        ref = model.motion_rep.inverse(batch["x0"][i:i+1, :n], is_normalized=True)
                        a, b = pred["posed_joints"][0], ref["posed_joints"][0]
                        if not all(torch.isfinite(x).all() for x in pred.values()):
                            raise FloatingPointError("Invalid generated motion")
                        generation.append({"weights": weights, "split": split, "seed": seed,
                            "sample_id": sample["sample_id"], "group": sample["group"], "frames": n,
                            "world_mpjpe_m": float(torch.linalg.vector_norm(a-b, dim=-1).mean()),
                            "root_relative_mpjpe_m": float(torch.linalg.vector_norm((a-a[:, :1])-(b-b[:, :1]), dim=-1).mean()),
                            "root_position_error_m": float(torch.linalg.vector_norm(pred["root_positions"]-ref["root_positions"],dim=-1).mean())})
                    print(f"generated {weights} {split} seed={seed} count={len(samples)}", flush=True)
        # Cross-group text shuffle with exactly the same x_t, heading, length and target.
        batch = batch_to(train, "cuda:0")
        rng = torch.Generator(device="cuda:0").manual_seed(20261102)
        noise = torch.randn(batch["x0"].shape, generator=rng, device="cuda:0")
        t = torch.full((len(train),), 999, device="cuda:0", dtype=torch.long)
        noisy = trainer.diffusion.q_sample(batch["x0"], t, noise)
        ablations = {}
        trainer.ema.eval()
        for name, text in (("correct", batch["text_feat"]), ("cross_group_shuffle", batch["text_feat"].roll(8,0)),
                           ("zero_text_out_of_training_distribution", torch.zeros_like(batch["text_feat"]))):
            condition = dict(batch, text_feat=text)
            prediction = forward(trainer.ema, noisy, condition, t)
            loss, _ = losses(prediction, batch["x0"], batch["x_pad_mask"])
            ablations[name] = float(loss)
    summaries = {}
    for weights in ("raw", "ema"):
        for split in ("train", "val"):
            rows = [r for r in generation if r["weights"] == weights and r["split"] == split]
            summaries[f"{weights}/{split}"] = {"sequences":len(rows), **{metric:{"mean":float(np.mean([r[metric] for r in rows])),
                "median":float(np.median([r[metric] for r in rows])), "p95":float(np.quantile([r[metric] for r in rows],.95))}
                for metric in ("world_mpjpe_m", "root_relative_mpjpe_m", "root_position_error_m")}}
    atomic_json(args.output / "per_sample_generation.json", generation)
    report = {"checkpoint":str(checkpoint), "checkpoint_sha256":sha(checkpoint), "checkpoint_step":trainer.step,
              "m2_fingerprint":ready["fingerprint"], "evaluation_script_sha256":sha(__file__),
              "curve_summary":summary, "fixed_denoising":fixed, "generation_summary":summaries,
              "ema_train_t999_text_ablation":ablations, "seeds":[0,42,1234], "postprocess":False,
              "limitations":["Only 32 training and 5 held-out events; val covers walk/jog only",
                 "Paired generation errors are overfit diagnostics, not distributional/generative benchmark scores",
                 "Zero-text was never trained; null-text ablation is an out-of-distribution control",
                 "No dynamics tracking or real-robot validation"]}
    atomic_json(args.output / "report.json", report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
