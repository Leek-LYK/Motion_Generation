#!/usr/bin/env python3
"""Sample a local M3 checkpoint using frozen cached text; no language model load."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import ImageDraw
from training.m3 import (atomic_json, batch_to, configure, export_samples, load_samples,
                         make_model, motion_image, sample_ddim, validate_inputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="latest.pt")
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--prompt-swap", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 2 <= args.ddim_steps <= 1000:
        raise ValueError("DDIM steps must be 2..1000")
    ready = validate_inputs()
    run = args.run_dir.resolve()
    state = torch.load(run / "checkpoints" / args.checkpoint, map_location="cpu", weights_only=False)
    if state["identity"]["m2_fingerprint"] != ready["fingerprint"]:
        raise ValueError("Checkpoint and data mismatch")
    configure(state["config"]["seed"])
    model = make_model(state["config"]["model"], run / "stats/motion").cuda().eval()
    model.load_state_dict(state["model" if args.weights == "raw" else "ema"])
    samples = load_samples(json.loads((run / "subset.json").read_text()), "train")
    representatives = []
    for sample in samples:
        if sample["group"] not in {x["group"] for x in representatives}:
            representatives.append(sample)
    if not args.prompt_swap:
        report = export_samples(model, representatives, args.output, args.seed, args.ddim_steps)
    else:
        # Every prompt sees the exact same initial noise, length and heading.
        from kimodo.exports.mujoco import MujocoQposConverter
        converter = MujocoQposConverter(model.motion_rep.skeleton)
        args.output.mkdir(parents=True)
        report = []
        with torch.inference_mode():
            for original in representatives:
                sample = dict(original, length=75, x0=torch.zeros(75, 417), first_heading_angle=torch.tensor(0.0))
                prediction = sample_ddim(model, batch_to([sample], "cuda:0"), args.seed, args.ddim_steps)
                decoded = model.motion_rep.inverse(prediction, is_normalized=True)
                qpos = converter.dict_to_qpos(decoded)[0]
                if not np.isfinite(qpos).all():
                    raise FloatingPointError("Invalid prompt-swap motion")
                name = original["group"]
                np.savez(args.output / f"{name}.npz", **{k: v[0].cpu().numpy() for k, v in decoded.items()})
                np.savetxt(args.output / f"{name}.csv", qpos, delimiter=",")
                xyz = decoded["posed_joints"][0].cpu().numpy()
                image = motion_image(xyz, xyz, model.motion_rep.skeleton.joint_parents.cpu().tolist(), original["text"])
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 27, image.width, 55), fill="white")
                draw.text((12, 30), "Generated only | same noise / 75 frames / heading 0 | no postprocess", fill="black")
                image.save(args.output / f"{name}.png")
                report.append({"text": original["text"], "group": name, "frames": 75, "heading": 0,
                               "seed": args.seed, "ground_truth_error": "not applicable: controlled prompt comparison"})
        atomic_json(args.output / "prompt_swap.json", report)
    print(json.dumps({"checkpoint_step": state["step"], "output": str(args.output), "report": report}, indent=2))


if __name__ == "__main__":
    main()
