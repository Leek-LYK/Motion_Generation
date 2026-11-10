#!/usr/bin/env python3
"""Freeze small, complete events from M2 without modifying any upstream data."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from training.m3 import INPUTS, atomic_json, configure, load_samples, make_model, motion_image, validate_inputs

PROMPTS = {
    "walk": ["A person walks forward at a moderate pace.",
             "A person walks forward in short steps at a normal pace with their arms swinging back and forth."],
    "jog": ["A person jogs forward at a steady pace.",
            "A person jogs forward in a straight line at a steady pace."],
    "jump": ["A person jumps forward from a standing position.",
             "A person jumps backward.", "A person jumps to the right while facing forward.",
             "A person jumps to the left while facing forward."],
    "raise_arms": ["A person raises their arms towards both sides in a t pose and stands idle.",
                   "A person raises their arms and stands with them extended to their sides.",
                   "A person raises their arms while standing, facing forward."],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    ready = validate_inputs()
    manifest = {"schema": "m3-small-subset-v1", "m2_fingerprint": ready["fingerprint"],
                "seed": 20261101, "min_frames": 60, "max_frames": 90,
                "selection": "fixed short descriptions; complete events; exclude mirrored IDs; unique source motions",
                "human_semantic_review": "pending; reference previews provided; numeric pilot permitted",
                "train": [], "val": []}
    for split, wanted in (("train", 8), ("val", 4)):
        frame = pd.read_parquet(INPUTS / f"{split}.parquet")
        frame = frame[frame.num_frames.between(60, 90) & ~frame.motion_id.str.endswith("_M")].copy()
        for group, texts in PROMPTS.items():
            pool = frame[frame.text.isin(texts)].copy()
            pool["rank"] = pool.sample_id.map(lambda x: hashlib.sha256((x + "20261101").encode()).hexdigest())
            pool = pool.sort_values("rank").drop_duplicates("motion_id").head(wanted)
            if split == "train" and len(pool) != wanted:
                raise ValueError(f"Insufficient {group} events: {len(pool)}")
            for row in pool.to_dict("records"):
                manifest[split].append({k: row[k] for k in ("sample_id", "text_id", "text", "motion_id", "num_frames", "source_sha256")})
                manifest[split][-1]["group"] = group
    if not manifest["val"]:
        raise ValueError("No held-out observation events")
    atomic_json(args.output, manifest)
    configure(20261101)
    model = make_model({"latent_dim": 64, "ff_size": 128, "num_layers": 1, "num_heads": 4})
    output = args.output.parent / "references"
    output.mkdir(exist_ok=True)
    samples = load_samples(manifest, "train")
    from kimodo.exports.mujoco import MujocoQposConverter
    converter = MujocoQposConverter(model.motion_rep.skeleton)
    with torch.inference_mode():
        for i, sample in enumerate(samples):
            decoded = model.motion_rep.inverse(sample["x0"][None], is_normalized=True)
            positions = decoded["posed_joints"][0].numpy()
            img = motion_image(positions, positions, model.motion_rep.skeleton.joint_parents.tolist(),
                               f"REFERENCE ONLY {i:02d}: {sample['text']}")
            img.save(output / f"{i:02d}_{sample['group']}.png")
            np.savez(output / f"{i:02d}_{sample['group']}.npz", **{k: v[0].numpy() for k, v in decoded.items()})
            np.savetxt(output / f"{i:02d}_{sample['group']}.csv", converter.dict_to_qpos(decoded)[0], delimiter=",")
    single = dict(manifest)
    single["train"] = manifest["train"][:1]
    single["val"] = manifest["val"][:4]
    atomic_json(args.output.with_name("single.json"), single)
    print(json.dumps({"manifest": str(args.output), "train": len(manifest["train"]), "val": len(manifest["val"]),
                      "groups": {g: sum(x["group"] == g for x in manifest["train"]) for g in PROMPTS}}, indent=2))


if __name__ == "__main__":
    main()
