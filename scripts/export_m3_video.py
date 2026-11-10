#!/usr/bin/env python3
"""One-command cached-text generation and traceable MuJoCo MP4 export."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from kimodo.assets import skeleton_asset_path
from kimodo.exports.mujoco import MujocoQposConverter
from training.m3 import INPUTS, atomic_json, batch_to, configure, make_model, sample_ddim, sha, validate_inputs
from training.video import render
from training.video_archive import freeze_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--checkpoint", default="latest.pt")
    p.add_argument("--weights", choices=("raw", "ema"), default="ema")
    p.add_argument("--prompt", required=True, help="Exact text in the frozen M2 cache")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--duration", type=float, help="Seconds; default matches a training event, otherwise 2.5")
    p.add_argument("--heading", type=float, help="Radians; default matches a training event, otherwise 0")
    p.add_argument("--ddim-steps", type=int, default=100)
    args = p.parse_args()
    if not 2 <= args.ddim_steps <= 1000:
        raise ValueError("Invalid DDIM step count")
    ready = validate_inputs()
    run = args.run_dir.resolve()
    relative_run = run.relative_to(ROOT / "runs")
    checkpoint = (run / "checkpoints" / args.checkpoint).resolve()
    checkpoint.relative_to(run / "checkpoints")
    checksum = sha(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if sha(checkpoint) != checksum:
        raise RuntimeError("Checkpoint changed while reading; retry")
    if state["identity"]["m2_fingerprint"] != ready["fingerprint"]:
        raise ValueError("M2 fingerprint mismatch")
    if state["identity"]["training_code_sha256"] != sha(ROOT / "training/m3.py"):
        raise ValueError("Training implementation changed")
    texts = pd.read_parquet(INPUTS / "texts.parquet")
    matches = texts[texts.text == args.prompt]
    if len(matches) != 1:
        raise ValueError("Prompt is not in the frozen cache. Encode new text with the verified LLM2Vec separately; no silent substitution.")
    row = matches.iloc[0]
    subset = json.loads((run / "subset.json").read_text())
    refs = [s for s in subset["train"] if s["text"] == args.prompt]
    train_rows = pd.read_parquet(INPUTS / "train.parquet")
    ref = train_rows[train_rows.sample_id == refs[0]["sample_id"]].iloc[0] if refs else None
    frames = round(args.duration * 30) if args.duration is not None else int(ref.num_frames) if ref is not None else 75
    heading = args.heading if args.heading is not None else float(ref.first_heading_angle) if ref is not None else 0.0
    if not 30 <= frames <= 300 or not np.isfinite(heading):
        raise ValueError("Use 1..10 seconds and a finite heading")
    stats_hashes = {}
    for file in (run / "stats/motion").rglob("*.npy"):
        relative = file.relative_to(run)
        stats_hashes[str(relative)] = sha(file)
        if sha(file) != sha(INPUTS / relative):
            raise ValueError(f"Run statistics differ from validated M2: {relative}")
    configure(state["config"]["seed"])
    model = make_model(state["config"]["model"], run / "stats/motion").cuda().eval()
    model.load_state_dict(state["model" if args.weights == "raw" else "ema"])
    embeddings = np.load(INPUTS / "text_encoder/embeddings.npy", mmap_mode="r")
    sample = {"x0": torch.zeros(frames, 417), "text_feat": torch.from_numpy(np.array(embeddings[int(row.text_row)], copy=True)),
              "length": frames, "first_heading_angle": torch.tensor(heading, dtype=torch.float32),
              "origin_xz": torch.zeros(2), "sample_id": "video_text_input", "text": args.prompt, "split": "inference"}
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S_%f%z")
    prompt_hash = hashlib.sha256(args.prompt.encode()).hexdigest()
    slug = re.sub(r"[^a-z0-9]+", "_", args.prompt.lower()).strip("_")[:55].rstrip("_")
    output = ROOT / "exports" / relative_run / f"step_{state['step']:06d}_{args.weights}_{checksum[:8]}" / f"{stamp}__{slug}__{prompt_hash[:8]}__seed{args.seed}"
    output.mkdir(parents=True, exist_ok=False)
    train_status = json.loads((run / "status.json").read_text())
    checkpoint_snapshot = freeze_model(run, checkpoint, checksum)
    xml_path = skeleton_asset_path("g1skel34", "xml", "g1.xml")
    metadata = {"schema": "kimodo-video-export-v1", "status": "generating", "exported_at": stamp,
                "timezone": "Asia/Shanghai", "training_run_id": str(relative_run), "training_run_path": str(run),
                "training_finished_at_utc": train_status.get("updated_at_utc"),
                "training_configuration": state["config"], "normalization_files_sha256": stats_hashes,
                "checkpoint": str(checkpoint), "checkpoint_sha256": checksum, "checkpoint_step": state["step"],
                "checkpoint_snapshot": str(checkpoint_snapshot),
                "weights": args.weights, "prompt": args.prompt, "prompt_sha256": prompt_hash,
                "text_id": str(row.text_id), "text_row": int(row.text_row), "prompt_seen_in_training": bool(refs),
                "reference_sample_id": str(ref.sample_id) if ref is not None else None,
                "seed": args.seed, "fps": 30, "frames": frames, "duration_s": frames / 30,
                "heading_radians": heading, "ddim_steps": args.ddim_steps, "cfg": "nocfg", "postprocess": False,
                "outside_m3_training_length": not 60 <= frames <= 90,
                "m2_fingerprint": ready["fingerprint"], "text_encoder_fingerprint": ready["text_config_fingerprint"],
                "source_code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in
                                       [Path(__file__), ROOT / "training/video.py", ROOT / "training/m3.py",
                                        ROOT / "training/video_archive.py"]},
                "robot_xml": str(xml_path), "robot_xml_sha256": sha(xml_path),
                "qpos_schema": "root xyz metres + quaternion wxyz + 29 joints rad; MuJoCo Z-up, X-forward",
                "physical_tracking_verified": False}
    atomic_json(output / "manifest.json", metadata)
    try:
        (output / "source").mkdir()
        for source in [Path(__file__), ROOT / "training/video.py", ROOT / "training/m3.py", ROOT / "training/video_archive.py"]:
            (output / "source" / source.name).write_bytes(source.read_bytes())
        np.save(output / "text_embedding.npy", sample["text_feat"].numpy())
        with torch.inference_mode():
            features = sample_ddim(model, batch_to([sample], "cuda:0"), args.seed, args.ddim_steps)
            decoded = model.motion_rep.inverse(features, is_normalized=True)
            qpos = MujocoQposConverter(model.motion_rep.skeleton).dict_to_qpos(decoded)[0]
            np.savez(output / "motion.npz", **{k: v[0].cpu().numpy() for k, v in decoded.items()})
            np.save(output / "normalized_features.npy", features[0].cpu().numpy())
            np.savetxt(output / "qpos.csv", qpos, delimiter=",")
        np.savetxt(output / "root_trajectory.csv", np.column_stack([np.arange(frames)/30, qpos[:, :3]]),
                   delimiter=",", header="time_s,x_m,y_m,z_m", comments="")
        metadata["render"] = render(qpos, output / "replay.mp4", metadata, xml_path)
        metadata["artifacts_sha256"] = {str(p.relative_to(output)): sha(p) for p in output.rglob("*")
                                         if p.is_file() and p.name != "manifest.json"}
        metadata["status"] = "complete"
        atomic_json(output / "manifest.json", metadata)
        from index_video_exports import build_index
        build_index(ROOT / "exports")
        print(json.dumps({"output": str(output), "video": str(output / "replay.mp4"),
                          "frames": frames, "checkpoint_step": state["step"]}, indent=2), flush=True)
    except Exception as error:
        atomic_json(output / "manifest.json", dict(metadata, status="failed", error=repr(error)))
        raise


if __name__ == "__main__":
    main()
