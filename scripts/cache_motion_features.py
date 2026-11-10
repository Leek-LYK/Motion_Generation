"""Build packed, unnormalized 417-D motion caches and train-only statistics."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import concurrent.futures as cf
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from data_pipeline.numeric_common import INDEX, OUTPUT, SPLITS, sha_file, fingerprint, atomic_json, atomic_parquet, source_versions

_builder = None


def init_worker():
    global _builder
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from data_pipeline.motion_features import SegmentFeatureBuilder
    _builder = SegmentFeatureBuilder()


def process_shard(task):
    from data_pipeline.motion_features import moments
    split, number, rows, fp = task
    name = f"{split}/shard_{number:05d}"
    final = OUTPUT / "motions" / name
    temp = final.with_name(final.name + ".building")
    started = time.monotonic()
    try:
        if temp.exists():
            shutil.rmtree(temp)
        temp.mkdir(parents=True)
        pieces, locals_, index = [], [], []
        previous = None
        local = root = None
        offset = 0
        for row in rows:
            if row["npz_path"] != previous:
                with np.load(ROOT / row["npz_path"], allow_pickle=False) as z:
                    local, root = z["local_rot_mats"], z["root_positions"]
                    if str(z["source_sha256"]) != row["source_sha256"]:
                        raise ValueError("Source provenance mismatch")
                previous = row["npz_path"]
            a, b = int(row["start_frame"]), int(row["end_frame_exclusive"])
            raw, lr, heading, origin = _builder(local[a:b], root[a:b])
            if len(raw) != row["num_frames"]:
                raise ValueError("Segment frame count mismatch")
            pieces.append(raw)
            if split == "train":
                locals_.append(lr)
            item = {k: row[k] for k in ("sample_id", "text_id", "text", "text_row", "split", "motion_id", "npz_path", "source_sha256", "start_frame", "end_frame_exclusive", "num_frames")}
            item.update(feature_path=f"motions/{name}/features.npy", frame_offset=offset,
                        first_heading_angle=heading, origin_x=float(origin[0]), origin_z=float(origin[1]))
            index.append(item)
            offset += len(raw)
        packed = np.concatenate(pieces)
        buffer = io.BytesIO()
        np.save(buffer, packed, allow_pickle=False)
        payload = buffer.getvalue()
        (temp / "features.npy").write_bytes(payload)
        pd.DataFrame(index).to_parquet(temp / "index.parquet", index=False)
        if split == "train":
            n, mean, m2 = moments(packed)
            nl, ml, sl = moments(np.concatenate(locals_))
            assert n == nl
            np.savez(temp / "moments.npz", count=n, mean=mean, m2=m2, local_mean=ml, local_m2=sl)
        result = {"shard": name, "fingerprint": fp, "samples": len(rows), "frames": len(packed),
                  "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                  "split": split, "elapsed_s": time.monotonic() - started}
        atomic_json(temp / "result.json", result)
        if final.exists():
            raise RuntimeError(f"Existing unverified shard: {final}")
        temp.rename(final)
        return result
    except Exception as error:
        return {"shard": name, "error": str(error), "traceback": traceback.format_exc()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit-shards", type=int, default=0)
    args = p.parse_args()
    if args.workers < 1 or args.limit_shards < 0:
        p.error("Invalid limits")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / "motion.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    index_ready = json.loads((INDEX / "READY").read_text())
    assert index_ready["structural_alignment_passed"]
    frames = [pd.read_parquet(INDEX / f"{s}.parquet") for s in SPLITS]
    full = pd.concat(frames, ignore_index=True)
    texts = full[["text_id", "text"]].drop_duplicates("text_id").sort_values("text_id").reset_index(drop=True)
    texts["text_row"] = np.arange(len(texts))
    text_rows = dict(zip(texts.text_id, texts.text_row))
    cfg = {"schema": "g1-model-inputs-v1", "alignment_fingerprint": index_ready["fingerprint"],
           "split_sha256": {s: sha_file(INDEX / f"{s}.parquet") for s in SPLITS},
           "feature_builder_sha256": sha_file(ROOT / "data_pipeline/motion_features.py"),
           "runner_sha256": sha_file(__file__), **source_versions(),
           "fps": 30, "feature_dim": 417, "global_root_dim": 5, "body_dim": 412, "local_root_dim": 4,
           "feature_storage": "unnormalized float32 packed NPY; normalize at Dataset read",
           "geometry_dtype": "float64", "root_policy": "subtract actual first-frame root XZ before FK; preserve height and heading",
           "temporal_policy": "recompute smooth root, world velocity, heading, contact separately per event at 30Hz",
           "heading_augmentation": "none in cache", "stats_policy": "population variance, valid frames only, selected train only",
           "normalization_epsilon": 1e-5, "shard_samples": 256,
           "human_semantic_review": "pending", "samples": len(full), "frames": int(full.num_frames.sum()), "unique_texts": len(texts)}
    cfg["fingerprint"] = fingerprint(cfg)
    config_path = OUTPUT / "config.json"
    if config_path.exists() and json.loads(config_path.read_text())["fingerprint"] != cfg["fingerprint"]:
        raise RuntimeError("Configuration changed; use a new output version")
    atomic_json(config_path, cfg)
    atomic_parquet(texts, OUTPUT / "texts.parquet")
    tasks, results, failures = [], {}, []
    for split, frame in zip(SPLITS, frames):
        frame = frame.sort_values(["motion_id", "event_index"])
        frame["text_row"] = frame.text_id.map(text_rows)
        rows = frame.to_dict("records")
        for number, start in enumerate(range(0, len(rows), 256)):
            batch = rows[start:start + 256]
            task = (split, number, batch, cfg["fingerprint"])
            name = f"{split}/shard_{number:05d}"
            path = OUTPUT / "motions" / name
            if (path / "result.json").exists():
                r = json.loads((path / "result.json").read_text())
                if r["fingerprint"] != cfg["fingerprint"] or not (path / "features.npy").is_file() or (path / "features.npy").stat().st_size != r["bytes"]:
                    raise RuntimeError(f"Invalid existing shard: {name}")
                results[name] = r
            else:
                tasks.append(task)
    total_shards = len(tasks) + len(results)
    if args.limit_shards:
        # Include each split in the pilot when possible.
        selected = []
        for s in SPLITS:
            selected.extend([t for t in tasks if t[0] == s][:max(1, args.limit_shards // 4)])
        tasks = selected[:args.limit_shards]
    start_time = time.monotonic()
    starting_done = sum(r["samples"] for r in results.values())
    stop = False
    def stop_signal(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    def progress(phase):
        done = sum(r["samples"] for r in results.values())
        elapsed = time.monotonic() - start_time
        rate = (done - starting_done) / max(elapsed, 1e-9)
        record = {"phase": phase, "pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                  "completed_samples": done, "total_samples": len(full), "completed_shards": len(results),
                  "total_shards": total_shards, "failed_shards": len(failures), "samples_per_second": rate,
                  "eta_seconds": (len(full) - done) / rate if rate else None,
                  "output_frames": sum(r["frames"] for r in results.values()), "elapsed_s": elapsed}
        atomic_json(OUTPUT / "motion_status.json", record)
        print(f"{phase}: {done}/{len(full)} samples, {len(results)}/{total_shards} shards, failed={len(failures)}, rate={rate:.1f}/s", flush=True)
    progress("running")
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"), initializer=init_worker) as pool:
        iterator = iter(tasks)
        active = {}
        def fill():
            while not stop and len(active) < args.workers * 2:
                task = next(iterator, None)
                if task is None:
                    break
                active[pool.submit(process_shard, task)] = task
        fill()
        last_report = time.monotonic()
        while active:
            done, _ = cf.wait(active, timeout=1, return_when=cf.FIRST_COMPLETED)
            for future in done:
                task = active.pop(future)
                r = future.result()
                if "error" in r:
                    failures.append(r)
                    atomic_json(OUTPUT / "motion_failures.json", failures)
                else:
                    results[r["shard"]] = r
            fill()
            if time.monotonic() - last_report >= 5:
                progress("stopping" if stop else "running")
                last_report = time.monotonic()
    if stop or args.limit_shards or failures or len(results) != total_shards:
        progress("stopped" if stop else ("failed" if failures else "pilot_complete"))
        return
    progress("computing_train_statistics")
    from data_pipeline.motion_features import merge_moments
    all_moments = (0, np.zeros(417), np.zeros(417))
    local_moments = (0, np.zeros(4), np.zeros(4))
    indexes = []
    for name in sorted(results):
        path = OUTPUT / "motions" / name
        indexes.append(pd.read_parquet(path / "index.parquet"))
        if results[name]["split"] == "train":
            with np.load(path / "moments.npz", allow_pickle=False) as z:
                n = int(z["count"])
                all_moments = merge_moments(all_moments, (n, z["mean"], z["m2"]))
                local_moments = merge_moments(local_moments, (n, z["local_mean"], z["local_m2"]))
    index = pd.concat(indexes, ignore_index=True)
    assert index.sample_id.is_unique and set(index.sample_id) == set(full.sample_id)
    n, mean, m2 = all_moments
    assert n == int(frames[0].num_frames.sum()) and n == local_moments[0]
    for group, m, variance in [("global_root", mean[:5], m2[:5] / n), ("body", mean[5:], m2[5:] / n),
                               ("local_root", local_moments[1], local_moments[2] / n)]:
        path = OUTPUT / "stats/motion" / group
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "mean.npy", m)
        np.save(path / "std.npy", np.sqrt(np.maximum(variance, 0)))
    for split in SPLITS:
        atomic_parquet(index[index.split == split], OUTPUT / f"{split}.parquet")
    atomic_parquet(index, OUTPUT / "samples.parquet")
    atomic_json(OUTPUT / "stats/report.json", {"train_samples": len(frames[0]), "train_frames": n,
                "population_variance": True, "epsilon": 1e-5, "other_splits_used": False,
                "train_index_sha256": cfg["split_sha256"]["train"], "fingerprint": cfg["fingerprint"]})
    atomic_json(OUTPUT / "MOTION_COMPLETE", {"fingerprint": cfg["fingerprint"], "samples": len(index),
                "frames": int(index.num_frames.sum()), "train_stats_frames": n, "shards": len(results)})
    progress("complete")


if __name__ == "__main__":
    main()
