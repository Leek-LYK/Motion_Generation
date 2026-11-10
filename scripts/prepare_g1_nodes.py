"""Offline SEED CSV -> 30 Hz Kimodo G1 nodes, with resumable state and progress."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sqlite3
import time
import traceback

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/bones_seed"
DEFAULT_OUTPUT = ROOT / "data/processed/g1_30hz_nodes_v1"
SCHEMA = "kimodo-g1-nodes-v1"
_worker = None


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def configuration():
    from importlib.metadata import version
    source = ROOT.parent / "kimodo/source"
    files = sorted((source / "kimodo").rglob("*.py"))
    files += sorted(p for p in (source / "kimodo/assets/skeletons/g1skel34").rglob("*")
                    if p.is_file() and p.suffix in (".p", ".xml"))
    fingerprint = {str(p.relative_to(source)): sha(p.read_bytes()) for p in files}
    split_dir = ROOT / "data/raw/kimodo_benchmark/splits"
    payload = {
        "schema": SCHEMA, "source_fps": 120, "target_fps": 30,
        "sampling": "source_frame_indices=0,4,8,...; no interpolation, filtering, or extrapolation",
        "world_axes": "Kimodo right-handed Y-up, +Z forward",
        "qpos_axes": "MuJoCo Z-up, +X forward; xyz meters, quaternion wxyz, 29 angles radians",
        "euler": "scipy Rotation.from_euler('xyz', degrees=True)",
        "mujoco_rest_zero": False, "calculation_dtype": "float64", "storage_dtype": "float32",
        "rotation_tolerance": 1e-4,
        "source_archive_sha256": "52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a",
        "metadata_sha256": sha((RAW / "metadata/seed_metadata_v004.parquet").read_bytes()),
        "splits_sha256": {p.name: sha(p.read_bytes()) for p in sorted(split_dir.glob("*.txt"))},
        "source_code_and_assets_sha256": fingerprint,
        "script_sha256": sha(Path(__file__).read_bytes()),
        "versions": {p: version(p) for p in ("torch", "numpy", "pandas", "scipy", "kimodo")},
    }
    payload["fingerprint"] = sha(json.dumps(payload, sort_keys=True).encode())
    return payload


def load_jobs():
    import pandas as pd
    data = pd.read_parquet(RAW / "metadata/seed_metadata_v004.parquet")
    split_map = {}
    for p in sorted((ROOT / "data/raw/kimodo_benchmark/splits").glob("*.txt")):
        for line in p.read_text().splitlines():
            key = line.strip()
            if not key:
                continue
            if key in split_map:
                raise ValueError(f"Overlapping official split: {key}")
            split_map[key] = p.stem.replace("_split_paths", "")
    jobs = []
    for row in data.to_dict("records"):
        path = Path(row["move_g1_path"])
        rel = path.relative_to("g1/csv")
        if ".." in path.parts:
            raise ValueError(f"Unsafe source path: {path}")
        key = str(rel.with_suffix(""))
        jobs.append({"key": key, "filename": row["filename"], "source": str(path),
                     "output": str(Path("motions") / rel.with_suffix(".npz")),
                     "expected_frames": int(row["move_duration_frames"]),
                     "split": split_map[key], "category": str(row["category"]),
                     "package": str(row["package"])})
    if len({r["key"] for r in jobs}) != len(jobs):
        raise ValueError("Duplicate source keys")
    return sorted(jobs, key=lambda r: (r["category"], r["key"]))


def initialize_worker(output, fingerprint):
    global _worker
    import torch
    from kimodo.skeleton.registry import build_skeleton
    from kimodo.exports.mujoco import MujocoQposConverter
    import xml.etree.ElementTree as ET
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    skeleton = build_skeleton(34)
    converter = MujocoQposConverter(skeleton)
    xml = ET.parse(converter.xml_path).getroot()
    joint_names = [j.get("name") for j in xml.find("worldbody").findall(".//joint")]
    if len(joint_names) != 29 or len(skeleton.bone_order_names) != 34:
        raise ValueError("Unexpected G1 topology")
    _worker = (Path(output), fingerprint, skeleton, converter, joint_names)


def convert(job):
    import numpy as np
    import pandas as pd
    import torch
    from scipy.spatial.transform import Rotation
    started = time.monotonic()
    try:
        output, fingerprint, sk, converter, joints = _worker
        source = RAW / job["source"]
        raw = source.read_bytes()
        source_hash = sha(raw)
        frame = pd.read_csv(io.BytesIO(raw))
        columns = (["Frame"] + ["root_translate" + x for x in "XYZ"]
                   + ["root_rotate" + x for x in "XYZ"] + [j + "_dof" for j in joints])
        if len(frame.columns) != 36 or set(frame.columns) != set(columns):
            raise ValueError(f"Unexpected CSV schema: {list(frame.columns)}")
        a = frame[columns].to_numpy(dtype=np.float64)
        n = len(a)
        if n != job["expected_frames"]:
            raise ValueError(f"Frame-count mismatch: CSV={n}, metadata={job['expected_frames']}")
        if not np.isfinite(a).all():
            raise ValueError("Nonfinite source CSV")
        if not np.array_equal(a[:, 0], np.arange(n)):
            raise ValueError("Frame numbers are not contiguous zero-based indices")
        indices = np.arange(0, n, 4, dtype=np.int64)
        if len(indices) < 2:
            raise ValueError("Clip too short for derived velocities")
        sampled = a[indices]
        quat = Rotation.from_euler("xyz", sampled[:, 4:7], degrees=True).as_quat()[:, [3, 0, 1, 2]]
        qpos = np.concatenate((sampled[:, 1:4] / 100, quat, np.deg2rad(sampled[:, 7:])), axis=-1)
        with torch.inference_mode():
            motion = converter.qpos_to_motion_dict(qpos, 30.0, mujoco_rest_zero=False)
        arrays = {}
        for name, value in motion.items():
            arr = value.detach().cpu().numpy()
            if not np.isfinite(arr).all():
                raise ValueError(f"Nonfinite output: {name}")
            arrays[name] = arr.astype(np.float32)
        t = len(indices)
        expected = {"posed_joints": (t, 34, 3), "local_rot_mats": (t, 34, 3, 3),
                    "global_rot_mats": (t, 34, 3, 3), "root_positions": (t, 3),
                    "smooth_root_pos": (t, 3), "global_root_heading": (t, 2), "foot_contacts": (t, 4)}
        for name, shape in expected.items():
            if arrays[name].shape != shape:
                raise ValueError(f"Unexpected shape: {name} {arrays[name].shape}")
        max_ortho = max_det = 0.0
        for name in ("local_rot_mats", "global_rot_mats"):
            r = arrays[name].astype(np.float64)
            max_ortho = max(max_ortho, float(np.max(np.abs(r.swapaxes(-2, -1) @ r - np.eye(3)))))
            max_det = max(max_det, float(np.max(np.abs(np.linalg.det(r) - 1))))
        if max_ortho > 1e-4 or max_det > 1e-4:
            raise ValueError(f"Invalid rotation matrices: orthogonality={max_ortho}, determinant={max_det}")
        arrays.update(qpos_mujoco=qpos.astype(np.float32), fps=np.array(30, np.int32),
                      source_fps=np.array(120, np.int32), source_frame_indices=indices,
                      timestamps=indices / 120.0, source_frame_count=np.array(n, np.int64),
                      node_names=np.asarray(sk.bone_order_names), joint_names=np.asarray(joints),
                      source_sha256=np.array(source_hash), pipeline_fingerprint=np.array(fingerprint),
                      source_relative_path=np.array(job["source"]), official_split=np.array(job["split"]))
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        payload = buffer.getvalue()
        # Validate the serialized representation before publishing.
        with np.load(io.BytesIO(payload), allow_pickle=False) as check:
            if check["posed_joints"].shape != (t, 34, 3) or str(check["pipeline_fingerprint"]) != fingerprint:
                raise ValueError("Serialized output validation failed")
        dest = output / job["output"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_name(dest.name + ".tmp")
        with temp.open("wb") as stream:
            stream.write(payload)
        os.replace(temp, dest)
        return {"key": job["key"], "status": "complete", "source_sha256": source_hash,
                "output_sha256": sha(payload), "bytes": len(payload), "frames": t,
                "source_frames": n, "max_orthogonality_error": max_ortho,
                "max_determinant_error": max_det, "elapsed_s": time.monotonic() - started}
    except Exception as error:
        return {"key": job["key"], "status": "failed", "error": str(error),
                "traceback": traceback.format_exc(), "elapsed_s": time.monotonic() - started}


def status(output):
    path = output / "status.json"
    if not path.exists():
        print("No conversion status yet")
        return
    data = json.loads(path.read_text())
    print(json.dumps(data, ensure_ascii=False, indent=2))


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = configuration()
    config_path = output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text())["fingerprint"] != config["fingerprint"]:
            raise RuntimeError("Pipeline fingerprint changed; use a new versioned output directory")
    elif (output / "manifest.sqlite").exists() or (output / "motions").exists():
        raise RuntimeError("Existing artifacts without configuration; refusing to overwrite")
    else:
        atomic_json(config_path, config)
    jobs = load_jobs()
    by_key = {j["key"]: j for j in jobs}
    db = sqlite3.connect(output / "manifest.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS motions (motion_id TEXT PRIMARY KEY, source_path TEXT, output_path TEXT, official_split TEXT, category TEXT, package TEXT, status TEXT, result_json TEXT)")
    db.executemany("INSERT OR IGNORE INTO motions VALUES (?,?,?,?,?,?,?,?)",
                   [(j["key"], j["source"], j["output"], j["split"], j["category"], j["package"], "pending", "{}") for j in jobs])
    db.commit()
    records = {r[0]: (r[1], json.loads(r[2])) for r in db.execute("SELECT motion_id,status,result_json FROM motions")}
    complete = failed = saved_frames = saved_bytes = 0
    pending = []
    for j in jobs:
        state, result = records[j["key"]]
        dest = output / j["output"]
        if state == "complete" and dest.is_file() and dest.stat().st_size == result.get("bytes"):
            complete += 1; saved_frames += result["frames"]; saved_bytes += result["bytes"]
        elif state == "failed" and not args.retry_failed:
            failed += 1
        else:
            db.execute("UPDATE motions SET status='pending',result_json='{}' WHERE motion_id=?", (j["key"],))
            pending.append(j)
    db.commit()
    if args.limit:
        # A deterministic stratified pilot: round-robin across metadata categories.
        groups = {}
        for j in pending:
            groups.setdefault(j["category"], []).append(j)
        selected = []
        index = 0
        while len(selected) < min(args.limit, len(pending)):
            for group in groups.values():
                if index < len(group) and len(selected) < args.limit:
                    selected.append(group[index])
            index += 1
        pending = selected
    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    processed = 0
    stop = False
    last_error = None
    def stop_signal(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, stop_signal)
    signal.signal(signal.SIGTERM, stop_signal)

    def report(phase):
        elapsed = time.monotonic() - start
        remaining = len(jobs) - complete - failed
        rate = processed / elapsed if elapsed else 0
        data = {"phase": phase, "pid": os.getpid(), "started_at_utc": started_at,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(), "total": len(jobs),
                "complete": complete, "failed": failed, "pending": remaining,
                "percent_complete": round(complete / len(jobs) * 100, 4), "workers": args.workers,
                "processed_this_run": processed, "elapsed_s": round(elapsed, 1),
                "motions_per_second": round(rate, 3),
                "eta_seconds": round(remaining / rate) if rate and remaining else None,
                "output_frames": saved_frames, "output_bytes": saved_bytes,
                "last_error": last_error, "pipeline_fingerprint": config["fingerprint"]}
        db.commit()
        atomic_json(output / "status.json", data)
        print(f"[{data['updated_at_utc']}] {phase}: {complete}/{len(jobs)} ({data['percent_complete']:.2f}%), failed={failed}, {rate:.2f} motions/s, ETA={data['eta_seconds']}s", flush=True)

    report("running")
    iterator = iter(pending)
    active = {}
    last_report = time.monotonic()
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                initializer=initialize_worker, initargs=(str(output), config["fingerprint"])) as pool:
        def fill():
            while not stop and len(active) < args.workers * 2:
                job = next(iterator, None)
                if job is None:
                    break
                active[pool.submit(convert, job)] = job
        fill()
        while active:
            done, _ = cf.wait(active, timeout=1, return_when=cf.FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = {"key": job["key"], "status": "failed", "error": str(error)}
                processed += 1
                if result["status"] == "complete":
                    complete += 1; saved_frames += result["frames"]; saved_bytes += result["bytes"]
                else:
                    failed += 1; last_error = {"motion_id": job["key"], "error": result["error"]}
                db.execute("UPDATE motions SET status=?,result_json=? WHERE motion_id=?",
                           (result["status"], json.dumps(result), job["key"]))
            fill()
            if time.monotonic() - last_report >= 5:
                report("stopping" if stop else "running")
                last_report = time.monotonic()
    phase = "stopped" if stop else ("pilot_complete" if args.limit else ("complete_with_failures" if failed else "complete"))
    report(phase)
    db.close()
    lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "status"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0:
        parser.error("workers must be positive; limit must be nonnegative")
    if args.command == "status":
        status(args.output)
    else:
        run(args)


if __name__ == "__main__":
    main()
