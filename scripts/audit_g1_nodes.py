"""Audit a finished conversion and publish READY only when all checks pass."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sqlite3

import numpy as np
import pandas as pd

from prepare_g1_nodes import ROOT, RAW, DEFAULT_OUTPUT, configuration, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=200)
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("sample-size must be positive")
    output = args.output.resolve()
    state = json.loads((output / "status.json").read_text())
    if state["phase"] in ("running", "stopping"):
        raise RuntimeError("Wait until the conversion process finishes before the final audit")
    config = json.loads((output / "config.json").read_text())
    if configuration()["fingerprint"] != config["fingerprint"]:
        raise RuntimeError("Pipeline fingerprint changed since conversion")
    metadata = pd.read_parquet(RAW / "metadata/seed_metadata_v004.parquet")
    expected_count = len(metadata)
    expected_source_frames = int(metadata["move_duration_frames"].sum())
    expected_frames = int(((metadata["move_duration_frames"] + 3) // 4).sum())
    with sqlite3.connect(f"file:{output / 'manifest.sqlite'}?mode=ro", uri=True) as db:
        rows = db.execute("SELECT motion_id,source_path,output_path,official_split,status,result_json FROM motions ORDER BY motion_id").fetchall()
    counts = Counter(r[4] for r in rows)
    completed = [r for r in rows if r[4] == "complete"]
    expected_paths = {r[2] for r in completed}
    actual_paths = {str(p.relative_to(output)) for p in (output / "motions").rglob("*.npz")}
    results = [json.loads(r[5]) for r in completed]
    report = {
        "pipeline_fingerprint": config["fingerprint"],
        "expected_motions": expected_count, "manifest_rows": len(rows), "status_counts": dict(counts),
        "split_counts": dict(Counter(r[3] for r in completed)),
        "expected_source_frames": expected_source_frames,
        "source_frames": sum(r["source_frames"] for r in results),
        "expected_output_frames": expected_frames, "output_frames": sum(r["frames"] for r in results),
        "output_bytes": sum(r["bytes"] for r in results),
        "missing_outputs": len(expected_paths - actual_paths), "unexpected_outputs": len(actual_paths - expected_paths),
        "max_orthogonality_error": max((r["max_orthogonality_error"] for r in results), default=None),
        "max_determinant_error": max((r["max_determinant_error"] for r in results), default=None),
        "sample_seed": 20261030, "samples": [],
    }
    sample = random.Random(20261030).sample(completed, min(args.sample_size, len(completed)))

    def check(row):
        motion_id, source_path, output_path, split, _, raw_result = row
        result = json.loads(raw_result)
        try:
            path = output / output_path
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != result["output_sha256"]:
                raise ValueError("Output SHA256 mismatch")
            if hashlib.sha256((RAW / source_path).read_bytes()).hexdigest() != result["source_sha256"]:
                raise ValueError("Source SHA256 mismatch")
            with np.load(path, allow_pickle=False) as data:
                t = result["frames"]
                assert data["posed_joints"].shape == (t, 34, 3)
                assert data["local_rot_mats"].shape == data["global_rot_mats"].shape == (t, 34, 3, 3)
                assert data["qpos_mujoco"].shape == (t, 36)
                assert len(data["node_names"]) == 34 and len(data["joint_names"]) == 29
                assert data["fps"].item() == 30 and data["source_fps"].item() == 120
                assert str(data["pipeline_fingerprint"]) == config["fingerprint"]
                assert str(data["source_sha256"]) == result["source_sha256"]
                assert str(data["official_split"]) == split
                assert np.array_equal(data["source_frame_indices"], np.arange(t) * 4)
                assert np.allclose(data["timestamps"], np.arange(t) / 30, atol=1e-12, rtol=0)
                assert data["source_frame_indices"][-1] < result["source_frames"]
                for key in data.files:
                    arr = data[key]
                    if arr.dtype.kind in "fciub" and not np.isfinite(arr).all():
                        raise ValueError(f"Nonfinite output: {key}")
                expected_root = data["qpos_mujoco"][:, [1, 2, 0]]
                assert np.allclose(data["root_positions"], expected_root, atol=5e-6, rtol=0)
            return {"motion_id": motion_id, "passed": True}
        except Exception as error:
            return {"motion_id": motion_id, "passed": False, "error": str(error)}

    with ThreadPoolExecutor(max_workers=8) as pool:
        report["samples"] = list(pool.map(check, sample))
    report["sample_count"] = len(sample)
    report["sample_passed"] = sum(r["passed"] for r in report["samples"])
    report["passed"] = (
        len(rows) == expected_count and counts.get("complete", 0) == expected_count
        and report["source_frames"] == expected_source_frames and report["output_frames"] == expected_frames
        and not report["missing_outputs"] and not report["unexpected_outputs"]
        and report["sample_passed"] == len(sample) and len(sample) > 0
    )
    atomic_json(output / "audit_report.json", report)
    if not report["passed"]:
        raise RuntimeError("Final audit failed; see audit_report.json; READY not created")
    atomic_json(output / "READY", {k: v for k, v in report.items() if k != "samples"})
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
