"""Independently read every train feature byte and verify frame-weighted stats."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import json
import hashlib
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from data_pipeline.numeric_common import OUTPUT, sha_file, atomic_json
from data_pipeline.motion_features import moments, merge_moments


def main():
    config = json.loads((OUTPUT / "config.json").read_text())
    assert json.loads((OUTPUT / "MOTION_COMPLETE").read_text())["fingerprint"] == config["fingerprint"]
    train = pd.read_parquet(OUTPUT / "train.parquet")
    aggregate = (0, np.zeros(417), np.zeros(417))
    checks = []
    groups = train.groupby("feature_path", sort=True)
    for i, (name, rows) in enumerate(groups):
        path = OUTPUT / name
        before = path.stat()
        expected = json.loads((path.parent / "result.json").read_text())
        assert expected["split"] == "train" and expected["fingerprint"] == config["fingerprint"]
        h = hashlib.sha256()
        with path.open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            assert version == (1, 0)
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            header_size = stream.tell()
            assert dtype == np.dtype("float32") and not fortran
            assert shape == (int(rows.num_frames.sum()), 417)
            stream.seek(0)
            h.update(stream.read(header_size))
            n = 0
            while True:
                block = stream.read(8192 * 417 * 4)
                if not block:
                    break
                h.update(block)
                arr = np.frombuffer(block, dtype=np.float32).reshape(-1, 417)
                assert np.isfinite(arr).all()
                aggregate = merge_moments(aggregate, moments(arr))
                n += len(arr)
            assert n == shape[0]
        assert h.hexdigest() == expected["sha256"]
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        checks.append({"path": name, "sha256": h.hexdigest(), "bytes": after.st_size, "mtime_ns": after.st_mtime_ns})
        if (i + 1) % 100 == 0:
            print(f"Train statistics audit {i+1}/{len(groups)} shards", flush=True)
    n, mean, m2 = aggregate
    stats = OUTPUT / "stats/motion"
    saved_mean = np.concatenate([np.load(stats / g / "mean.npy") for g in ("global_root", "body")])
    saved_std = np.concatenate([np.load(stats / g / "std.npy") for g in ("global_root", "body")])
    assert n == int(train.num_frames.sum())
    assert np.allclose(mean, saved_mean, atol=1e-9, rtol=1e-9)
    assert np.allclose(np.sqrt(m2 / n), saved_std, atol=1e-9, rtol=1e-9)
    report = {"passed": True, "fingerprint": config["fingerprint"], "train_frames": n,
              "train_samples": len(train), "train_index_sha256": sha_file(OUTPUT / "train.parquet"),
              "mean_max_abs_error": float(np.max(np.abs(mean - saved_mean))),
              "std_max_abs_error": float(np.max(np.abs(np.sqrt(m2/n) - saved_std))),
              "stats_files_sha256": {str(p.relative_to(OUTPUT)): sha_file(p) for p in sorted(stats.rglob("*.npy"))},
              "other_splits_used": False, "all_train_shards_sha256_verified": True, "shards": checks,
              "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(OUTPUT / "stats/independent_audit.json", report)
    print(json.dumps({k:v for k,v in report.items() if k!='shards'},indent=2),flush=True)


if __name__ == "__main__":
    main()
