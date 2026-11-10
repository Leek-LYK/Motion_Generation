"""Validate alignment against actual NPZ clocks and test the Dataset/DataLoader."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from build_text_motion_index import DEFAULT_OUTPUT, frame_range, atomic_json, digest
from data_pipeline.text_motion_dataset import G1TextMotionDataset, collate_text_motion, FIELDS


def main():
    output = DEFAULT_OUTPUT
    ready_path = output / "READY"
    if ready_path.exists():
        ready_path.unlink()
    config = json.loads((output / "config.json").read_text())
    assert config["script_sha256"] == digest(ROOT / "scripts/build_text_motion_index.py")
    assert config["metadata_sha256"] == digest(ROOT / "data/raw/bones_seed/metadata/seed_metadata_v004.parquet")
    labels_path = ROOT / "data/raw/bones_seed/metadata/seed_metadata_v002_temporal_labels.jsonl"
    assert config["labels_sha256"] == digest(labels_path)
    assert config["node_manifest_sha256"] == digest(ROOT / config["nodes_relative_path"] / "manifest.sqlite")
    df = pd.read_parquet(output / "segments.parquet")
    motions = pd.read_parquet(output / "motions.parquet")
    assert len(df) == 352703 and len(motions) == 142220 and df.sample_id.is_unique
    assert set(df.motion_id) == set(motions.motion_id)
    valid = df[df.alignment_valid]
    assert ((valid.start_frame >= 0) & (valid.end_frame_exclusive <= valid.source_output_frames) & (valid.num_frames == valid.end_frame_exclusive - valid.start_frame) & (valid.num_frames > 0)).all()
    active = motions[motions.split != "excluded"]
    assert active.groupby("source_group_id").split.nunique().max() == 1
    trainval = motions[motions.split.isin(["train", "val"])]
    assert trainval.groupby("validation_group_id").split.nunique().max() == 1
    assert trainval.groupby("take_name").split.nunique().max() == 1
    assert trainval.groupby("mirror_key").split.nunique().max() == 1
    assert trainval.groupby("source_sha256").split.nunique().max() == 1
    official_tests = motions.official_split.isin(["test_content", "test_repetition"])
    assert (motions.loc[official_tests, "split"] == motions.loc[official_tests, "official_split"]).all()
    tol = config["boundary_tolerance_seconds"]
    cases = [
        (0, 1, (0, 30), ""), (.1, .2, (3, 6), ""), (.101, .2, (4, 6), ""),
        (-.01, .1, (0, 3), ""), (1.5, 2.01, (45, 60), ""),
        (.001, .01, (1, 1), "empty_frame_interval"),
        (0, 2.1, (0, 0), "outside_source_time_tolerance"),
        (2, 2.01, (0, 0), "interval_outside_source"),
        (.5, .5, (0, 0), "nonpositive_interval"),
        (float("nan"), 1, (0, 0), "nonfinite_time"),
    ]
    for start, end, expected, reason in cases:
        result = frame_range(start, end, 2, 60, tol)
        assert result[:2] == expected and result[-1] == reason, (start, end, result)
    selected = df[df.profile_eligible]
    split_ids = set()
    for split in ("train", "val", "test_content", "test_repetition"):
        subset = pd.read_parquet(output / f"{split}.parquet")
        assert set(subset.sample_id) == set(selected.loc[selected.split == split, "sample_id"])
        assert subset.num_frames.between(config["profile_min_frames"], config["profile_max_frames"]).all()
        assert not (split_ids & set(subset.sample_id))
        split_ids.update(subset.sample_id)
    assert len(split_ids) == len(selected)
    print("Full index and grouping invariants passed; checking actual NPZ slices", flush=True)

    # Cover all categories plus repaired boundaries, quarantined, short, and long events.
    samples = []
    for _, group in df.groupby("category"):
        samples.extend(group.sort_values("sample_id").head(5).index.tolist())
    for mask in (df.boundary_clipped, df.split == "excluded", df.num_frames < 30, df.num_frames > 300):
        samples.extend(df[mask].sort_values("sample_id").head(20).index.tolist())
    samples = list(dict.fromkeys(samples))
    extra = df.drop(index=samples).sample(n=200 - len(samples), random_state=20261030).index.tolist()
    samples.extend(extra)
    assert len(samples) == 200
    sample_rows = df.loc[samples].to_dict("records")
    needed = {r["filename"] for r in sample_rows}
    annotations = {}
    for line in labels_path.open():
        record = json.loads(line)
        if record["filename"] in needed:
            annotations[record["filename"]] = record

    def check(row):
        try:
            event = annotations[row["filename"]]["events"][row["event_index"]]
            assert event["description"] == row["text_original"]
            assert event["description"].strip() == row["text"]
            assert float(event["start_time"]) == row["raw_start_s"]
            assert float(event["end_time"]) == row["raw_end_s"]
            path = ROOT / row["npz_path"]
            assert digest(path) == row["npz_sha256"]
            with np.load(path, allow_pickle=False) as z:
                timestamps = z["timestamps"]
                assert len(timestamps) == row["source_output_frames"]
                assert str(z["source_sha256"]) == row["source_sha256"]
                assert str(z["pipeline_fingerprint"]) == config["nodes_fingerprint"]
                a = int(np.searchsorted(timestamps, row["aligned_start_s"] - 1e-10, side="left"))
                b = int(np.searchsorted(timestamps, row["aligned_end_s"] - 1e-10, side="left"))
                assert (a, b) == (row["start_frame"], row["end_frame_exclusive"]), (a, b)
                assert b - a == row["num_frames"]
                for key in FIELDS:
                    arr = z[key][a:b]
                    assert len(arr) == b - a and np.isfinite(arr).all(), key
            return {"sample_id": row["sample_id"], "passed": True}
        except Exception as error:
            return {"sample_id": row["sample_id"], "passed": False, "error": repr(error)}

    with ThreadPoolExecutor(max_workers=8) as pool:
        checked = list(pool.map(check, sample_rows))
    if not all(r["passed"] for r in checked):
        atomic_json(output / "sample_validation.json", {"passed": False, "samples": checked})
        raise RuntimeError("Actual NPZ alignment validation failed")

    torch.set_num_threads(1)
    dataset = G1TextMotionDataset("train", cache_megabytes=32, require_ready=False)
    lengths = [r["num_frames"] for r in dataset.rows]
    ids = [int(np.argmin(lengths)), int(np.argmax(lengths)), 0, len(dataset) // 2]
    records = [dataset[i] for i in ids]
    batch = collate_text_motion(records)
    assert torch.equal(batch["valid_mask"].sum(-1), batch["lengths"])
    for i, record in enumerate(records):
        for key in FIELDS:
            assert torch.equal(batch["motion"][key][i, :record["length"]], record["motion"][key])
    assert batch["motion"]["local_rot_mats"].shape[:2] == batch["valid_mask"].shape
    loader = DataLoader(Subset(dataset, list(range(12))), batch_size=4, num_workers=2,
                        collate_fn=collate_text_motion, multiprocessing_context="spawn")
    iterator = iter(loader)
    workers = list(iterator._workers)
    batches = list(iterator)
    assert len(batches) == 3
    for multi in batches:
        assert torch.equal(multi["valid_mask"].sum(-1), multi["lengths"])
        assert len(multi["texts"]) == 4
    for worker in workers:
        worker.join(timeout=10)
        assert worker.exitcode == 0, f"Worker did not exit cleanly: {worker.pid}, {worker.exitcode}"
    del iterator
    report = {"fingerprint": config["fingerprint"], "passed": True, "samples_checked": len(checked),
              "samples_passed": sum(r["passed"] for r in checked), "frame_range_cases": len(cases),
              "collate_variable_lengths": batch["lengths"].tolist(), "dataloader_spawn_workers": 2,
              "dataloader_batches_consumed": len(batches), "dataloader_worker_exitcodes": [w.exitcode for w in workers],
              "dataloader_batch_shape": list(multi["motion"]["posed_joints"].shape),
              "dataset_code_sha256": digest(ROOT / "data_pipeline/text_motion_dataset.py"), "samples": checked}
    atomic_json(output / "sample_validation.json", report)
    atomic_json(ready_path, {"fingerprint": config["fingerprint"], "structural_alignment_passed": True,
                           "annotation_events": len(df), "profile_selected": len(selected),
                           "split_sample_counts": selected.split.value_counts().to_dict(),
                           "human_semantic_review": "pending", "normalized_features_ready": False,
                           "text_embeddings_ready": False, "sample_checks": len(checked),
                           "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    atomic_json(output / "status.json", {"phase": "ready", "source_motions_processed": len(motions),
                                        "events_processed": len(df), "updated_at_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
