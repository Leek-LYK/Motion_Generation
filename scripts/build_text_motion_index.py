"""Build auditable text/30-Hz-motion indexes; leave node NPZ files untouched."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NODES = ROOT / "data/processed/g1_30hz_nodes_v1"
RAW = ROOT / "data/raw/bones_seed"
DEFAULT_OUTPUT = ROOT / "data/processed/g1_text_motion_v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def save_parquet(frame, path):
    tmp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


class UnionFind:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, k):
        if self.parent[k] != k:
            self.parent[k] = self.find(self.parent[k])
        return self.parent[k]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def unite_equal(uf, rows, field):
    first = {}
    for r in rows:
        value = r[field]
        if value in first:
            uf.union(r["motion_id"], first[value])
        else:
            first[value] = r["motion_id"]


def frame_range(start, end, duration, frames, tolerance):
    """Half-open seconds -> half-open frame indexes, with recorded boundary repair."""
    if not math.isfinite(start) or not math.isfinite(end):
        return 0, 0, start, end, False, "nonfinite_time"
    if end <= start:
        return 0, 0, start, end, False, "nonpositive_interval"
    if start < -tolerance or end > duration + tolerance:
        return 0, 0, start, end, False, "outside_source_time_tolerance"
    if end <= 0 or start >= duration:
        return 0, 0, start, end, False, "interval_outside_source"
    clipped_start = max(0.0, start)
    clipped_end = min(duration, end)
    # 1e-9 target frames only compensates arithmetic noise at exact grid boundaries.
    a = min(frames, max(0, math.ceil(clipped_start * 30 - 1e-9)))
    b = min(frames, max(0, math.ceil(clipped_end * 30 - 1e-9)))
    changed = clipped_start != start or clipped_end != end
    return a, b, clipped_start, clipped_end, changed, "" if b > a else "empty_frame_interval"


def build(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "build.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ready = json.loads((NODES / "READY").read_text())
    if not ready["passed"]:
        raise RuntimeError("Node dataset is not validated")
    labels = RAW / "metadata/seed_metadata_v002_temporal_labels.jsonl"
    metadata = RAW / "metadata/seed_metadata_v004.parquet"
    config = {
        "schema": "g1-text-motion-index-v1", "fps": 30, "source_fps": 120,
        "nodes_relative_path": str(NODES.relative_to(ROOT)),
        "nodes_fingerprint": ready["pipeline_fingerprint"],
        "metadata_sha256": digest(metadata), "labels_sha256": digest(labels),
        "node_manifest_sha256": digest(NODES / "manifest.sqlite"),
        "script_sha256": digest(__file__), "validation_group_fraction": args.val_fraction,
        "seed": args.seed, "profile_min_frames": args.min_frames, "profile_max_frames": args.max_frames,
        "boundary_tolerance_seconds": 1 / 30 + 1e-6,
        "time_contract": "[start,end); source duration N/120; target k/30; no interpolation",
        "test_policy": "official test splits unchanged; training mirror/exact-copy relatives of tests quarantined",
        "validation_grouping": "union of take_name, source-performance mirror key and exact source SHA256 within remaining official train",
        "text_policy": "retain original text; strip leading/trailing whitespace only; no paraphrasing",
        "human_semantic_review": "pending",
    }
    config["fingerprint"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (output / "config.json").exists():
        old = json.loads((output / "config.json").read_text())
        if old["fingerprint"] != config["fingerprint"]:
            raise RuntimeError("Configuration changed: use a new output version")
        if (output / "READY").exists():
            print("Validated index already exists:", output)
            return
    atomic_json(output / "config.json", config)
    started = time.monotonic()

    def progress(phase, motions=0, events=0):
        state = {"phase": phase, "pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                 "source_motions_processed": motions, "events_processed": events,
                 "elapsed_seconds": round(time.monotonic() - started, 1)}
        atomic_json(output / "status.json", state)
        print(json.dumps(state), flush=True)

    progress("loading_metadata")
    meta = pd.read_parquet(metadata)
    by_name = {r["filename"]: r for r in meta.to_dict("records")}
    with sqlite3.connect(f"file:{NODES / 'manifest.sqlite'}?mode=ro", uri=True) as db:
        node_rows = db.execute("SELECT motion_id,source_path,output_path,official_split,status,result_json FROM motions ORDER BY motion_id").fetchall()
    motions = []
    for key, src, dest, split, status, result_json in node_rows:
        if status != "complete":
            raise RuntimeError(f"Incomplete node motion: {key}")
        row = by_name[Path(key).name]
        result = json.loads(result_json)
        if result["source_frames"] != int(row["move_duration_frames"]):
            raise RuntimeError(f"Source frame mismatch: {key}")
        name = row["filename"]
        mirror_key = name[:-2] if bool(row["is_mirror"]) and name.endswith("_M") else name
        motions.append({
            "motion_id": key, "filename": name, "npz_path": str((NODES / dest).relative_to(ROOT)),
            "raw_source_path": str((RAW / src).relative_to(ROOT)),
            "source_sha256": result["source_sha256"], "npz_sha256": result["output_sha256"],
            "source_frames": result["source_frames"], "frames": result["frames"],
            "source_duration_s": result["source_frames"] / 120,
            "official_split": split, "category": str(row["category"]), "package": str(row["package"]),
            "take_name": str(row["take_name"]), "content_name": str(row["content_name"]),
            "actor_uid": str(row["actor_uid"]), "is_mirror": bool(row["is_mirror"]),
            "mirror_key": mirror_key, "overview_text": str(row["content_natural_desc_4"]),
        })
    if len(motions) != ready["expected_motions"]:
        raise RuntimeError("Node manifest/metadata count mismatch")

    progress("grouping_and_leakage_audit")
    uf = UnionFind(r["motion_id"] for r in motions)
    unite_equal(uf, motions, "mirror_key")
    unite_equal(uf, motions, "source_sha256")
    group_splits = defaultdict(set)
    take_splits = defaultdict(set)
    for r in motions:
        r["source_group_id"] = uf.find(r["motion_id"])
        group_splits[r["source_group_id"]].add(r["official_split"])
        take_splits[r["take_name"]].add(r["official_split"])
    conflicts = []
    train_candidates = []
    for r in motions:
        splits = group_splits[r["source_group_id"]]
        r["exclusion_reason"] = ""
        r["split"] = r["official_split"]
        r["validation_group_id"] = ""
        if len(splits) > 1:
            conflicts.append({k: r[k] for k in ("motion_id", "filename", "official_split", "source_group_id", "mirror_key")})
        if r["official_split"] == "train":
            if splits & {"test_content", "test_repetition"}:
                r["split"] = "excluded"
                r["exclusion_reason"] = "mirror_or_exact_duplicate_of_official_test"
            else:
                train_candidates.append(r)
    val_uf = UnionFind(r["motion_id"] for r in train_candidates)
    for field in ("take_name", "mirror_key", "source_sha256"):
        unite_equal(val_uf, train_candidates, field)
    components = defaultdict(list)
    for r in train_candidates:
        components[val_uf.find(r["motion_id"])].append(r["motion_id"])
    group_ids = {key: hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest() for key, values in components.items()}
    for r in train_candidates:
        gid = group_ids[val_uf.find(r["motion_id"])]
        r["validation_group_id"] = gid
        fraction = int(hashlib.sha256(f"{args.seed}|{gid}".encode()).hexdigest(), 16) / 2**256
        r["split"] = "val" if fraction < args.val_fraction else "train"
    motion_by_name = {r["filename"]: r for r in motions}
    motion_df = pd.DataFrame(motions)
    save_parquet(motion_df, output / "motions.parquet")
    save_parquet(pd.DataFrame(conflicts), output / "official_split_conflicts.parquet")

    progress("aligning_events")
    segments = []
    label_names = set()
    event_overlap = event_gap = event_order_issues = 0
    for count, line in enumerate(labels.open(encoding="utf-8"), 1):
        record = json.loads(line)
        name = record["filename"]
        if name in label_names:
            raise ValueError(f"Duplicate annotation record: {name}")
        label_names.add(name)
        m = motion_by_name[name]
        if int(record["num_events"]) != len(record["events"]):
            raise ValueError(f"num_events mismatch: {name}")
        previous_end = previous_start = None
        for event_index, event in enumerate(record["events"]):
            s, e = float(event["start_time"]), float(event["end_time"])
            text_original = event["description"]
            if not isinstance(text_original, str):
                raise ValueError(f"Non-string description: {name}/{event_index}")
            text = text_original.strip()
            a, b, sa, ea, clipped, invalid = frame_range(s, e, m["source_duration_s"], m["frames"], config["boundary_tolerance_seconds"])
            if not text:
                invalid = "empty_description"
            overlap = previous_end is not None and s < previous_end - 1e-6
            gap = previous_end is not None and s > previous_end + 1 / 30
            order_issue = previous_start is not None and s < previous_start - 1e-6
            event_overlap += int(overlap); event_gap += int(gap); event_order_issues += int(order_issue)
            previous_end = max(previous_end, e) if previous_end is not None else e
            previous_start = s
            frames = b - a
            reason = invalid or m["exclusion_reason"]
            if not reason:
                if frames < args.min_frames:
                    reason = "below_profile_min_frames"
                elif frames > args.max_frames:
                    reason = "above_profile_max_frames"
            sid_payload = json.dumps([m["motion_id"], event_index, s, e, text_original], ensure_ascii=False)
            segments.append({
                "sample_id": hashlib.sha256(sid_payload.encode()).hexdigest()[:24],
                "text_id": hashlib.sha256(text.encode()).hexdigest(), "text": text,
                "text_original": text_original, "event_index": event_index,
                **{k: m[k] for k in ("motion_id", "filename", "npz_path", "source_sha256", "npz_sha256", "source_group_id", "validation_group_id", "official_split", "split", "category", "package", "is_mirror")},
                "raw_start_s": s, "raw_end_s": e, "aligned_start_s": sa, "aligned_end_s": ea,
                "source_duration_s": m["source_duration_s"], "source_output_frames": m["frames"],
                "start_frame": a, "end_frame_exclusive": b, "num_frames": frames, "fps": 30,
                "boundary_clipped": clipped, "start_clipped_s": sa - s, "end_clipped_s": e - ea,
                "overlaps_previous_event": overlap, "gap_before_event": gap, "event_order_issue": order_issue,
                "alignment_valid": not bool(invalid), "profile_eligible": not bool(reason), "exclusion_reason": reason,
            })
        if count % 10000 == 0:
            progress("aligning_events", count, len(segments))
    if label_names != set(motion_by_name):
        raise ValueError("Annotation and node motion coverage differs")
    df = pd.DataFrame(segments)
    if not df["sample_id"].is_unique:
        raise ValueError("Duplicate sample ID")
    valid = df[df["alignment_valid"]]
    if not ((valid.start_frame >= 0) & (valid.end_frame_exclusive <= valid.source_output_frames) & (valid.num_frames > 0)).all():
        raise ValueError("Frame range invariant failed")
    trainval = motion_df[motion_df.split.isin(["train", "val"])]
    if (trainval.groupby("validation_group_id").split.nunique() > 1).any():
        raise ValueError("Validation group leakage")
    active_motions = motion_df[motion_df.split != "excluded"]
    if (active_motions.groupby("source_group_id").split.nunique() > 1).any():
        raise ValueError("Mirror or exact-source leakage in derived splits")

    progress("writing_indexes", len(label_names), len(df))
    save_parquet(df, output / "segments.parquet")
    selected = df[df.profile_eligible]
    for split in ("train", "val", "test_content", "test_repetition"):
        save_parquet(selected[selected.split == split], output / f"{split}.parquet")
    save_parquet(df[~df.profile_eligible], output / "excluded.parquet")
    texts = df[["text_id", "text"]].drop_duplicates("text_id").sort_values("text_id")
    save_parquet(texts, output / "texts.parquet")
    review = selected[selected.split.isin(["train", "val"])].sort_values("sample_id")
    review = review.drop_duplicates(["category", "motion_id"]).groupby("category", group_keys=False).head(10).copy()
    review["human_review_status"] = "pending"
    review["human_review_notes"] = ""
    review[["sample_id", "motion_id", "split", "category", "text", "npz_path", "raw_start_s", "raw_end_s", "start_frame", "end_frame_exclusive", "boundary_clipped", "human_review_status", "human_review_notes"]].to_csv(output / "review_samples.csv", index=False)
    report = {
        "fingerprint": config["fingerprint"], "source_motions": len(motion_df), "annotation_events": len(df),
        "source_split_counts": motion_df.split.value_counts().to_dict(),
        "official_source_split_counts": motion_df.official_split.value_counts().to_dict(),
        "profile_segment_split_counts": selected.split.value_counts().to_dict(),
        "all_segment_split_counts": df.split.value_counts().to_dict(),
        "alignment_valid": int(df.alignment_valid.sum()), "alignment_invalid": int((~df.alignment_valid).sum()),
        "profile_selected": len(selected), "excluded_from_profile": int((~df.profile_eligible).sum()),
        "exclusion_reason_counts": df.loc[~df.profile_eligible, "exclusion_reason"].value_counts().to_dict(),
        "boundary_clipped_events": int(df.boundary_clipped.sum()),
        "max_end_clip_s": float(df.end_clipped_s.max()),
        "overlapping_events": event_overlap, "events_with_gap": event_gap, "event_order_issues": event_order_issues,
        "unique_texts": len(texts), "validation_groups": int(trainval.validation_group_id.nunique()),
        "train_val_group_overlap": 0, "active_source_group_split_overlap": 0,
        "official_source_groups_cross_splits": sum(len(s) > 1 for s in group_splits.values()),
        "official_take_names_cross_splits": sum(len(s) > 1 for s in take_splits.values()),
        "human_review_samples": len(review), "human_semantic_review": "pending",
        "profile_category_counts": selected.groupby(["split", "category"]).size().reset_index(name="samples").to_dict("records"),
    }
    atomic_json(output / "alignment_audit.json", report)
    atomic_json(output / "INDEX_BUILT", {"fingerprint": config["fingerprint"], "sample_validation": "pending"})
    progress("index_built_pending_validation", len(label_names), len(df))
    print(json.dumps({k: v for k, v in report.items() if k != "profile_category_counts"}, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20261030)
    p.add_argument("--min-frames", type=int, default=30)
    p.add_argument("--max-frames", type=int, default=300)
    args = p.parse_args()
    if not 0 < args.val_fraction < 1 or not 1 <= args.min_frames <= args.max_frames:
        p.error("Invalid split fraction or frame limits")
    build(args)


if __name__ == "__main__":
    main()
