#!/usr/bin/env python3
"""Freeze nested M4 train subsets and a balanced, caption-isolated validation set."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from training.m3 import INPUTS, atomic_json, sha, validate_inputs

PATTERNS = {
    "walk": r"\bwalk(?:s|ing)?\b",
    "jog": r"\b(?:jog(?:s|ging)?|run(?:s|ning)?)\b",
    "jump": r"\bjump(?:s|ing)?\b",
    "raise_arms": r"\b(?:rais(?:e|es|ing)|lift(?:s|ing)?)\b(?:\s+[\w-]+){0,5}\s+(?:arms|hands)\b",
}
EXCLUDE = r"\b(?:crate|cupboard|flute|recorder|nail|fence|valve|burger|danc\w*|twirl\w*|object|chainsaw|table|chair|wall|ladder|stair\w*|box|rope|gun|weapon|platform|obstacle|cigarette|smok\w*|climb\w*|ledge|bench|stool)\b"


def rank(value):
    return hashlib.sha256((str(value) + ":M4:20261102").encode()).hexdigest()


def diverse_pick(rows, count, anchors=(), per_text_cap=8):
    """Round-robin captions; unique source performance, at most eight events per exact caption."""
    rows = sorted(rows, key=lambda x: rank(x["sample_id"]))
    buckets = {}
    for row in rows:
        buckets.setdefault(row["text"], []).append(row)
    order = sorted(buckets, key=rank)
    selected, used_sources, used_motions, texts = [], set(), set(), Counter()
    def add(text):
        for row in buckets.get(text, []):
            if row["source_group_id"] in used_sources or row["motion_id"] in used_motions:
                continue
            if texts[text] >= per_text_cap:
                break
            selected.append(row)
            used_sources.add(row["source_group_id"])
            used_motions.add(row["motion_id"])
            texts[text] += 1
            return True
        return False
    for text in anchors:
        if not add(text):
            raise ValueError("Could not preserve seen-caption anchor")
    while len(selected) < count:
        before = len(selected)
        for text in order:
            if len(selected) == count:
                break
            add(text)
        if len(selected) == before:
            raise ValueError(f"Only {len(selected)} eligible events, requested {count}")
    return selected


def slim(row, group, cohort=None):
    keys = ["sample_id", "text_id", "text", "motion_id", "num_frames", "source_sha256",
            "source_group_id", "validation_group_id", "take_name", "actor_uid"]
    result = {key: str(row[key]) if key not in ("num_frames",) else int(row[key]) for key in keys}
    result["group"] = group
    if cohort:
        result["cohort"] = cohort
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=Path, required=True)
    args = p.parse_args()
    experiment = args.experiment.resolve()
    experiment.relative_to(ROOT / "runs")
    output = experiment / "data"
    if output.exists():
        raise FileExistsError(output)
    ready = validate_inputs()
    motion_meta = pd.read_parquet(ROOT / "data/processed/g1_text_motion_v1/motions.parquet",
                    columns=["motion_id", "source_group_id", "validation_group_id", "take_name", "actor_uid", "is_mirror"])
    candidates = {}
    audit = {"candidate_counts": {}, "classification_patterns": PATTERNS, "exclude_pattern": EXCLUDE,
             "per_caption_cap": 8, "per_source_performance_cap": 1, "length_frames": [60, 90],
             "official_test_accessed": False, "human_semantic_review": "pending"}
    for split in ("train", "val"):
        frame = pd.read_parquet(INPUTS / f"{split}.parquet").merge(motion_meta, on="motion_id", validate="many_to_one")
        frame = frame[frame.num_frames.between(60, 90) & ~frame.is_mirror]
        frame = frame[~frame.text.str.contains(EXCLUDE, flags=re.I, regex=True)]
        for group, pattern in PATTERNS.items():
            q = frame[frame.text.str.contains(pattern, flags=re.I, regex=True)]
            for other, op in PATTERNS.items():
                if other != group:
                    q = q[~q.text.str.contains(op, flags=re.I, regex=True)]
            if group == "raise_arms":
                q = q[~q.text.str.contains(r"lifting their hands off their head", flags=re.I, regex=True)]
            candidates[split, group] = q.to_dict("records")
            audit["candidate_counts"][f"{split}/{group}"] = {"events": len(q), "captions": q.text.nunique()}
    train_a, train_b, validation = [], [], []
    group_audit = {}
    for group in PATTERNS:
        train_rows, val_rows = candidates["train", group], candidates["val", group]
        train_texts = {r["text"] for r in train_rows}
        val_counts = Counter(r["text"] for r in val_rows)
        # Reserve sufficient common captions for 8 held-out new performances.
        shared = sorted(train_texts & set(val_counts), key=lambda t: (-min(8, val_counts[t]), rank(t)))
        seen_texts = []
        for text in shared:
            seen_texts.append(text)
            try:
                seen = diverse_pick([r for r in val_rows if r["text"] in seen_texts], 8)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Insufficient seen-caption validation for {group}")
        new = diverse_pick([r for r in val_rows if r["text"] not in seen_texts], 8)
        new_texts = {r["text"] for r in new}
        eligible = [r for r in train_rows if r["text"] not in new_texts]
        large = diverse_pick(eligible, 256, anchors=seen_texts)
        small = large[:64]
        if not set(seen_texts) <= {r["text"] for r in small}:
            raise AssertionError("A lacks seen-caption anchors")
        train_a.extend(slim(r, group) for r in small)
        train_b.extend(slim(r, group) for r in large)
        validation.extend(slim(r, group, "seen_text_new_motion") for r in seen)
        validation.extend(slim(r, group, "unseen_text_same_class") for r in new)
        group_audit[group] = {"train_A":64, "train_B":256, "val_seen":8, "val_unseen":8,
                             "A_unique_captions":len({r['text'] for r in small}),
                             "B_unique_captions":len({r['text'] for r in large}),
                             "val_seen_unique_captions":len({r['text'] for r in seen}),
                             "val_unseen_unique_captions":len(new_texts)}
    for name, rows in (("A", train_a), ("B", train_b)):
        for field in ("sample_id", "motion_id", "source_group_id", "validation_group_id"):
            if {r[field] for r in rows} & {r[field] for r in validation}:
                raise AssertionError(f"{name}/validation leakage: {field}")
        if {r['text'] for r in rows} & {r['text'] for r in validation if r['cohort']=='unseen_text_same_class'}:
            raise AssertionError("Held-out text leaked into training")
    if len({r['sample_id'] for r in train_b}) != 1024 or len({r['sample_id'] for r in validation}) != 64:
        raise AssertionError("Duplicate samples across classes")
    common = {"schema":"m4-subset-v1", "m2_fingerprint":ready['fingerprint'],
              "normalization":"fixed M2 full-training statistics; subset size counts denoiser training examples",
              "human_semantic_review":"pending; rules, numeric checks and reference sheets are separate evidence",
              "selection_script_sha256":sha(__file__), "val":validation}
    atomic_json(output / "A.json", dict(common, train=train_a))
    atomic_json(output / "BCD.json", dict(common, train=train_b))
    audit.update(passed=True, groups=group_audit, train_A=256, train_BCD=1024, validation=64,
                 nested_A_in_B=True, source_group_overlap=0, validation_group_overlap=0,
                 heldout_caption_overlap=0, m2_fingerprint=ready['fingerprint'])
    atomic_json(output / "audit.json", audit)
    pd.DataFrame(validation).assign(review_status="pending").to_csv(output / "validation_review.csv", index=False)
    print(json.dumps(audit,indent=2),flush=True)


if __name__ == "__main__":
    main()
