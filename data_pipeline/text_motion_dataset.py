"""Read text-aligned node segments without duplicating motion files."""
from collections import OrderedDict
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "data/processed/g1_text_motion_v1"
FIELDS = ("posed_joints", "local_rot_mats", "global_rot_mats", "root_positions", "qpos_mujoco", "timestamps", "source_frame_indices")


class G1TextMotionDataset(Dataset):
    """Unnormalized node motion + exact event text; no random time cropping.

    Source-world positions are preserved. Recompute crop-dependent features before
    training; full-clip smooth-root/velocity/contact caches are not returned here.
    """

    def __init__(self, split="train", index_dir=DEFAULT_INDEX, project_root=ROOT,
                 min_frames=None, max_frames=None, cache_megabytes=128, require_ready=True):
        if split not in ("train", "val", "test_content", "test_repetition"):
            raise ValueError(f"Unsupported split: {split}")
        self.index_dir = Path(index_dir).resolve()
        self.project_root = Path(project_root).resolve()
        self.config = json.loads((self.index_dir / "config.json").read_text())
        if require_ready:
            ready = json.loads((self.index_dir / "READY").read_text())
            if not ready["structural_alignment_passed"] or ready["fingerprint"] != self.config["fingerprint"]:
                raise ValueError("Training index is not validated")
        columns = ["sample_id", "text_id", "text", "motion_id", "npz_path", "source_sha256",
                   "split", "start_frame", "end_frame_exclusive", "num_frames", "fps",
                   "raw_start_s", "raw_end_s", "aligned_start_s", "aligned_end_s", "boundary_clipped", "profile_eligible"]
        frame = pd.read_parquet(self.index_dir / f"{split}.parquet", columns=columns)
        if not frame["profile_eligible"].all() or not (frame["split"] == split).all():
            raise ValueError("Invalid rows in split index")
        if min_frames is not None:
            frame = frame[frame.num_frames >= min_frames]
        if max_frames is not None:
            frame = frame[frame.num_frames <= max_frames]
        self.rows = frame.to_dict("records")
        if cache_megabytes < 0:
            raise ValueError("cache_megabytes must be nonnegative")
        self.cache_limit = int(cache_megabytes * 1024**2)
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._cache_pid = os.getpid()

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_cache=OrderedDict(), _cache_bytes=0, _cache_pid=None)
        return state

    def _load(self, row):
        if self._cache_pid != os.getpid():
            self._cache.clear()
            self._cache_bytes = 0
            self._cache_pid = os.getpid()
        key = row["npz_path"]
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key][0]
        path = (self.project_root / key).resolve()
        path.relative_to(self.project_root)
        with np.load(path, allow_pickle=False) as npz:
            if int(npz["fps"]) != 30 or int(npz["source_fps"]) != 120:
                raise ValueError(f"Unexpected FPS: {path}")
            if str(npz["pipeline_fingerprint"]) != self.config["nodes_fingerprint"]:
                raise ValueError(f"Motion pipeline mismatch: {path}")
            if str(npz["source_sha256"]) != row["source_sha256"]:
                raise ValueError(f"Motion provenance mismatch: {path}")
            arrays = {k: npz[k] for k in FIELDS}
            arrays["node_names"] = tuple(npz["node_names"].tolist())
        size = sum(v.nbytes for v in arrays.values() if isinstance(v, np.ndarray))
        if size <= self.cache_limit:
            while self._cache and self._cache_bytes + size > self.cache_limit:
                _, (_, old_size) = self._cache.popitem(last=False)
                self._cache_bytes -= old_size
            self._cache[key] = (arrays, size)
            self._cache_bytes += size
        return arrays

    def __getitem__(self, index):
        row = self.rows[index]
        arrays = self._load(row)
        a, b = int(row["start_frame"]), int(row["end_frame_exclusive"])
        if not 0 <= a < b <= len(arrays["timestamps"]) or b - a != row["num_frames"]:
            raise ValueError(f"Invalid stored frame interval: {row['sample_id']}")
        motion = {k: torch.from_numpy(arrays[k][a:b].copy()) for k in FIELDS}
        timestamps = motion["timestamps"]
        if not ((timestamps >= row["aligned_start_s"] - 1e-10) & (timestamps < row["aligned_end_s"] + 1e-10)).all():
            raise ValueError(f"Time alignment mismatch: {row['sample_id']}")
        return {
            "motion": motion, "text": row["text"], "text_id": row["text_id"],
            "sample_id": row["sample_id"], "motion_id": row["motion_id"], "split": row["split"],
            "length": b - a, "fps": 30, "node_names": arrays["node_names"],
            "source_start_frame_30hz": a, "source_end_frame_30hz_exclusive": b,
            "annotation_start_s": row["raw_start_s"], "annotation_end_s": row["raw_end_s"],
            "boundary_clipped": bool(row["boundary_clipped"]),
        }


def collate_text_motion(samples):
    """Batch variable lengths. True in valid_mask denotes real, unpadded frames."""
    if not samples:
        raise ValueError("Empty batch")
    lengths = torch.tensor([s["length"] for s in samples], dtype=torch.long)
    maximum = int(lengths.max())
    batch_size = len(samples)
    names = samples[0]["node_names"]
    if any(s["node_names"] != names or s["fps"] != 30 for s in samples):
        raise ValueError("Mixed skeletons or FPS")
    motion = {}
    for field in FIELDS:
        example = samples[0]["motion"][field]
        padded = torch.zeros((batch_size, maximum) + tuple(example.shape[1:]), dtype=example.dtype)
        if field.endswith("rot_mats"):
            padded[:] = torch.eye(3, dtype=example.dtype)
        elif field == "qpos_mujoco":
            padded[..., 3] = 1
        elif field == "source_frame_indices":
            padded.fill_(-1)
        for i, sample in enumerate(samples):
            padded[i, :sample["length"]] = sample["motion"][field]
        motion[field] = padded
    return {
        "motion": motion, "texts": [s["text"] for s in samples],
        "text_ids": [s["text_id"] for s in samples], "sample_ids": [s["sample_id"] for s in samples],
        "motion_ids": [s["motion_id"] for s in samples], "splits": [s["split"] for s in samples],
        "lengths": lengths, "valid_mask": torch.arange(maximum)[None, :] < lengths[:, None],
        "node_names": names, "fps": 30,
    }
