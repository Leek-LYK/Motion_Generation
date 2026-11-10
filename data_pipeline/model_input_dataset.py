"""Numerical model inputs: normalized motion, frozen text features, and masks."""
from collections import OrderedDict
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .numeric_common import OUTPUT, SPLITS


class G1ModelInputDataset(Dataset):
    def __init__(self, split="train", input_dir=OUTPUT, min_frames=None, max_frames=None,
                 require_ready=True, max_mapped_shards=8):
        if split not in SPLITS or max_mapped_shards < 1:
            raise ValueError("Invalid split or cache limit")
        self.directory = Path(input_dir).resolve()
        self.config = json.loads((self.directory / "config.json").read_text())
        if require_ready:
            ready = json.loads((self.directory / "READY").read_text())
            if not ready["passed"] or ready["fingerprint"] != self.config["fingerprint"]:
                raise ValueError("Numerical dataset is not validated")
        frame = pd.read_parquet(self.directory / f"{split}.parquet")
        if not (frame.split == split).all():
            raise ValueError("Unexpected split rows")
        if min_frames is not None:
            frame = frame[frame.num_frames >= min_frames]
        if max_frames is not None:
            frame = frame[frame.num_frames <= max_frames]
        self.rows = frame.to_dict("records")
        stats = self.directory / "stats/motion"
        self.mean = np.concatenate([np.load(stats / g / "mean.npy") for g in ("global_root", "body")]).astype(np.float32)
        std = np.concatenate([np.load(stats / g / "std.npy") for g in ("global_root", "body")]).astype(np.float32)
        self.scale = np.sqrt(std**2 + np.float32(self.config["normalization_epsilon"]))
        if self.mean.shape != (417,) or not np.isfinite(self.scale).all() or not (self.scale > 0).all():
            raise ValueError("Invalid normalization statistics")
        text_index = pd.read_parquet(self.directory / "texts.parquet").sort_values("text_row")
        self.text_ids = text_index.text_id.tolist()
        self.max_mapped_shards = max_mapped_shards
        self._pid = os.getpid()
        self._shards = OrderedDict()
        self._text = None

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_shards=OrderedDict(), _text=None, _pid=None)
        return state

    def __getitem__(self, index):
        if self._pid != os.getpid():
            self._shards.clear()
            self._text = None
            self._pid = os.getpid()
        row = self.rows[index]
        path = row["feature_path"]
        if path not in self._shards:
            resolved = (self.directory / path).resolve()
            resolved.relative_to(self.directory)
            self._shards[path] = np.load(resolved, mmap_mode="r", allow_pickle=False)
            while len(self._shards) > self.max_mapped_shards:
                self._shards.popitem(last=False)
        self._shards.move_to_end(path)
        a = int(row["frame_offset"])
        length = int(row["num_frames"])
        raw = np.asarray(self._shards[path][a:a + length])
        if raw.shape != (length, 417):
            raise ValueError("Feature slice shape mismatch")
        normalized = (raw - self.mean) / self.scale
        if self._text is None:
            self._text = np.load(self.directory / "text_encoder/embeddings.npy", mmap_mode="r", allow_pickle=False)
        text_row = int(row["text_row"])
        if self.text_ids[text_row] != row["text_id"]:
            raise ValueError("Text ID/embedding row mismatch")
        text = np.array(self._text[text_row], copy=True)
        if text.shape != (1, 4096) or not np.isfinite(normalized).all() or not np.isfinite(text).all():
            raise ValueError("Nonfinite or malformed model input")
        return {"x0": torch.from_numpy(normalized.copy()), "text_feat": torch.from_numpy(text),
                "first_heading_angle": torch.tensor(row["first_heading_angle"], dtype=torch.float32),
                "length": length, "sample_id": row["sample_id"], "text_id": row["text_id"],
                "motion_id": row["motion_id"], "text": row["text"], "split": row["split"],
                "origin_xz": torch.tensor([row["origin_x"], row["origin_z"]], dtype=torch.float32)}

    def unnormalize(self, x):
        return x * torch.as_tensor(self.scale, device=x.device, dtype=x.dtype) + torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)


def collate_model_inputs(samples):
    if not samples:
        raise ValueError("Empty batch")
    lengths = torch.tensor([s["length"] for s in samples], dtype=torch.long)
    maximum = int(lengths.max())
    x0 = torch.zeros(len(samples), maximum, 417, dtype=torch.float32)
    for i, sample in enumerate(samples):
        x0[i, :sample["length"]] = sample["x0"]
    return {"x0": x0, "x_pad_mask": torch.arange(maximum)[None] < lengths[:, None],
            "text_feat": torch.stack([s["text_feat"] for s in samples]),
            "text_feat_pad_mask": torch.ones(len(samples), 1, dtype=torch.bool),
            "first_heading_angle": torch.stack([s["first_heading_angle"] for s in samples]),
            "lengths": lengths, "origin_xz": torch.stack([s["origin_xz"] for s in samples]),
            "sample_ids": [s["sample_id"] for s in samples], "texts": [s["text"] for s in samples],
            "splits": [s["split"] for s in samples]}
