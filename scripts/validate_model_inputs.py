"""Audit full numerical caches, normalization, DataLoader and denoiser contract."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from data_pipeline.numeric_common import OUTPUT, INDEX, SPLITS, sha_file, atomic_json, source_versions
from data_pipeline.model_input_dataset import G1ModelInputDataset, collate_model_inputs
from data_pipeline.motion_features import SegmentFeatureBuilder, moments, merge_moments
from kimodo.motion_rep import KimodoMotionRep
from kimodo.model.diffusion import Diffusion
from kimodo.model.twostage_denoiser import TwostageDenoiser


def main():
    torch.set_num_threads(2)
    config = json.loads((OUTPUT / "config.json").read_text())
    motion_done = json.loads((OUTPUT / "MOTION_COMPLETE").read_text())
    text_done = json.loads((OUTPUT / "text_encoder/COMPLETE").read_text())
    assert motion_done["fingerprint"] == config["fingerprint"]
    text_config = json.loads((OUTPUT / "text_encoder/config.json").read_text())
    assert text_done["fingerprint"] == text_config["fingerprint"]
    assert text_config["model_inputs_fingerprint"] == config["fingerprint"]
    assert text_config["texts_sha256"] == sha_file(OUTPUT / "texts.parquet")
    current_source = source_versions()
    assert current_source["source_sha256"] == config["source_sha256"]
    assert current_source["versions"] == config["versions"]
    assert sha_file(ROOT / "data_pipeline/motion_features.py") == config["feature_builder_sha256"]
    assert sha_file(ROOT / "scripts/cache_motion_features.py") == config["runner_sha256"]
    ready_path = OUTPUT / "READY"
    if ready_path.exists():
        ready_path.unlink()
    def progress(stage, extra=None):
        r = {"phase": stage, "updated_at_utc": datetime.now(timezone.utc).isoformat()}
        if extra:
            r.update(extra)
        atomic_json(OUTPUT / "validation_status.json", r)
        print(stage, extra or "", flush=True)
    progress("checking_indexes_and_embeddings")
    frames = {s: pd.read_parquet(OUTPUT / f"{s}.parquet") for s in SPLITS}
    source = {s: pd.read_parquet(INDEX / f"{s}.parquet") for s in SPLITS}
    for s in SPLITS:
        assert sha_file(INDEX / f"{s}.parquet") == config["split_sha256"][s]
        assert set(frames[s].sample_id) == set(source[s].sample_id)
        assert int(frames[s].num_frames.sum()) == int(source[s].num_frames.sum())
    all_rows = pd.concat(frames.values(), ignore_index=True)
    assert len(all_rows) == config["samples"] and all_rows.sample_id.is_unique
    texts = pd.read_parquet(OUTPUT / "texts.parquet").sort_values("text_row")
    assert set(texts.text_id) == set(all_rows.text_id)
    assert (all_rows.text_id == all_rows.text_row.map(dict(zip(texts.text_row, texts.text_id)))).all()
    text_path = OUTPUT / "text_encoder/embeddings.npy"
    assert sha_file(text_path) == text_done["sha256"]
    embeddings = np.load(text_path, mmap_mode="r")
    assert embeddings.shape == (len(texts), 1, 4096) and embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all() and (np.linalg.norm(embeddings[:, 0], axis=-1) > 0).all()
    assert text_done["baseline_exact_equal"] and text_done["repeat_exact_equal"]
    stats_report = json.loads((OUTPUT / "stats/report.json").read_text())
    assert not stats_report["other_splits_used"]
    assert stats_report["train_frames"] == int(frames["train"].num_frames.sum())
    progress("verifying_independent_train_statistics_audit")
    audit_path = OUTPUT / "stats/independent_audit.json"
    if not audit_path.exists():
        from check_train_statistics import main as audit_statistics
        audit_statistics()
    audit = json.loads(audit_path.read_text())
    assert audit["passed"] and audit["fingerprint"] == config["fingerprint"]
    assert audit["all_train_shards_sha256_verified"] and not audit["other_splits_used"]
    assert audit["train_index_sha256"] == sha_file(OUTPUT / "train.parquet")
    assert {r["path"] for r in audit["shards"]} == set(frames["train"].feature_path)
    for record in audit["shards"]:
        path = OUTPUT / record["path"]
        stat = path.stat()
        assert (stat.st_size, stat.st_mtime_ns) == (record["bytes"], record["mtime_ns"]), "Training cache changed after its full audit"
        assert json.loads((path.parent / "result.json").read_text())["sha256"] == record["sha256"]
    for path, expected_hash in audit["stats_files_sha256"].items():
        assert sha_file(OUTPUT / path) == expected_hash
    count = audit["train_frames"]
    stats = OUTPUT / "stats/motion"
    assert count == stats_report["train_frames"]
    for group, dim in (("global_root", 5), ("body", 412), ("local_root", 4)):
        for name in ("mean", "std"):
            a = np.load(stats / group / f"{name}.npy")
            assert a.shape == (dim,) and np.isfinite(a).all()
            if name == "std":
                assert (a >= 0).all()
    progress("checking_roundtrip_and_dataset")
    builder = SegmentFeatureBuilder()
    rep = KimodoMotionRep(builder.skeleton, fps=30, stats_path=str(stats))
    checks = []
    rng = random.Random(20261101)
    for split in SPLITS:
        ds = G1ModelInputDataset(split, require_ready=False)
        picks = rng.sample(range(len(ds)), 8)
        for i in picks:
            row = ds.rows[i]
            sample = ds[i]
            data = np.load(OUTPUT / row["feature_path"], mmap_mode="r")
            a, n = int(row["frame_offset"]), int(row["num_frames"])
            raw = np.array(data[a:a + n])
            actual = rep.normalize(torch.from_numpy(raw))
            assert torch.allclose(actual, sample["x0"], atol=1e-5, rtol=1e-6)
            restored = rep.unnormalize(sample["x0"])
            norm_error = float((restored - torch.from_numpy(raw)).abs().max())
            assert norm_error < 1e-5
            assert torch.equal(sample["text_feat"], torch.from_numpy(np.array(embeddings[int(row["text_row"])])))
            with np.load(ROOT / row["npz_path"], allow_pickle=False) as z:
                start, end = int(row["start_frame"]), int(row["end_frame_exclusive"])
                recomputed, _, heading, origin = builder(z["local_rot_mats"][start:end], z["root_positions"][start:end])
                expected_positions = z["posed_joints"][start:end].copy()
            assert np.array_equal(raw, recomputed)
            expected_positions[..., 0] -= origin[0]
            expected_positions[..., 2] -= origin[1]
            decoded = rep.inverse(sample["x0"][None], is_normalized=True)
            position_error = float(np.max(np.abs(decoded["posed_joints"][0].numpy() - expected_positions)))
            assert position_error < 1e-4
            checks.append({"sample_id": sample["sample_id"], "split": split,
                           "normalization_max_error": norm_error, "position_max_error_m": position_error})
    train = G1ModelInputDataset("train", require_ready=False)
    lengths = [r["num_frames"] for r in train.rows]
    examples = [train[int(np.argmin(lengths))], train[int(np.argmax(lengths))]]
    batch = collate_model_inputs(examples)
    assert torch.equal(batch["x_pad_mask"].sum(-1), batch["lengths"])
    assert batch["text_feat"].shape == (2, 1, 4096) and batch["text_feat_pad_mask"].all()
    assert torch.count_nonzero(batch["x0"][~batch["x_pad_mask"]]) == 0
    loader = DataLoader(Subset(train, range(8)), batch_size=4, num_workers=2,
                        collate_fn=collate_model_inputs, multiprocessing_context="spawn")
    iterator = iter(loader)
    workers = list(iterator._workers)
    batches = list(iterator)
    for worker in workers:
        worker.join(timeout=10)
        assert worker.exitcode == 0
    assert len(batches) == 2
    del iterator
    progress("checking_random_denoiser_forward_backward")
    torch.manual_seed(20261101)
    denoiser = TwostageDenoiser(rep, motion_mask_mode="concat", ckpt_path=None,
                              llm_shape=[1, 4096], use_text_mask=False, latent_dim=64, ff_size=128,
                              num_layers=1, num_heads=4, activation="gelu", dropout=0.0,
                              pe_dropout=0.0, norm_first=False, num_text_tokens_override=50,
                              input_first_heading_angle=True)
    denoiser.train()
    diffusion = Diffusion(1000)
    t = torch.tensor([0, 999], dtype=torch.long)
    noisy = diffusion.q_sample(batch["x0"], t)
    prediction = denoiser(noisy, batch["x_pad_mask"], batch["text_feat"], batch["text_feat_pad_mask"], t,
                         first_heading_angle=batch["first_heading_angle"])
    assert prediction.shape == batch["x0"].shape and torch.isfinite(prediction).all()
    per_element = torch.nn.functional.smooth_l1_loss(prediction, batch["x0"], reduction="none")
    loss = (per_element * batch["x_pad_mask"][..., None]).sum() / (batch["x_pad_mask"].sum() * 417)
    loss.backward()
    grads = [p.grad for p in denoiser.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and any(torch.count_nonzero(g) for g in grads)
    report = {"passed": True, "fingerprint": config["fingerprint"],
              "samples": len(all_rows), "frames": int(all_rows.num_frames.sum()), "text_vectors": len(texts),
              "split_samples": {s: len(frames[s]) for s in SPLITS}, "train_statistics_frames": count,
              "train_statistics_independently_verified": True, "roundtrip_samples": checks,
              "dataloader_worker_exitcodes": [w.exitcode for w in workers],
              "denoiser_input_shape": list(batch["x0"].shape), "text_input_shape": list(batch["text_feat"].shape),
              "random_denoiser_loss": float(loss.detach()), "random_denoiser_gradients_finite": True,
              "optimizer_steps": 0, "text_config_fingerprint": text_done["fingerprint"],
              "dataset_sha256": sha_file(ROOT / "data_pipeline/model_input_dataset.py"),
              "human_semantic_review": "pending", "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(OUTPUT / "validation_report.json", report)
    atomic_json(ready_path, {k: v for k, v in report.items() if k != "roundtrip_samples"})
    progress("complete")
    print(json.dumps({k: v for k, v in report.items() if k != "roundtrip_samples"}, indent=2))


if __name__ == "__main__":
    main()
