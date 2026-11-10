"""Offline, frozen LLM2Vec embeddings with batch_size=1 and resumable rows."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "4"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                  TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1", TEXT_ENCODER_MODE="local")

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from data_pipeline.numeric_common import OUTPUT, KIMODO, sha_file, fingerprint, atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    if args.limit < 0:
        p.error("limit must be nonnegative")
    directory = OUTPUT / "text_encoder"
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    common = json.loads((OUTPUT / "config.json").read_text())
    texts = pd.read_parquet(OUTPUT / "texts.parquet").sort_values("text_row")
    assert np.array_equal(texts.text_row.to_numpy(), np.arange(len(texts)))
    total = len(texts)
    started = time.monotonic()
    completed = 0
    starting_count = 0
    def progress(phase, extra=None):
        elapsed = time.monotonic() - started
        rate = (completed - starting_count) / max(elapsed, 1e-9)
        r = {"phase": phase, "pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(),
             "completed_texts": completed, "total_texts": total, "texts_per_second": rate,
             "eta_seconds": (total - completed) / rate if rate else None, "elapsed_s": elapsed}
        if extra:
            r.update(extra)
        atomic_json(OUTPUT / "text_status.json", r)
        print(f"{phase}: {completed}/{total}, {rate:.2f} texts/s" + (f" {extra}" if extra else ""), flush=True)
    progress("fingerprinting_local_weights")
    model_root = KIMODO / "text_encoder"
    roots = [model_root / "meta-llama/Meta-Llama-3-8B-Instruct",
             model_root / "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
             model_root / "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"]
    hashes = {}
    for root in roots:
        for file in sorted(root.rglob("*")):
            if file.is_file() and "original" not in file.relative_to(root).parts and file.suffix in (".json", ".safetensors", ".py", ".model"):
                hashes[str(file.relative_to(model_root))] = sha_file(file)
                if file.suffix == ".safetensors":
                    progress("fingerprinting_local_weights", {"verified_file": str(file.relative_to(model_root))})
    from importlib.metadata import version
    import torch
    from transformers.utils.logging import disable_progress_bar
    from kimodo.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder
    disable_progress_bar()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this configured text caching run")
    os.environ["TEXT_ENCODERS_DIR"] = str(model_root)
    os.environ["TEXT_ENCODER_DEVICE"] = "cuda:0"
    cfg = {"schema": "frozen-llm2vec-cache-v1", "model_inputs_fingerprint": common["fingerprint"],
           "texts_sha256": sha_file(OUTPUT / "texts.parquet"), "model_file_sha256": hashes,
           "kimodo_source_sha256": common["source_sha256"],
           "versions": {n: version(n) for n in ("torch", "transformers", "peft", "accelerate", "safetensors")},
           "device": torch.cuda.get_device_name(0), "cuda_build": torch.version.cuda,
           "base_dtype": "bfloat16", "storage_dtype": "float32", "internal_batch_size": 1,
           "prompt_policy": "unchanged local Kimodo wrapper; empty instruction; preserve existing local-path template behavior",
           "pooling": "mean", "max_length": 512, "doc_max_length": 400, "skip_instruction": True,
           "shape": [total, 1, 4096], "frozen": True}
    cfg["fingerprint"] = fingerprint(cfg)
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text())["fingerprint"] != cfg["fingerprint"]:
        raise RuntimeError("Encoder or text identity changed: refusing to mix embeddings")
    atomic_json(config_path, cfg)
    if (directory / "COMPLETE").exists():
        done = json.loads((directory / "COMPLETE").read_text())
        assert done["fingerprint"] == cfg["fingerprint"]
        assert sha_file(directory / "embeddings.npy") == done["sha256"]
        completed = total
        progress("complete")
        return
    progress("loading_frozen_encoder")
    encoder = LLM2VecEncoder("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
                            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
                            "bfloat16", 4096, device="cuda:0")
    assert not any(p.requires_grad for p in encoder.model.parameters())
    assert encoder.model.pooling_mode == "mean" and encoder.model.skip_instruction
    assert encoder.model.max_length == 512 and encoder.model.doc_max_length == 400
    progress("checking_baseline_and_token_lengths")
    reference = torch.load(KIMODO / "experiments/llm2vec_adapter_audit/embeddings_candidate.pt", map_location="cpu", weights_only=True)
    actual, _ = encoder(reference["prompts"])
    current = actual[:, 0].detach().cpu()
    expected = reference["enabled"].cpu()
    delta = (current - expected).abs()
    baseline = {"prompts": reference["prompts"], "shape": list(current.shape),
                "exact_equal": bool(torch.equal(current, expected)), "max_abs_error": float(delta.max()),
                "mean_abs_error": float(delta.mean())}
    atomic_json(directory / "baseline_validation.json", baseline)
    if not baseline["exact_equal"]:
        raise RuntimeError("New encoder output differs from the verified local baseline")
    raw_texts = texts.text.tolist()
    token_lengths = []
    for offset in range(0, total, 256):
        ids = encoder.model.tokenizer(raw_texts[offset:offset + 256], add_special_tokens=False,
                                      padding=False, truncation=False)["input_ids"]
        token_lengths.extend(len(v) for v in ids)
    if max(token_lengths) > 400:
        atomic_json(directory / "token_length_failures.json", {"rows": [i for i, n in enumerate(token_lengths) if n > 400]})
        raise RuntimeError("Some text would be truncated by LLM2Vec; resolve before caching")
    atomic_json(directory / "token_audit.json", {"texts": total, "min_tokens": min(token_lengths),
                "max_tokens": max(token_lengths), "over_doc_max_length": 0,
                "tokenizer": str(encoder.model.tokenizer.name_or_path),
                "model_name_or_path": str(encoder.model.model.config._name_or_path)})
    checkpoint = directory / "checkpoint.json"
    partial = directory / "embeddings.partial.npy"
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        assert state["fingerprint"] == cfg["fingerprint"]
        completed = int(state["completed"])
        array = np.load(partial, mmap_mode="r+", allow_pickle=False)
        assert list(array.shape) == cfg["shape"] and array.dtype == np.float32
    else:
        if partial.exists():
            raise RuntimeError("Partial cache without checkpoint; inspect before resuming")
        array = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=tuple(cfg["shape"]))
        atomic_json(checkpoint, {"fingerprint": cfg["fingerprint"], "completed": 0})
    starting_count = completed
    started = time.monotonic()
    stop = False
    def stop_signal(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    end = min(total, completed + args.limit) if args.limit else total
    progress("encoding")
    while completed < end and not stop:
        limit = min(end, completed + 32)
        encoded, lengths = encoder(raw_texts[completed:limit])
        values = encoded.detach().cpu().numpy().astype(np.float32)
        assert values.shape == (limit - completed, 1, 4096) and np.isfinite(values).all()
        assert (np.linalg.norm(values[:, 0], axis=-1) > 0).all() and all(n == 1 for n in lengths)
        array[completed:limit] = values
        array.flush()
        completed = limit
        atomic_json(checkpoint, {"fingerprint": cfg["fingerprint"], "completed": completed})
        progress("encoding")
    if completed != total:
        progress("stopped" if stop else "pilot_complete")
        return
    ids = sorted(set([0, total // 2, total - 1]))
    repeat, _ = encoder([raw_texts[i] for i in ids])
    repeated = repeat.detach().cpu().numpy()
    assert np.array_equal(array[ids], repeated), "Cached embeddings are not deterministic"
    assert np.isfinite(array).all()
    array.flush()
    del array
    final = directory / "embeddings.npy"
    os.replace(partial, final)
    record = {"fingerprint": cfg["fingerprint"], "texts": total, "shape": cfg["shape"],
              "sha256": sha_file(final), "repeat_rows": ids, "repeat_exact_equal": True,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "baseline_exact_equal": True}
    atomic_json(directory / "COMPLETE", record)
    progress("complete")


if __name__ == "__main__":
    main()
