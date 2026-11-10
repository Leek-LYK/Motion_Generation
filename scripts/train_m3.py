#!/usr/bin/env python3
"""Finite-budget M3 run with TensorBoard, resume and raw-motion snapshots."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import argparse
import fcntl
import json
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.tensorboard import SummaryWriter
from training.m3 import (INPUTS, Trainer, atomic_json, configure, export_samples, load_samples,
                         sha, validate_inputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/m3_tiny.json")
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.steps, args.eval_every, args.sample_every, args.log_every) < 1 or not 2 <= args.sample_steps <= 1000:
        raise ValueError("Invalid run budget or cadence")
    run = args.run_dir.resolve()
    run.relative_to(ROOT / "runs")
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    latest = run / "checkpoints/latest.pt"
    if not args.resume and (latest.exists() or (run / "run_config.json").exists()):
        raise FileExistsError("Use a new run directory or --resume")
    ready = validate_inputs()
    config = json.loads(args.config.read_text())
    subset = json.loads(args.subset.read_text())
    if subset["m2_fingerprint"] != ready["fingerprint"]:
        raise ValueError("Subset/M2 mismatch")
    configure(config["seed"])
    identity = {"m2_fingerprint": ready["fingerprint"], "subset_sha256": sha(args.subset),
                "text_config_fingerprint": ready["text_config_fingerprint"],
                "training_code_sha256": sha(ROOT / "training/m3.py")}
    train, val = load_samples(subset, "train"), load_samples(subset, "val")
    trainer = Trainer(config, train, identity, args.device)
    if args.resume:
        trainer.restore(latest)
    else:
        atomic_json(run / "run_config.json", {"config": config, "identity": identity,
                    "train_samples": len(train), "val_samples": len(val),
                    "parameters": sum(p.numel() for p in trainer.model.parameters()),
                    "initialization": "random; ckpt_path=None; frozen cached text; no LLM loaded",
                    "human_semantic_review": subset["human_semantic_review"]})
        atomic_json(run / "subset.json", subset)
        for name in ("global_root", "body", "local_root"):
            destination = run / "stats/motion" / name
            destination.mkdir(parents=True, exist_ok=True)
            for stat in ("mean.npy", "std.npy"):
                (destination / stat).write_bytes((INPUTS / "stats/motion" / name / stat).read_bytes())
    writer = SummaryWriter(str(run / "tensorboard"), flush_secs=10,
                           purge_step=trainer.step + 1 if args.resume else None)
    writer.add_text("experiment/config", json.dumps(config, indent=2), trainer.step)
    writer.add_text("experiment/samples", "\n\n".join(f"{s['group']}: {s['text']}" for s in train), trainer.step)
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    representatives = []
    for sample in train:
        if sample["group"] not in {s["group"] for s in representatives}:
            representatives.append(sample)
    fixed_train = [s for representative in representatives
                   for s in [x for x in train if x["group"] == representative["group"]][:2]][:8]
    start = time.monotonic()
    initial_step = trainer.step
    last_values = {}

    def status(phase, **extra):
        atomic_json(run / "status.json", {"phase": phase, "pid": os.getpid(), "step": trainer.step,
                    "target_steps": args.steps, "elapsed_s_this_process": time.monotonic() - start,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(), "last_train": last_values, **extra})

    def evaluate():
        values = trainer.evaluate(fixed_train)
        for prefix, result in (("fixed_train", values), ("heldout_val", trainer.evaluate(val[:8])),
                               ("fixed_train_ema", trainer.evaluate(fixed_train, trainer.ema))):
            for key, value in result.items():
                writer.add_scalar(f"{prefix}/{key}", value, trainer.step)
        if values["loss"] < trainer.best:
            trainer.best = values["loss"]
            trainer.save(run / "checkpoints/best_train.pt")
        trainer.save(latest)
        atomic_json(run / "last_evaluation.json", {"step": trainer.step, "fixed_train": values,
                                                   "best_train_loss": trainer.best})
        writer.flush()
        return values

    def sample():
        for tag, model in (("raw", trainer.model), ("ema", trainer.ema)):
            metrics = export_samples(model, representatives, run / "samples" / f"step_{trainer.step:06d}" / tag,
                                     config["seed"] + 200, args.sample_steps, writer, trainer.step, tag)
            for key, value in metrics.items():
                writer.add_scalar(f"generation_{tag}/{key}", value, trainer.step)
        writer.flush()

    try:
        status("starting")
        if trainer.step == 0:
            evaluate()
            sample()
        status("training")
        with (run / "metrics.jsonl").open("a", buffering=1) as metrics_file:
            while trainer.step < args.steps and not stop[0]:
                tick = time.monotonic()
                last_values = trainer.update()
                last_values["step_seconds"] = time.monotonic() - tick
                if trainer.step % args.log_every == 0:
                    for key, value in last_values.items():
                        writer.add_scalar(f"train/{key}", value, trainer.step)
                    writer.add_scalar("system/peak_allocated_gib", torch.cuda.max_memory_allocated() / 1024**3, trainer.step)
                    writer.add_scalar("train/learning_rate", config["learning_rate"], trainer.step)
                    metrics_file.write(json.dumps({"step": trainer.step, **last_values}) + "\n")
                    status("training")
                    print(f"step={trainer.step} loss={last_values['loss']:.6f} root_grad={last_values['root_grad_norm']:.4f} "
                          f"body_grad={last_values['body_grad_norm']:.4f} seconds={last_values['step_seconds']:.3f}", flush=True)
                if trainer.step % args.eval_every == 0:
                    status("evaluating")
                    values = evaluate()
                    print(f"fixed_train step={trainer.step} loss={values['loss']:.6f}", flush=True)
                if trainer.step % args.sample_every == 0:
                    status("sampling")
                    sample()
                    status("training")
            evaluate()
            if not stop[0] and trainer.step % args.sample_every:
                sample()
        status("stopped" if stop[0] else "budget_completed", best_train_loss=trainer.best,
               updates_this_process=trainer.step - initial_step)
    except Exception as error:
        status("failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    main()
