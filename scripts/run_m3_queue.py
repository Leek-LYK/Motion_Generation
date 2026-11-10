#!/usr/bin/env python3
"""Run a gated single-clip pilot followed by the finite 32-clip training budget."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_pipeline.numeric_common import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    experiment.relative_to(ROOT / "runs")
    lock = (experiment / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not json.loads((experiment / "verification/report.json").read_text())["passed"]:
        raise ValueError("M3 implementation verification has not passed")
    child = [None]
    stopped = [False]

    def stop(*_):
        stopped[0] = True
        if child[0] is not None and child[0].poll() is None:
            child[0].send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def status(phase, **extra):
        atomic_json(experiment / "queue_status.json", {"phase": phase, "pid": os.getpid(),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra})

    for name, subset, budget, sample_every in (("single_clip", "single.json", 1000, 250),
                                              ("text32", "train32.json", 10000, 500)):
        if stopped[0]:
            break
        run = experiment / name
        command = [sys.executable, "-B", "-u", str(ROOT / "scripts/train_m3.py"),
                   "--subset", str(experiment / "subsets" / subset), "--run-dir", str(run),
                   "--steps", str(budget), "--sample-every", str(sample_every)]
        if (run / "checkpoints/latest.pt").exists():
            command.append("--resume")
        with (experiment / f"{name}.log").open("ab", buffering=0) as log:
            child[0] = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL)
            status(name, child_pid=child[0].pid, command=command)
            print(f"Started {name}, pid={child[0].pid}, budget={budget}", flush=True)
            code = child[0].wait()
        if stopped[0]:
            break
        if code:
            status("failed", stage=name, exit_code=code)
            raise SystemExit(code)
        if name == "single_clip":
            initial = json.loads((run / "samples/step_000000/raw/metrics.json").read_text())[0]
            final = json.loads((run / "samples/step_001000/raw/metrics.json").read_text())[0]
            passed = final["finite_qpos"] and final["world_mpjpe_m"] < min(0.15, initial["world_mpjpe_m"] * 0.3)
            gate = {"passed": passed, "criterion": "raw pure-noise DDIM world MPJPE <0.15m and <30% initialization",
                    "initial": initial, "final": final, "physical_tracking_verified": False}
            atomic_json(experiment / "single_clip_gate.json", gate)
            if not passed:
                status("single_clip_needs_review", gate=gate)
                raise SystemExit(2)
    status("stopped" if stopped[0] else "budget_completed")


if __name__ == "__main__":
    main()
