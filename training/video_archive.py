"""Content-addressed local checkpoint snapshots for repeatable video tests."""
import os
from pathlib import Path
import shutil

from training.m3 import ROOT, atomic_json, sha


def freeze_model(run, checkpoint, expected_sha):
    directory = ROOT / "exports/_models" / expected_sha
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "checkpoint.pt"
    if not destination.exists():
        temp = directory / f"checkpoint.{os.getpid()}.tmp"
        shutil.copyfile(checkpoint, temp)
        if sha(temp) != expected_sha:
            temp.unlink()
            raise RuntimeError("Checkpoint changed during snapshot")
        os.replace(temp, destination)
    elif sha(destination) != expected_sha:
        raise RuntimeError("Existing model snapshot has changed")
    files = [run / "run_config.json", run / "subset.json", ROOT / "pyproject.toml", ROOT / "uv.lock"]
    files += list((run / "stats/motion").rglob("*.npy"))
    for source in files:
        rel = source.relative_to(run) if source.is_relative_to(run) else Path("environment") / source.name
        target = directory / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != sha(source):
                # The same model can be exported with a later renderer environment.
                if rel.parts[0] == "environment":
                    continue
                raise RuntimeError(f"Snapshot context changed: {rel}")
        else:
            shutil.copyfile(source, target)
    if not (directory / "model.json").exists():
        atomic_json(directory / "model.json", {"source_run": str(run), "original_checkpoint": str(checkpoint),
                    "sha256": expected_sha, "snapshot": str(destination),
                    "policy": "local independent copy, deduplicated by full checkpoint SHA256; do not overwrite"})
    return destination
