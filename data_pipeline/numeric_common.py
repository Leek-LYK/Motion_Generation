"""Shared paths and atomic metadata helpers for numerical model inputs."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data/processed/g1_text_motion_v1"
OUTPUT = ROOT / "data/processed/g1_model_inputs_v1"
KIMODO = ROOT.parent / "kimodo"
SPLITS = ("train", "val", "test_content", "test_repetition")


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def atomic_parquet(frame, path):
    path = Path(path)
    temp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def source_versions():
    from importlib.metadata import version
    files = sorted((KIMODO / "source/kimodo").rglob("*.py"))
    files += sorted(p for p in (KIMODO / "source/kimodo/assets/skeletons/g1skel34").rglob("*")
                    if p.is_file() and p.suffix in (".p", ".xml"))
    return {"source_sha256": {str(p.relative_to(KIMODO)): sha_file(p) for p in files},
            "versions": {p: version(p) for p in ("torch", "numpy", "scipy", "kimodo")}}
