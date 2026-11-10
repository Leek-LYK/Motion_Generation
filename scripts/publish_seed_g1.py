"""Copy an extracted G1 archive to project storage, then publish it atomically."""

import argparse
import concurrent.futures
import fcntl
import hashlib
import os
import shutil
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/bones_seed"
SCRATCH_RECORD = ROOT / "logs/g1_extraction_scratch_path.txt"
STAGING = RAW / ".extracting/g1"
DESTINATION = RAW / "g1"


def digest(path: Path) -> bytes:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.digest()


def copy_file(source: Path, source_root: Path, stage_only: bool) -> str:
    target = STAGING / source.relative_to(source_root)
    if target.exists() and target.stat().st_size == source.stat().st_size and digest(target) == digest(source):
        return "reused"
    temporary = target.with_name(f".{target.name}.copying")
    shutil.copyfile(source, temporary)
    if temporary.stat().st_size != source.stat().st_size:
        temporary.unlink()
        if stage_only:
            return "unstable"
        raise RuntimeError(f"Incomplete copy: {source}")
    os.replace(temporary, target)
    return "copied"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-only", action="store_true", help="Copy stable files without publishing the directory")
    args = parser.parse_args()
    scratch = Path(SCRATCH_RECORD.read_text(encoding="utf-8").strip()).resolve()
    if scratch.parent != Path("/tmp") or not scratch.name.startswith("kimodo-g1-extract."):
        raise RuntimeError(f"Unexpected extraction directory: {scratch}")
    source_root = scratch / "g1"
    if not source_root.is_dir():
        raise RuntimeError(f"Extracted G1 directory not found: {source_root}")
    if DESTINATION.exists():
        raise RuntimeError(f"Refusing to overwrite existing destination: {DESTINATION}")

    lock_path = ROOT / "logs/g1_publish.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        files = []
        for root, directories, names in os.walk(source_root, followlinks=False):
            current = Path(root)
            for name in directories:
                if (current / name).is_symlink():
                    raise RuntimeError(f"Unexpected symbolic link in archive: {current / name}")
            (STAGING / current.relative_to(source_root)).mkdir(parents=True, exist_ok=True)
            for name in names:
                source = current / name
                if source.is_symlink() or not source.is_file():
                    raise RuntimeError(f"Unexpected non-regular file in archive: {source}")
                if args.stage_only and time.time() - source.stat().st_mtime < 30:
                    continue
                files.append(source)

        print(f"Copying {len(files)} files to project storage", flush=True)
        counts = {"copied": 0, "reused": 0, "unstable": 0}
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            for index, result in enumerate(pool.map(lambda path: copy_file(path, source_root, args.stage_only), files), 1):
                counts[result] += 1
                if index % 5000 == 0:
                    print(f"{index}/{len(files)} complete; {counts}", flush=True)

        if args.stage_only:
            print(f"Staging pass complete; {counts}", flush=True)
            return
        if DESTINATION.exists():
            raise RuntimeError(f"Destination appeared during copy: {DESTINATION}")
        STAGING.rename(DESTINATION)
        print(f"Published {DESTINATION}; {counts}", flush=True)


if __name__ == "__main__":
    main()
