"""Resume the pinned BONES-SEED G1 archive download and verify it before use."""

import hashlib
import os
from pathlib import Path

os.environ.setdefault("MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS", "16")
os.environ.setdefault("MODELSCOPE_DOWNLOAD_PART_SIZE_MB", "256")

from modelscope_hub._download import _parallel_download  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw/bones_seed"
TARGET = RAW_DIR / "g1.tar.gz"
STAGING = RAW_DIR / ".download_parts/g1.tar.gz"
SIZE = 23_499_973_647
SHA256 = "52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a"
URL = (
    "https://modelscope.cn/api/v1/datasets/bones-studio/seed/repo"
    "?Revision=master&FilePath=g1.tar.gz"
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if TARGET.exists():
        if TARGET.stat().st_size == SIZE and file_hash(TARGET) == SHA256:
            print(f"Already verified: {TARGET}")
            return
        raise RuntimeError(f"Existing target does not match the pinned archive: {TARGET}")
    if STAGING.exists():
        raise RuntimeError(f"Inspect the previously assembled staging file: {STAGING}")
    STAGING.parent.mkdir(parents=True, exist_ok=True)
    print("Resuming BONES-SEED G1 archive download", flush=True)
    digest = _parallel_download(URL, STAGING, SIZE, headers=None, cookies=None)
    if STAGING.stat().st_size != SIZE or digest != SHA256:
        raise RuntimeError(f"Downloaded archive failed SHA-256 verification: {STAGING}")
    STAGING.rename(TARGET)
    print(f"Verified and published: {TARGET}\nSHA256: {digest}")


if __name__ == "__main__":
    main()
