"""Download and extract ATRW from LILA into data/atrw/.

Usage:
    python scripts/download_atrw.py              # re-ID only (~220 MB)
    python scripts/download_atrw.py reid detection pose

Source: https://lila.science/datasets/atrw (CC BY-NC-SA 4.0)
"""

import sys
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/public-datasets-lila/cvwc2019"

FILES = {
    "reid": [
        "train/atrw_reid_train.tar.gz",
        "train/atrw_anno_reid_train.tar.gz",
        "test/atrw_reid_test.tar.gz",
    ],
    "detection": [
        "train/atrw_detection_train.tar.gz",
        "train/atrw_anno_detection_train.tar.gz",
        "test/atrw_detection_test.tar.gz",
    ],
    "pose": [
        "train/atrw_pose_train.tar.gz",
        "train/atrw_pose_val.tar.gz",
        "train/atrw_anno_pose_train.tar.gz",
        "test/atrw_pose_test.tar.gz",
    ],
}

OUT = Path(__file__).resolve().parent.parent / "data" / "atrw"


def progress(blocks, block_size, total):
    done = blocks * block_size
    if total > 0:
        print(f"\r  {min(done, total) / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)


def fetch(rel_path):
    name = Path(rel_path).name
    dest_dir = OUT / name.removesuffix(".tar.gz")
    if dest_dir.exists():
        print(f"skip {name} (already extracted)")
        return
    archive = OUT / name
    if not archive.exists():
        print(f"download {name}")
        urllib.request.urlretrieve(f"{BASE}/{rel_path}", archive, progress)
        print()
    print(f"extract  {name}")
    with tarfile.open(archive) as tar:
        tar.extractall(dest_dir, filter="data")
    archive.unlink()


def main():
    groups = sys.argv[1:] or ["reid"]
    for g in groups:
        if g not in FILES:
            sys.exit(f"unknown group {g!r}; choose from {list(FILES)}")
    OUT.mkdir(parents=True, exist_ok=True)
    for g in groups:
        for rel_path in FILES[g]:
            fetch(rel_path)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
