#!/usr/bin/env python
"""Download + extract the DFG-TSD images (a single ~7.5 GB tar.bz2, direct, no auth).

Source: https://go.vicos.si/dfgimages -> data.vicos.si/skokec/villard/JPEGImages.tar.bz2
License: CC BY-NC-SA 4.0 (academic/non-commercial; cite Tabernik & Skočaj, 2019).

The archive extracts JPEGImages/*.jpg FLAT (DFG's train/test split lives in the annotation
json, not in folders). We extract into data/dfg/images/ and flatten so a record's file_name
(e.g. "0000001.jpg") resolves to data/dfg/images/0000001.jpg — matching prepare_dfg's path.

Resumable (curl -C -), idempotent (skips if the expected image count is already present).

Usage (run on the workstation where training happens):
  python scripts/fetch_dfg.py                       # download + extract + verify
  python scripts/fetch_dfg.py --keep-archive        # keep the 7.5 GB tar.bz2 after extract
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

URL = "https://go.vicos.si/dfgimages"          # 301 -> data.vicos.si/.../JPEGImages.tar.bz2
EXPECTED_MIN_IMAGES = 6900                       # DFG has 6957 images (train+test)


def _count_jpgs(d: Path) -> int:
    return sum(1 for _ in d.glob("*.jpg")) if d.exists() else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/dfg", help="dataset root (holds train.json etc.)")
    ap.add_argument("--images-dir", default=None, help="default: <root>/images")
    ap.add_argument("--keep-archive", action="store_true",
                    help="keep the tar.bz2 after extract (default: delete to save 7.5 GB)")
    args = ap.parse_args()

    root = Path(args.root)
    images = Path(args.images_dir) if args.images_dir else root / "images"
    archive = root / "JPEGImages.tar.bz2"
    images.mkdir(parents=True, exist_ok=True)

    have = _count_jpgs(images)
    if have >= EXPECTED_MIN_IMAGES:
        print(f"[skip] {have} images already in {images} (>= {EXPECTED_MIN_IMAGES})")
        return

    # 1) download (resumable). curl is the most robust for a single 7.5 GB file.
    print(f"[1/3] downloading DFG images (~7.5 GB) -> {archive} (resumable)")
    rc = subprocess.run(["curl", "-L", "-C", "-", "--fail", "--retry", "5",
                         "-o", str(archive), URL]).returncode
    if rc != 0 or not archive.exists():
        sys.exit(f"[erro] download falhou (curl rc={rc}). Retomável: rode de novo.")

    # 2) extract (bz2), flattening JPEGImages/ into images/
    print(f"[2/3] extracting {archive.name} -> {images}")
    with tarfile.open(archive, "r:bz2") as tf:
        for m in tf:
            if not m.isfile() or not m.name.lower().endswith((".jpg", ".jpeg")):
                continue
            m.name = Path(m.name).name          # flatten (drop JPEGImages/ prefix)
            tf.extract(m, images)

    # 3) verify + cleanup
    n = _count_jpgs(images)
    print(f"[3/3] {n} images extracted in {images}")
    if n < EXPECTED_MIN_IMAGES:
        sys.exit(f"[erro] só {n} imagens (esperava >= {EXPECTED_MIN_IMAGES}) — extração incompleta?")
    if not args.keep_archive:
        archive.unlink(missing_ok=True)
        print(f"  removed {archive.name} (use --keep-archive to keep)")
    print(f"[ok] DFG images ready. Next: prepare_dfg.py (done) -> tile -> train (cheap arms).")


if __name__ == "__main__":
    main()
