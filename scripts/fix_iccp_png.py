#!/usr/bin/env python
"""Fix libpng iCCP warnings by re-saving PNG files without problematic ICC metadata.

This script reads PNG paths from one or more meta list files and rewrites those PNGs
in-place using OpenCV. Re-encoding removes problematic embedded ICC chunks in most cases.

Example:
  python scripts/fix_iccp_png.py \
    --meta data_list/paint_mix_train.txt data_list/paint_mix_train_val.txt \
    --root .
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def collect_png_paths(meta_files, root: Path):
    paths = []
    seen = set()
    for meta in meta_files:
        with open(meta, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                rel = line.strip()
                if not rel:
                    continue
                p = (root / rel).resolve()
                if p.suffix.lower() != ".png":
                    continue
                key = str(p)
                if key in seen:
                    continue
                seen.add(key)
                paths.append(p)
    return paths


def rewrite_png(path: Path) -> bool:
    data = cv2.imdecode(
        np.fromfile(str(path), dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    if data is None:
        return False
    ok, encoded = cv2.imencode(".png", data)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def main():
    parser = argparse.ArgumentParser(description="Fix PNG iCCP warnings by rewriting PNG files")
    parser.add_argument("--meta", nargs="+", required=True, help="Meta list files")
    parser.add_argument("--root", default=".", help="Project root for resolving relative paths")
    parser.add_argument("--dry_run", action="store_true", help="Only count files, do not rewrite")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    meta_files = [Path(m).resolve() for m in args.meta]

    png_paths = collect_png_paths(meta_files, root)
    exists = [p for p in png_paths if p.exists()]
    missing = [p for p in png_paths if not p.exists()]

    print(f"PNG entries in meta: {len(png_paths)}")
    print(f"Existing PNG files: {len(exists)}")
    print(f"Missing PNG files: {len(missing)}")

    if args.dry_run:
        return

    ok_cnt = 0
    fail_cnt = 0
    for p in exists:
        if rewrite_png(p):
            ok_cnt += 1
        else:
            fail_cnt += 1
            print(f"[FAIL] {p}")

    print(f"Rewrite done. success={ok_cnt}, fail={fail_cnt}")


if __name__ == "__main__":
    main()
