import argparse
import os
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_image_paths(root: Path, exts: Sequence[str]) -> Iterable[Path]:
    exts_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in exts_set:
                yield p


def is_big_enough(p: Path, min_size: int) -> bool:
    img = cv2.imdecode(
        cv2.UMat(open(p, "rb").read()).get(),
        cv2.IMREAD_COLOR,
    )
    # fallback if UMat path fails on some builds
    if img is None:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    return min(h, w) >= min_size


def write_list(paths: List[Path], out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(p.as_posix()) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Generate meta_info_file txt for DDColor LabDataset.")
    ap.add_argument("--root", type=str, required=True, help="Root folder containing images")
    ap.add_argument("--out", type=str, required=True, help="Output txt path")
    ap.add_argument("--ext", type=str, nargs="*", default=sorted(IMG_EXTS), help="Allowed extensions")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle list")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="Limit to first N images after shuffle (0=all)")
    ap.add_argument("--min-size", type=int, default=256, help="Filter images with min(H,W) < min-size")
    ap.add_argument(
        "--no-size-check",
        action="store_true",
        help="Skip decoding to check size (faster, but may include tiny/bad images)",
    )
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="If >0, also write a *_val.txt split with this ratio.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    paths = list(iter_image_paths(root, args.ext))

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(paths)

    if not args.no_size_check:
        kept = []
        for p in paths:
            try:
                if is_big_enough(p, args.min_size):
                    kept.append(p)
            except Exception:
                continue
        paths = kept

    if args.limit and args.limit > 0:
        paths = paths[: args.limit]

    if args.val_ratio and args.val_ratio > 0:
        n_total = len(paths)
        n_val = int(n_total * args.val_ratio)
        val_paths = paths[:n_val]
        train_paths = paths[n_val:]

        write_list(train_paths, out)
        val_out = out.with_name(out.stem + "_val" + out.suffix)
        write_list(val_paths, val_out)
        print(f"Wrote train={len(train_paths)} to {out}")
        print(f"Wrote val={len(val_paths)} to {val_out}")
    else:
        write_list(paths, out)
        print(f"Wrote {len(paths)} paths to {out}")


if __name__ == "__main__":
    main()
