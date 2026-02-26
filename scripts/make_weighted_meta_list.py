import argparse
import os
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2


DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def iter_image_paths(root: Path, exts: Sequence[str]) -> List[Path]:
    exts_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    paths: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in exts_set:
                paths.append(p)
    return paths


def is_big_enough(p: Path, min_size: int) -> bool:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    return min(h, w) >= min_size


def write_list(paths: List[Path], out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(p.as_posix() + "\n")


def main():
    ap = argparse.ArgumentParser(description="Make weighted mixed meta_info_file from two roots (e.g., target thickpaint + aux mixed).")
    ap.add_argument("--target-root", required=True)
    ap.add_argument("--aux-root", required=True)
    ap.add_argument("--out", required=True, help="Output train txt path")
    ap.add_argument("--ext", nargs="*", default=list(DEFAULT_EXTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-size", type=int, default=256)
    ap.add_argument("--no-size-check", action="store_true")
    ap.add_argument("--val-ratio", type=float, default=0.05)

    ap.add_argument("--target-ratio", type=float, default=0.6, help="Fraction of target samples in final mix")
    ap.add_argument("--total", type=int, default=0, help="Total number of samples in final mix (0=use all available)"
                   )
    ap.add_argument("--oversample-target", action="store_true", help="Allow repeating target samples if needed")
    ap.add_argument("--oversample-aux", action="store_true", help="Allow repeating aux samples if needed")

    args = ap.parse_args()

    target_root = Path(args.target_root)
    aux_root = Path(args.aux_root)
    out = Path(args.out)

    if not target_root.exists():
        raise SystemExit(f"target-root not found: {target_root}")
    if not aux_root.exists():
        raise SystemExit(f"aux-root not found: {aux_root}")

    target_paths = iter_image_paths(target_root, args.ext)
    aux_paths = iter_image_paths(aux_root, args.ext)

    if not args.no_size_check:
        target_paths = [p for p in target_paths if is_big_enough(p, args.min_size)]
        aux_paths = [p for p in aux_paths if is_big_enough(p, args.min_size)]

    random.seed(args.seed)
    random.shuffle(target_paths)
    random.shuffle(aux_paths)

    if len(target_paths) == 0 or len(aux_paths) == 0:
        raise SystemExit(f"Empty after filtering: target={len(target_paths)}, aux={len(aux_paths)}")

    if args.total and args.total > 0:
        total = args.total
    else:
        total = len(target_paths) + len(aux_paths)

    target_n = int(round(total * float(args.target_ratio)))
    aux_n = total - target_n

    def take(paths: List[Path], n: int, oversample: bool) -> List[Path]:
        if n <= len(paths):
            return paths[:n]
        if not oversample:
            return paths[:]  # will be shorter than requested
        out_list = paths[:]
        while len(out_list) < n:
            need = n - len(out_list)
            out_list.extend(random.sample(paths, k=min(need, len(paths))))
        return out_list[:n]

    picked_target = take(target_paths, target_n, args.oversample_target)
    picked_aux = take(aux_paths, aux_n, args.oversample_aux)

    mixed = picked_target + picked_aux
    random.shuffle(mixed)

    # split val
    val_n = int(len(mixed) * float(args.val_ratio))
    val_paths = mixed[:val_n]
    train_paths = mixed[val_n:]

    write_list(train_paths, out)
    val_out = out.with_name(out.stem + "_val" + out.suffix)
    write_list(val_paths, val_out)

    print(f"target_available={len(target_paths)}, aux_available={len(aux_paths)}")
    print(f"picked_target={len(picked_target)}, picked_aux={len(picked_aux)}")
    print(f"train={len(train_paths)} -> {out}")
    print(f"val={len(val_paths)} -> {val_out}")


if __name__ == "__main__":
    main()
