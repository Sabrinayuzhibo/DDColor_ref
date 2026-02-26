import argparse
import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, Tuple


DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def sanitize(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff\uff00-\uffef]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s[:80] if len(s) > 80 else s


def file_sha1_short(p: Path, n: int = 10) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def iter_images(root: Path, exts: Tuple[str, ...]) -> Iterable[Path]:
    exts_set = {e.lower() for e in exts}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in exts_set:
                yield p


def unique_target_path(dst_dir: Path, base_name: str, ext: str, src: Path) -> Path:
    candidate = dst_dir / f"{base_name}{ext}"
    if not candidate.exists():
        return candidate

    # collision: add short content hash
    suffix = file_sha1_short(src)
    candidate2 = dst_dir / f"{base_name}__{suffix}{ext}"
    if not candidate2.exists():
        return candidate2

    # extremely unlikely: add counter
    for i in range(2, 10000):
        cand = dst_dir / f"{base_name}__{suffix}_{i}{ext}"
        if not cand.exists():
            return cand
    raise RuntimeError(f"Unable to find unique name for {src}")


def main():
    ap = argparse.ArgumentParser(description="Flatten nested image folders into a single directory.")
    ap.add_argument("--root", required=True, help="Root folder containing many subfolders of images")
    ap.add_argument(
        "--dst",
        default=None,
        help="Destination folder. Default: same as --root (flatten into root)",
    )
    ap.add_argument("--ext", nargs="*", default=list(DEFAULT_EXTS), help="Image extensions")
    ap.add_argument(
        "--mode",
        choices=("move", "copy"),
        default="move",
        help="Whether to move or copy images into destination",
    )
    ap.add_argument(
        "--prefix",
        choices=("parent", "relpath", "none"),
        default="parent",
        help="How to prefix filenames to reduce collisions",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without making changes",
    )
    args = ap.parse_args()

    root = Path(args.root)
    dst = Path(args.dst) if args.dst else root

    if not root.exists():
        raise SystemExit(f"Root not found: {root}")
    dst.mkdir(parents=True, exist_ok=True)

    moved = 0
    skipped = 0

    for src in iter_images(root, tuple(args.ext)):
        # skip images already in destination root when flattening into root
        if src.parent.resolve() == dst.resolve():
            skipped += 1
            continue

        if args.prefix == "parent":
            prefix = sanitize(src.parent.name)
        elif args.prefix == "relpath":
            rel = src.parent.relative_to(root)
            prefix = sanitize(str(rel).replace(os.sep, "_"))
        else:
            prefix = ""

        stem = sanitize(src.stem)
        base_name = f"{prefix}__{stem}" if prefix else stem
        ext = src.suffix.lower()

        target = unique_target_path(dst, base_name, ext, src)

        if args.dry_run:
            print(f"{args.mode}: {src} -> {target}")
            moved += 1
            continue

        if args.mode == "move":
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(target))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(target))
        moved += 1

        if moved % 500 == 0:
            print(f"processed {moved} images...", flush=True)

    print(f"done. {args.mode}d={moved}, skipped_in_dst={skipped}, dst={dst}")


if __name__ == "__main__":
    main()
