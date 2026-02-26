#!/usr/bin/env python
"""Prune missing image paths from meta list files.

Usage:
  python scripts/prune_meta_missing.py --root . --meta data_list/a.txt data_list/b.txt
"""

import argparse
from pathlib import Path


def prune_one(meta_path: Path, root: Path):
    with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    keep = []
    removed = []
    for rel in lines:
        p = (root / rel).resolve()
        if p.exists():
            keep.append(rel)
        else:
            removed.append(rel)

    backup = meta_path.with_suffix(meta_path.suffix + ".bak")
    with open(backup, "w", encoding="utf-8", newline="\n") as f:
        for rel in lines:
            f.write(rel + "\n")

    with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
        for rel in keep:
            f.write(rel + "\n")

    return len(lines), len(keep), len(removed), backup, removed


def main():
    parser = argparse.ArgumentParser(description="Prune missing files from meta lists")
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--meta", nargs="+", required=True, help="Meta list files")
    parser.add_argument("--show_removed", action="store_true", help="Print removed entries")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    metas = [Path(m).resolve() for m in args.meta]

    total_removed = 0
    for m in metas:
        total, keep, removed, backup, removed_list = prune_one(m, root)
        total_removed += removed
        print(f"{m}: total={total}, keep={keep}, removed={removed}, backup={backup}")
        if args.show_removed and removed_list:
            for rel in removed_list:
                print(f"  - {rel}")

    print(f"Done. Total removed: {total_removed}")


if __name__ == "__main__":
    main()
