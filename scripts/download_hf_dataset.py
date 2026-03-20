import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Optional


def import_hf_datasets_module():
    """Import huggingface datasets package, avoiding local ./datasets shadowing."""
    cwd_abs = os.path.abspath(os.getcwd())
    repo_root_abs = os.path.abspath(Path(__file__).resolve().parent.parent)
    original_sys_path = list(sys.path)
    try:
        cleaned = []
        for p in sys.path:
            abs_p = os.path.abspath(p or cwd_abs)
            if abs_p in (cwd_abs, repo_root_abs):
                continue
            cleaned.append(p)
        sys.path = cleaned
        module = importlib.import_module("datasets")
    finally:
        sys.path = original_sys_path

    if not hasattr(module, "load_dataset"):
        raise SystemExit(
            "Imported 'datasets' but it is not huggingface-datasets package. "
            "Please run this script from scripts/ or ensure pip package 'datasets' is installed in current env."
        )
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset and export images to local folder."
    )
    parser.add_argument(
        "--dataset",
        default="vollerei-id/anime_cartoon2",
        help="Dataset repo id on Hugging Face, e.g. vollerei-id/anime_cartoon2",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional dataset config name (subset)",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split to export, e.g. train / validation / test; empty means all splits",
    )
    parser.add_argument(
        "--cache_dir",
        default="./hf_cache",
        help="Cache directory used by load_dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="./datasets",
        help="Root output folder where images are exported",
    )
    parser.add_argument(
        "--image_column",
        default=None,
        help="Image column name. If omitted, script auto-detects first Image feature",
    )
    parser.add_argument(
        "--ext",
        default="png",
        help="Saved image extension, e.g. png or jpg",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap for number of samples per split, for quick testing",
    )
    return parser


def pick_image_column(dataset, image_feature_type, preferred: Optional[str]) -> str:
    if preferred:
        if preferred not in dataset.column_names:
            raise SystemExit(f"image column not found: {preferred}")
        return preferred

    for name, feat in dataset.features.items():
        if isinstance(feat, image_feature_type):
            return name
    raise SystemExit(
        f"No image feature found in columns: {dataset.column_names}. "
        "Use --image_column to set one manually."
    )


def sanitize_stem(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "item"


def export_one_split(split_name: str, dataset, image_column: str, output_dir: Path, ext: str, max_samples: Optional[int]) -> int:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    total = len(dataset)
    limit = total if max_samples is None else min(total, max_samples)

    for idx in range(limit):
        row = dataset[idx]
        image = row[image_column]

        if image is None or not hasattr(image, "save"):
            continue

        stem = None
        for key in ("id", "image_id", "file_name", "filename", "name"):
            if key in row and row[key] is not None:
                stem = sanitize_stem(str(row[key]))
                break
        if not stem:
            stem = f"{idx:08d}"

        file_path = split_dir / f"{stem}.{ext}"
        if file_path.exists():
            file_path = split_dir / f"{stem}_{idx:08d}.{ext}"

        image.save(file_path)

    return limit


def main() -> None:
    args = build_parser().parse_args()
    hf_datasets = import_hf_datasets_module()

    print("Loading dataset from Hugging Face...")
    loaded = hf_datasets.load_dataset(
        path=args.dataset,
        name=args.name,
        split=args.split,
        cache_dir=args.cache_dir,
    )

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if isinstance(loaded, hf_datasets.DatasetDict):
        split_map = dict(loaded.items())
    elif isinstance(loaded, hf_datasets.Dataset):
        split_map = {args.split or "train": loaded}
    else:
        raise SystemExit(f"Unexpected dataset type: {type(loaded)}")

    exported_total = 0
    for split_name, split_ds in split_map.items():
        image_column = pick_image_column(split_ds, hf_datasets.Image, args.image_column)
        exported = export_one_split(
            split_name=split_name,
            dataset=split_ds,
            image_column=image_column,
            output_dir=output_root,
            ext=args.ext,
            max_samples=args.max_samples,
        )
        exported_total += exported
        print(
            f"split={split_name}, image_column={image_column}, exported={exported}, "
            f"target={output_root / split_name}"
        )

    print(f"Done. Total exported images: {exported_total}")


if __name__ == "__main__":
    main()