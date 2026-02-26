import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Download DDColor weights via ModelScope snapshot_download")
    parser.add_argument(
        "--model_id",
        default="damo/cv_ddcolor_image-colorization",
        help="ModelScope model id, default matches README",
    )
    parser.add_argument(
        "--cache_dir",
        default="./modelscope",
        help="Where to store downloaded model assets",
    )
    args = parser.parse_args()

    from modelscope.hub.snapshot_download import snapshot_download

    model_dir = snapshot_download(args.model_id, cache_dir=args.cache_dir)
    model_dir_abs = os.path.abspath(model_dir)
    print(f"model assets saved to: {model_dir_abs}")

    candidates: list[str] = []
    for root, _, files in os.walk(model_dir_abs):
        for name in files:
            lower = name.lower()
            if lower in ("pytorch_model.pt", "model.pt") or lower.endswith((".pt", ".pth")):
                candidates.append(os.path.join(root, name))

    if not candidates:
        print("No .pt/.pth weights found under the downloaded directory.")
        return

    print("weight candidates:")
    for p in candidates:
        print("-", p)


if __name__ == "__main__":
    main()
