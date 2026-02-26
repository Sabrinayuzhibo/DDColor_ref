import os
from pathlib import Path
import sys
import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from infer_style_transfer import _MODNetTorchMatte


def _safe_write(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def _matte_to_u8(matte: np.ndarray) -> np.ndarray:
    return np.clip(matte * 255.0, 0, 255).astype(np.uint8)


def _overlay_matte(img_bgr: np.ndarray, matte: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    m = np.clip(matte, 0.0, 1.0).reshape(h, w, 1).astype(np.float32)
    color = np.zeros_like(img_bgr, dtype=np.float32)
    color[:, :, 1] = 255.0
    out = img_bgr.astype(np.float32) * (1.0 - 0.45 * m) + color * (0.45 * m)
    return np.clip(out, 0, 255).astype(np.uint8)


def _foreground_cutout(img_bgr: np.ndarray, matte: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    m = np.clip(matte, 0.0, 1.0).reshape(h, w, 1).astype(np.float32)
    bg = np.full_like(img_bgr, 255, dtype=np.float32)
    out = img_bgr.astype(np.float32) * m + bg * (1.0 - m)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    root = Path(__file__).resolve().parents[1]
    ckpt = root / "pretrain" / "modnet_webcam_portrait_matting.ckpt"
    out_dir = root / "results_modnet_matte_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_dir = root / "assets" / "test_images"
    picks = [
        test_dir / "Audrey Hepburn.jpg",
        test_dir / "Ansel Adams _ Moore Photography.jpeg",
    ]
    picks.extend(sorted([p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])[:6])

    uniq = []
    seen = set()
    for p in picks:
        if p.exists() and p.name not in seen:
            uniq.append(p)
            seen.add(p.name)

    matte_model = _MODNetTorchMatte(
        ckpt_path=str(ckpt),
        device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
        ref_size=512,
        repo_root=None,
    )

    lines = []
    for p in uniq:
        img = cv2.imread(str(p))
        if img is None:
            continue
        matte = matte_model.predict_matte(img)
        matte_u8 = _matte_to_u8(matte)
        overlay = _overlay_matte(img, matte)
        cutout = _foreground_cutout(img, matte)

        stem = p.stem
        _safe_write(out_dir / f"{stem}__matte.png", matte_u8)
        _safe_write(out_dir / f"{stem}__overlay.png", overlay)
        _safe_write(out_dir / f"{stem}__cutout.png", cutout)

        fg_ratio = float((matte > 0.5).mean())
        lines.append(f"{p.name}\tmin={matte.min():.4f}\tmax={matte.max():.4f}\tmean={matte.mean():.4f}\tfg@0.5={fg_ratio:.4f}")

    report = out_dir / "modnet_stats.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved MODNet demos to: {out_dir}")
    print(f"Stats: {report}")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
