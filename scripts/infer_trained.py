#!/usr/bin/env python
"""Infer with a trained BasicSR-style checkpoint (e.g. net_g_latest.pth).

Example:
  python scripts/infer_trained.py \
    --ckpt experiments/train_ddcolor_condA_ms_tokens/models/net_g_latest.pth \
    --input assets/test_images \
    --output results_trained
"""

import argparse
import os
import sys
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ddcolor import DDColor, build_ddcolor_model


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _iter_images(input_path: str):
    if os.path.isfile(input_path):
        yield input_path
        return

    for name in sorted(os.listdir(input_path)):
        full = os.path.join(input_path, name)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in _IMG_EXTS:
            yield full


def _predict_ab(model, img_bgr: np.ndarray, input_size: int, device: torch.device):
    height, width = img_bgr.shape[:2]
    img_f = (img_bgr / 255.0).astype(np.float32)
    orig_l = cv2.cvtColor(img_f, cv2.COLOR_BGR2Lab)[:, :, :1]

    img_resized = cv2.resize(img_f, (input_size, input_size))
    img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
    img_gray_lab = np.zeros((input_size, input_size, 3), dtype=np.float32)
    img_gray_lab[:, :, :1] = img_l
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

    tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
    output_ab = model(tensor_gray_rgb)
    output_ab = F.interpolate(output_ab, size=(height, width))[0].float().cpu().numpy().transpose(1, 2, 0)
    return orig_l, output_ab


def _auto_scale_ab(output_ab: np.ndarray, target_p90: float, min_scale: float, max_scale: float):
    chroma = np.sqrt(np.sum(output_ab * output_ab, axis=-1))
    p90 = float(np.percentile(chroma, 90))
    if p90 < 1e-6:
        scale = max_scale
    else:
        scale = float(np.clip(target_p90 / p90, min_scale, max_scale))
    return output_ab * scale, scale


def _edge_suppress_ab(orig_l: np.ndarray, output_ab: np.ndarray, strength: float, edge_threshold: float):
    if strength <= 0:
        return output_ab

    l = (orig_l[:, :, 0] / 100.0).astype(np.float32)
    grad_x = cv2.Sobel(l, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(l, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    grad = grad / (float(grad.max()) + 1e-6)

    edge = np.clip((grad - edge_threshold) / (1.0 - edge_threshold + 1e-6), 0.0, 1.0)
    edge = cv2.GaussianBlur(edge, (0, 0), 1.0)

    a_smooth = cv2.bilateralFilter(output_ab[:, :, 0].astype(np.float32), d=7, sigmaColor=18, sigmaSpace=9)
    b_smooth = cv2.bilateralFilter(output_ab[:, :, 1].astype(np.float32), d=7, sigmaColor=18, sigmaSpace=9)
    ab_smooth = np.stack([a_smooth, b_smooth], axis=-1)

    blend = np.clip(edge * strength, 0.0, 1.0)[..., None]
    return output_ab * (1.0 - blend) + ab_smooth * blend


def main():
    parser = argparse.ArgumentParser(description="Infer using trained net_g_*.pth")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to BasicSR generator checkpoint (e.g. net_g_latest.pth)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="assets/test_images",
        help="Input image file or folder",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_trained",
        help="Output folder",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
        help="Resize size fed to the model (square)",
    )
    parser.add_argument(
        "--ab_scale",
        type=float,
        default=1.0,
        help="Scale factor for predicted ab channels (e.g. 1.2~1.8 for more vivid colors)",
    )
    parser.add_argument(
        "--ab_auto",
        action="store_true",
        help="Auto-adjust ab scale per image to reduce under/over saturation",
    )
    parser.add_argument(
        "--ab_auto_target_p90",
        type=float,
        default=42.0,
        help="Target P90 chroma magnitude for --ab_auto",
    )
    parser.add_argument("--ab_auto_min", type=float, default=1.0, help="Min auto scale")
    parser.add_argument("--ab_auto_max", type=float, default=2.4, help="Max auto scale")
    parser.add_argument(
        "--edge_suppress",
        type=float,
        default=0.35,
        help="Edge color-bleed suppression strength in [0,1], 0 to disable",
    )
    parser.add_argument(
        "--edge_threshold",
        type=float,
        default=0.12,
        help="Edge threshold for suppression (lower=more aggressive)",
    )

    # These must match your training config/checkpoint.
    parser.add_argument("--model_size", type=str, default="large", choices=["tiny", "large"])
    parser.add_argument("--decoder_type", type=str, default="MultiScaleColorDecoder")
    parser.add_argument("--num_queries", type=int, default=256)
    parser.add_argument("--num_scales", type=int, default=3)
    parser.add_argument("--dec_layers", type=int, default=9)

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_ddcolor_model(
        DDColor,
        model_path=args.ckpt,
        input_size=args.input_size,
        model_size=args.model_size,
        decoder_type=args.decoder_type,
        device=device,
        num_queries=args.num_queries,
        num_scales=args.num_scales,
        dec_layers=args.dec_layers,
    )

    image_paths = list(_iter_images(args.input))
    if len(image_paths) == 0:
        raise SystemExit(f"No images found under: {args.input}")

    for img_path in tqdm(image_paths, desc="Colorizing"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Failed to read: {img_path}")
            continue

        with torch.inference_mode():
            orig_l, output_ab = _predict_ab(model, img, args.input_size, device)

        if args.ab_auto:
            output_ab, _ = _auto_scale_ab(
                output_ab,
                target_p90=args.ab_auto_target_p90,
                min_scale=args.ab_auto_min,
                max_scale=args.ab_auto_max,
            )
        elif args.ab_scale != 1.0:
            output_ab = output_ab * args.ab_scale

        output_ab = _edge_suppress_ab(orig_l, output_ab, args.edge_suppress, args.edge_threshold)
        output_ab = np.clip(output_ab, -110.0, 110.0)

        output_lab = np.concatenate((orig_l, output_ab), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
        out = (output_bgr * 255.0).round().astype(np.uint8)
        out_path = os.path.join(args.output, os.path.basename(img_path))
        cv2.imwrite(out_path, out)

    print(f"Done. Wrote results to: {args.output}")


if __name__ == "__main__":
    main()
