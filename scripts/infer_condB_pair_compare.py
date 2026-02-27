#!/usr/bin/env python
"""
专用 condB 推理脚本：灰度内容图 + 两张参考图，对比条件 tokens。

设计目标（对齐 Agent.md）：
- 使用训练时同款 condB dense/grid conditioner（MultiScaleDenseTokenConditioner）。
- 对每张原始灰度图，建立独立文件夹，内含：
  - content_original.png
  - content_gray.png（模型真实看到的输入灰度）
  - outputs/ref_<key1>.png、outputs/ref_<key2>.png（两张参考图对应的上色结果）
- 在 output 根目录统一维护 ref_lowres/ 目录，保存参考图的低分辨率版本，便于快速浏览。
- 对每张图生成 condition_compare.txt，比较两次 cond tokens 的差异：
  - 分尺度统计 L2 差异与各自范数
  - 给出“差异过小 / 正常 / 过大”的文字提示

当前版本不启用人脸分割/五官语义掩码，仅使用 dense/grid 无掩码 conditioner；
若后续启用掩码模型，再补充保存 mask 可视化到子目录。
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ddcolor import DDColor, build_ddcolor_model  # noqa: E402
from basicsr.archs.ddcolor_arch_utils.region_tokens import (  # noqa: E402
    MultiScaleDenseTokenConditioner,
)


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _safe_name(name: str) -> str:
    """Make a filename-safe stem (Windows-friendly)."""
    bad = '<>:"/\\|?*'
    out = "".join("_" if c in bad else c for c in name)
    out = out.strip().rstrip(".")
    return out if out else "unnamed"


def _hash8(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def _unique_key_for_path(p: Path, used: set) -> str:
    """Create a stable, collision-free key from a file path."""
    base = _safe_name(p.stem)
    key = base
    if key in used:
        key = f"{base}__{_hash8(str(p.resolve()))}"
    used.add(key)
    return key


def _list_images(path: str):
    p = Path(path)
    if p.is_file():
        return [p]
    files = [x for x in sorted(p.iterdir()) if x.is_file() and x.suffix.lower() in _IMG_EXTS]
    return files


def _safe_write(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"Failed to encode image: {path}")
    try:
        encoded.tofile(str(path))
    except Exception:
        ok2 = cv2.imwrite(str(path), img)
        if not ok2:
            raise IOError(f"Failed to write image: {path}")


def _to_gray_rgb_tensor(img_bgr: np.ndarray, input_size: int, device: torch.device):
    """和现有 infer_style_transfer 保持一致的灰度输入构造逻辑。"""
    img = (img_bgr / 255.0).astype(np.float32)
    img_resized = cv2.resize(img, (input_size, input_size))
    img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
    img_gray_lab = np.concatenate([img_l, np.zeros_like(img_l), np.zeros_like(img_l)], axis=-1)
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
    tensor = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
    return tensor, img_gray_rgb


@torch.no_grad()
def _build_cond_from_reference(
    model: torch.nn.Module,
    conditioner: torch.nn.Module,
    ref_bgr: np.ndarray,
    input_size: int,
    device: torch.device,
    cond_gain: float = 1.0,
):
    """从参考图构造 (cond_tokens_per_scale, cond_pos_per_scale)。"""
    ref = (ref_bgr / 255.0).astype(np.float32)
    ref_resized = cv2.resize(ref, (input_size, input_size))

    ref_rgb = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2RGB)
    ref_tensor = torch.from_numpy(ref_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(device)

    ref_in = model.normalize(ref_tensor) if hasattr(model, "normalize") else ref_tensor
    _ = model.encoder(ref_in)
    hooks = model.encoder.hooks
    ref_feats = [hooks[1].feature, hooks[2].feature, hooks[3].feature]

    cond_tokens_per_scale, cond_pos_per_scale = conditioner(ref_feats)
    if cond_gain != 1.0 and cond_tokens_per_scale is not None:
        g = float(cond_gain)
        cond_tokens_per_scale = [t * g for t in cond_tokens_per_scale]
    return cond_tokens_per_scale, cond_pos_per_scale


@torch.no_grad()
def _infer_one_with_cond(
    model: torch.nn.Module,
    content_bgr: np.ndarray,
    content_gray_tensor: torch.Tensor,
    cond_tokens_per_scale,
    cond_pos_per_scale,
    input_size: int,
    device: torch.device,
):
    """用给定的 cond tokens 对单张内容图推理一次。"""
    h, w = content_bgr.shape[:2]
    content_f = (content_bgr / 255.0).astype(np.float32)
    orig_l = cv2.cvtColor(content_f, cv2.COLOR_BGR2Lab)[:, :, :1]

    out_ab = model(
        content_gray_tensor,
        cond_tokens=cond_tokens_per_scale,
        cond_pos=cond_pos_per_scale,
    )
    out_ab = F.interpolate(out_ab, size=(h, w))[0].float().cpu().numpy().transpose(1, 2, 0)

    out_lab = np.concatenate((orig_l, out_ab), axis=-1)
    out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
    out_img = (out_bgr * 255.0).round().astype(np.uint8)
    return out_img


def _compare_condition_tokens(cond_a, cond_b):
    """比较两组 cond_tokens_per_scale 的差异，返回 per-scale 统计信息和文本结论。"""
    if cond_a is None or cond_b is None:
        return {}, "cond tokens 缺失，无法比较。"

    if not isinstance(cond_a, (list, tuple)) or not isinstance(cond_b, (list, tuple)):
        return {}, "内部格式异常：cond tokens 不是 list/tuple。"

    num_scales = min(len(cond_a), len(cond_b))
    stats = {}
    diffs = []
    norms = []

    for i in range(num_scales):
        ta = cond_a[i]
        tb = cond_b[i]
        if ta.shape != tb.shape:
            stats[f"scale_{i}"] = {
                "error": f"shape mismatch: {tuple(ta.shape)} vs {tuple(tb.shape)}",
            }
            continue
        # 形状: (S_k, B, C)，通常 B=1；我们对全部元素做 RMS
        diff = (ta - tb).pow(2).mean().sqrt().item()
        na = ta.pow(2).mean().sqrt().item()
        nb = tb.pow(2).mean().sqrt().item()
        avg_norm = 0.5 * (na + nb)

        stats[f"scale_{i}"] = {
            "rms_diff": float(diff),
            "norm_a": float(na),
            "norm_b": float(nb),
            "avg_norm": float(avg_norm),
        }
        diffs.append(diff)
        norms.append(avg_norm if avg_norm > 0 else 1.0)

    if not diffs:
        return stats, "没有有效尺度的 cond tokens 统计。"

    mean_rel = float(np.mean([d / n for d, n in zip(diffs, norms)]))

    if mean_rel < 0.05:
        summary = f"两次条件 tokens 差异相对范数均值约 {mean_rel:.4f}，整体偏小，可能参考语义几乎未区分。"
    elif mean_rel > 0.8:
        summary = f"两次条件 tokens 差异相对范数均值约 {mean_rel:.4f}，非常大，可能导致风格/色彩分布强烈偏移。"
    else:
        summary = f"两次条件 tokens 差异相对范数均值约 {mean_rel:.4f}，处于中等区间。"

    return stats, summary


def main():
    parser = argparse.ArgumentParser(
        description="DDColor cond-B inference with two references and conditioner comparison",
    )
    parser.add_argument("--ckpt", required=True, type=str, help="net_g checkpoint (net_g_*.pth)")
    parser.add_argument("--content", required=True, type=str, help="Content image file/folder (grayscale photos)")
    parser.add_argument("--ref1", required=True, type=str, help="First reference image file")
    parser.add_argument("--ref2", required=True, type=str, help="Second reference image file")
    parser.add_argument("--output", default="results_condB_pair_compare", type=str)
    parser.add_argument("--input_size", default=512, type=int)
    parser.add_argument("--model_size", default="large", choices=["tiny", "large"])
    parser.add_argument("--num_queries", default=256, type=int)
    parser.add_argument("--num_scales", default=3, type=int)
    parser.add_argument("--dec_layers", default=9, type=int)
    parser.add_argument("--cond_gain1", default=1.0, type=float, help="Gain for first reference conditioner")
    parser.add_argument("--cond_gain2", default=1.0, type=float, help="Gain for second reference conditioner")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ddcolor_model(
        DDColor,
        model_path=args.ckpt,
        input_size=args.input_size,
        model_size=args.model_size,
        decoder_type="MultiScaleColorDecoder",
        device=device,
        num_queries=args.num_queries,
        num_scales=args.num_scales,
        dec_layers=args.dec_layers,
    )
    model.eval()

    # 与训练配置一致的 dense/grid conditioner
    conditioner = MultiScaleDenseTokenConditioner(
        num_scales=3,
        hidden_dim=256,
        grid_size=16,
    ).to(device)
    conditioner.eval()

    content_list = _list_images(args.content)
    if not content_list:
        raise SystemExit(f"No content images found: {args.content}")

    ref1_path = Path(args.ref1)
    ref2_path = Path(args.ref2)
    if not ref1_path.is_file():
        raise SystemExit(f"ref1 not found or not a file: {ref1_path}")
    if not ref2_path.is_file():
        raise SystemExit(f"ref2 not found or not a file: {ref2_path}")

    ref1_img = cv2.imread(str(ref1_path))
    ref2_img = cv2.imread(str(ref2_path))
    if ref1_img is None:
        raise SystemExit(f"Failed to read ref1 image: {ref1_path}")
    if ref2_img is None:
        raise SystemExit(f"Failed to read ref2 image: {ref2_path}")

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    # 统一 low-res 参考图目录，方便浏览
    ref_lowres_dir = out_root / "ref_lowres"
    ref_lowres_dir.mkdir(parents=True, exist_ok=True)
    used_ref_keys = set()

    ref1_key = _unique_key_for_path(ref1_path, used_ref_keys)
    ref2_key = _unique_key_for_path(ref2_path, used_ref_keys)

    for key, img in [(ref1_key, ref1_img), (ref2_key, ref2_img)]:
        lr = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        _safe_write(ref_lowres_dir / f"{key}.png", lr)

    used_content_keys = set()

    for c_path in content_list:
        c_img = cv2.imread(str(c_path))
        if c_img is None:
            print(f"[WARN] skip unreadable content: {c_path}")
            continue

        content_key = _unique_key_for_path(Path(c_path), used_content_keys)
        c_dir = out_root / content_key
        outputs_dir = c_dir / "outputs"
        c_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # 保存原图与灰度输入可视化
        _safe_write(c_dir / "content_original.png", c_img)
        content_gray_tensor, content_gray_rgb_dbg = _to_gray_rgb_tensor(c_img, args.input_size, device)
        content_gray_bgr_dbg = cv2.cvtColor(
            (content_gray_rgb_dbg * 255.0).round().astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        _safe_write(c_dir / "content_gray.png", content_gray_bgr_dbg)

        # 构造两次 conditioner
        cond1_tokens, cond1_pos = _build_cond_from_reference(
            model,
            conditioner,
            ref1_img,
            args.input_size,
            device,
            cond_gain=args.cond_gain1,
        )
        cond2_tokens, cond2_pos = _build_cond_from_reference(
            model,
            conditioner,
            ref2_img,
            args.input_size,
            device,
            cond_gain=args.cond_gain2,
        )

        # 推理两次
        out1 = _infer_one_with_cond(
            model,
            c_img,
            content_gray_tensor,
            cond1_tokens,
            cond1_pos,
            args.input_size,
            device,
        )
        out2 = _infer_one_with_cond(
            model,
            c_img,
            content_gray_tensor,
            cond2_tokens,
            cond2_pos,
            args.input_size,
            device,
        )

        _safe_write(outputs_dir / f"{ref1_key}.png", out1)
        _safe_write(outputs_dir / f"{ref2_key}.png", out2)

        # 对比 cond tokens 差异
        stats, summary = _compare_condition_tokens(cond1_tokens, cond2_tokens)
        report_lines = []
        report_lines.append(f"content: {c_path}")
        report_lines.append(f"ref1: {ref1_path} (key={ref1_key}, gain={args.cond_gain1})")
        report_lines.append(f"ref2: {ref2_path} (key={ref2_key}, gain={args.cond_gain2})")
        report_lines.append("")
        report_lines.append("=== per-scale stats ===")
        for k in sorted(stats.keys()):
            v = stats[k]
            if "error" in v:
                report_lines.append(f"{k}: {v['error']}")
            else:
                report_lines.append(
                    f"{k}: rms_diff={v['rms_diff']:.6f}, "
                    f"norm_a={v['norm_a']:.6f}, norm_b={v['norm_b']:.6f}, avg_norm={v['avg_norm']:.6f}"
                )
        report_lines.append("")
        report_lines.append("=== summary ===")
        report_lines.append(summary)
        report_lines.append("")
        report_lines.append(
            "解释：若差异远小于范数，说明两张参考在条件空间中几乎重合；"
            "若差异与范数同量级甚至更大，说明参考风格/语义在条件空间中差异很大。"
        )

        report_path = c_dir / "condition_compare.txt"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Done. Structured condB pair results saved under: {out_root}")
    print(
        "命名说明：\n"
        f"- 根目录 {out_root.name}: 表示 condB 双参考对比推理结果。\n"
        "- ref_lowres/: 保存所有参考图的 256×256 预览，用于快速浏览。\n"
        "- 每个内容图建立子目录 <content_key>/，其中 content_key 源自原文件名，"
        "若重复则追加短 hash 保证唯一；目录内包含原图、灰度输入以及 outputs/ 下两张参考的上色结果和 condition_compare.txt 对比报告。"
    )


if __name__ == "__main__":
    main()

