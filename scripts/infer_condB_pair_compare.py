#!/usr/bin/env python
"""
专用 condB 推理脚本：灰度内容图 + 两张参考图，对比条件 tokens。

设计目标（对齐 Agent.md）：
- 使用训练时同款 condB dense/grid conditioner（MultiScaleDenseTokenConditioner）。
- 对每张原始灰度图，建立独立文件夹，内含：
  - content_original.png
    - dif.png（两张参考输出结果的像素差异可视化）
  - outputs/ref_<key1>.png、outputs/ref_<key2>.png（两张参考图对应的上色结果）
- 在 output 根目录统一维护 ref_lowres/ 目录，保存参考图的低分辨率版本，便于快速浏览。
- 对每张图生成 condition_compare.txt，比较两次 cond tokens 的差异：
  - 分尺度统计 L2 差异与各自范数
  - 给出“差异过小 / 正常 / 过大”的文字提示

当前版本不启用人脸分割/五官语义掩码，仅使用 dense/grid 无掩码 conditioner；
若后续启用掩码模型，再补充保存 mask 可视化到子目录。

新增（推理侧增强参考区分）:
- 可选对 cond tokens 去均值（突出结构差异）
- 可选将两路 cond tokens RMS 归一到同一目标值
- 可选对两路 cond 做“对比拉开”增强（contrast boost）
- 可选在解码后的 Lab-ab 空间继续做激进增强（相对基线增益 / 双分支对比拉开 / 色度放大）
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Optional

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


def _resolve_cond_ckpt_path(net_g_ckpt: str, cond_ckpt: str = None):
    """Infer conditioner checkpoint path from a given net_g checkpoint, unless overridden."""
    if cond_ckpt:
        return cond_ckpt if os.path.isfile(cond_ckpt) else None
    if not net_g_ckpt:
        return None
    ckpt_path = Path(net_g_ckpt)
    name = ckpt_path.name
    if not name.startswith("net_g_"):
        return None
    cand = ckpt_path.with_name(name.replace("net_g_", "net_c_", 1))
    return str(cand) if cand.exists() else None


def _default_output_name_from_ckpt(net_g_ckpt: str):
    """Auto-generate output root as results_<experiment_name> from ckpt path."""
    try:
        ckpt_path = Path(net_g_ckpt).resolve()
        # Typical path: experiments/<exp_name>/models/net_g_xxx.pth
        if ckpt_path.parent.name == "models" and ckpt_path.parent.parent.name:
            exp_name = ckpt_path.parent.parent.name
        else:
            exp_name = ckpt_path.stem
        safe_exp = _safe_name(exp_name)
        return f"results_{safe_exp}"
    except Exception:
        return "results_condB_pair_compare"


def _load_conditioner_weights(conditioner: torch.nn.Module, ckpt_path: str, device: torch.device):
    """Load a BasicSR-style conditioner checkpoint (net_c_*.pth) into `conditioner`."""
    loaded = torch.load(ckpt_path, map_location=device)
    if isinstance(loaded, dict) and "params" in loaded:
        state_dict = loaded["params"]
    else:
        state_dict = loaded

    # Some modules (e.g., grid_conditioner.input_proj) are lazily initialized
    # on first forward; their keys therefore appear in the checkpoint but not
    # yet in the freshly constructed module. To avoid an initial
    # 'unexpected keys' warning, pre-create input_proj convs from the
    # checkpoint shapes when possible before loading weights.
    try:
        grid_cond = getattr(conditioner, "grid_conditioner", None)
        has_proj_keys = any(k.startswith("grid_conditioner.input_proj.") for k in state_dict.keys())
        if has_proj_keys and grid_cond is not None and getattr(grid_cond, "input_proj", None) is None:
            idx_shapes = {}
            for k in state_dict.keys():
                if k.startswith("grid_conditioner.input_proj.") and k.endswith(".weight"):
                    parts = k.split(".")
                    try:
                        idx = int(parts[2])
                    except Exception:
                        continue
                    wt = state_dict[k]
                    idx_shapes[idx] = (int(wt.shape[1]), int(wt.shape[0]))

            if idx_shapes:
                max_idx = max(idx_shapes.keys())
                projs = []
                for i in range(max_idx + 1):
                    if i in idx_shapes:
                        in_ch, out_ch = idx_shapes[i]
                    else:
                        out_ch = getattr(grid_cond, "hidden_dim", 256)
                        in_ch = out_ch
                    conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size=1)
                    conv = conv.to(device)
                    projs.append(conv)
                grid_cond.input_proj = torch.nn.ModuleList(projs)
                print(f"[INFO] Pre-created grid_conditioner.input_proj with {len(projs)} convs from checkpoint shapes.")
    except Exception as e:
        print(f"[WARN] Failed to precreate input_proj from checkpoint: {e}")

    # Now load weights; any remaining unexpected/missing keys will be reported.
    missing, unexpected = conditioner.load_state_dict(state_dict, strict=False)
    if unexpected:
        unexpected_list = list(unexpected)
        print(
            f"[WARN] Ignoring unexpected conditioner keys: {unexpected_list[:8]}... "
            f"(total {len(unexpected_list)})"
        )

    if missing:
        print(
            f"[WARN] Conditioner missing {len(missing)} keys from checkpoint; "
            f"model may not exactly match training config."
        )
    return missing, unexpected


def _maybe_override_cond_gate(model: torch.nn.Module, override_raw: Optional[float]):
    """Optionally override cond gate logits for debugging reference influence strength."""
    if override_raw is None:
        return
    dec = getattr(model, "decoder", None)
    color_dec = getattr(dec, "color_decoder", None) if dec is not None else None
    gate_param = getattr(color_dec, "cond_gate_logit", None) if color_dec is not None else None
    if gate_param is None:
        print("[WARN] override_cond_gate_raw requested but cond gates were not found on model.")
        return
    with torch.no_grad():
        gate_param.fill_(float(override_raw))
        gate_sigmoid = torch.sigmoid(gate_param.detach().float().cpu()).numpy().tolist()
    print(f"[INFO] override cond gates raw={float(override_raw):.4f}, sigmoid={gate_sigmoid}")


def _print_cond_gate(model: torch.nn.Module):
    """Print current cond gate logits/sigmoid (if present)."""
    dec = getattr(model, "decoder", None)
    color_dec = getattr(dec, "color_decoder", None) if dec is not None else None
    gate_param = getattr(color_dec, "cond_gate_logit", None) if color_dec is not None else None
    if gate_param is None:
        print("[INFO] cond gates: not found on model.")
        return
    with torch.no_grad():
        raw = gate_param.detach().float().cpu().numpy().tolist()
        sig = torch.sigmoid(gate_param.detach().float().cpu()).numpy().tolist()
    print(f"[INFO] cond gates raw={raw}, sigmoid={sig}")


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
def _predict_ab_with_cond(
    model: torch.nn.Module,
    content_bgr: np.ndarray,
    content_gray_tensor: torch.Tensor,
    cond_tokens_per_scale,
    cond_pos_per_scale,
):
    """Predict Lab ab channels for one content image with provided cond tokens."""
    h, w = content_bgr.shape[:2]
    out_ab = model(
        content_gray_tensor,
        cond_tokens=cond_tokens_per_scale,
        cond_pos=cond_pos_per_scale,
    )
    out_ab = F.interpolate(out_ab, size=(h, w))[0].float().cpu().numpy().transpose(1, 2, 0)
    return out_ab


def _lab_ab_to_bgr(content_bgr: np.ndarray, out_ab: np.ndarray):
    """Compose original L + predicted ab into final uint8 BGR image."""
    content_f = (content_bgr / 255.0).astype(np.float32)
    orig_l = cv2.cvtColor(content_f, cv2.COLOR_BGR2Lab)[:, :, :1]
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


def _compute_pixel_diff_stats(img1_bgr: np.ndarray, img2_bgr: np.ndarray):
    """Compute pixel-level difference statistics between two uint8 BGR outputs."""
    d = cv2.absdiff(img1_bgr, img2_bgr)  # uint8, HWC
    d_f = d.astype(np.float32)
    per_pixel_mean = d_f.mean(axis=2)

    stats = {
        "mean_abs_diff_b": float(d_f[:, :, 0].mean()),
        "mean_abs_diff_g": float(d_f[:, :, 1].mean()),
        "mean_abs_diff_r": float(d_f[:, :, 2].mean()),
        "mean_abs_diff_all": float(per_pixel_mean.mean()),
        "max_abs_diff_all": float(per_pixel_mean.max()),
        "p95_abs_diff_all": float(np.percentile(per_pixel_mean, 95)),
        "ratio_abs_diff_gt_5": float((per_pixel_mean > 5.0).mean()),
        "ratio_abs_diff_gt_10": float((per_pixel_mean > 10.0).mean()),
        "ratio_abs_diff_gt_20": float((per_pixel_mean > 20.0).mean()),
    }
    return stats, per_pixel_mean


def _make_diff_vis(per_pixel_mean: np.ndarray):
    """Create a visible heatmap for per-pixel output difference."""
    if per_pixel_mean.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    d = per_pixel_mean.astype(np.float32)
    vmax = float(d.max())
    if vmax <= 1e-8:
        norm = np.zeros_like(d, dtype=np.uint8)
    else:
        norm = np.clip(d / vmax * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    return heat


def _compute_ab_diff_stats(ab1: np.ndarray, ab2: np.ndarray):
    """Compute difference stats in Lab ab space (float32)."""
    delta = np.abs(ab1 - ab2).astype(np.float32)
    per_pixel = delta.mean(axis=2)
    stats = {
        "mean_abs_diff_ab": float(per_pixel.mean()),
        "p95_abs_diff_ab": float(np.percentile(per_pixel, 95)),
        "max_abs_diff_ab": float(per_pixel.max()),
        "ratio_abs_diff_ab_gt_1": float((per_pixel > 1.0).mean()),
        "ratio_abs_diff_ab_gt_2": float((per_pixel > 2.0).mean()),
        "ratio_abs_diff_ab_gt_5": float((per_pixel > 5.0).mean()),
    }
    return stats


def _aggressive_postprocess_ab_pair(
    ab1: np.ndarray,
    ab2: np.ndarray,
    ab_base: Optional[np.ndarray],
    ab_ref_delta_gain: float,
    ab_pair_contrast_boost: float,
    ab_chroma_gain: float,
    ab_clip: float,
):
    """Aggressive inference-only AB postprocess to convert cond diff into stronger pixel diff."""
    out1 = ab1.astype(np.float32, copy=True)
    out2 = ab2.astype(np.float32, copy=True)

    if ab_base is not None and ab_ref_delta_gain != 1.0:
        g = float(ab_ref_delta_gain)
        out1 = ab_base + g * (out1 - ab_base)
        out2 = ab_base + g * (out2 - ab_base)

    b = float(ab_pair_contrast_boost)
    if b != 0.0:
        m = 0.5 * (out1 + out2)
        out1 = m + (1.0 + b) * (out1 - m)
        out2 = m + (1.0 + b) * (out2 - m)

    c = float(ab_chroma_gain)
    if c != 1.0:
        out1 = out1 * c
        out2 = out2 * c

    clipv = float(max(1.0, ab_clip))
    out1 = np.clip(out1, -clipv, clipv)
    out2 = np.clip(out2, -clipv, clipv)
    return out1, out2


def _cond_tokens_rms(tokens_per_scale):
    vals = []
    for t in tokens_per_scale:
        vals.append(float(t.pow(2).mean().sqrt().item()))
    return vals


def _normalize_cond_tokens_rms(tokens_per_scale, target_rms: float):
    out = []
    eps = 1e-8
    tgt = float(target_rms)
    for t in tokens_per_scale:
        cur = t.pow(2).mean().sqrt()
        scale = tgt / (cur + eps)
        out.append(t * scale)
    return out


def _center_cond_tokens(tokens_per_scale):
    # token shape: (S_k, B, C), center over token dimension S_k
    return [t - t.mean(dim=0, keepdim=True) for t in tokens_per_scale]


def _contrast_boost_pair(cond1_tokens, cond2_tokens, boost: float):
    b = float(boost)
    if b <= 0:
        return cond1_tokens, cond2_tokens

    out1 = []
    out2 = []
    for t1, t2 in zip(cond1_tokens, cond2_tokens):
        delta = t1 - t2
        out1.append(t1 + b * delta)
        out2.append(t2 - b * delta)
    return out1, out2


def _postprocess_cond_pair(
    cond1_tokens,
    cond2_tokens,
    center_tokens: bool,
    target_rms: Optional[float],
    contrast_boost: float,
):
    out1 = cond1_tokens
    out2 = cond2_tokens

    if center_tokens:
        out1 = _center_cond_tokens(out1)
        out2 = _center_cond_tokens(out2)

    if target_rms is not None and target_rms > 0:
        out1 = _normalize_cond_tokens_rms(out1, target_rms)
        out2 = _normalize_cond_tokens_rms(out2, target_rms)

    out1, out2 = _contrast_boost_pair(out1, out2, contrast_boost)
    return out1, out2


def main():
    parser = argparse.ArgumentParser(
        description="DDColor cond-B inference with two references and conditioner comparison",
    )
    parser.add_argument("--ckpt", required=True, type=str, help="net_g checkpoint (net_g_*.pth)")
    parser.add_argument("--content", required=True, type=str, help="Content image file/folder (grayscale photos)")
    parser.add_argument("--ref1", required=True, type=str, help="First reference image file")
    parser.add_argument("--ref2", required=True, type=str, help="Second reference image file")
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help=(
            "Output root directory. If omitted, auto-uses results_<experiment_name> "
            "parsed from --ckpt path."
        ),
    )
    parser.add_argument("--input_size", default=512, type=int)
    parser.add_argument("--model_size", default="large", choices=["tiny", "large"])
    parser.add_argument("--num_queries", default=256, type=int)
    parser.add_argument("--num_scales", default=3, type=int)
    parser.add_argument("--dec_layers", default=9, type=int)
    parser.add_argument("--cond_gain1", default=1.0, type=float, help="Gain for first reference conditioner")
    parser.add_argument("--cond_gain2", default=1.0, type=float, help="Gain for second reference conditioner")
    parser.add_argument(
        "--contrast_boost",
        default=0.0,
        type=float,
        help=(
            "Inference-only cond separation boost for two refs. "
            "Applied as cond1 += b*(cond1-cond2), cond2 += b*(cond2-cond1)."
        ),
    )
    parser.add_argument(
        "--center_cond_tokens",
        action="store_true",
        help="Inference-only: center cond tokens over token dimension at each scale before decoding.",
    )
    parser.add_argument(
        "--target_cond_rms",
        default=None,
        type=float,
        help="Inference-only: normalize each cond-token scale RMS to this value before decode.",
    )
    parser.add_argument(
        "--ab_ref_delta_gain",
        default=1.0,
        type=float,
        help=(
            "Inference-only aggressive AB amplification relative to no-cond baseline. "
            "1.0 means disabled; >1.0 strengthens reference effect."
        ),
    )
    parser.add_argument(
        "--ab_pair_contrast_boost",
        default=0.0,
        type=float,
        help=(
            "Inference-only AB pair separation after decode. "
            "Applied around pair mean with factor (1 + boost)."
        ),
    )
    parser.add_argument(
        "--ab_chroma_gain",
        default=1.0,
        type=float,
        help="Inference-only global chroma gain on AB channels after postprocess.",
    )
    parser.add_argument(
        "--ab_clip",
        default=110.0,
        type=float,
        help="Inference-only AB value clipping range [-ab_clip, ab_clip] after postprocess.",
    )
    parser.add_argument(
        "--override_cond_gate_raw",
        default=None,
        type=float,
        help="Debug: override model decoder cond gate logits (raw). Larger => stronger ref influence. Example: 1.5",
    )
    parser.add_argument(
        "--cond_ckpt",
        default=None,
        type=str,
        help="Optional conditioner checkpoint path (net_c_*.pth). If omitted, auto-resolve from --ckpt",
    )
    parser.add_argument(
        "--allow_random_conditioner",
        action="store_true",
        help=(
            "Allow running without net_c checkpoint (will use randomly initialized conditioner; "
            "usually reference will have little/no effect)."
        ),
    )

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
    _maybe_override_cond_gate(model, args.override_cond_gate_raw)
    _print_cond_gate(model)

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

    # 加载训练好的 net_c（若存在）；否则默认不允许随机 conditioner，避免“参考图不起作用”的误用
    cond_ckpt_path = _resolve_cond_ckpt_path(args.ckpt, args.cond_ckpt)
    if cond_ckpt_path is not None:
        _load_conditioner_weights(conditioner, cond_ckpt_path, device)
        print(f"[INFO] Loaded conditioner weights: {cond_ckpt_path}")
        # 关键：grid_conditioner.input_proj 是“懒初始化”的 ModuleList。
        # 若直接 load_state_dict，会把 input_proj.* 当成 unexpected keys 忽略，导致投影层权重没加载。
        # 这里用一次真实 ref feats 走通 conditioner，构建 input_proj，然后再 load 一次，把权重补齐。
        try:
            _ = _build_cond_from_reference(
                model,
                conditioner,
                ref1_img,
                args.input_size,
                device,
                cond_gain=1.0,
            )
        except Exception as e:
            raise SystemExit(f"Failed to warmup conditioner for lazy modules: {e}")
        missing2, unexpected2 = _load_conditioner_weights(conditioner, cond_ckpt_path, device)
        if unexpected2 or missing2:
            print(
                f"[WARN] After warmup+reload, missing={len(missing2)} unexpected={len(unexpected2)} "
                f"(this may indicate a real arch/config mismatch)."
            )
        else:
            print("[INFO] Conditioner weights fully loaded after warmup+reload.")
    else:
        msg = (
            "[ERROR] Conditioner checkpoint (net_c_*.pth) not found.\n"
            "Ref-guided condB inference requires trained net_c weights; "
            "otherwise the conditioner is random and reference images have almost no effect.\n"
            f"  - net_g ckpt: {args.ckpt}\n"
            f"  - cond_ckpt (override): {args.cond_ckpt}\n"
            "Expected auto-resolve: same folder, net_g_* -> net_c_*\n"
            "If you *really* want to run with a random conditioner (NOT recommended), "
            "pass --allow_random_conditioner."
        )
        if args.allow_random_conditioner:
            print(msg.replace("[ERROR]", "[WARN]"))
        else:
            raise SystemExit(msg)

    output_name = args.output if args.output else _default_output_name_from_ckpt(args.ckpt)
    out_root = Path(output_name)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] output root: {out_root}")

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

        # 保存原图
        _safe_write(c_dir / "content_original.png", c_img)
        content_gray_tensor, _ = _to_gray_rgb_tensor(c_img, args.input_size, device)

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

        raw_stats, raw_summary = _compare_condition_tokens(cond1_tokens, cond2_tokens)
        raw_rms1 = _cond_tokens_rms(cond1_tokens)
        raw_rms2 = _cond_tokens_rms(cond2_tokens)

        cond1_tokens, cond2_tokens = _postprocess_cond_pair(
            cond1_tokens,
            cond2_tokens,
            center_tokens=args.center_cond_tokens,
            target_rms=args.target_cond_rms,
            contrast_boost=args.contrast_boost,
        )

        post_stats, post_summary = _compare_condition_tokens(cond1_tokens, cond2_tokens)
        post_rms1 = _cond_tokens_rms(cond1_tokens)
        post_rms2 = _cond_tokens_rms(cond2_tokens)

        # 先解码 AB（两路参考）
        ab1 = _predict_ab_with_cond(
            model,
            c_img,
            content_gray_tensor,
            cond1_tokens,
            cond1_pos,
        )
        ab2 = _predict_ab_with_cond(
            model,
            c_img,
            content_gray_tensor,
            cond2_tokens,
            cond2_pos,
        )

        # 可选：无条件基线 AB（用于“相对基线”的参考增益）
        ab_base = None
        if args.ab_ref_delta_gain != 1.0:
            zero_cond = [torch.zeros_like(t) for t in cond1_tokens]
            ab_base = _predict_ab_with_cond(
                model,
                c_img,
                content_gray_tensor,
                zero_cond,
                cond1_pos,
            )

        raw_ab_stats = _compute_ab_diff_stats(ab1, ab2)
        ab1, ab2 = _aggressive_postprocess_ab_pair(
            ab1,
            ab2,
            ab_base=ab_base,
            ab_ref_delta_gain=args.ab_ref_delta_gain,
            ab_pair_contrast_boost=args.ab_pair_contrast_boost,
            ab_chroma_gain=args.ab_chroma_gain,
            ab_clip=args.ab_clip,
        )
        post_ab_stats = _compute_ab_diff_stats(ab1, ab2)

        # 再转成最终 BGR
        out1 = _lab_ab_to_bgr(c_img, ab1)
        out2 = _lab_ab_to_bgr(c_img, ab2)

        _safe_write(outputs_dir / f"{ref1_key}.png", out1)
        _safe_write(outputs_dir / f"{ref2_key}.png", out2)

        pixel_stats, per_pixel_mean = _compute_pixel_diff_stats(out1, out2)
        diff_vis = _make_diff_vis(per_pixel_mean)
        _safe_write(c_dir / "dif.png", diff_vis)

        # 对比 cond tokens 差异
        report_lines = []
        report_lines.append(f"content: {c_path}")
        report_lines.append(f"ref1: {ref1_path} (key={ref1_key}, gain={args.cond_gain1})")
        report_lines.append(f"ref2: {ref2_path} (key={ref2_key}, gain={args.cond_gain2})")
        report_lines.append(
            "postprocess: "
            f"center_cond_tokens={args.center_cond_tokens}, "
            f"target_cond_rms={args.target_cond_rms}, "
            f"contrast_boost={args.contrast_boost}, "
            f"ab_ref_delta_gain={args.ab_ref_delta_gain}, "
            f"ab_pair_contrast_boost={args.ab_pair_contrast_boost}, "
            f"ab_chroma_gain={args.ab_chroma_gain}, "
            f"ab_clip={args.ab_clip}"
        )
        report_lines.append("")
        report_lines.append("=== raw cond stats (before postprocess) ===")
        report_lines.append(f"ref1_rms_per_scale={raw_rms1}")
        report_lines.append(f"ref2_rms_per_scale={raw_rms2}")
        for k in sorted(raw_stats.keys()):
            v = raw_stats[k]
            if "error" in v:
                report_lines.append(f"{k}: {v['error']}")
            else:
                report_lines.append(
                    f"{k}: rms_diff={v['rms_diff']:.6f}, "
                    f"norm_a={v['norm_a']:.6f}, norm_b={v['norm_b']:.6f}, avg_norm={v['avg_norm']:.6f}"
                )
        report_lines.append(f"raw_summary: {raw_summary}")
        report_lines.append("")
        report_lines.append("=== post cond stats (after postprocess) ===")
        report_lines.append(f"ref1_rms_per_scale={post_rms1}")
        report_lines.append(f"ref2_rms_per_scale={post_rms2}")
        for k in sorted(post_stats.keys()):
            v = post_stats[k]
            if "error" in v:
                report_lines.append(f"{k}: {v['error']}")
            else:
                report_lines.append(
                    f"{k}: rms_diff={v['rms_diff']:.6f}, "
                    f"norm_a={v['norm_a']:.6f}, norm_b={v['norm_b']:.6f}, avg_norm={v['avg_norm']:.6f}"
                )
        report_lines.append("")
        report_lines.append("=== summary ===")
        report_lines.append(post_summary)
        report_lines.append("")
        report_lines.append("=== ab diff stats (decoder output) ===")
        report_lines.append(
            f"raw_mean_abs_diff_ab={raw_ab_stats['mean_abs_diff_ab']:.6f}, "
            f"raw_p95_abs_diff_ab={raw_ab_stats['p95_abs_diff_ab']:.6f}, "
            f"raw_max_abs_diff_ab={raw_ab_stats['max_abs_diff_ab']:.6f}"
        )
        report_lines.append(
            f"post_mean_abs_diff_ab={post_ab_stats['mean_abs_diff_ab']:.6f}, "
            f"post_p95_abs_diff_ab={post_ab_stats['p95_abs_diff_ab']:.6f}, "
            f"post_max_abs_diff_ab={post_ab_stats['max_abs_diff_ab']:.6f}"
        )
        report_lines.append(
            f"post_ratio_abs_diff_ab_gt_1={post_ab_stats['ratio_abs_diff_ab_gt_1']:.6f}, "
            f"post_ratio_abs_diff_ab_gt_2={post_ab_stats['ratio_abs_diff_ab_gt_2']:.6f}, "
            f"post_ratio_abs_diff_ab_gt_5={post_ab_stats['ratio_abs_diff_ab_gt_5']:.6f}"
        )
        report_lines.append("")
        report_lines.append("=== pixel diff stats (between output images) ===")
        report_lines.append(
            f"mean_abs_diff_b={pixel_stats['mean_abs_diff_b']:.6f}, "
            f"mean_abs_diff_g={pixel_stats['mean_abs_diff_g']:.6f}, "
            f"mean_abs_diff_r={pixel_stats['mean_abs_diff_r']:.6f}"
        )
        report_lines.append(
            f"mean_abs_diff_all={pixel_stats['mean_abs_diff_all']:.6f}, "
            f"p95_abs_diff_all={pixel_stats['p95_abs_diff_all']:.6f}, "
            f"max_abs_diff_all={pixel_stats['max_abs_diff_all']:.6f}"
        )
        report_lines.append(
            f"ratio_abs_diff_gt_5={pixel_stats['ratio_abs_diff_gt_5']:.6f}, "
            f"ratio_abs_diff_gt_10={pixel_stats['ratio_abs_diff_gt_10']:.6f}, "
            f"ratio_abs_diff_gt_20={pixel_stats['ratio_abs_diff_gt_20']:.6f}"
        )
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
        "若重复则追加短 hash 保证唯一；目录内包含原图、dif.png（两张输出像素差热力图）、"
        "outputs/ 下两张参考的上色结果和 condition_compare.txt 对比报告（含 cond 差异与像素差异统计）。"
    )


if __name__ == "__main__":
    main()

