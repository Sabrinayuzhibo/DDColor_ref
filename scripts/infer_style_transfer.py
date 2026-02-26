#!/usr/bin/env python
"""Reference-guided style transfer inference for DDColor cond-B.

Output arrangement per sample:
  - result image
  - optional triptych: [content_gray | reference | result]

Examples:
  1) Single pair:
     python scripts/infer_style_transfer.py \
       --ckpt experiments/train_ddcolor_condB_style/models/net_g_latest.pth \
       --content assets/test_images/Audrey\ Hepburn.jpg \
       --reference assets/test_images/Ansel\ Adams\ _\ Moore\ Photography.jpeg \
       --output results_condB_pair

  2) Folder + fixed reference file:
     python scripts/infer_style_transfer.py \
       --ckpt experiments/train_ddcolor_condB_style/models/net_g_latest.pth \
       --content assets/test_images \
       --reference assets/test_images/Audrey\ Hepburn.jpg \
       --output results_condB_fixedref
"""

import argparse
import math
import os
import sys
from pathlib import Path
from PIL import Image

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ddcolor import DDColor, build_ddcolor_model
from basicsr.archs.ddcolor_arch_utils.region_tokens import MultiScaleRegionTokenConditioner, MultiScaleDenseTokenConditioner, RegionTokenSpec

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class _MODNetONNXMatte:
    def __init__(self, model_path: str, ref_size: int = 512):
        self.model_path = str(model_path)
        self.ref_size = int(ref_size)
        self.net = cv2.dnn.readNetFromONNX(self.model_path)

    @staticmethod
    def _get_scale_factor(im_h: int, im_w: int, ref_size: int):
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            else:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = max(32, im_rw - im_rw % 32)
        im_rh = max(32, im_rh - im_rh % 32)
        return (im_rw / im_w), (im_rh / im_h)

    def predict_matte(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Empty input image for MODNet matte prediction")

        im = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        im = (im.astype(np.float32) - 127.5) / 127.5
        im_h, im_w, _ = im.shape
        x_scale, y_scale = self._get_scale_factor(im_h, im_w, self.ref_size)
        im_rs = cv2.resize(im, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_AREA)

        blob = np.transpose(im_rs, (2, 0, 1))[None, ...].astype(np.float32)
        self.net.setInput(blob)
        matte = self.net.forward()
        matte = np.squeeze(matte).astype(np.float32)
        matte = cv2.resize(matte, dsize=(im_w, im_h), interpolation=cv2.INTER_AREA)
        return np.clip(matte, 0.0, 1.0)


class _MODNetTorchMatte:
    def __init__(self, ckpt_path: str, device: torch.device, ref_size: int = 512, repo_root: str = None):
        self.ckpt_path = str(ckpt_path)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.ref_size = int(ref_size)

        if repo_root is None:
            repo_root = os.path.join(str(Path.home()), ".cache", "torch", "hub", "ZHKKKe_MODNet_master")
        self.repo_root = str(repo_root)
        if not os.path.isdir(self.repo_root):
            raise FileNotFoundError(f"MODNet source repo not found: {self.repo_root}")

        if self.repo_root not in sys.path:
            sys.path.insert(0, self.repo_root)

        import torchvision.transforms as transforms
        from src.models.modnet import MODNet

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        base_model = MODNet(backbone_pretrained=False)
        if self.device.type == 'cuda':
            self.model = torch.nn.DataParallel(base_model).to(self.device)
            weights = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(weights)
        else:
            self.model = base_model.to(self.device)
            weights = torch.load(self.ckpt_path, map_location=self.device)
            if isinstance(weights, dict) and any(k.startswith('module.') for k in weights.keys()):
                weights = {k.replace('module.', '', 1): v for k, v in weights.items()}
            self.model.load_state_dict(weights)
        self.model.eval()

    @staticmethod
    def _resize_hw(im_h: int, im_w: int, ref_size: int):
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            else:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w
        im_rw = max(32, im_rw - im_rw % 32)
        im_rh = max(32, im_rh - im_rh % 32)
        return im_rh, im_rw

    def predict_matte(self, img_bgr: np.ndarray) -> np.ndarray:
        im = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        im_h, im_w, _ = im.shape
        im_rh, im_rw = self._resize_hw(im_h, im_w, self.ref_size)

        im_pil = Image.fromarray(im)
        im_tensor = self.transform(im_pil)[None, :, :, :].to(self.device)
        im_tensor = F.interpolate(im_tensor, size=(im_rh, im_rw), mode='area')

        with torch.no_grad():
            _, _, matte = self.model(im_tensor, True)
            matte = F.interpolate(matte, size=(im_h, im_w), mode='area')

        matte_np = matte[0, 0].detach().float().cpu().numpy()
        return np.clip(matte_np, 0.0, 1.0)


def _gate_with_matte(mask: np.ndarray, matte: np.ndarray, gate_strength: float = 1.0, matte_min: float = 0.02):
    if matte is None:
        return mask
    m = np.clip(matte.astype(np.float32), 0.0, 1.0)
    if matte_min > 0:
        m = np.where(m >= float(matte_min), m, 0.0)
    gs = float(np.clip(gate_strength, 0.0, 1.0))
    gate = (1.0 - gs) + gs * m
    return np.clip(mask.astype(np.float32) * gate.astype(np.float32), 0.0, 1.0)


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


def _build_masks_from_ab(img_ab: np.ndarray, num_regions: int = 6):
    a = img_ab[:, :, 0]
    b = img_ab[:, :, 1]
    chroma = np.sqrt(a * a + b * b)
    angle = np.arctan2(b, a)

    r = int(num_regions)
    bin_idx = np.floor((angle + math.pi) / (2.0 * math.pi) * r).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, r - 1)

    neutral_id = r - 1
    bin_idx = np.where(chroma < 3.0, neutral_id, bin_idx)

    masks = np.zeros((r, img_ab.shape[0], img_ab.shape[1]), dtype=np.float32)
    for rid in range(r):
        masks[rid] = (bin_idx == rid).astype(np.float32)
    return masks


def _build_masks_portrait_heuristic(img_rgb: np.ndarray, num_regions: int = 6):
    if num_regions != 6:
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        img_ab = img_lab[:, :, 1:3] - 128.0
        return _build_masks_from_ab(img_ab, num_regions=num_regions)

    h, w, _ = img_rgb.shape
    rgb_u8 = np.clip(img_rgb * 255.0, 0, 255).astype(np.uint8) if img_rgb.dtype != np.uint8 else img_rgb
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    if face_cascade.empty():
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        img_ab = img_lab[:, :, 1:3] - 128.0
        return _build_masks_from_ab(img_ab, num_regions=num_regions)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
    if len(faces) == 0:
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        img_ab = img_lab[:, :, 1:3] - 128.0
        return _build_masks_from_ab(img_ab, num_regions=num_regions)

    x, y, fw, fh = max(faces, key=lambda t: t[2] * t[3])
    masks = np.zeros((6, h, w), dtype=np.float32)
    occupied = np.zeros((h, w), dtype=np.uint8)

    def add_mask(binary, idx):
        nonlocal occupied
        b = (binary > 0).astype(np.uint8)
        b = np.where(occupied == 0, b, 0).astype(np.uint8)
        masks[idx] = b.astype(np.float32)
        occupied = np.clip(occupied + b, 0, 1)

    skin = np.zeros((h, w), dtype=np.uint8)
    center = (int(x + 0.5 * fw), int(y + 0.58 * fh))
    axes = (max(8, int(0.34 * fw)), max(8, int(0.43 * fh)))
    cv2.ellipse(skin, center, axes, 0, 0, 360, 1, -1)
    add_mask(skin, 0)

    hair = np.zeros((h, w), dtype=np.uint8)
    hx1 = max(0, int(x - 0.18 * fw))
    hx2 = min(w, int(x + 1.18 * fw))
    hy1 = max(0, int(y - 0.50 * fh))
    hy2 = min(h, int(y + 0.25 * fh))
    hair[hy1:hy2, hx1:hx2] = 1
    add_mask(hair, 1)

    eyes = np.zeros((h, w), dtype=np.uint8)
    eye_region = gray[max(0, y):min(h, y + int(0.62 * fh)), max(0, x):min(w, x + fw)]
    if not eye_cascade.empty() and eye_region.size > 0:
        det = eye_cascade.detectMultiScale(eye_region, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
        for ex, ey, ew, eh in det[:2]:
            ex1 = max(0, x + ex)
            ey1 = max(0, y + ey)
            ex2 = min(w, ex1 + ew)
            ey2 = min(h, ey1 + eh)
            eyes[ey1:ey2, ex1:ex2] = 1
    if eyes.sum() == 0:
        ey1 = max(0, int(y + 0.25 * fh))
        ey2 = min(h, int(y + 0.45 * fh))
        ex1 = max(0, int(x + 0.18 * fw))
        ex2 = min(w, int(x + 0.82 * fw))
        eyes[ey1:ey2, ex1:ex2] = 1
    add_mask(eyes, 3)

    lips = np.zeros((h, w), dtype=np.uint8)
    lcenter = (int(x + 0.50 * fw), int(y + 0.80 * fh))
    laxes = (max(6, int(0.12 * fw)), max(4, int(0.05 * fh)))
    cv2.ellipse(lips, lcenter, laxes, 0, 0, 360, 1, -1)
    add_mask(lips, 2)

    cloth = np.zeros((h, w), dtype=np.uint8)
    cx1 = max(0, int(x - 0.40 * fw))
    cx2 = min(w, int(x + 1.40 * fw))
    cy1 = min(h, int(y + 0.92 * fh))
    cloth[cy1:h, cx1:cx2] = 1
    add_mask(cloth, 4)

    masks[5] = (occupied == 0).astype(np.float32)
    return masks


def _region_stat_ab(img_bgr: np.ndarray, masks: np.ndarray):
    img = (img_bgr / 255.0).astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    ab = lab[:, :, 1:3] - 128.0

    num_regions = masks.shape[0]
    stats = np.zeros((num_regions, 2), dtype=np.float32)
    areas = np.zeros((num_regions,), dtype=np.float32)
    for rid in range(num_regions):
        m = masks[rid] > 0.5
        cnt = int(m.sum())
        if cnt > 0:
            vals = ab[m]
            stats[rid] = np.median(vals, axis=0)
            areas[rid] = float(cnt)
    return stats, areas


def _feather_mask(mask: np.ndarray, erode: int = 2, blur: int = 5):
    m = (mask > 0.5).astype(np.uint8)
    if erode > 0:
        k = erode * 2 + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        m = cv2.erode(m, kernel, iterations=1)
    m = m.astype(np.float32)
    if blur > 0:
        b = blur * 2 + 1
        m = cv2.GaussianBlur(m, (b, b), 0)
    return np.clip(m, 0.0, 1.0)


def _apply_region_color_transfer(output_bgr: np.ndarray,
                                 ref_bgr: np.ndarray,
                                 out_masks: np.ndarray,
                                 ref_masks: np.ndarray,
                                 strength: float = 0.8,
                                 min_area_frac: float = 0.005,
                                 max_shift: float = 28.0,
                                 include_regions=(0, 1, 2, 3),
                                 mask_erode: int = 2,
                                 mask_blur: int = 5,
                                 region_weights=None,
                                 region_min_area=None,
                                 region_max_shift=None,
                                 out_matte: np.ndarray = None,
                                 ref_matte: np.ndarray = None,
                                 matte_gate_strength: float = 1.0,
                                 matte_min: float = 0.02):
    h, w = output_bgr.shape[:2]
    total = float(h * w)

    out_img = (output_bgr / 255.0).astype(np.float32)
    out_lab = cv2.cvtColor(out_img, cv2.COLOR_BGR2Lab)
    out_ab = out_lab[:, :, 1:3] - 128.0

    ref_means, ref_areas = _region_stat_ab(ref_bgr, ref_masks)
    out_means, out_areas = _region_stat_ab(output_bgr, out_masks)

    if region_weights is None:
        region_weights = {0: 0.55, 1: 1.0, 2: 1.05, 3: 0.95}
    if region_min_area is None:
        region_min_area = {0: 0.010, 1: 0.006, 2: 0.001, 3: 0.001}
    if region_max_shift is None:
        region_max_shift = {0: 14.0, 1: 24.0, 2: 18.0, 3: 16.0}

    if out_matte is not None:
        for rid in range(out_masks.shape[0]):
            out_masks[rid] = _gate_with_matte(out_masks[rid], out_matte, gate_strength=matte_gate_strength, matte_min=matte_min)
    if ref_matte is not None:
        for rid in range(ref_masks.shape[0]):
            ref_masks[rid] = _gate_with_matte(ref_masks[rid], ref_matte, gate_strength=matte_gate_strength, matte_min=matte_min)

    num_regions = min(out_masks.shape[0], ref_masks.shape[0])
    for rid in range(num_regions):
        if rid not in include_regions:
            continue
        min_area_r = float(region_min_area.get(rid, min_area_frac))
        if out_areas[rid] / total < min_area_r:
            continue
        if ref_areas[rid] / total < min_area_r:
            continue

        delta = ref_means[rid] - out_means[rid]
        max_shift_r = float(region_max_shift.get(rid, max_shift))
        delta = np.clip(delta, -max_shift_r, max_shift_r)
        w_r = float(region_weights.get(rid, 1.0))

        m_soft = _feather_mask(out_masks[rid], erode=int(mask_erode), blur=int(mask_blur))
        if float(m_soft.max()) <= 0.0:
            continue
        out_ab = out_ab + (m_soft[:, :, None] * (strength * w_r * delta[None, None, :]))

    out_ab = np.clip(out_ab, -128.0, 127.0)
    out_lab[:, :, 1:3] = out_ab + 128.0
    out_bgr_new = cv2.cvtColor(out_lab, cv2.COLOR_Lab2BGR)
    out_u8 = np.clip(out_bgr_new * 255.0, 0, 255).round().astype(np.uint8)
    return out_u8


def _get_region_profile(profile: str):
    profile = (profile or 'balanced').lower()
    if profile == 'hair_priority':
        return {
            'region_weights': {0: 0.30, 1: 2.40, 2: 1.05, 3: 0.90},
            'region_min_area': {0: 0.010, 1: 0.003, 2: 0.001, 3: 0.001},
            'region_max_shift': {0: 12.0, 1: 30.0, 2: 16.0, 3: 14.0},
        }

    # balanced (default)
    return {
        'region_weights': {0: 0.55, 1: 1.0, 2: 1.05, 3: 0.95},
        'region_min_area': {0: 0.010, 1: 0.006, 2: 0.001, 3: 0.001},
        'region_max_shift': {0: 14.0, 1: 24.0, 2: 18.0, 3: 16.0},
    }


def _to_gray_rgb_tensor(img_bgr: np.ndarray, input_size: int, device: torch.device):
    img = (img_bgr / 255.0).astype(np.float32)
    img_resized = cv2.resize(img, (input_size, input_size))
    img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
    img_gray_lab = np.concatenate([img_l, np.zeros_like(img_l), np.zeros_like(img_l)], axis=-1)
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
    tensor = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
    return tensor, img_gray_rgb


def _build_cond_from_reference(model, conditioner, ref_bgr: np.ndarray, input_size: int, device: torch.device, token_mode: str = 'dense', mask_mode: str = 'portrait_heuristic', cond_gain: float = 1.0):
    ref = (ref_bgr / 255.0).astype(np.float32)
    ref_resized = cv2.resize(ref, (input_size, input_size))

    ref_rgb = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2RGB)
    ref_tensor = torch.from_numpy(ref_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
    with torch.no_grad():
        ref_in = model.normalize(ref_tensor) if hasattr(model, "normalize") else ref_tensor
        _ = model.encoder(ref_in)
        hooks = model.encoder.hooks
        ref_feats = [hooks[1].feature, hooks[2].feature, hooks[3].feature]
        if str(token_mode).lower() == 'dense':
            cond_tokens, cond_pos = conditioner(ref_feats)
        else:
            if mask_mode == 'portrait_heuristic':
                masks_np = _build_masks_portrait_heuristic(ref_rgb, num_regions=6)
            else:
                ref_ab = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2Lab)[:, :, 1:3]
                masks_np = _build_masks_from_ab(ref_ab, num_regions=6)
            masks = torch.from_numpy(masks_np).unsqueeze(0).to(device)  # (1,R,H,W)
            cond_tokens, cond_pos = conditioner(ref_feats, masks)
        if cond_gain != 1.0:
            cond_tokens = cond_tokens * float(cond_gain)
    return cond_tokens, cond_pos


def _infer_one(model, conditioner, content_bgr: np.ndarray, ref_bgr: np.ndarray, input_size: int, device: torch.device, token_mode: str = 'dense', mask_mode: str = 'portrait_heuristic', cond_gain: float = 1.0):
    h, w = content_bgr.shape[:2]
    content_f = (content_bgr / 255.0).astype(np.float32)
    orig_l = cv2.cvtColor(content_f, cv2.COLOR_BGR2Lab)[:, :, :1]

    content_gray_tensor, content_gray_rgb = _to_gray_rgb_tensor(content_bgr, input_size, device)
    cond_tokens, cond_pos = _build_cond_from_reference(model, conditioner, ref_bgr, input_size, device, token_mode=token_mode, mask_mode=mask_mode, cond_gain=cond_gain)

    with torch.no_grad():
        out_ab = model(content_gray_tensor, cond_tokens=cond_tokens, cond_pos=cond_pos)
        out_ab = F.interpolate(out_ab, size=(h, w))[0].float().cpu().numpy().transpose(1, 2, 0)

    out_lab = np.concatenate((orig_l, out_ab), axis=-1)
    out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
    out_img = (out_bgr * 255.0).round().astype(np.uint8)

    content_gray_bgr = cv2.cvtColor((content_gray_rgb * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    content_gray_bgr = cv2.resize(content_gray_bgr, (w, h))

    return out_img, content_gray_bgr


def _resolve_cond_ckpt_path(net_g_ckpt: str, cond_ckpt: str = None):
    if cond_ckpt:
        return cond_ckpt if os.path.isfile(cond_ckpt) else None
    if not net_g_ckpt:
        return None
    ckpt_path = Path(net_g_ckpt)
    name = ckpt_path.name
    if not name.startswith('net_g_'):
        return None
    cand = ckpt_path.with_name(name.replace('net_g_', 'net_c_', 1))
    return str(cand) if cand.exists() else None


def _load_conditioner_weights(conditioner: torch.nn.Module, ckpt_path: str, device: torch.device):
    loaded = torch.load(ckpt_path, map_location=device)
    if isinstance(loaded, dict) and 'params' in loaded:
        state_dict = loaded['params']
    else:
        state_dict = loaded
    conditioner.load_state_dict(state_dict, strict=True)


def main():
    parser = argparse.ArgumentParser(description="DDColor cond-B style transfer inference")
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--content", required=True, type=str, help="Content image file/folder")
    parser.add_argument("--reference", required=True, type=str, help="Reference image file/folder")
    parser.add_argument("--output", default="results_condB", type=str)
    parser.add_argument("--input_size", default=512, type=int)
    parser.add_argument("--model_size", default="large", choices=["tiny", "large"])
    parser.add_argument("--num_queries", default=256, type=int)
    parser.add_argument("--num_scales", default=3, type=int)
    parser.add_argument("--dec_layers", default=9, type=int)
    parser.add_argument("--save_triptych", action="store_true", help="Save [content_gray|ref|result] panel")
    parser.add_argument("--pair_mode", default="cycle", choices=["cycle", "same_name"], help="Folder pairing strategy")
    parser.add_argument("--token_mode", default="dense", choices=["dense", "region"], help="Condition token mode")
    parser.add_argument("--mask_mode", default="portrait_heuristic", choices=["portrait_heuristic", "ab_angle"], help="Reference mask building mode (only for token_mode=region)")
    parser.add_argument("--cond_gain", default=1.0, type=float, help="Inference-time gain on reference condition tokens; >1 strengthens reference influence")
    parser.add_argument("--cond_ckpt", default=None, type=str, help="Optional conditioner checkpoint path (net_c_*.pth). If omitted, auto-resolve from --ckpt")
    parser.add_argument("--region_color_transfer", action="store_true", help="Apply semantic region-wise color transfer post-process")
    parser.add_argument("--region_transfer_strength", default=0.85, type=float, help="Region color transfer strength")
    parser.add_argument("--region_transfer_min_area", default=0.005, type=float, help="Minimum area ratio for a region to be transferred")
    parser.add_argument("--region_transfer_max_shift", default=28.0, type=float, help="Maximum per-channel ab shift for region transfer")
    parser.add_argument("--region_transfer_mask_erode", default=2, type=int, help="Erode pixels for region masks to reduce color bleeding")
    parser.add_argument("--region_transfer_mask_blur", default=5, type=int, help="Gaussian blur radius for soft region blending")
    parser.add_argument("--region_profile", default="balanced", choices=["balanced", "hair_priority"], help="Region-transfer profile preset")
    parser.add_argument("--modnet_onnx", default=None, type=str, help="Optional MODNet ONNX path for portrait matte prior")
    parser.add_argument("--modnet_ckpt", default=None, type=str, help="Optional MODNet ckpt path for portrait matte prior (PyTorch backend)")
    parser.add_argument("--modnet_repo", default=None, type=str, help="Optional MODNet repository root containing src/models/modnet.py")
    parser.add_argument("--modnet_ref_size", default=512, type=int, help="MODNet preprocessing reference size")
    parser.add_argument("--modnet_gate_strength", default=1.0, type=float, help="Mask gating strength by MODNet matte in [0,1]")
    parser.add_argument("--modnet_matte_min", default=0.02, type=float, help="Minimum matte value to keep as foreground")
    parser.add_argument("--modnet_bg_preserve", default=0.0, type=float, help="Optional blend to preserve grayscale background using (1-matte)")

    args = parser.parse_args()

    if str(args.token_mode).lower() != 'dense':
        raise SystemExit('Mask-based inference is disabled: only --token_mode dense is allowed.')
    if bool(args.region_color_transfer):
        raise SystemExit('Mask-based region color transfer is disabled for inference policy.')
    if args.modnet_ckpt or args.modnet_onnx:
        raise SystemExit('Mask-based MODNet matte inference is disabled for inference policy.')
    if args.mask_mode != 'ab_angle':
        print('[WARN] mask_mode is ignored in dense no-mask inference policy.')

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

    if args.token_mode == 'dense':
        conditioner = MultiScaleDenseTokenConditioner(
            num_scales=3,
            hidden_dim=256,
            grid_size=16,
        ).to(device)
    else:
        conditioner = MultiScaleRegionTokenConditioner(
            spec=RegionTokenSpec(),
            num_scales=3,
            hidden_dim=256,
            include_area_frac=True,
        ).to(device)
    cond_ckpt_path = _resolve_cond_ckpt_path(args.ckpt, args.cond_ckpt)
    if cond_ckpt_path is not None:
        _load_conditioner_weights(conditioner, cond_ckpt_path, device)
        print(f"[INFO] Loaded conditioner weights: {cond_ckpt_path}")
    else:
        print("[WARN] Conditioner checkpoint not found. Using randomly initialized conditioner for inference.")
    conditioner.eval()

    modnet = None
    if args.modnet_ckpt:
        modnet = _MODNetTorchMatte(
            ckpt_path=args.modnet_ckpt,
            device=device,
            ref_size=args.modnet_ref_size,
            repo_root=args.modnet_repo,
        )
    elif args.modnet_onnx:
        modnet = _MODNetONNXMatte(args.modnet_onnx, ref_size=args.modnet_ref_size)

    content_list = _list_images(args.content)
    ref_list = _list_images(args.reference)
    if not content_list:
        raise SystemExit(f"No content images found: {args.content}")
    if not ref_list:
        raise SystemExit(f"No reference images found: {args.reference}")

    out_dir = Path(args.output)
    trip_dir = out_dir / "triptych"
    out_dir.mkdir(parents=True, exist_ok=True)
    content_matte_cache = {}
    ref_matte_cache = {}

    for idx, c_path in enumerate(content_list):
        if Path(args.reference).is_file():
            r_path = ref_list[0]
        else:
            if args.pair_mode == "same_name":
                candidate = Path(args.reference) / c_path.name
                r_path = candidate if candidate.exists() else ref_list[idx % len(ref_list)]
            else:
                r_path = ref_list[idx % len(ref_list)]

        c_img = cv2.imread(str(c_path))
        r_img = cv2.imread(str(r_path))
        if c_img is None or r_img is None:
            print(f"[WARN] skip unreadable pair: {c_path} | {r_path}")
            continue

        out_img, content_gray_bgr = _infer_one(
            model,
            conditioner,
            c_img,
            r_img,
            args.input_size,
            device,
            token_mode=args.token_mode,
            mask_mode=args.mask_mode,
            cond_gain=args.cond_gain,
        )

        content_matte = None
        ref_matte = None
        if modnet is not None:
            c_key = str(c_path)
            r_key = str(r_path)
            if c_key not in content_matte_cache:
                content_matte_cache[c_key] = modnet.predict_matte(c_img)
            if r_key not in ref_matte_cache:
                ref_matte_cache[r_key] = modnet.predict_matte(r_img)
            content_matte = content_matte_cache[c_key]
            ref_matte = ref_matte_cache[r_key]

        if args.region_color_transfer:
            ref_rgb = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
            out_rgb = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
            ref_masks = _build_masks_portrait_heuristic(ref_rgb, num_regions=6)
            out_masks = _build_masks_portrait_heuristic(out_rgb, num_regions=6)
            prof = _get_region_profile(args.region_profile)
            out_img = _apply_region_color_transfer(
                output_bgr=out_img,
                ref_bgr=r_img,
                out_masks=out_masks,
                ref_masks=ref_masks,
                strength=float(args.region_transfer_strength),
                min_area_frac=float(args.region_transfer_min_area),
                max_shift=float(args.region_transfer_max_shift),
                include_regions=(0, 1, 2, 3),
                mask_erode=int(args.region_transfer_mask_erode),
                mask_blur=int(args.region_transfer_mask_blur),
                region_weights=prof['region_weights'],
                region_min_area=prof['region_min_area'],
                region_max_shift=prof['region_max_shift'],
                out_matte=content_matte,
                ref_matte=ref_matte,
                matte_gate_strength=float(args.modnet_gate_strength),
                matte_min=float(args.modnet_matte_min),
            )

        if modnet is not None and float(args.modnet_bg_preserve) > 0:
            alpha = np.clip(content_matte, 0.0, 1.0)[:, :, None].astype(np.float32)
            preserve = float(np.clip(args.modnet_bg_preserve, 0.0, 1.0))
            bg_w = (1.0 - alpha) * preserve
            out_f = out_img.astype(np.float32)
            gray_f = content_gray_bgr.astype(np.float32)
            out_img = np.clip(out_f * (1.0 - bg_w) + gray_f * bg_w, 0, 255).astype(np.uint8)

        stem = c_path.stem
        suffix = c_path.suffix if c_path.suffix else ".png"
        out_name = f"{stem}__ref_{r_path.stem}{suffix}"
        _safe_write(out_dir / out_name, out_img)

        if args.save_triptych:
            ref_resized = cv2.resize(r_img, (c_img.shape[1], c_img.shape[0]))
            panel = np.concatenate([content_gray_bgr, ref_resized, out_img], axis=1)
            _safe_write(trip_dir / out_name, panel)

    print(f"Done. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
