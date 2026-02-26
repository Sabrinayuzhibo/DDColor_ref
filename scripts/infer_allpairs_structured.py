#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from infer_style_transfer import (
    DDColor,
    MultiScaleDenseTokenConditioner,
    _infer_one,
    _list_images,
    _load_conditioner_weights,
    _resolve_cond_ckpt_path,
    _safe_write,
    build_ddcolor_model,
)


def _safe_name(name: str) -> str:
    bad = '<>:"/\\|?*'
    out = ''.join('_' if c in bad else c for c in name)
    return out.strip().rstrip('.')


def _make_diff_panel(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    if img_a.shape[:2] != img_b.shape[:2]:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_AREA)
    abs_diff = cv2.absdiff(img_a, img_b)
    heat = cv2.applyColorMap(cv2.cvtColor(abs_diff, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_TURBO)
    panel = np.concatenate([img_a, img_b, heat], axis=1)
    return panel


def main():
    parser = argparse.ArgumentParser(description='All-pairs structured inference for DDColor')
    parser.add_argument('--ckpt', required=True, type=str)
    parser.add_argument('--cond_ckpt', default=None, type=str)
    parser.add_argument('--content', required=True, type=str)
    parser.add_argument('--reference', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--input_size', default=512, type=int)
    parser.add_argument('--model_size', default='large', choices=['tiny', 'large'])
    parser.add_argument('--num_queries', default=256, type=int)
    parser.add_argument('--num_scales', default=3, type=int)
    parser.add_argument('--dec_layers', default=9, type=int)
    parser.add_argument('--token_mode', default='dense', choices=['dense'])
    parser.add_argument('--cond_gain', default=1.0, type=float)
    parser.add_argument('--override_cond_gate_raw', default=None, type=float,
                        help='Optional raw gate value override for all decoder cond gates (e.g. 0.0=>sigmoid 0.5, 1.0=>0.73)')
    parser.add_argument('--save_ref_diff', action='store_true', help='Save pairwise diff panels when at least two reference images exist')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_ddcolor_model(
        DDColor,
        model_path=args.ckpt,
        input_size=args.input_size,
        model_size=args.model_size,
        decoder_type='MultiScaleColorDecoder',
        device=device,
        num_queries=args.num_queries,
        num_scales=args.num_scales,
        dec_layers=args.dec_layers,
    )
    model.eval()

    if args.override_cond_gate_raw is not None:
        gate_applied = False
        try:
            gate_param = model.decoder.color_decoder.cond_layer_gates
            with torch.no_grad():
                gate_param.fill_(float(args.override_cond_gate_raw))
            gate_applied = True
            gate_sigmoid = torch.sigmoid(gate_param.detach().float().cpu()).tolist()
            gate_sigmoid = [round(float(v), 4) for v in gate_sigmoid]
            print(f'[INFO] override cond gates raw={float(args.override_cond_gate_raw):.4f}, sigmoid={gate_sigmoid}')
        except Exception:
            gate_applied = False
        if not gate_applied:
            print('[WARN] override_cond_gate_raw requested but cond gates were not found on model.')

    conditioner = MultiScaleDenseTokenConditioner(
        num_scales=3,
        hidden_dim=256,
        grid_size=16,
    ).to(device)
    cond_ckpt_path = _resolve_cond_ckpt_path(args.ckpt, args.cond_ckpt)
    if cond_ckpt_path is not None:
        _load_conditioner_weights(conditioner, cond_ckpt_path, device)
        print(f'[INFO] Loaded conditioner weights: {cond_ckpt_path}')
    else:
        print('[WARN] Conditioner checkpoint not found. Using randomly initialized conditioner for inference.')
    conditioner.eval()

    content_list = _list_images(args.content)
    ref_list = _list_images(args.reference)
    if not content_list:
        raise SystemExit(f'No content images found: {args.content}')
    if not ref_list:
        raise SystemExit(f'No reference images found: {args.reference}')

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    for c_path in content_list:
        c_img = cv2.imread(str(c_path))
        if c_img is None:
            print(f'[WARN] skip unreadable content: {c_path}')
            continue

        content_key = _safe_name(c_path.stem)
        c_dir = out_root / content_key
        ref_lowres_dir = c_dir / 'ref_lowres'
        outputs_dir = c_dir / 'outputs'
        diff_dir = c_dir / 'diff_pairs'
        c_dir.mkdir(parents=True, exist_ok=True)
        ref_lowres_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        if args.save_ref_diff:
            diff_dir.mkdir(parents=True, exist_ok=True)

        _safe_write(c_dir / 'content_original.png', c_img)

        pair_outputs = {}
        for r_path in ref_list:
            r_img = cv2.imread(str(r_path))
            if r_img is None:
                print(f'[WARN] skip unreadable ref: {r_path}')
                continue

            r_key = _safe_name(r_path.stem)
            ref_lr = cv2.resize(r_img, (args.input_size, args.input_size), interpolation=cv2.INTER_AREA)
            _safe_write(ref_lowres_dir / f'{r_key}.png', ref_lr)

            out_img, _ = _infer_one(
                model,
                conditioner,
                c_img,
                r_img,
                args.input_size,
                device,
                token_mode='dense',
                cond_gain=args.cond_gain,
            )

            _safe_write(outputs_dir / f'{r_key}.png', out_img)
            pair_outputs[r_key] = out_img

        if args.save_ref_diff and len(pair_outputs) >= 2:
            keys = list(pair_outputs.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    k1, k2 = keys[i], keys[j]
                    panel = _make_diff_panel(pair_outputs[k1], pair_outputs[k2])
                    _safe_write(diff_dir / f'{k1}__vs__{k2}.png', panel)

    print(f'Done. Results saved to: {out_root}')


if __name__ == '__main__':
    main()
