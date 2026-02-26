#!/usr/bin/env python
"""Plot training curves from BasicSR logs.

Example:
  python scripts/plot_train_curves.py \
    --logs experiments/train_ddcolor_condA_ms_tokens/train_train_ddcolor_condA_ms_tokens_20260218_182524.log \
           experiments/train_ddcolor_condA_ms_tokens/train_train_ddcolor_condA_ms_tokens_20260219_171027.log \
    --labels iter3000 iter20000 \
    --out experiments/train_ddcolor_condA_ms_tokens/curves_3000_vs_20000.png
"""

import argparse
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt


LINE_RE = re.compile(
    r"iter:\s*([\d,]+).*?"
    r"l_g_pix:\s*([\deE+\-.]+).*?"
    r"l_g_percep:\s*([\deE+\-.]+).*?"
    r"l_g_gan:\s*([\deE+\-.]+).*?"
    r"l_d:\s*([\deE+\-.]+)"
)


def _parse_log(log_path: str) -> Dict[str, List[float]]:
    data = {"iter": [], "l_g_pix": [], "l_g_percep": [], "l_g_gan": [], "l_d": []}
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            it = int(m.group(1).replace(",", ""))
            data["iter"].append(it)
            data["l_g_pix"].append(float(m.group(2)))
            data["l_g_percep"].append(float(m.group(3)))
            data["l_g_gan"].append(float(m.group(4)))
            data["l_d"].append(float(m.group(5)))
    return data


def _smooth(values: List[float], win: int) -> List[float]:
    if win <= 1 or len(values) < win:
        return values
    out = []
    acc = 0.0
    q = []
    for v in values:
        q.append(v)
        acc += v
        if len(q) > win:
            acc -= q.pop(0)
        out.append(acc / len(q))
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot BasicSR train curves")
    parser.add_argument("--logs", nargs="+", required=True, help="One or more log files")
    parser.add_argument("--labels", nargs="*", default=None, help="Legend labels")
    parser.add_argument("--out", required=True, help="Output png path")
    parser.add_argument("--smooth", type=int, default=20, help="Moving average window")
    args = parser.parse_args()

    labels = args.labels if args.labels else [os.path.basename(p) for p in args.logs]
    if len(labels) != len(args.logs):
        raise ValueError("--labels number must match --logs number")

    parsed = [(_parse_log(p), l) for p, l in zip(args.logs, labels)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    keys = ["l_g_pix", "l_g_percep", "l_g_gan", "l_d"]

    for ax, key in zip(axes.flatten(), keys):
        for data, label in parsed:
            x = data["iter"]
            y = _smooth(data[key], args.smooth)
            if len(x) == 0:
                continue
            ax.plot(x, y, label=label, linewidth=1.6)
        ax.set_title(key)
        ax.set_xlabel("iter")
        ax.grid(alpha=0.25)
        ax.legend()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    print(f"Saved curve figure to: {args.out}")


if __name__ == "__main__":
    main()
