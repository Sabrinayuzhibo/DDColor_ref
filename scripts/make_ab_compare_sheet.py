import random
from pathlib import Path
import cv2
import numpy as np

A = Path('results_condB_nomask_abangle_v1')
B = Path('results_condB_nomask_abangle_softprior_v1')
OUT = Path('results_condB_nomask_ab_compare')
OUT.mkdir(parents=True, exist_ok=True)

files = sorted([p.name for p in A.glob('*') if p.is_file() and (A / p.name).exists() and (B / p.name).exists()])
if not files:
    raise SystemExit('No shared files between A and B')

sample = files[:6]
for name in sample:
    ia = cv2.imread(str(A / name))
    ib = cv2.imread(str(B / name))
    if ia is None or ib is None:
        continue
    h = min(ia.shape[0], ib.shape[0])
    w = min(ia.shape[1], ib.shape[1])
    ia = cv2.resize(ia, (w, h))
    ib = cv2.resize(ib, (w, h))

    pad = 36
    canvas = np.full((h + pad, w * 2, 3), 245, dtype=np.uint8)
    canvas[pad:, :w] = ia
    canvas[pad:, w:] = ib
    cv2.putText(canvas, 'A: no hard mask', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20,20,20), 2, cv2.LINE_AA)
    cv2.putText(canvas, 'B: no hard mask + MODNet soft prior', (w + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (20,20,20), 2, cv2.LINE_AA)
    cv2.imwrite(str(OUT / name), canvas)

print(f'Wrote compare sheets: {OUT} (count={len(sample)})')
