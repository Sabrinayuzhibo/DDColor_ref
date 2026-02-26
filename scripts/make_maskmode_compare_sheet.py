from pathlib import Path
import cv2
import numpy as np

A_ROOT = Path('results_condB_transfer2k_allpairs_structured')
B_ROOT = Path('results_condB_transfer2k_allpairs_structured_faceparsing')
OUT = Path('results_condB_transfer2k_maskmode_compare')
OUT.mkdir(parents=True, exist_ok=True)

content_dirs = [d for d in sorted(A_ROOT.iterdir()) if d.is_dir() and (B_ROOT / d.name).is_dir()]
count = 0
for cdir in content_dirs:
    a_out = cdir / 'outputs'
    b_out = B_ROOT / cdir.name / 'outputs'
    if not a_out.exists() or not b_out.exists():
        continue

    names = sorted([p.name for p in a_out.glob('*.png') if (b_out / p.name).exists()])
    if not names:
        continue

    name = names[0]
    ia = cv2.imread(str(a_out / name))
    ib = cv2.imread(str(b_out / name))
    if ia is None or ib is None:
        continue

    h = min(ia.shape[0], ib.shape[0])
    w = min(ia.shape[1], ib.shape[1])
    ia = cv2.resize(ia, (w, h))
    ib = cv2.resize(ib, (w, h))

    pad = 38
    canvas = np.full((h + pad, w * 2, 3), 245, dtype=np.uint8)
    canvas[pad:, :w] = ia
    canvas[pad:, w:] = ib
    cv2.putText(canvas, 'A: ab_angle', (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, 'B: face_parsing', (w + 10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (25, 25, 25), 2, cv2.LINE_AA)

    out_name = f"{cdir.name}__ref_{Path(name).stem}.png"
    cv2.imwrite(str(OUT / out_name), canvas)
    count += 1

print(f'Wrote compare sheets: {OUT} (count={count})')
