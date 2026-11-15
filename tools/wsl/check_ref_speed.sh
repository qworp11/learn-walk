#!/bin/bash
source /root/hlc-venv/bin/activate
python - <<'EOF'
import numpy as np

d = np.load("/mnt/g/LEARN WALK/data/motions/npz_g1/walk1_subject1.npz")
pos = d["body_pos_w"][:, 0, :]   # [T, 3] корень
T = pos.shape[0]
fps = 50.0
print("клип:", T, "кадров =", T / fps, "сек")

v = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1) * fps  # м/с по XY
v = np.concatenate([[0], v])

print(f"скорость: mean={v.mean():.2f} м/с, p50={np.percentile(v,50):.2f}, "
      f"p95={np.percentile(v,95):.2f}, max={v.max():.2f}")

# сегменты «стоит» (<0.15 м/с дольше 1 с)
still = v < 0.15
segs, s = [], None
for i, x in enumerate(still):
    if x and s is None:
        s = i
    elif not x and s is not None:
        if i - s >= int(fps):
            segs.append((s / fps, i / fps))
        s = None
if s is not None and T - s >= int(fps):
    segs.append((s / fps, T / fps))

print(f"пауз-сегментов (>=1с стоит): {len(segs)}")
for a, b in segs[:12]:
    print(f"  стоит {a:6.1f}с -> {b:6.1f}с  ({b-a:.1f} с)")

# первые 30 сек: сколько идёт / стоит
for name, a, b in [("начало 0-10с", 0, 10), ("10-20с", 10, 20), ("20-30с", 20, 30)]:
    if b * fps <= T:
        w = v[int(a*fps):int(b*fps)]
        print(f"{name}: mean {w.mean():.2f} м/с, стоит {100*still[int(a*fps):int(b*fps)].mean():.0f}%")
EOF
