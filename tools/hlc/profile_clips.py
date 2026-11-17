#!/bin/bash
source /root/hlc-venv/bin/activate
python - <<'PYEOF'
import numpy as np, math

def profile(name):
    d = np.load(f"/mnt/g/LEARN WALK/data/motions/npz_g1/{name}.npz")
    pos = d["body_pos_w"][:, 0]
    v = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1) * 50  # м/с по кадрам
    T = len(v)
    # когда клип разгоняется: первый кадр, где скорость > 0.3 в течение 50 кадров подряд
    moving = v > 0.3
    first = None
    for i in range(T - 50):
        if moving[i:i + 50].all():
            first = i
            break
    print(f"{name}: T={T}")
    for t0 in (0, 50, 100, 200, 400, 800):
        seg = v[t0:t0 + 50]
        print(f"   кадры {t0:4d}-{t0+49:4d}: средняя скорость {seg.mean():.2f} м/с")
    print(f"   первая непрерывно-идущая фаза: кадр {first}")
    # медианная скорость идущих кадров
    print(f"   медиана скорости: {np.median(v[v > 0.05]):.2f} м/с (у идущих кадров)")

for n in ("walk3_subject1", "walk1_subject1", "run2_subject1", "dance1_subject1"):
    try:
        profile(n)
    except FileNotFoundError as e:
        print(n, "нет файла")
PYEOF
