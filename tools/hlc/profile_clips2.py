#!/bin/bash
# Профиль скоростей якоря клипов walk/run/sprint: где начинается шаг?
/root/hlc-venv/bin/python - <<'PYEOF'
import numpy as np, glob, os

for name in ("walk3", "run1", "sprint1"):
    hits = glob.glob(f"/mnt/g/LEARN WALK/data/motions/npz_g1/{name}*.npz") \
        + glob.glob(f"/root/hlc-data/motions/npz_g1/{name}*.npz")
    if not hits:
        # ищем любые клипы с похожими именами
        hits = [p for p in
                glob.glob("/mnt/g/LEARN WALK/data/motions/npz_g1/*.npz")
                if name[:4] in os.path.basename(p).lower()]
    if not hits:
        print(f"{name}: файл не найден")
        continue
    p = hits[0]
    d = np.load(p)
    pos = d["body_pos_w"][:, 0, :2]          # якорь = тело 0? (проверим по pelvis)
    # якорь в тренировке - pelvis (тело 0 в body_pos_w? берём тело 0 как раньше)
    v = np.linalg.norm(np.diff(pos, axis=0), axis=1) / 0.02
    T = len(v)
    print(f"{os.path.basename(p)}: T={T} v_max={v.max():.2f} v_mean={v.mean():.2f}")
    # первый кадр устойчивого шага: v>0.25 на протяжении 50 кадров
    mf = (v > 0.25).astype(np.float64)
    cs = np.cumsum(mf)
    win = cs[49:] - np.concatenate([[0.0], cs[:-50]])
    idx = np.nonzero(win >= 50)[0]
    first25 = int(idx[0]) if idx.size else -1
    mf = (v > 0.15).astype(np.float64)
    cs = np.cumsum(mf)
    win = cs[49:] - np.concatenate([[0.0], cs[:-50]])
    idx = np.nonzero(win >= 50)[0]
    first15 = int(idx[0]) if idx.size else -1
    print(f"  шаг(v>0.25,устойч.50) с кадра {first25}; (v>0.15) с кадра {first15}")
    # срез скорости по ходу клипа
    for i in range(0, T, max(T // 15, 1)):
        print(f"    кадр {i:5d}: v={v[i]:5.2f}")
PYEOF
