#!/bin/bash
# Тонкий профиль: где у клипов НАСТОЯЩАЯ походка (политика её подхватывает)?
/root/hlc-venv/bin/python - <<'PYEOF'
import numpy as np

for name, lo, hi in (("walk3_subject1", 200, 1400),
                     ("run1_subject2", 300, 900),
                     ("sprint1_subject2", 0, 500)):
    d = np.load(f"/mnt/g/LEARN WALK/data/motions/npz_g1/{name}.npz")
    pos = d["body_pos_w"][:, 0, :2]
    v = np.linalg.norm(np.diff(pos, axis=0), axis=1) / 0.02
    print(f"--- {name} ---")
    for thr in (0.3, 0.35, 0.4):
        mf = (v > thr).astype(np.float64)
        cs = np.cumsum(mf)
        win = cs[49:] - np.concatenate([[0.0], cs[:-50]])
        idx = np.nonzero(win >= 50)[0]
        print(f"  устойчиво >{thr}: с кадра {int(idx[0]) if idx.size else -1}")
    print("  срез v по кадрам:")
    for i in range(lo, min(hi, len(v)), 25):
        bar = "#" * int(v[i] * 20)
        print(f"    {i:5d}: {v[i]:5.2f} {bar}")
PYEOF
