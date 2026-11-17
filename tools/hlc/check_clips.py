#!/bin/bash
source /root/hlc-venv/bin/activate
python - <<'PYEOF'
import numpy as np, glob, os, math
for name in ("walk", "stand", "sit_synth", "run", "sprint"):
    p = f"/mnt/g/LEARN WALK/data/motions/npz_g1/{name}.npz"
    if not os.path.exists(p):
        cands = glob.glob(f"/mnt/g/LEARN WALK/data/motions/npz_g1/*{name}*.npz")
        if not cands:
            print(f"{name}: НЕТ ФАЙЛА"); continue
        p = cands[0]
    d = np.load(p)
    pos = d["body_pos_w"][:, 0]      # pelvis
    quat = d["body_quat_w"][:, 0]    # (w,x,y,z)
    T = pos.shape[0]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    v = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1).mean() * 50
    disp = np.linalg.norm(pos[-1, :2] - pos[0, :2])
    print(f"{os.path.basename(p):18s} T={T:5d}  yaw0={math.degrees(yaw[0]):7.1f}°  "
          f"yaw_end={math.degrees(yaw[-1]):7.1f}°  speed={v:.2f} м/с  disp={disp:.2f} м")
PYEOF
