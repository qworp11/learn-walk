#!/bin/bash
/root/hlc-venv/bin/python - <<'PYEOF'
import numpy as np
d = np.load("/mnt/g/LEARN WALK/data/motions/npz_g1/sit_synth.npz")
z = d["body_pos_w"][:, 0, 2]
T = len(z)
print("T =", T)
for i in range(0, T, 25):
    print(f"  кадр {i:3d}: z={z[i]:.3f}")
PYEOF
