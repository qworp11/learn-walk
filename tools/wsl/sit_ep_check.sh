#!/bin/bash
source /root/hlc-venv/bin/activate
python - <<'PYEOF'
import numpy as np
q = np.load("/mnt/g/LEARN WALK/results/rollouts/s9_sit/final/combined.npy")
for ep in range(3):
    seg = q[ep*500:(ep+1)*500, 2]
    hold = seg[125:225]  # окно удержания референса (2.5-4.5с)
    print(f"эп{ep}: min z в удержании {hold.min():.3f}, z в конце (9.9с) {seg[-10:].mean():.3f}, max z {seg.max():.3f}")
PYEOF
