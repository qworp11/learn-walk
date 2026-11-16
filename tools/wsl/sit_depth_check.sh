#!/bin/bash
source /root/hlc-venv/bin/activate
python - <<'PYEOF'
import numpy as np
d = np.load("/mnt/g/LEARN WALK/data/motions/npz_g1/sit_synth.npz")
ref_z = d["body_pos_w"][:, 0, 2]
print("референс z по таймлайну клипа:")
for t_sec, fr in [(0.6, 30), (2.0, 100), (3.0, 150), (3.6, 180), (4.0, 200), (5.0, 250), (6.5, 325)]:
    print(f"  {t_sec:4.1f}s (кадр {fr}): ref_z = {ref_z[fr]:.3f}")
q = np.load("/mnt/g/LEARN WALK/results/rollouts/s9_sit/final/combined.npy")
pol_z = q[:, 2]
print("политика vs референс в удержании (кадры 150-200):")
for fr in range(150, 201, 10):
    print(f"  кадр {fr}: пол {pol_z[fr]:.3f}  реф {ref_z[fr]:.3f}  diff {pol_z[fr]-ref_z[fr]:+.3f}")
print(f"мин z политики в удержании: {pol_z[130:220].min():.3f}")
PYEOF
