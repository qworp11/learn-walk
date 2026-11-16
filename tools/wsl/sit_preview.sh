#!/bin/bash
# Превью-кадры s9_sit + числовая проверка позы по combined.npy
V="/mnt/g/LEARN WALK/results/videos/final/s9_sit_final.mp4"
O="/mnt/g/LEARN WALK/results/videos/final"
for f in 30 180 780; do
  ffmpeg -y -loglevel error -i "$V" -vf "select=eq(n\,$f)" -vframes 1 \
    "$O/s9_sit_final_frame$f.png"
done
ls -la "$O" | grep s9_sit
source /root/hlc-venv/bin/activate
python - <<'PYEOF'
import numpy as np
q = np.load("/mnt/g/LEARN WALK/results/rollouts/s9_sit/final/combined.npy")
# пелвис - свободный сустав: qpos[0:3] = xyz
for f in [30, 180, 780]:
    print(f"frame {f}: pelvis_z = {q[f][2]:.3f}")
PYEOF
echo PREVIEW_DONE
