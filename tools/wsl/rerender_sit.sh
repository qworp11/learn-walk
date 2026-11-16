#!/bin/bash
# Перерендер сравнительного видео s9_sit после фикса заворота референса
set -u
PROJ="/mnt/g/LEARN WALK"
Q="$PROJ/results/rollouts/s9_sit/final"
ls -la "$Q/combined.npy" || { echo "НЕТ combined.npy"; exit 1; }
source /root/hlc-venv/bin/activate
export MUJOCO_GL=egl
python "$PROJ/src/hlc/render_video.py" \
  --qpos "$Q/combined.npy" \
  --compare_npz "$PROJ/data/motions/npz_g1/sit_synth.npz" \
  --out "$PROJ/results/videos/final/s9_sit_final.mp4" \
  --width 640 --height 360
echo "RERENDER_EXIT=$?"
