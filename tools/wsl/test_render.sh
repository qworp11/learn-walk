#!/bin/bash
# Тест видео-рендера: референс dance1_subject1, 3 секунды, 640x360
set -e
source /root/hlc-venv/bin/activate
export MUJOCO_GL=egl
export WARP_CACHE=/root/wp_kernel_cache
python '/mnt/g/LEARN WALK/src/hlc/render_video.py' \
  --npz '/mnt/g/LEARN WALK/data/motions/npz_g1/dance1_subject1.npz' \
  --out '/mnt/g/LEARN WALK/results/videos/test_ref.mp4' \
  --width 640 --height 360 --max_frames 150
ls -la '/mnt/g/LEARN WALK/results/videos/'
