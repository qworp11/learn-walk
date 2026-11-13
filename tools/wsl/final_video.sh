#!/bin/bash
# Финальный артефакт этапа: ждёт TRAIN_DONE, делает роллаут финальной политики
# на CUDA + сравнительное видео 640x360. Usage: final_video.sh <stage> <motion>
set -u
STAGE="$1"; MOTION="$2"
LOGROOT="/root/hlc-logs/$STAGE"
PROJ="/mnt/g/LEARN WALK"
OUT="$PROJ/results/videos/final"
mkdir -p "$OUT"

# ждём завершения тренировки (максимум 2 часа)
for i in $(seq 1 120); do
  ls "$LOGROOT"/run_*/TRAIN_DONE >/dev/null 2>&1 && break
  sleep 60
done
ls "$LOGROOT"/run_*/TRAIN_DONE >/dev/null 2>&1 || { echo "FINAL_SKIP: не дождались TRAIN_DONE"; exit 1; }

RUN=$(ls -dt "$LOGROOT"/run_* | head -1)
CKPT=$(ls -v "$RUN"/model_*.pt | tail -1)
echo "финальный чекпоинт: $CKPT"

source /root/hlc-venv/bin/activate
export MUJOCO_GL=egl
Q="$PROJ/results/rollouts/$STAGE/final"
mkdir -p "$Q"
python "$PROJ/src/hlc/policy_rollout.py" --ckpt "$CKPT" --motion "$MOTION" \
  --out_dir "$Q" --episodes 3 --device cuda --play 2>&1 | grep -E 'ep[0-9]|ROLLOUT'

python - <<PYEOF
import glob
import numpy as np
parts = sorted(glob.glob("$Q/qpos_ep*.npy"))
arrs = [np.load(p) for p in parts]
n = min(a.shape[0] for a in arrs)
np.save("$Q/combined.npy", np.concatenate([a[:n] for a in arrs], axis=0))
print("combined frames:", n * len(arrs), "episodes:", len(arrs), "min len:", n)
PYEOF

python "$PROJ/src/hlc/render_video.py" \
  --qpos "$Q/combined.npy" \
  --compare_npz "$PROJ/data/motions/npz_g1/$MOTION.npz" \
  --out "$OUT/${STAGE}_final.mp4" \
  --width 640 --height 360 \
  2>&1 | tail -2
echo "FINAL_VIDEO_DONE $OUT/${STAGE}_final.mp4"
