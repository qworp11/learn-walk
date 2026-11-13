#!/bin/bash
# Превью-воркер: на каждом новом чекпоинте делает короткое видео
# «текущая политика vs референс» (роллаут на CPU, тренировку не трогает).
# Usage: preview_worker.sh <stage> [motion]
# Выход: results/videos/previews/<stage>/model_<iter>.mp4
set -u
STAGE="${1:-s1_balance}"
MOTION="${2:-}"
PROJ="/mnt/g/LEARN WALK"
LOGROOT="/root/hlc-logs/$STAGE"
PREV_DIR="$PROJ/results/videos/previews/$STAGE"
mkdir -p "$PREV_DIR"
PROCESSED="/root/preview_processed_$STAGE.txt"
touch "$PROCESSED"
source /root/hlc-venv/bin/activate
export MUJOCO_GL=egl
export WARP_CACHE=/root/wp_kernel_cache

# motion берём из cmd.json последнего запуска, если не задан
if [ -z "$MOTION" ]; then
  MOTION=$(python - <<'PYEOF'
import glob, json, os, pathlib
runs = sorted(glob.glob("/root/hlc-logs/s1_balance/run_*/cmd.json"))
if runs:
    c = json.load(open(runs[-1]))
    p = c.get("env_overrides", {}).get("trajectory_path", "")
    print(pathlib.Path(p).stem)
PYEOF
)
fi
echo "PREVIEW_WORKER старт: stage=$STAGE motion=$MOTION"

while true; do
  # последний run с чекпоинтами
  RUN=$(ls -dt "$LOGROOT"/run_* 2>/dev/null | head -1)
  if [ -n "$RUN" ]; then
    for CKPT in $(ls -v "$RUN"/model_*.pt 2>/dev/null); do
      BASE=$(basename "$CKPT")
      grep -qxF "$CKPT" "$PROCESSED" && continue
      echo "[$(date -u +%H:%M:%S)] превью для $BASE"
      OUT_Q="$PROJ/results/rollouts/$STAGE/prev_$BASE"
      mkdir -p "$OUT_Q"
      python "$PROJ/src/hlc/policy_rollout.py" \
        --ckpt "$CKPT" --motion "$MOTION" \
        --out_dir "$OUT_Q" --episodes 2 --device cpu --play \
        > /dev/null 2>&1
      # склейка эпизодов в один npy (первый ряд + rest, разделитель не нужен)
      python - <<PYEOF
import glob
import numpy as np
parts = sorted(glob.glob("$OUT_Q/qpos_ep*.npy"))
if parts:
    arrs = [np.load(p) for p in parts]
    n = min(a.shape[0] for a in arrs)
    # выравниваем по самому короткому, чтобы панели были синхронны
    cut = np.concatenate([a[:n] for a in arrs], axis=0)
    np.save("$OUT_Q/combined.npy", cut)
    print("combined frames:", cut.shape[0])
PYEOF
      if [ -f "$OUT_Q/combined.npy" ]; then
        python "$PROJ/src/hlc/render_video.py" \
          --qpos "$OUT_Q/combined.npy" \
          --compare_npz "$PROJ/data/motions/npz_g1/$MOTION.npz" \
          --out "$PREV_DIR/$BASE.mp4" \
          --width 480 --height 270 --max_frames 180 \
          > /dev/null 2>&1 \
          && echo "[$(date -u +%H:%M:%S)] OK $PREV_DIR/$BASE.mp4"
      fi
      echo "$CKPT" >> "$PROCESSED"
    done
  fi
  # штатное завершение тренировки — заканчиваем работу
  ls "$LOGROOT"/run_*/TRAIN_DONE >/dev/null 2>&1 && { echo "TRAIN_DONE — превью-воркер стоп"; break; }
  sleep 60
done
