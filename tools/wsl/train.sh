#!/bin/bash
# Обучение этапа: train.sh <stage> [train_args...]
# Смотрит configs/stages.json на предмет nworld/max_iterations/motion.
# Логи: /root/hlc-logs/<stage>/run_<N>; чекпоинты: /root/hlc-logs/<stage>/run_<N>/model_*.pt
set -e
source /root/hlc-venv/bin/activate
export WP_CACHE_DIR=/root/wp_kernel_cache
export MUJOCO_GL=egl
PROJ="/mnt/g/LEARN WALK"
STAGE="$1"; shift || true
python "$PROJ/tools/hlc/run_stage.py" "$STAGE" "$@"
