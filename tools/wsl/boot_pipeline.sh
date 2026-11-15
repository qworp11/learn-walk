#!/bin/bash
# Бут-страп конвейера: не даёт плодить дубли при автозапуске.
# Использование: bash boot_pipeline.sh <stage1> <stage2> ...
set -u
source /root/hlc-venv/bin/activate
PROJ="/mnt/g/LEARN WALK"
LOCK="/root/pipeline.lock"

# уже живой оркестратор? — выходим
if pgrep -f "stage_orchestrator.sh" >/dev/null 2>&1; then
  echo "$(date -u +%FT%TZ) конвейер уже работает — автозапуск пропущен" >> /root/boot_pipeline.log
  exit 0
fi
touch "$LOCK"
echo "$(date -u +%FT%TZ) автозапуск конвейера: $*" >> /root/boot_pipeline.log
nohup bash "$PROJ/tools/wsl/stage_orchestrator.sh" "$@" \
  > /root/orchestrator.out 2>&1 &

# --- демоны наблюдения: dashboard :8080, TensorBoard :6006, live-стрим :8081 ---
if ! pgrep -f "dashboard.py" >/dev/null 2>&1; then
  nohup python "$PROJ/src/hlc/dashboard.py" > /root/dashboard.out 2>&1 &
  echo "$(date -u +%FT%TZ) dashboard запущен" >> /root/boot_pipeline.log
fi
if ! pgrep -f "live_stream.py" >/dev/null 2>&1; then
  # свежая копия в нативную ФС: EGL-рендер стабильнее и без кэша DrvFS
  cp "$PROJ/src/hlc/live_stream.py" /root/live_stream.py
  nohup python /root/live_stream.py > /root/live_stream.out 2>&1 &
  echo "$(date -u +%FT%TZ) live_stream запущен" >> /root/boot_pipeline.log
fi
if ! pgrep -f "tensorboard" >/dev/null 2>&1; then
  nohup tensorboard --logdir /root/hlc-logs --port 6006 --bind_all \
    > /root/tensorboard.out 2>&1 &
  echo "$(date -u +%FT%TZ) tensorboard запущен" >> /root/boot_pipeline.log
fi
