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
