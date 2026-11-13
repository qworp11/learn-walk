#!/bin/bash
# Watchdog v2: следит за тренировкой этапа.
# Прогресс = число строк "Iteration time" в логе (формат rsl_rl может меняться,
# счётчик строк растёт монотонно — надёжнее регэкспов по номеру итерации).
# - процесс умер и TRAIN_DONE нет → рестарт с последнего чекпоинта (backoff)
# - прогресс не двигается 30+ мин → kill hlc_train + run_stage → рестарт
# - TRAIN_DONE → штатный выход
# Запуск: bash watchdog.sh <stage> [max_restarts]
set -u
source /root/hlc-venv/bin/activate
STAGE="$1"; shift || true
MAX_RESTARTS="${1:-5}"
PROJ="/mnt/g/LEARN WALK"
HB="/mnt/g/LEARN WALK/heartbeat.json"
LOGROOT="/root/hlc-logs/$STAGE"
PGREP_PAT="run_stage.py $STAGE"

restarts=0
backoff=300
last_count=-1
same_count=0

# убить тренировку целиком: hlc_train + его run_stage (pkill -f в этой WSL
# ненадёжен — работаем по явным PID)
kill_train() {
  for p in $(pgrep -f "hlc_train.py" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  for p in $(pgrep -f "run_stage.py $STAGE" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  return 0
}

progress_count() {
  # самый свежий лог этапа; бинарный вывод grep не пугает (-a)
  local L
  L=$(ls -t "$LOGROOT"/run_*/train_stdout.log 2>/dev/null | head -1)
  if [ -n "$L" ]; then
    grep -ac "Iteration time" "$L" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

while true; do
  now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  gpu=$(nvidia-smi --query-gpu=temperature.gpu,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)
  IFS=', ' read -r temp vram util power <<< "$gpu"

  pid=$(pgrep -f "$PGREP_PAT" | head -1)
  # pgrep матчит и зомби — проверяем, что процесс реально работает
  if [ -n "$pid" ] && [ "$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null)" == "Z" ]; then
    pid=""
  fi
  alive="false"
  [ -n "$pid" ] && alive="true"

  run_dir=$(ls -dt "$LOGROOT"/run_* 2>/dev/null | head -1)
  iter=$(progress_count)
  nan="no"
  if [ -n "$run_dir" ] && [ -n "$(ls -t "$run_dir"/train_stdout.log 2>/dev/null | head -1)" ]; then
    if grep -aqE "nan|NaN" <(tail -c 100000 "$run_dir"/train_stdout.log 2>/dev/null); then nan="maybe"; fi
  fi

  # Зависание: счётчик итераций не растёт 3 проверки подряд (30+ мин)
  if [ "$iter" == "$last_count" ] && [ "$alive" == "true" ]; then
    same_count=$((same_count+1))
  else
    same_count=0
    last_count="$iter"
  fi

  python - "$HB" "$now" "$STAGE" "$alive" "$pid" "$temp" "$vram" "$util" "$power" "$run_dir" "$iter" "$nan" "$restarts" <<'PYEOF'
import json, sys
path, now, stage, alive, pid, temp, vram, util, power, run_dir, it, nan, restarts = sys.argv[1:]
try:
    hb = json.load(open(path))
except Exception:
    hb = {}
hb.update({
  "time": now, "stage": stage, "train_alive": alive == "true", "pid": pid,
  "gpu": {"temp_c": temp, "vram_mb": vram, "util": util, "power_w": power},
  "iteration_count": it, "nan_suspect": nan, "restarts": restarts,
  "run_dir": run_dir,
})
json.dump(hb, open(path, "w"), indent=2)
PYEOF

  if [ "$alive" != "true" ]; then
    if ls "$LOGROOT"/run_*/TRAIN_DONE >/dev/null 2>&1; then
      echo "$now: тренировка завершилась штатно (TRAIN_DONE) — watchdog стоп." | tee -a "$LOGROOT/watchdog.log"
      break
    fi
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
      echo "$now: ПЕРЕСТАНОВКИ ИЗЛОЖИЛИСЬ ($restarts) — стоп. Виталик, требуется ручной разбор." | tee -a "$LOGROOT/watchdog.log"
      break
    fi
    restarts=$((restarts+1))
    echo "$now: train мёртв, рестарт #$restarts (backoff ${backoff}s)" | tee -a "$LOGROOT/watchdog.log"
    sleep "$backoff"
    backoff=$((backoff*2))
    bash "$PROJ/tools/wsl/train.sh" "$STAGE" || true
  elif [ "$same_count" -ge 3 ]; then
    echo "$now: прогресс ($iter итераций) не двигается 30+ мин — kill + resume" | tee -a "$LOGROOT/watchdog.log"
    kill_train
    same_count=0
    sleep 30
    bash "$PROJ/tools/wsl/train.sh" "$STAGE" || true
  fi

  sleep 600
done
