#!/bin/bash
# Watchdog: следит за тренировкой, рестартит при падении, пишет heartbeat.json.
# Запуск: bash watchdog.sh <stage> [max_restarts]
# Каждые 10 минут:
#  - жив ли процесс train (pgrep по run_stage.py <stage>)
#  - темп./VRAM GPU
#  - последние метрики из train_stdout.log (reward/KL/entropy/NaN)
#  - если умер: resume с последнего чекпоинта, экспоненциальный backoff, максимум 5 попыток
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
last_iter=-1
same_iter_count=0

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

  # Последний лог тренировки
  run_dir=$(ls -dt "$LOGROOT"/run_* 2>/dev/null | head -1)
  iter=""; reward=""; nan="no"; kl=""; entropy=""
  if [ -n "$run_dir" ]; then
    lastline=$(tail -c 200000 "$run_dir/train_stdout.log" 2>/dev/null | grep -E "^(  iteration|iteration)" | tail -1)
    # rsl_rl печатает разные форматы — парсим по-простому
    iter=$(tail -c 200000 "$run_dir/train_stdout.log" 2>/dev/null | grep -oE "Iteration [0-9]+" | tail -1 | grep -oE "[0-9]+")
    [ -z "$iter" ] && iter=$(tail -c 200000 "$run_dir/train_stdout.log" 2>/dev/null | grep -oE "iteration:[0-9]+" | tail -1 | cut -d: -f2)
    if grep -qE "nan|NaN|inf" <(tail -c 100000 "$run_dir/train_stdout.log" 2>/dev/null); then nan="maybe"; fi
  fi

  # Зависание: итерация не меняется 30+ минут
  if [ "$iter" == "$last_iter" ] && [ "$alive" == "true" ]; then
    same_iter_count=$((same_iter_count+1))
  else
    same_iter_count=0
    last_iter="$iter"
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
  "last_iteration": it, "nan_suspect": nan, "restarts": restarts,
  "run_dir": run_dir,
})
json.dump(hb, open(path, "w"), indent=2)
PYEOF

  if [ "$alive" != "true" ]; then
    # Штатное завершение (все итерации прошли) — не рестартуем
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
    touch "$LOGROOT"/run_*/.keep_training 2>/dev/null
    bash "$PROJ/tools/wsl/train.sh" "$STAGE" || true
  elif [ "$same_iter_count" -ge 3 ]; then
    echo "$now: итерация $iter не двигается 30+ мин — kill + resume" | tee -a "$LOGROOT/watchdog.log"
    kill -9 "$pid" 2>/dev/null
    same_iter_count=0
    sleep 30
    touch "$LOGROOT"/run_*/.keep_training 2>/dev/null
    bash "$PROJ/tools/wsl/train.sh" "$STAGE" || true
  fi

  sleep 600
done
