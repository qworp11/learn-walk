#!/bin/bash
# Оркестратор лестницы этапов: обучает этапы по порядку из stages.json.
# На каждом этапе: watchdog + превью-воркер → ждёт TRAIN_DONE → финальное видео
# → следующий этап. Проваленный этап (5 рестартов) помечает FAILED и идёт дальше.
# Usage: stage_orchestrator.sh <stage1> <stage2> ...
set -u
source /root/hlc-venv/bin/activate
PROJ="/mnt/g/LEARN WALK"
REPORT="$PROJ/results/orchestrator_report.txt"
MAX_WAIT_MINS=600  # предохранитель на этап (10 ч)

echo "=== ОРКЕСТРАТОР старт $(date -u +%FT%TZ) ===" >> "$REPORT"
echo "этапы: $*" >> "$REPORT"

# ждём освобождения GPU (не стартовать следующий этап поверх текущего)
for i in $(seq 1 30); do
  pgrep -f "hlc_train.py" >/dev/null || break
  sleep 60
done

for STAGE in "$@"; do
  echo "[$(date -u +%H:%M)] --- этап $STAGE ---" >> "$REPORT"

  # конфиг этапа
  MOTION=$(python "/mnt/g/LEARN WALK/tools/hlc/stage_motion.py" "$STAGE")
  if [ -z "$MOTION" ]; then
    echo "$STAGE: нет в stages.json — СКИП" >> "$REPORT"; continue
  fi

  # этап уже завершён ранее? — только финальное видео
  if ls /root/hlc-logs/"$STAGE"/run_*/TRAIN_DONE >/dev/null 2>&1; then
    echo "$STAGE: TRAIN_DONE уже есть — только финальное видео" >> "$REPORT"
    bash "$PROJ/tools/wsl/final_video.sh" "$STAGE" "$MOTION" \
      > "/root/final_video_$STAGE.out" 2>&1
    echo "$STAGE: финальное видео готово" >> "$REPORT"
    continue
  fi

  # пускаем watchdog (рестарт при краше/зависании) и превью-воркер
  nohup bash "$PROJ/tools/wsl/watchdog.sh" "$STAGE" 5 \
    > "/root/watchdog_$STAGE.out" 2>&1 &
  nohup bash "$PROJ/tools/wsl/preview_worker.sh" "$STAGE" "$MOTION" \
    > "/root/preview_worker_$STAGE.out" 2>&1 &

  # тренировка (run_stage сам резюмит последний чекпоинт)
  bash "$PROJ/tools/wsl/train.sh" "$STAGE" &
  TRAIN_PID=$!

  # ждём TRAIN_DONE / израсходованные рестарты / таймаут
  DONE=0
  for i in $(seq 1 "$MAX_WAIT_MINS"); do
    if ls /root/hlc-logs/"$STAGE"/run_*/TRAIN_DONE >/dev/null 2>&1; then
      DONE=1; break
    fi
    if grep -q "ПЕРЕСТАНОВКИ ИЗЛОЖИЛИСЬ" "/root/hlc-logs/$STAGE/watchdog.log" 2>/dev/null; then
      break
    fi
    if ! kill -0 "$TRAIN_PID" 2>/dev/null && ! pgrep -f "watchdog.sh $STAGE" >/dev/null; then
      # ни тренировки, ни watchdogа — а TRAIN_DONE нет: что-то сломалось
      break
    fi
    sleep 60
  done

  if [ "$DONE" == "1" ]; then
    echo "$STAGE: TRAIN_DONE ✓" >> "$REPORT"
    # финальное видео этапа (роллаут + сравнение с референсом)
    bash "$PROJ/tools/wsl/final_video.sh" "$STAGE" "$MOTION" \
      > "/root/final_video_$STAGE.out" 2>&1
    echo "$STAGE: финальное видео готово" >> "$REPORT"
  else
    echo "$STAGE: НЕ ЗАВЕРШЁН штатно — FAILED (см. /root/hlc-logs/$STAGE/watchdog.log)" >> "$REPORT"
    for p in $(pgrep -f "watchdog.sh $STAGE" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
    for p in $(pgrep -f "preview_worker.sh $STAGE" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
    for p in $(pgrep -f "hlc_train.py" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
    for p in $(pgrep -f "run_stage.py $STAGE" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  fi
done

echo "=== ОРКЕСТРАТОР конец $(date -u +%FT%TZ) ===" >> "$REPORT"
