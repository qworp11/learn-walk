#!/bin/bash
# Наблюдатель тренировки s10_unified: будит агента на model_6000 (предфинал)
# или раньше, если процесс тренировки умер. Раз в 30 с, потолок 4.5 ч.
LOG=/root/hlc-logs/s10_unified/run_0
OUT=/root/hlc-logs/s10_unified_train.out
for i in $(seq 1 540); do
  sleep 30
  if ! pgrep -f "hlc_train.py" > /dev/null; then
    if [ -f "$LOG/TRAIN_DONE" ]; then
      echo "WATCH: тренировка ЗАВЕРШЕНА успешно"
    else
      echo "WATCH: ПРОЦЕСС ТРЕНИРОВКИ УМЕР РАНО! Последние строки лога:"
      tail -40 "$OUT"
    fi
    exit 0
  fi
  if [ -f "$LOG/model_6000.pt" ]; then
    echo "WATCH: model_6000.pt готов - предфинальная проверка"
    tail -8 "$OUT"
    exit 0
  fi
done
echo "WATCH: таймаут наблюдения 4.5ч - тренировка ещё идёт?"
tail -8 "$OUT"