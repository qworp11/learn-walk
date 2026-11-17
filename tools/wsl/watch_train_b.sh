#!/bin/bash
# Наблюдатель коррекции s10_unified_b: будит агента на model_10500
# (предфинал) или при смерти процесса. Раз в 30 с, потолок 4.5 ч.
LOG=/root/hlc-logs/s10_unified_b/run_0
OUT=/root/hlc-logs/s10_unified_b_train.out
for i in $(seq 1 540); do
  sleep 30
  if [ -f "$LOG/TRAIN_DONE" ]; then
    echo "WATCH_B: коррекция ЗАВЕРШЕНА успешно"
    tail -8 "$OUT"
    exit 0
  fi
  if ! pgrep -f "s10_unified_b/run_0" > /dev/null; then
    echo "WATCH_B: ПРОЦЕСС УМЕР (guardian_b мог перезапустить - проверь его лог)"
    tail -8 "$OUT"
    tail -5 /root/hlc-logs/s10_unified_b_guardian.out
    exit 0
  fi
  if [ -f "$LOG/model_10500.pt" ]; then
    echo "WATCH_B: model_10500.pt готов - предфинальная проверка"
    tail -8 "$OUT"
    exit 0
  fi
done
echo "WATCH_B: таймаут 4.5ч - тренировка ещё идёт?"
tail -8 "$OUT"
