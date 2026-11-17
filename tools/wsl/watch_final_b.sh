#!/bin/bash
# Финальный будильник s10_unified_b: разбудить агента на TRAIN_DONE
# или при смерти процесса. Раз в 30 с, потолок 40 мин.
LOG=/root/hlc-logs/s10_unified_b/run_0
OUT=/root/hlc-logs/s10_unified_b_train.out
for i in $(seq 1 80); do
  sleep 30
  if [ -f "$LOG/TRAIN_DONE" ]; then
    echo "WATCH_FINAL: коррекция ЗАВЕРШЕНА (финал готов)"
    tail -6 "$OUT"
    exit 0
  fi
  if ! pgrep -f "s10_unified_b/run_0" > /dev/null; then
    echo "WATCH_FINAL: ПРОЦЕСС УМЕР (смотри guardian_b лог)"
    tail -6 "$OUT"
    tail -5 /root/hlc-logs/s10_unified_b_guardian.out
    exit 0
  fi
done
echo "WATCH_FINAL: таймаут 40 мин - что-то не так"
tail -6 "$OUT"
