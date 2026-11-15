#!/bin/bash
# Ждёт завершения run_a (TRAIN_DONE) и проверяет старт s3_run + warm-start
RUN_A=/root/hlc-logs/s2_walk_run_a/run_0
for i in $(seq 1 200); do
  if [ -f "$RUN_A/TRAIN_DONE" ]; then
    echo "RUN_A_TRAIN_DONE в $(date -u +%H:%M)Z"
    LOG3=""
    for j in $(seq 1 20); do
      LOG3=$(ls -t /root/hlc-logs/s3_run/run_*/train_stdout.log 2>/dev/null | head -1)
      [ -n "$LOG3" ] && break
      sleep 30
    done
    if [ -n "$LOG3" ]; then
      echo "--- s3_run лог: $LOG3 ---"
      sleep 120
      grep -m1 "WARM-START" "$LOG3" || echo "WARM-START строки нет"
      grep "Learning iteration" "$LOG3" | tail -1
      grep "Mean reward" "$LOG3" | tail -1
    else
      echo "s3_run не появился за 10 минут после TRAIN_DONE"
    fi
    exit 0
  fi
  sleep 60
done
echo "таймаут: TRAIN_DONE не появился за 200 минут"
