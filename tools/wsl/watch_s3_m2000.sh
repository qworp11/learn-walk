#!/bin/bash
# Ждёт 2000-й итерации s3_run и репортит: награда, NaN, ошибки трекинга, GPU
LOG=""
for i in $(seq 1 120); do
  LOG=$(ls -t /root/hlc-logs/s3_run/run_*/train_stdout.log 2>/dev/null | head -1)
  [ -n "$LOG" ] && grep -q "Learning iteration 2000/6000" "$LOG" && break
  sleep 60
done
if [ -z "$LOG" ]; then echo "s3_run лог так и не появился"; exit 1; fi
echo "=== 2000-я итерация достигнута ($(date -u +%H:%M)Z) ==="
grep "Learning iteration" "$LOG" | tail -1
grep "Mean reward" "$LOG" | tail -6
echo "nan в логе: $(grep -ac 'nan' "$LOG")"
grep -E "error_anchor_pos|error_anchor_rot" "$LOG" | tail -2
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
