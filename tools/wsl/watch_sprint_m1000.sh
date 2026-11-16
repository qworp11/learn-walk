#!/bin/bash
# Ждёт 1000-й итерации s3_sprint и репортит: награда, NaN, GPU
LOG=""
for i in $(seq 1 120); do
  LOG=$(ls -t /root/hlc-logs/s3_sprint/run_*/train_stdout.log 2>/dev/null | head -1)
  [ -n "$LOG" ] && grep -q "Learning iteration 1000/6000" "$LOG" && break
  sleep 60
done
if [ -z "$LOG" ]; then echo "s3_sprint лог так и не появился"; exit 1; fi
echo "=== спринт: 1000-я итерация ($(date -u +%H:%M)Z) ==="
grep "Learning iteration" "$LOG" | tail -1
grep "Mean reward" "$LOG" | tail -5
echo "nan в логе: $(grep -ac 'nan' "$LOG")"
grep "Mean episode length" "$LOG" | tail -1
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
