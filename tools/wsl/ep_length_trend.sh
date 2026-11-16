#!/bin/bash
# Тренд длины эпизодов s3_run: первая / середина / последняя записи
LOG=$(ls -t /root/hlc-logs/s3_run/run_*/train_stdout.log 2>/dev/null | head -1)
grep "Mean episode length" "$LOG" > /tmp/ep_len.txt
N=$(wc -l < /tmp/ep_len.txt)
echo "всего записей: $N"
echo "первая (итерация ~1):    $(head -1 /tmp/ep_len.txt)"
echo "середина (итерация ~$((N / 2))):  $(sed -n "$((N / 2))p" /tmp/ep_len.txt)"
echo "последняя (итерация ~$N): $(tail -1 /tmp/ep_len.txt)"
