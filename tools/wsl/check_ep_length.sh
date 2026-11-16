#!/bin/bash
# Блок метрик последней итерации: длина эпизодов + ошибки
LOG=$(ls -t /root/hlc-logs/s3_run/run_*/train_stdout.log 2>/dev/null | head -1)
echo "лог: $LOG"
tail -80 "$LOG" | grep -iE "length" | tail -3
echo "--- ошибки ---"
tail -80 "$LOG" | grep -iE "error" | tail -5
