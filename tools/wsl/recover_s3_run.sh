#!/bin/bash
# Восстановление s3_run после фикса run_stage.py:
# старый watchdog (backoff на битом коде) → стоп; фиксированный train + свежий watchdog
echo "старые процессы s3_run:"
for p in $(pgrep -f "watchdog.sh s3_run"); do kill -9 "$p" && echo "killed watchdog $p"; done
for p in $(pgrep -f "train.sh s3_run"); do kill -9 "$p" 2>/dev/null && echo "killed train $p"; done
for p in $(pgrep -f "run_stage.py s3_run"); do kill -9 "$p" 2>/dev/null && echo "killed run_stage $p"; done
for p in $(pgrep -f "hlc_train.py"); do kill -9 "$p" 2>/dev/null && echo "killed hlc_train $p"; done
sleep 2
source /root/hlc-venv/bin/activate
cd "/mnt/g/LEARN WALK"
nohup bash tools/wsl/train.sh s3_run > /root/train_s3_run_launch.out 2>&1 &
echo "train.sh запущен: $!"
nohup bash tools/wsl/watchdog.sh s3_run 5 > /root/watchdog_s3_run.out 2>&1 &
echo "watchdog запущен: $!"
sleep 45
echo "=== лог s3_run ==="
LOG=$(ls -t /root/hlc-logs/s3_run/run_*/train_stdout.log 2>/dev/null | head -1)
echo "лог: $LOG"
[ -n "$LOG" ] && grep -m2 -E "WARM-START|RESUME" "$LOG"
[ -n "$LOG" ] && tail -4 "$LOG"
