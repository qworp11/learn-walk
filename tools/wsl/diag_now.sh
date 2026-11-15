#!/bin/bash
# Экстренная диагностика: жива ли тренировка, где она застряла
echo "=== время ==="
date -u
echo "=== процессы ==="
pgrep -af "run_stage|hlc_train|watchdog.sh|train.sh" | grep -v pgrep
echo "=== хвост тренировочного лога run_a ==="
tail -6 /root/hlc-logs/s2_walk_run_a/run_0/train_stdout.log
echo "=== чекпоинты run_a (последние) ==="
ls -t /root/hlc-logs/s2_walk_run_a/run_0/ | head -6
echo "=== watchdog.log (хвост) ==="
tail -8 /root/hlc-logs/s2_walk_run_a/watchdog.log
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
