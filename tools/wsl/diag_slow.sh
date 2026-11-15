#!/bin/bash
# диагностика замедления тренировки
echo '=== процессы (CPU%) ==='
top -bn1 | grep -E 'python|PID' | head -8
echo '=== load ==='
uptime
echo '=== хвост лога run_a ==='
tail -25 /root/hlc-logs/s2_walk_run_a/run_0/train_stdout.log
echo '=== последние номера итераций ==='
grep -oE 'Learning iteration [0-9]+' /root/hlc-logs/s2_walk_run_a/run_0/train_stdout.log | tail -5
echo '=== GPU ==='
nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
echo '=== nvidia-smi pmon (5 с) ==='
timeout 6 nvidia-smi pmon -c 3 2>/dev/null | head -15
