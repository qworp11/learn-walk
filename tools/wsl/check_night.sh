#!/bin/bash
# Предсонный чек: всё живое? диск, RAM, GPU, стрим, прогресс
echo "время: $(date -u +%H:%M)Z"
echo "=== процессы ==="
pgrep -af "stage_orchestrator|train.sh s3_sprint|watchdog.sh s3_sprint|live_stream.py" | grep -v pgrep
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
echo "=== диск G: ==="
df -h /mnt/g | tail -1
echo "=== RAM WSL ==="
free -m | awk 'NR==2 {print "свободно " $7 " МБ из " $2 " МБ"}'
echo "=== стрим :8081 ==="
curl -s --max-time 4 http://localhost:8081/stream 2>/dev/null | head -c 40 >/dev/null && echo "стрим жив" || echo "СТРИМ НЕ ОТВЕЧАЕТ"
echo "=== спринт сейчас ==="
LOG=$(ls -t /root/hlc-logs/s3_sprint/run_*/train_stdout.log 2>/dev/null | head -1)
grep "Learning iteration" "$LOG" | tail -1
grep "Mean reward" "$LOG" | tail -2
