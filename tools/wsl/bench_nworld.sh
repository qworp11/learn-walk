#!/bin/bash
# Бенчмарк масштабирования nworld на 1080 Ti
source /root/hlc-venv/bin/activate
for N in 512 1024 2048 3072 4096; do
  echo "=== nworld=$N ==="
  timeout 900 python '/mnt/g/LEARN WALK/src/hlc/s0_smoke.py' --nworld $N --steps 200 2>&1 | grep -E "steps/s|peak mem|finite|SMOKE_OK|Error|error"
  nvidia-smi --query-gpu=memory.used,temperature.gpu,utilization.gpu --format=csv,noheader
done
