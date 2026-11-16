#!/bin/bash
# Статус s9_sit run_2: жив ли процесс, TRAIN_DONE, финальные метрики, чекпоинты
echo "=== процесс ==="
ps aux | grep -E "hlc_trai[n]" | grep -v grep || echo "процесса нет (завершён?)"
echo "=== run_2 содержимое ==="
ls /root/hlc-logs/s9_sit/run_2/ | tail -12
echo "=== TRAIN_DONE? ==="
ls /root/hlc-logs/s9_sit/run_2/TRAIN_DONE 2>/dev/null && echo "ЕСТЬ" || echo "нет"
echo "=== финальные метрики ==="
grep -E "Mean reward|Mean episode length" /root/hlc-logs/s9_sit/run_2/train_stdout.log | tail -4
grep -E "error_anchor_pos|error_body_pos|error_anchor_rot" /root/hlc-logs/s9_sit/run_2/train_stdout.log | tail -3
grep -E "Time elapsed|ETA" /root/hlc-logs/s9_sit/run_2/train_stdout.log | tail -2
echo "=== инциденты ==="
grep -c "narrowphase overflow - please" /root/hlc-logs/s9_sit/run_2/train_stdout.log
grep -wiEc "nan" /root/hlc-logs/s9_sit/run_2/train_stdout.log
