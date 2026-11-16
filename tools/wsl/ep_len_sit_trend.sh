#!/bin/bash
# Тренд длины эпизодов s9_sit run_1: первые 5, середина, последние 5
grep "Mean episode length" /root/hlc-logs/s9_sit/run_1/train_stdout.log \
  | sed 's/.*: //' > /tmp/ep_len_sit.txt
N=$(wc -l < /tmp/ep_len_sit.txt)
echo "всего точек: $N"
echo "первые 5:  $(head -5 /tmp/ep_len_sit.txt | tr '\n' ' ')"
echo "середина:  $(sed -n "$((N/2-2)),$((N/2+2))p" /tmp/ep_len_sit.txt | tr '\n' ' ')"
echo "последние: $(tail -5 /tmp/ep_len_sit.txt | tr '\n' ' ')"
