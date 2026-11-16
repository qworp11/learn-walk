#!/bin/bash
# Качество трекинга s9_sit run_2: последний полный блок метрик
LOG=/root/hlc-logs/s9_sit/run_2/train_stdout.log
awk '/Iteration time/{n++} {if (n>=1) print}' "$LOG" | tail -60 | \
  grep -E 'error_anchor_pos|error_anchor_rot|error_body_pos|error_body_rot|error_joint_pos|error_joint_vel|error_body_ang_vel|error_body_lin_vel|Mean reward|Mean episode|Iteration time|Time elapsed|ETA' | tail -20
echo "--- самопересечения и лимиты ---"
tail -60 "$LOG" | grep -E 'self_collisions|joint_limit' | tail -4
