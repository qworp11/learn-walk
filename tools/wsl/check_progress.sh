#!/bin/bash
# Сводка прогресса тренировки: usage: check_progress.sh <stage> [run_N]
STAGE="${1:-s1_balance}"
LOGROOT="/root/hlc-logs/$STAGE"
if [ -n "$2" ]; then
  L="$LOGROOT/$2/train_stdout.log"
else
  L=$(ls -t "$LOGROOT"/run_*/train_stdout.log 2>/dev/null | head -1)
fi
echo "log: $L"
grep -aoE 'Total reward: [-0-9.e]+' "$L" | tail -5
echo === anchor error ===
grep -aoE 'error_anchor_pos: [0-9.e]+' "$L" | tail -3
grep -aoE 'error_anchor_rot: [0-9.e]+' "$L" | tail -3
echo === timing ===
grep -aoE 'Iteration time: [0-9.]+' "$L" | tail -2
grep -aoE 'Time elapsed: [0-9:]+' "$L" | tail -1
grep -aoE 'ETA: [0-9:]+' "$L" | tail -1
echo === nan count ===
grep -acE 'NaN|nan' "$L" || true
echo === reward trend ===
grep -aoE 'Mean reward: [-0-9.e]+' "$L" | cut -d' ' -f3 | awk 'NR%25==1' | tr '\n' ' '
echo
echo === ep len trend ===
grep -aoE 'Mean episode length: [0-9]+' "$L" | cut -d' ' -f4 | awk 'NR%25==1' | tr '\n' ' '
echo
