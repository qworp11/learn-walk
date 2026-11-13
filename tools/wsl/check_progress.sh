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
grep -oE 'Total reward: [-0-9.e]+' "$L" | tail -5
echo === anchor error ===
grep -oE 'error_anchor_pos: [0-9.e]+' "$L" | tail -3
grep -oE 'error_anchor_rot: [0-9.e]+' "$L" | tail -3
echo === timing ===
grep -oE 'Iteration time: [0-9.]+' "$L" | tail -2
grep -oE 'Time elapsed: [0-9:]+' "$L" | tail -1
grep -oE 'ETA: [0-9:]+' "$L" | tail -1
echo === nan count ===
grep -cE 'NaN|nan' "$L" || true
