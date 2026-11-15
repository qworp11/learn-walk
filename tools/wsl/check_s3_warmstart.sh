#!/bin/bash
LOG=$(ls -t /root/hlc-logs/s3_run/run_*/train_stdout.log 2>/dev/null | head -1)
if [ -z "$LOG" ]; then
  echo "s3_run лог ещё не появился"
  exit 0
fi
echo "лог: $LOG"
echo "=== warm-start ==="
grep -m1 "WARM-START" "$LOG" || echo "строки WARM-START пока нет"
echo "=== первые итерации и награда ==="
grep "Learning iteration" "$LOG" | tail -2
grep "Mean reward" "$LOG" | tail -3
echo "=== хвост ==="
tail -3 "$LOG"
