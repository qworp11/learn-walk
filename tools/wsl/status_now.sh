#!/bin/bash
echo "время: $(date -u +%H:%M)Z"
echo "=== s3_run ==="
[ -f /root/hlc-logs/s3_run/run_0/TRAIN_DONE ] && echo "TRAIN_DONE есть" || echo "TRAIN_DONE нет"
LOG=$(ls -t /root/hlc-logs/s3_run/run_*/train_stdout.log 2>/dev/null | head -1)
grep "Learning iteration" "$LOG" | tail -1
grep "Mean reward" "$LOG" | tail -3
echo "=== s3_sprint ==="
SLOG=$(ls -t /root/hlc-logs/s3_sprint/run_*/train_stdout.log 2>/dev/null | head -1)
if [ -n "$SLOG" ]; then
  echo "лог: $SLOG"
  grep -m1 "WARM-START" "$SLOG" || echo "warm-start строка ещё не появилась"
  grep "Mean reward" "$SLOG" | tail -2
else
  echo "лога ещё нет (оркестратор рендерит финальное видео бега)"
fi
echo "=== оркестратор (хвост) ==="
tail -4 "/mnt/g/LEARN WALK/results/orchestrator_report.txt"
