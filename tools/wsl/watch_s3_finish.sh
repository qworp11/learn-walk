#!/bin/bash
# Ждёт TRAIN_DONE у s3_run, репортит финал, затем проверяет старт s3_sprint
for i in $(seq 1 90); do
  [ -f /root/hlc-logs/s3_run/run_0/TRAIN_DONE ] && break
  sleep 60
done
LOG=$(ls -t /root/hlc-logs/s3_run/run_*/train_stdout.log 2>/dev/null | head -1)
echo "=== s3_run ФИНИШ ($(date -u +%H:%M)Z) ==="
grep "Mean reward" "$LOG" | tail -4
echo "nan в логе: $(grep -ac 'nan' "$LOG")"
sleep 180
echo "=== переход к s3_sprint ==="
SLOG=$(ls -t /root/hlc-logs/s3_sprint/run_*/train_stdout.log 2>/dev/null | head -1)
if [ -n "$SLOG" ]; then
  grep -m1 "WARM-START" "$SLOG" || echo "warm-start строка ещё не появилась"
  grep "Mean reward" "$SLOG" | tail -2
else
  echo "s3_sprint лог ещё не появился (оркестратор рендерит финальное видео)"
fi
tail -3 "/mnt/g/LEARN WALK/results/orchestrator_report.txt"
