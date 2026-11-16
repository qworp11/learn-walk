#!/bin/bash
# Ждёт TRAIN_DONE у s3_sprint, репортит финал, затем проверяет старт s4_jump
for i in $(seq 1 240); do
  [ -f /root/hlc-logs/s3_sprint/run_0/TRAIN_DONE ] && break
  sleep 60
done
LOG=$(ls -t /root/hlc-logs/s3_sprint/run_*/train_stdout.log 2>/dev/null | head -1)
echo "=== s3_sprint ФИНИШ ($(date -u +%H:%M)Z) ==="
grep "Mean reward" "$LOG" | tail -4
echo "nan в логе: $(grep -ac 'nan' "$LOG")"
grep "Mean episode length" "$LOG" | tail -1
sleep 180
echo "=== переход к s4_jump ==="
JLOG=$(ls -t /root/hlc-logs/s4_jump/run_*/train_stdout.log 2>/dev/null | head -1)
if [ -n "$JLOG" ]; then
  grep -m1 "WARM-START" "$JLOG" || echo "warm-start строка ещё не появилась"
  grep "Mean reward" "$JLOG" | tail -2
else
  echo "s4_jump лог ещё не появился (оркестратор рендерит финальное видео)"
fi
tail -3 "/mnt/g/LEARN WALK/results/orchestrator_report.txt"
