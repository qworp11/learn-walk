#!/bin/bash
# Ждёт TRAIN_DONE этапа $1, репортит финал; если задан $2 — проверяет его старт
# $3 — максимум минут ожидания (по умолчанию 720 = 12 часов)
STAGE="$1"; NEXT="$2"; MAXW=${3:-720}
for i in $(seq 1 $((MAXW * 60))); do
  ls /root/hlc-logs/$STAGE/run_*/TRAIN_DONE >/dev/null 2>&1 && break
  sleep 60
done
RUN=$(ls -d /root/hlc-logs/$STAGE/run_* 2>/dev/null | sort -V | tail -1)
LOG="$RUN/train_stdout.log"
echo "=== $STAGE ФИНИШ ($(date -u +%H:%M)Z) ==="
grep "Learning iteration" "$LOG" | tail -1
grep "Mean reward" "$LOG" | tail -4
echo "nan в логе: $(grep -ac 'nan' "$LOG")"
grep "Mean episode length" "$LOG" | tail -1
sleep 180
if [ -n "$NEXT" ] && [ "$NEXT" != "done" ]; then
  echo "=== переход к $NEXT ==="
  NLOG=$(ls -t /root/hlc-logs/$NEXT/run_*/train_stdout.log 2>/dev/null | head -1)
  if [ -n "$NLOG" ]; then
    grep -m1 "WARM-START" "$NLOG" || echo "warm-start строки нет (для s6_fall это норма)"
    grep "Mean reward" "$NLOG" | tail -2
  else
    echo "$NEXT лог ещё не появился (оркестратор рендерит финальное видео)"
  fi
else
  echo "=== ЛЕСТНИЦА ЗАВЕРШЕНА ==="
  ls "/mnt/g/LEARN WALK/results/videos/final/" 2>/dev/null | tail -10
fi
tail -4 "/mnt/g/LEARN WALK/results/orchestrator_report.txt"
