#!/bin/bash
# Проверка перехода run_a -> s3_run
echo "=== процессы (тренировка/оркестратор/стрим) ==="
pgrep -af "run_stage|stage_orchestrator|live_stream" | head -6
echo
echo "=== этапы в /root/hlc-logs ==="
ls /root/hlc-logs/
echo
for st in s2_walk_run_a s3_run; do
  d="/root/hlc-logs/$st"
  echo "=== $st ==="
  ls "$d" 2>/dev/null || { echo "нет каталога"; echo; continue; }
  for run in "$d"/run_*; do
    [ -d "$run" ] || continue
    echo "--- $(basename "$run") ---"
    if [ -f "$run/TRAIN_DONE" ]; then echo "TRAIN_DONE: ЕСТЬ"; else echo "TRAIN_DONE: нет"; fi
    log="$run/train_stdout.log"
    if [ -f "$log" ]; then
      grep -m1 "WARM-START" "$log"
      grep "Learning iteration" "$log" | tail -1
      grep "Mean reward" "$log" | tail -1
      echo "хвост лога:"
      tail -2 "$log"
    else
      echo "нет train_stdout.log"
    fi
  done
  echo
done
echo "=== отчёт оркестратора (хвост) ==="
tail -6 "/mnt/g/LEARN WALK/results/orchestrator_report.txt" 2>/dev/null || echo "нет отчёта"
