#!/bin/bash
echo "=== run_a TRAIN_DONE ==="
ls /root/hlc-logs/s2_walk_run_a/run_0/TRAIN_DONE 2>/dev/null && echo "ЕСТЬ — run_a завершён" || echo "ещё нет"
echo "=== последний чекпоинт run_a ==="
ls -t /root/hlc-logs/s2_walk_run_a/run_0/model_*.pt | head -2
echo "=== оркестратор (хвост отчёта) ==="
tail -4 "/mnt/g/LEARN WALK/results/orchestrator_report.txt"
echo "=== финальное видео run_a ==="
tail -3 /root/final_video_s2_walk_run_a.out 2>/dev/null || echo "ещё не началось"
