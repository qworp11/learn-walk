#!/bin/bash
echo '=== все процессы конвейера ==='
ps aux | grep -E 'orchestr|watchdog|preview|sync_results|hlc_train|run_stage' | grep -v grep
echo '=== orchestrator_report хвост ==='
tail -12 '/mnt/g/LEARN WALK/results/orchestrator_report.txt'
echo '=== watchdog лог ==='
tail -12 /root/watchdog.out 2>/dev/null || echo 'нет файла'
echo '=== orchestrator stdout ==='
tail -12 /root/orchestrator.out 2>/dev/null || echo 'нет файла'
echo '=== heartbeat ==='
cat '/mnt/g/LEARN WALK/heartbeat.json' 2>/dev/null | head -20
echo '=== ошибки в train log ==='
grep -nE 'Traceback|CUDA|OOM|Killed|Error' /root/hlc-logs/s2_walk_run_a/run_0/train_stdout.log | tail -10
echo '=== mtime лога ==='
stat -c '%y %n' /root/hlc-logs/s2_walk_run_a/run_0/train_stdout.log
date -u '+now: %FT%TZ'
