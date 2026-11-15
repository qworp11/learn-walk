#!/bin/bash
echo '=== watchdog.log run_a ==='
cat /root/hlc-logs/s2_walk_run_a/watchdog.log 2>/dev/null || echo 'нет'
echo '=== watchdog stdout (рестарты) ==='
cat /root/watchdog_s2_walk_run_a.out 2>/dev/null | head -40 || echo 'нет'
echo '=== heartbeat mtime ==='
stat -c '%y' '/mnt/g/LEARN WALK/heartbeat.json'
date -u '+now: %FT%TZ'
echo '=== watchdog жив? ==='
ps aux | grep 'watchdog.sh' | grep -v grep
