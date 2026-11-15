#!/bin/bash
echo '=== train.sh / все bash скрипты конвейера ==='
ps aux | grep -E 'train\.sh|\.sh' | grep -v grep
echo '=== dmesg OOM ==='
dmesg 2>/dev/null | grep -iE 'killed process|oom' | tail -8
echo '=== RAM ==='
free -h
echo '=== top mem ==='
ps aux --sort=-%mem | head -6
