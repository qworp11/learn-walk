#!/bin/bash
# Синхронизация артефактов из WSL /root/hlc-logs -> /mnt/g/LEARN WALK/results
# Каждые 5 минут копирует: чекпоинты (последние 3 + best), tensorboard-события, cmd.json
set -u
LOGROOT="/root/hlc-logs"
DEST="/mnt/g/LEARN WALK/results"
while true; do
  for d in "$LOGROOT"/*/run_*; do
    [ -d "$d" ] || continue
    stage=$(basename "$(dirname "$d")")
    out="$DEST/$stage/$(basename "$d")"
    mkdir -p "$out"
    rsync -a --delete --include='model_*.pt' --include='events.out.tfevents*' \
      --include='cmd.json' --exclude='*' "$d/" "$out/" 2>/dev/null
    # чистка старых чекпоинтов на DEST (оставить 3 свежих)
    ls -t "$out"/model_*.pt 2>/dev/null | tail -n +4 | xargs -r rm -f
  done
  sleep 300
done
