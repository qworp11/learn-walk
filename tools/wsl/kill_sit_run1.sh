#!/bin/bash
# Глушим s9_sit run_1 (попытка 2) и всех наблюдателей по PID, консервируем чекпоинты
for pat in "hlc_trai[n]" "run_stage.py s9_si[t]" "train_sit.s[h]" \
           "watch_sit_healt[h]" "watch_stage_finish.sh s9_si[t]"; do
  for p in $(pgrep -f "$pat"); do
    echo "kill $p ($pat)"
    kill "$p" 2>/dev/null
  done
done
sleep 3
echo "--- живые: ---"
ps aux | grep -E "hlc_trai[n]|run_stag[e]|watch_sit|watch_stage" | grep -v grep || echo "нет"
cd /root/hlc-logs/s9_sit/run_1
for f in model_*.pt; do
  [ -e "$f" ] && mv "$f" "$f.stuck" && echo "консервирован $f"
done
ls
