#!/bin/bash
# Санитар-наблюдатель s9_sit run_1: ранний выход при NaN / overflow /
# застое (эпизоды < 5 шагов спустя 15 мин), иначе полный отчёт через 30 мин.
LOG=/root/hlc-logs/s9_sit/run_1/train_stdout.log
for i in $(seq 1 30); do
  sleep 60
  if ! kill -0 22532 2>/dev/null && ! pgrep -f "hlc_trai[n]" >/dev/null; then
    echo "SANIT: тренировка умерла"; exit 1
  fi
  if grep -q "narrowphase overflow - please" "$LOG" 2>/dev/null; then
    echo "SANIT: narrowphase overflow есть — патч nconmax не сработал"; exit 2
  fi
  if grep -wiEq "nan" "$LOG" 2>/dev/null; then
    echo "SANIT: NaN в логе"; exit 3
  fi
  if [ "$i" -ge 15 ]; then
    EP=$(grep "Mean episode length" "$LOG" | tail -1 | grep -oE '[0-9.]+$')
    RW=$(grep "Mean reward" "$LOG" | tail -1 | grep -oE '[-0-9.]+$')
    echo "SANIT ${i}м: ep_len=$EP rew=$RW"
    if [ "$i" -ge 15 ] && [ -n "$EP" ] && \
       awk "BEGIN{exit !($EP < 5)}"; then
      echo "SANIT: эпизоды всё ещё <$EP шага — застой"; exit 4
    fi
  fi
done
echo "SANIT: 30 мин без инцидентов, финальные метрики:"
grep -E "Mean reward|Mean episode length" "$LOG" | tail -4
