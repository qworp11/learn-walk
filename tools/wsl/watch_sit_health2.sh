#!/bin/bash
# Санитар s9_sit run_2 (play=True, курс с кадра 0): ранний выход при
# NaN / overflow / застое; эпизоды должны расти (умирает глубже в спуске).
LOG=/root/hlc-logs/s9_sit/run_2/train_stdout.log
for i in $(seq 1 40); do
  sleep 60
  if ! pgrep -f "hlc_trai[n]" >/dev/null; then
    echo "SANIT: тренировка умерла"; exit 1
  fi
  if grep -q "narrowphase overflow - please" "$LOG" 2>/dev/null; then
    echo "SANIT: narrowphase overflow — патч nconmax не сработал"; exit 2
  fi
  if grep -wiEq "nan" "$LOG" 2>/dev/null; then
    echo "SANIT: NaN в логе"; exit 3
  fi
  if [ "$i" -ge 5 ]; then
    EP=$(grep "Mean episode length" "$LOG" | tail -1 | grep -oE '[0-9.]+$')
    RW=$(grep "Mean reward" "$LOG" | tail -1 | grep -oE '[-0-9.]+$')
    echo "SANIT ${i}м: ep_len=$EP rew=$RW"
    if [ "$i" -ge 20 ] && [ -n "$EP" ] && \
       awk "BEGIN{exit !($EP < 40)}"; then
      echo "SANIT: эпизоды <$EP за 20 мин — застой курса с кадра 0"; exit 4
    fi
  fi
done
echo "SANIT: 40 мин без инцидентов, последние метрики:"
grep -E "Mean reward|Mean episode length" "$LOG" | tail -4
