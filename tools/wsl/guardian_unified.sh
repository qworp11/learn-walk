#!/bin/bash
# АВТОХРАНИТЕЛЬ тренировки s10_unified. Живёт внутри WSL (setsid), работает
# БЕЗ агента: 1) зеркалит чекпоинты на диск G: каждые 60 с; 2) если тренировка
# умерла и нет TRAIN_DONE - рестартует ПОЛНЫМ resume с последнего чекпоинта
# (макс 3 попытки; max_iterations = 8000 - latest, т.к. при resume это
# ДОПОЛНИТЕЛЬНЫЕ итерации - проверено микро-тестом s99_resume_test);
# 3) если рестартованный процесс дошёл до 8000 - сам пишет TRAIN_DONE.
# Лог: /root/hlc-logs/s10_unified_guardian.out
# ПРАВИЛО ДЛЯ АГЕНТА: при пробуждении СНАЧАЛА читать этот лог и pgrep,
# чтобы не запустить вторую копию тренировки в тот же run-каталог.
RUN=/root/hlc-logs/s10_unified/run_0
ROOT=/root/hlc-logs/s10_unified
OUT=/root/hlc-logs/s10_unified_train.out
DST="/mnt/g/LEARN WALK/backups/unified_ckpts"
TARGET=8000
mkdir -p "$DST"
RESTARTS=0
echo "$(date '+%F %T') guardian стартует"
cp "$RUN/cmd.json" "$DST/" 2>/dev/null

finalize() {
  touch "$RUN/TRAIN_DONE"
  cp "$RUN/cmd.json" "$DST/" 2>/dev/null
  cp "$OUT" "$DST/train_stdout_final.log" 2>/dev/null
  touch "$DST/TRAIN_DONE"
  echo "$(date '+%F %T') ФИНАЛ: TRAIN_DONE + лог скопированы на G:, выхожу"
  exit 0
}

while true; do
  # --- 1) финал: DONE - копируем лог и выходим ---
  if [ -f "$RUN/TRAIN_DONE" ]; then
    if [ ! -f "$DST/TRAIN_DONE" ]; then
      cp "$RUN/cmd.json" "$DST/" 2>/dev/null
      cp "$OUT" "$DST/train_stdout_final.log" 2>/dev/null
      touch "$DST/TRAIN_DONE"
      echo "$(date '+%F %T') TRAIN_DONE: финал скопирован на G:, выхожу"
    fi
    exit 0
  fi
  # --- 2) зеркало чекпоинтов на G: (каждый цикл, пока живы) ---
  for f in "$RUN"/model_*.pt; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    if [ ! -e "$DST/$b" ]; then
      if cp "$f" "$DST/$b.tmp" && mv "$DST/$b.tmp" "$DST/$b"; then
        echo "$(date '+%F %T') mirror $b"
      fi
    fi
  done
  # --- 3) процесс жив? ---
  if pgrep -f "s10_unified/run_0" > /dev/null; then
    sleep 60
    continue
  fi
  # --- 4) мёртв: завершён или падение? ---
  LATEST=$(ls -1 "$RUN"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
  LATEST=${LATEST:-0}
  if [ "$LATEST" -ge $((TARGET - 1)) ]; then
    finalize   # дошёл до конца без TRAIN_DONE (рестартованный запуск)
  fi
  if [ -z "$(ls "$RUN"/model_*.pt 2>/dev/null)" ]; then
    echo "$(date '+%F %T') нет чекпоинтов для рестарта - нужен человек"
    touch "$ROOT/GUARDIAN_GAVE_UP"
    exit 1
  fi
  if [ "$RESTARTS" -ge 3 ]; then
    echo "$(date '+%F %T') СДАЮСЬ после 3 рестартов (последний model_$LATEST.pt) - нужен человек"
    touch "$ROOT/GUARDIAN_GAVE_UP"
    exit 1
  fi
  REMAIN=$((TARGET - LATEST))
  RESTARTS=$((RESTARTS+1))
  echo "$(date '+%F %T') РЕСТАРТ #$RESTARTS: полный resume с model_$LATEST.pt, доп. итераций: $REMAIN"
  source /root/hlc-venv/bin/activate
  cd /root
  setsid nohup python "/mnt/g/LEARN WALK/src/hlc/hlc_train.py" \
    --env G1WholeBodyTracking --seed 0 \
    --logdir "$RUN" \
    --env_overrides '{"trajectory_path": "/mnt/g/LEARN WALK/data/motions/npz_g1/unified_all.npz"}' \
    --overrides "{\"nworld\": 1024, \"max_iterations\": $REMAIN}" \
    --wp_kernel_cache_dir /root/wp_kernel_cache \
    --resume_path "$RUN/model_$LATEST.pt" \
    >> "$OUT" 2>&1 &
  echo $! > "$ROOT/restart_pid.txt"
  sleep 150   # дать рестарту загрузиться; живость проверим в следующем цикле
done
