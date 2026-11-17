#!/bin/bash
# Проверка ПОЛНОГО resume (БЕЗ HLC_WARM_START): микро-прогон 3 итерации,
# затем рестарт с resume_path - убеждаемся, что загружается всё и счётчик
# итераций продолжается. Одноразовый тест, гонится рядом с тренировкой.
source /root/hlc-venv/bin/activate
cd /root
T=/root/hlc-logs/s99_resume_test
rm -rf "$T"; mkdir -p "$T/run_0"
echo "=== ПРОГОН 1: свежий, до 3 итераций ==="
python "/mnt/g/LEARN WALK/src/hlc/hlc_train.py" \
  --env G1WholeBodyTracking --seed 0 \
  --logdir "$T/run_0" \
  --env_overrides '{"trajectory_path": "/mnt/g/LEARN WALK/data/motions/npz_g1/unified_all.npz"}' \
  --overrides '{"nworld": 64, "max_iterations": 3, "save_interval": 2}' \
  --wp_kernel_cache_dir /root/wp_kernel_cache \
  > "$T/run1.out" 2>&1
echo "run1 exit=$?"
ls "$T/run_0" | grep model_ || echo "НЕТ ЧЕКПОИНТОВ в прогоне 1"
echo "=== ПРОГОН 2: полный resume с model_2.pt, до 6 итераций ==="
python "/mnt/g/LEARN WALK/src/hlc/hlc_train.py" \
  --env G1WholeBodyTracking --seed 0 \
  --logdir "$T/run_0" \
  --env_overrides '{"trajectory_path": "/mnt/g/LEARN WALK/data/motions/npz_g1/unified_all.npz"}' \
  --overrides '{"nworld": 64, "max_iterations": 6, "save_interval": 2}' \
  --wp_kernel_cache_dir /root/wp_kernel_cache \
  --resume_path "$T/run_0/model_2.pt" \
  > "$T/run2.out" 2>&1
echo "run2 exit=$?"
grep -iE "load|resum|warm|iteration" "$T/run2.out" | head -15
ls "$T/run_0" | grep model_ | sort
echo "=== СТРОКИ СЧЁТЧИКА ИТЕРАЦИЙ из run2 ==="
grep -E "Learning iteration" "$T/run2.out" | head -8
echo "RESUME_TEST_DONE"
