#!/bin/bash
cd "/mnt/g/LEARN WALK"
cat >> PROGRESS.md << 'EOF'
- 2026-11-15 21:11 UTC - **Третий баг (перехвачен до ущерба)**: s3_run
  (первый настоящий warm-start) падал с NameError: run_stage не создавал
  run_dir, когда у этапа нет СВОИХ чекпоинтов (resume задан, run_dir нет).
  CPU-тест warm-start проверял загрузку весов, но не скрипт запуска —
  урок: тестировать entry-point целиком. Починено (warm_start -> новый
  run_dir), s3_run перезапущен вручную + свежий watchdog; WARM-START load
  подтверждён вживую (s2_walk model_5499 -> бег), ETA ~2:36.
- 2026-11-15 21:40 UTC - run_a завершён штатно: 6000/6000, rew 26.3
  (15.5 на старте resume). Финальное видео s2_walk_run_a_final.mp4 готово.
  Стрим: JPEG q60 + энкодер-поток + polling 15мс (решение оставить как
  есть, 20 шаг/с — потолок: физика 30мс + EGL-рендер 17мс на CPU).
  Примечание: heartbeat.json замирает, пока watchdog сидит внутри
  синхронного train.sh после рестарта — не авария, помнить при диагностике.
EOF
echo "дописано"
tail -3 PROGRESS.md
