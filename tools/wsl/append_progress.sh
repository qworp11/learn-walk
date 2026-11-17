#!/bin/bash
cd "/mnt/g/LEARN WALK"
cat >> PROGRESS.md << 'EOF'
- 2026-11-17 ~05:20 UTC - s10_unified: страховка от сбоев и лимитов агента.
  ПОЛНЫЙ resume ПРОВЕРЕН (s99_resume_test, nworld=64): рестарт БЕЗ
  HLC_WARM_START с --resume_path грузит актор+критик+оптимизатор+
  счётчик итераций ("Learning iteration 2/8" - продолжил с 2;
  чекпоинты 4/6/7 записаны). Семантика: при resume max_iterations =
  ДОПОЛНИТЕЛЬНЫЕ итерации (итого = resume_iter + max_iterations).
  => guardian_unified.sh: живёт в WSL под setsid, НЕ зависит от агента.
  Каждые 60 с зеркалит model_*.pt на backups/unified_ckpts (диск G:);
  при смерти тренировки рестартует полным resume с последнего
  чекпоинта (макс 3 попытки; доп. итераций = 8000 - latest);
  при выходе на 8000 сам пишет TRAIN_DONE и копирует финальный лог.
  Лог: /root/hlc-logs/s10_unified_guardian.out.
  ПРАВИЛО ДЛЯ АГЕНТА: при пробуждении СНАЧАЛА ps aux | grep guardian
  и лог guardian, потом действия (иначе вторая копия тренировки).
  ДИАГНОСТИКА: pgrep через PowerShell-кавычки может ВРАТЬ (кавычки
  становятся частью паттерна; pgrep -fc "[g]uardian_unified" -> 0
  при живом процессе) - верить ps aux | grep, не pgrep -fc.
  Godot-трек отложен Виталиком: game_server.py написан, UDP-связка
  Windows<->WSL проверена (прямой IP, ping->pong), ждёт отладки
  после финала s10. Конвенция yaw для стирринга: слать yaw камеры
  Godot (рад, ось +Y) как есть - совпадает с radians(azimuth) game.py.
EOF
echo OK
