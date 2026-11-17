#!/bin/bash
set -e
G="/mnt/g/LEARN WALK"

# 1) скрипт тренировки-продолжения: полный resume с 7999, +3000 итераций
cat > /root/train_unified_b.sh <<'EOS'
#!/bin/bash
# Продолжение s10_unified: resume с model_7999 (БЕЗ warm-start = полный),
# +3000 итераций; цель - вернуть танец и спринт.
source /root/hlc-venv/bin/activate
LOGDIR=/root/hlc-logs/s10_unified_b/run_0
mkdir -p "$LOGDIR"
cat > "$LOGDIR/cmd.json" <<'EOF'
{"stage": "s10_unified_b",
 "cmd": "full resume from s10_unified/run_0/model_7999.pt, +3000 iters (target 11000)",
 "warm_start": false}
EOF
cd /root
python "/mnt/g/LEARN WALK/src/hlc/hlc_train.py" \
  --env G1WholeBodyTracking --seed 0 \
  --logdir "$LOGDIR" \
  --env_overrides '{"trajectory_path": "/mnt/g/LEARN WALK/data/motions/npz_g1/unified_all.npz"}' \
  --overrides '{"nworld": 1024, "max_iterations": 3000}' \
  --wp_kernel_cache_dir /root/wp_kernel_cache \
  --resume_path /root/hlc-logs/s10_unified/run_0/model_7999.pt \
  > /root/hlc-logs/s10_unified_b_train.out 2>&1
RC=$?
echo "TRAIN_EXIT=$RC"
if [ "$RC" = "0" ]; then
  cp /root/hlc-logs/s10_unified_b_train.out "$LOGDIR/train_stdout.log"
  touch "$LOGDIR/TRAIN_DONE"
fi
EOS
chmod +x /root/train_unified_b.sh

# 2) guardian_b: клон guardian_unified.sh с путями _b, зеркалом _b, целью 11000
sed -e 's|s10_unified|s10_unified_b|g' \
    -e 's|TARGET=8000|TARGET=11000|' \
    -e 's|unified_ckpts|unified_b_ckpts|g' \
    "$G/tools/wsl/guardian_unified.sh" > /root/guardian_unified_b.sh
grep -E "^RUN=|^ROOT=|^OUT=|^DST=|^TARGET=" /root/guardian_unified_b.sh

# 3) запуск обоих в отрыве от этой сессии
setsid nohup bash /root/train_unified_b.sh < /dev/null > /dev/null 2>&1 &
sleep 2
setsid nohup bash /root/guardian_unified_b.sh >> /root/hlc-logs/s10_unified_b_guardian.out 2>&1 < /dev/null &
sleep 10
echo "--- тренировка:"; pgrep -af "s10_unified_b/run_0" | head -2 || echo НЕ_НАЙДЕНА
echo "--- guardian_b лог:"; tail -3 /root/hlc-logs/s10_unified_b_guardian.out
echo SETUP_DONE
