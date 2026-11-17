#!/bin/bash
# Единая модель: 8 клипов в unified_all.npz, warm-start актёра от s3_run.
source /root/hlc-venv/bin/activate
export HLC_WARM_START=1
LOGDIR=/root/hlc-logs/s10_unified/run_0
mkdir -p "$LOGDIR"
cat > "$LOGDIR/cmd.json" <<'EOF'
{
  "stage": "s10_unified",
  "cmd": ["hlc_train.py", "--env", "G1WholeBodyTracking", "--seed", "0",
          "--env_overrides", {"trajectory_path":
            "/mnt/g/LEARN WALK/data/motions/npz_g1/unified_all.npz"},
          "--overrides", {"nworld": 1024, "max_iterations": 8000},
          "--resume_path", "/root/hlc-logs/s3_run/run_0/model_5999.pt"],
  "warm_start": true,
  "note": "единая модель на 8 клипах (stand/walk/run/sprint/jump/fall/dance/sit x20)"
}
EOF
cd /root
python "/mnt/g/LEARN WALK/src/hlc/hlc_train.py" \
  --env G1WholeBodyTracking --seed 0 \
  --logdir "$LOGDIR" \
  --env_overrides '{"trajectory_path": "/mnt/g/LEARN WALK/data/motions/npz_g1/unified_all.npz"}' \
  --overrides '{"nworld": 1024, "max_iterations": 8000}' \
  --wp_kernel_cache_dir /root/wp_kernel_cache \
  --resume_path /root/hlc-logs/s3_run/run_0/model_5999.pt \
  > /root/hlc-logs/s10_unified_train.out 2>&1
RC=$?
echo "TRAIN_EXIT=$RC"
if [ "$RC" = "0" ]; then
  cp /root/hlc-logs/s10_unified_train.out "$LOGDIR/train_stdout.log"
  touch "$LOGDIR/TRAIN_DONE"
fi
