#!/bin/bash
# Конвертация LAFAN1-G1 CSV -> NPZ для трекинг-таска
set -e
source /root/hlc-venv/bin/activate
DATA=/mnt/g/LEARN\ WALK/data
OUT=$DATA/motions/npz_g1
mkdir -p "$OUT"
cd "$DATA/motions/lafan1_g1"
for f in *.csv; do
  base="${f%.csv}"
  if [ -f "$OUT/$base.npz" ]; then echo "SKIP $base"; continue; fi
  echo "CONVERT $base"
  python -m mujoco_playground.tasks.whole_body.tracking.csv_to_npz \
    --input_csv="$DATA/motions/lafan1_g1/$f" \
    --output_npz="$OUT/$base.npz" || echo "FAIL $base"
done
echo "CONVERT_DONE $(ls "$OUT" | wc -l) npz files"
