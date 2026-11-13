#!/bin/bash
# Сверка числа кадров CSV vs NPZ (детект дублей/склейки)
cd "/mnt/g/LEARN WALK/data/motions/lafan1_g1"
source /root/hlc-venv/bin/activate
python - <<'EOF'
import glob, os
import numpy as np
import csv
for p in sorted(glob.glob("*.csv"))[:8]:
    base = p[:-4]
    with open(p) as f:
        r = csv.reader(f)
        hdr = next(r)
        n = sum(1 for _ in r)
    z = np.load(f"/mnt/g/LEARN WALK/data/motions/npz_g1/{base}.npz")
    key = "qpos" if "qpos" in z.files else "joint_pos"
    print(f"{base:34s} csv_rows={n:6d} npz_frames={z[key].shape[0]:6d} cols={len(hdr)}")
EOF
