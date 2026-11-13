#!/bin/bash
# Сравнение содержимого CSV разных субъектов + fps в NPZ
source /root/hlc-venv/bin/activate
python - <<'EOF'
import hashlib
import numpy as np

# 1) Совпадает ли контент dance1_subject1.csv и dance1_subject2.csv?
def h(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]

base = "/mnt/g/LEARN WALK/data/motions/lafan1_g1/"
pairs = [("dance1_subject1.csv", "dance1_subject2.csv"),
         ("dance1_subject1.csv", "dance1_subject3.csv"),
         ("run1_subject2.csv", "run1_subject5.csv")]
for a, b in pairs:
    print(f"{a} vs {b}: md5 {h(base+a)} == {h(base+b)} -> {h(base+a)==h(base+b)}")

# 2) отличаются ли данные (не заголовок): первые 10 строк данных
import csv
rows_a = list(csv.reader(open(base + "dance1_subject1.csv")))[1:11]
rows_b = list(csv.reader(open(base + "dance1_subject2.csv")))[1:11]
print("first data rows equal:", rows_a == rows_b)

# 3) fps в NPZ
for name in ["dance1_subject1", "walk1_subject1", "run1_subject2"]:
    z = np.load(f"/mnt/g/LEARN WALK/data/motions/npz_g1/{name}.npz")
    print(name, "fps:", float(z["fps"]), "frames:", z["joint_pos"].shape[0])
EOF
