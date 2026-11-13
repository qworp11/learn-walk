"""fps и соотношение кадров CSV/NPZ."""
import csv
import glob

import numpy as np

for name in ["dance1_subject1", "walk1_subject1", "run1_subject2"]:
    z = np.load(f"/mnt/g/LEARN WALK/data/motions/npz_g1/{name}.npz")
    print(name, "fps:", np.asarray(z["fps"]).ravel(), "frames:", z["joint_pos"].shape[0])

csvp = "/mnt/g/LEARN WALK/data/motions/lafan1_g1/dance1_subject1.csv"
with open(csvp) as f:
    r = csv.reader(f)
    hdr = next(r)
    rows = list(r)
print("csv data rows:", len(rows), "| header cols:", len(hdr))
print("first row:", rows[0][:8])
