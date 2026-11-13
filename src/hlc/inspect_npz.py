"""Проверка структуры сконвертированных NPZ-референсов."""
import glob
import sys

import numpy as np

paths = sorted(glob.glob("/mnt/g/LEARN WALK/data/motions/npz_g1/*.npz"))
print("files:", len(paths))
d = np.load(paths[0])
for k in sorted(d.files):
    a = d[k]
    print(f"  {k}: shape={a.shape} dtype={a.dtype}")
# Сводка по длинам всех файлов
for p in paths[:40]:
    d = np.load(p)
    key = "qpos" if "qpos" in d.files else sorted(d.files)[0]
    n = d[key].shape[0]
    print(f"{p.split('/')[-1]:32s} frames={n}")
