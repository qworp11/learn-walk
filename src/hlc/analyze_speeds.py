"""Измерение скоростей референсов LAFAN1-G1 (средняя горизонтальная скорость root).

Выход: data/motions/speeds.json — подобрать этапы под gate-критерии (S2: 0.8-1.6 м/с).
"""
import csv
import json
from pathlib import Path

import numpy as np

DATA = Path(r"G:\LEARN WALK\data\motions\lafan1_g1")
FPS = 30.0


def analyze(path):
    rows = []
    with open(path) as f:
        for row in csv.reader(f):
            if row:
                rows.append([float(x) for x in row])
    a = np.array(rows)
    pos = a[:, :3]
    q = a[:, 3:7]  # xyzw
    dt = 1.0 / FPS
    vel = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1) / dt
    speed_mean = vel.mean()
    speed_p95 = np.percentile(vel, 95)
    dur = len(pos) * dt
    dist = np.linalg.norm(pos[-1, :2] - pos[0, :2])
    height_mean = pos[:, 2].mean()
    return {"mean": round(float(speed_mean), 3), "p95": round(float(speed_p95), 3),
            "dur_s": round(dur, 1), "net_m": round(float(dist), 2),
            "z_mean": round(float(height_mean), 3)}


def main():
    out = {}
    for f in sorted(DATA.glob("*.csv")):
        out[f.stem] = analyze(f)
    (DATA.parent / "speeds.json").write_text(json.dumps(out, indent=1))
    # групповое саммари
    groups = {}
    for k, v in out.items():
        g = k.rsplit("_subject", 1)[0]
        groups.setdefault(g, []).append(v["mean"])
    for g, vs in sorted(groups.items()):
        print(f"{g:28s} mean_speed {np.mean(vs):.2f} м/с (n={len(vs)})")


if __name__ == "__main__":
    main()
