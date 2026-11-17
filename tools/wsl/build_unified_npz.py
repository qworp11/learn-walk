#!/bin/bash
# Склейка клипов всех скиллов в один unified_all.npz для единой модели.
# sit_synth дублируется x20 - бесшовный цикл (клип in-place) + доля сэмплов.
source /root/hlc-venv/bin/activate
python - <<'PYEOF'
import json
import numpy as np

BASE = "/mnt/g/LEARN WALK/data/motions/npz_g1"
CLIPS = [  # (имя сегмента, файл, повторов)
    ("stand",  "dance1_subject1.npz",       1),
    ("walk",   "walk3_subject1.npz",        1),
    ("run",    "run2_subject1.npz",         1),
    ("sprint", "sprint1_subject2.npz",      1),
    ("jump",   "jumps1_subject1.npz",       1),
    ("fall",   "fallAndGetUp2_subject2.npz",1),
    ("dance",  "dance2_subject1.npz",       1),
    ("sit",    "sit_synth.npz",            20),
]

ACC_KEYS = ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
            "body_lin_vel_w", "body_ang_vel_w"]
parts = {k: [] for k in ACC_KEYS}
segments = []
t0 = 0
for name, fname, reps in CLIPS:
    z = np.load(f"{BASE}/{fname}")
    assert z["fps"].item() == 50.0, fname
    if reps == 1:
        for k in ACC_KEYS:
            parts[k].append(z[k])
        segments.append({"skill": name, "file": fname, "start": t0,
                         "frames": int(z["joint_pos"].shape[0])})
        t0 += int(z["joint_pos"].shape[0])
    else:
        T = int(z["joint_pos"].shape[0])
        for r in range(reps):
            for k in ACC_KEYS:
                parts[k].append(z[k])
            segments.append({"skill": name, "file": fname, "start": t0,
                             "frames": T, "copy": r})
            t0 += T
        # записать в сегменты и "основной" диапазон навыка
        segments.append({"skill": name, "file": fname, "start": t0 - reps * T,
                         "frames": reps * T, "master": True})

out = {k: np.concatenate(parts[k], axis=0) for k in ACC_KEYS}
out["fps"] = np.array([50.0])
dst = f"{BASE}/unified_all.npz"
np.savez(dst, **out)
meta = {"total_frames": int(out["joint_pos"].shape[0]),
        "duration_s": out["joint_pos"].shape[0] / 50.0,
        "segments": segments}
json.dump(meta, open(f"{BASE}/unified_segments.json", "w"),
          ensure_ascii=False, indent=1)
print("TOTAL", meta["total_frames"], "кадров =",
      round(meta["duration_s"] / 60, 1), "мин")
for s in meta["segments"]:
    if s.get("master") or s.get("copy", 0) == 0:
        print(f"  {s['skill']:7s} старт {s['start']:6d} кадров {s['frames']}")
print("BUILD_DONE")
PYEOF
echo "BUILD_EXIT=$?"