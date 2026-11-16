#!/bin/bash
# Инвентаризация для игры: формат npz, API mujoco, чекпоинты всех этапов
source /root/hlc-venv/bin/activate
python - <<'EOF'
import numpy as np, glob, json, os
npz_path = "/mnt/g/LEARN WALK/data/motions/npz_g1/walk1_subject1.npz"
d = np.load(npz_path)
print("=== NPZ keys ===")
for k in d.keys():
    try:
        print(" ", k, d[k].shape, d[k].dtype)
    except Exception as e:
        print(" ", k, type(d[k]), e)

import mujoco
print("=== mujoco version:", mujoco.__version__, "===")
from mujoco_playground.robots.g1.sim import scene as g1_scene
spec = g1_scene.g1_scene_spec()
print("spec type:", type(spec))
print("spec attrs:", [a for a in dir(spec) if not a.startswith("_")][:30])

print("=== чекпоинты этапов ===")
for st in ["s1_balance","s2_walk","s2_walk_run_a","s3_run","s3_sprint",
           "s4_jump","s6_fall","s6_fall2","s8_dance"]:
    runs = sorted(glob.glob(f"/root/hlc-logs/{st}/run_*"))
    if not runs:
        print(f"  {st}: НЕТ"); continue
    r = runs[-1]
    cks = sorted(glob.glob(r+"/model_*.pt"))
    cmd = os.path.join(r, "cmd.json")
    cj = {}
    if os.path.exists(cmd):
        cj = json.load(open(cmd)).get("env_overrides", {})
    last = os.path.basename(cks[-1]) if cks else None
    print(f"  {st}: ckpts={len(cks)} last={last} "
          f"traj={os.path.basename(cj.get('trajectory_path','?'))} "
          f"nworld={cj.get('nworld','?')}")
EOF
