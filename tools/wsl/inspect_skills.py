#!/bin/bash
# Проверка совместимости скиллов перед объединением в одну модель:
# 1) финальные чекпоинты всех стадий - размерности актёра/критика/нормализатора
# 2) npz клипы - fps, число кадров, размерности (совместимость склейки)
source /root/hlc-venv/bin/activate
python - <<'PYEOF'
import glob, json, os, sys
import numpy as np
import torch

LOG = "/root/hlc-logs"
stages = sorted(os.listdir(LOG))

print("=== ЧЕКПОИНТЫ ===")
info = {}
for st in stages:
    runs = sorted(glob.glob(f"{LOG}/{st}/run_*"))
    done = [r for r in runs if os.path.exists(r + "/TRAIN_DONE")]
    run = done[-1] if done else (runs[-1] if runs else None)
    if run is None:
        print(f"{st}: НЕТ ПРОГОНОВ")
        continue
    pts = sorted(glob.glob(run + "/model_*.pt"),
                 key=lambda p: int(p.rsplit("_", 1)[1][:-3]))
    ck = pts[-1]
    d = torch.load(ck, weights_only=False, map_location="cpu")
    keys = list(d.keys()) if isinstance(d, dict) else type(d).__name__
    # find actor dims
    def dims(sd):
        w = [k for k in sd if k.endswith("weight")]
        if not w:
            return "?"
        first = sd[w[0]].shape
        last = sd[w[-1]].shape
        return f"{first}->{last} ({len(w)} слоёв)"
    sd = d.get("model_state_dict", d.get("state_dict", d)) \
        if isinstance(d, dict) else d
    actor_w = [k for k in sd if "actor" in k and k.endswith("weight")]
    critic_w = [k for k in sd if "critic" in k and k.endswith("weight")]
    norm_keys = [k for k in sd if "mean" in k or "std" in k or "var" in k]
    cmd = {}
    cj = os.path.join(run, "cmd.json")
    if os.path.exists(cj):
        cmd = json.load(open(cj))
    traj = cmd.get("env_overrides", {}).get("trajectory_path", "?")
    print(f"{st}: {os.path.basename(ck)} actor={dims({k: sd[k] for k in actor_w})} "
          f"critic={dims({k: sd[k] for k in critic_w})} norm={norm_keys[:2]}")
    print(f"    нормализатор mean={tuple(sd[norm_keys[0]].shape) if norm_keys else '?'} "
          f"трей={os.path.basename(traj)}")

print()
print("=== КЛИПЫ ===")
seen = set()
for st in stages:
    runs = sorted(glob.glob(f"{LOG}/{st}/run_*"))
    done = [r for r in runs if os.path.exists(r + "/TRAIN_DONE")]
    run = done[-1] if done else (runs[-1] if runs else None)
    cj = os.path.join(run, "cmd.json") if run else None
    if not cj or not os.path.exists(cj):
        continue
    traj = json.load(open(cj)).get("env_overrides", {}).get("trajectory_path", "")
    if traj in seen or not traj:
        continue
    seen.add(traj)
    z = np.load(traj)
    ks = list(z.keys())
    out = f"{os.path.basename(traj)}: "
    for k in ("fps", "time_step_total", "dt"):
        if k in z:
            out += f"{k}={z[k]} "
    for k in ks:
        if k not in ("fps", "time_step_total", "dt"):
            out += f"{k}{z[k].shape} "
    print(out)
print("INSPECT_DONE")
PYEOF
echo "INSPECT_EXIT=$?"