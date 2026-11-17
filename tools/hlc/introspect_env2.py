#!/bin/bash
source /root/hlc-venv/bin/activate
python - <<'PYEOF'
import torch
from mujoco_playground._src.rl import env_registry

entry = env_registry.get("G1WholeBodyTracking")
agent_cfg = entry.runner_cfg
agent_cfg.nworld = 1
env = entry.env_builder(nworld=1, device="cpu", trajectory_path=None,
                        play=True, episode_length_secs=3600.0)
tl = env.task_layer
print("task_layer attrs:", [a for a in dir(tl) if not a.startswith("_")])
for attr in ("components", "producers", "consumers", "_components", "_producers"):
    o = getattr(tl, attr, None)
    if o is not None:
        print(f"task_layer.{attr} ->", type(o))
        try:
            for k, v in o.items():
                print(f"   [{k}] {type(v).__name__}")
        except Exception as e:
            print("   not dict:", e)
# ищем рекурсивно MotionReferenceProducer
import mujoco_playground.tasks.whole_body.tracking.motion_command as mc

def find(obj, cls, path="", depth=0, seen=None):
    if seen is None:
        seen = set()
    if id(obj) in seen or depth > 4:
        return
    seen.add(id(obj))
    if isinstance(obj, cls):
        print("FOUND:", path)
        return path
    for name in dir(obj):
        if name.startswith("__"):
            continue
        try:
            child = getattr(obj, name)
        except Exception:
            continue
        if isinstance(child, (str, int, float, bool, type(None))):
            continue
        r = find(child, cls, f"{path}.{name}", depth + 1, seen)
        if r:
            return r
    return None

p = find(env, mc.MotionReferenceProducer)
print("producer path:", p)
if p:
    prod = eval(f"env{p}")
    print("producer type:", type(prod).__name__)
    print("time_steps:", prod.time_steps)
    print("motion T:", prod.motion.time_step_total)
PYEOF
