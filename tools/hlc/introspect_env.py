#!/bin/bash
# Интроспекция env: путь до MotionReferenceProducer, запись qpos/qvel, камера
source /root/hlc-venv/bin/activate
python - <<'PYEOF'
import torch
from mujoco_playground._src.rl import env_registry

entry = env_registry.get("G1WholeBodyTracking")
agent_cfg = entry.runner_cfg
agent_cfg.nworld = 1
# проверка episode_length_secs kwarg
env = entry.env_builder(nworld=1, device="cpu", trajectory_path=None,
                        play=True, episode_length_secs=3600.0)
print("episode_length kwarg OK")
print("env attrs:", [a for a in dir(env) if not a.startswith("_")][:30])
# ищем task layer / компоненты
for attr in ("task_layer", "task", "components", "producers"):
    o = getattr(env, attr, None)
    print(f"env.{attr} ->", type(o))
PYEOF
