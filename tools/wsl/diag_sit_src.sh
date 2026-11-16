#!/bin/bash
# Диагностика s9_sit: как именно работают терминации, чем спавнит
# MotionResetter, и rollout play=False (как на тренировке) со следованием z.
source /root/hlc-venv/bin/activate
python - <<'EOF'
import inspect, dataclasses
from mujoco_playground._src.rl import env_registry
e = env_registry.get("G1WholeBodyTracking")
mod = inspect.getmodule(e.env_cfg_fn)
g = vars(mod)
mdp = g.get("mdp"); rmod = g.get("reset")
print("mdp:", getattr(mdp, "__file__", None))
for fn in ("termination_bad_anchor_pos_z_only",
           "termination_bad_anchor_ori",
           "termination_bad_motion_body_pos_z_only"):
    f = getattr(mdp, fn, None)
    print(f"=== {fn} ===")
    print(inspect.getsource(f) if f else "НЕ НАЙДЕНА")
print("=== MotionResetter (первые 4500 симв) ===")
print(inspect.getsource(rmod.MotionResetter)[:4500])
print("=== константы ===")
for name in ("POSE_RANGE", "VELOCITY_RANGE", "JOINT_POSITION_RANGE",
             "_CONTROL_TIMESTEP", "_SIMULATION_TIMESTEP"):
    v = g.get(name, rmod.__dict__.get(name, "нет"))
    print(name, "=", v)
EOF
