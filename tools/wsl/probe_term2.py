#!/bin/bash
# Зонд 2: какой ИМЕННО termination checker срабатывает на 1-м шаге walk(t_start)?
source /root/hlc-venv/bin/activate
python - <<'PYEOF'
import math, sys, torch
sys.path.insert(0, "/mnt/g/LEARN WALK/src/hlc")
import game, steering
from tensordict import TensorDict

def fresh_obs(venv):
    env = venv.unwrapped
    measurements = env.device_layer.get_measurements()
    features = env.task_layer.compute_all_features(measurements)
    observations = env._compute_observations_from_features(features)
    td = TensorDict(dict(observations), batch_size=[env.nworld])
    return td, features

def all_checkers(venv, features, label):
    tl = venv.unwrapped.task_layer
    print(f"  --- чекеры ({label}) ---", flush=True)
    for tc in tl._termination_checkers:
        try:
            keys = tc.required_features_keys()
            sub = {k: features[k] for k in keys if k in features}
            res = tc.check_termination(sub)
            name = getattr(tc, "name", lambda: type(tc).__name__)()
            bad = "<<< СРАБОТАЛ" if res.is_terminated() or res.is_truncated() else ""
            print(f"    {name}: terminated={res.is_terminated()} "
                  f"truncated={res.is_truncated()} {bad}", flush=True)
        except Exception as e:
            print(f"    {type(tc).__name__}: ошибка {e}", flush=True)

game.steering.install()
venv, policy = game.load_skill("walk")
prod = steering.attach(venv)
ts = steering.pick_start_frame(prod)
print(f"  t_start={ts}", flush=True)
venv_obs, _ = venv.reset()
steering.begin(venv, (0.0, 0.0), 0.0, t_start=ts)
td, feats = fresh_obs(venv)
all_checkers(venv, feats, "ДО шага")
with torch.no_grad():
    a = policy(td)
print(f"  |action| max={a.abs().max().item():.2f} mean={a.abs().mean().item():.3f}",
      flush=True)
o, r, d, _ = venv.step(a)
print(f"  done после 1-го шага: {bool(d[0])}", flush=True)
td2, feats2 = fresh_obs(venv)
all_checkers(venv, feats2, "ПОСЛЕ шага")
# ключевые числа после шага
f = feats2
print(f"  anchor z: ref={f['motion/anchor_pos_ref'][0,-1]:.3f} "
      f"robot={f['motion/anchor_pos_robot'][0,-1]:.3f}", flush=True)
print(f"  EE rel z: {[round(x,3) for x in f['motion/body_pos_relative'][0,:,-1].tolist()]}",
      flush=True)
print(f"  EE robot z: {[round(x,3) for x in f['motion/body_pos_robot'][0,:,-1].tolist()]}",
      flush=True)
q = venv.unwrapped.device_layer.sim.data.qpos
print(f"  qpos root: {q[0,:3].tolist()} qvel norm: "
      f"{venv.unwrapped.device_layer.sim.data.qvel[0].norm().item():.2f}", flush=True)
print("PROBE2_DONE", flush=True)
PYEOF
echo "PROBE2_EXIT=$?"
