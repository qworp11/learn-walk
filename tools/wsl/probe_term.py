#!/bin/bash
# Зонд: какой терминатор срабатывает на 1-м шаге после begin(t_start) / транспланта?
source /root/hlc-venv/bin/activate
python - <<'PYEOF'
import math, sys, torch
sys.path.insert(0, "/mnt/g/LEARN WALK/src/hlc")
import game, steering
from tensordict import TensorDict
import mujoco_playground.tasks.whole_body.tracking.mdp as mdp

EE_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link",
            "left_wrist_yaw_link", "right_wrist_yaw_link")

def pose(venv):
    q = venv.unwrapped.device_layer.sim.data.qpos
    p = q[0, :3].detach().cpu().numpy().copy()
    quat = q[0, 3:7].detach().cpu().numpy().copy()
    yaw = math.atan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                     1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
    return p, yaw

def fresh_obs(venv):
    """Пересчитать obs из ТЕКУЩЕГО состояния сима (без шага физики)."""
    env = venv.unwrapped
    measurements = env.device_layer.get_measurements()
    features = env.task_layer.compute_all_features(measurements)
    observations = env._compute_observations_from_features(features)
    td = TensorDict(dict(observations), batch_size=[env.nworld])
    return td, features

def term_report(features):
    az_r = features["motion/anchor_pos_ref"][0, -1].item()
    az_b = features["motion/anchor_pos_robot"][0, -1].item()
    print(f"    anchor z: ref={az_r:.3f} robot={az_b:.3f} "
          f"diff={abs(az_r-az_b):.3f} (порог 0.25)", flush=True)
    g = torch.zeros(1, 3); g[0, 2] = -1.0
    qr = features["motion/anchor_quat_ref"][0].unsqueeze(0)
    qb = features["motion/anchor_quat_robot"][0].unsqueeze(0)
    pm = mdp.pg_math.quat_rotate_inverse(qr, g)[0, 2].item()
    pb = mdp.pg_math.quat_rotate_inverse(qb, g)[0, 2].item()
    print(f"    ori gravity diff: {abs(pm-pb):.3f} (порог 0.8)", flush=True)
    ee = features["motion/body_pos_relative"][0, :, -1]
    eer = features["motion/body_pos_robot"][0, :, -1]
    for i, n in enumerate(EE_NAMES):
        print(f"    EE {n:22s} z: ref={ee[i].item():5.2f} robot={eer[i].item():5.2f} "
              f"diff={abs(ee[i].item()-eer[i].item()):.3f}", flush=True)

game.steering.install()
print("=== A. walk, begin(t_start), obs от reset (устаревший путь) ===", flush=True)
venv, policy = game.load_skill("walk")
prod = steering.attach(venv)
ts = steering.pick_start_frame(prod)
print(f"  t_start={ts}", flush=True)
venv_obs, _ = venv.reset()
steering.begin(venv, (0.0, 0.0), 0.0, t_start=ts)
with torch.no_grad():
    a = policy(venv_obs)
o, r, d, _ = venv.step(a)
print(f"  шаг со СТАРЫМ obs: done={bool(d[0])}", flush=True)

print("=== A2. walk, fresh obs после begin(t_start) ===", flush=True)
venv_obs, _ = venv.reset()
steering.begin(venv, (0.0, 0.0), 0.0, t_start=ts)
td, feats = fresh_obs(venv)
term_report(feats)
with torch.no_grad():
    a = policy(td)
o, r, d, _ = venv.step(a)
print(f"  шаг с FRESH obs: done={bool(d[0])}", flush=True)
p, _ = pose(venv)
print(f"  поза: {p.round(3)}", flush=True)

print("=== B. stand после транспланта из ходьбы ===", flush=True)
venv2, policy2 = game.load_skill("stand")
prod2 = steering.attach(venv2)
venv2_obs, _ = venv2.reset()
steering.transplant(venv2, venv)
pB, yawB = pose(venv2)
steering.begin(venv2, (pB[0], pB[1]), yawB)
td2, feats2 = fresh_obs(venv2)
term_report(feats2)
with torch.no_grad():
    a2 = policy2(td2)
o2, r2, d2, _ = venv2.step(a2)
print(f"  шаг с FRESH obs: done={bool(d2[0])}", flush=True)
p2, _ = pose(venv2)
print(f"  поза: {p2.round(3)}", flush=True)

print("=== C. walk 150 шагов после транспланта stand->walk (fresh obs) ===", flush=True)
venv_obs, _ = venv.reset()
steering.transplant(venv, venv2)
p3, yaw3 = pose(venv)
steering.begin(venv, (p3[0], p3[1]), yaw3, t_start=steering.pick_start_frame(prod))
cur, _ = fresh_obs(venv)
alive = 0
for i in range(150):
    steering.update(prod, yaw3, steering.MAX_RATE_WALK)
    steering.advance_pin(prod)
    with torch.no_grad():
        a = policy(cur)
    cur, r, d, _ = venv.step(a)
    alive += 1
    if bool(d[0]):
        break
p4, _ = pose(venv)
dist = float(((p4[:2] - p3[:2]) ** 2).sum()) ** 0.5
print(f"  выжил {alive}/150 шагов, дистанция {dist:.2f} м", flush=True)
print("PROBE_DONE", flush=True)
PYEOF
echo "PROBE_EXIT=$?"
