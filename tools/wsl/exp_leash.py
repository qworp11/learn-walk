#!/bin/bash
# E8: короткий поводок - pin КАЖДЫЙ шаг ставится перед роботом на margin
# по курсу psi (в тренировке якорь всегда был чуть впереди - это штатный
# сигнал "иди вперёд"). Пробуем margin и точку старта.
source /root/hlc-venv/bin/activate
python - <<'PYEOF'
import math, sys, torch
sys.path.insert(0, "/mnt/g/LEARN WALK/src/hlc")
import game, steering

def pose(venv):
    q = venv.unwrapped.device_layer.sim.data.qpos
    return q[0, :3].detach().cpu().numpy().copy()

def trial(t_start, margin, steps, tag):
    venv.reset()
    steering.begin(venv, (0.0, 0.0), 0.0, t_start=t_start, blend_steps=50)
    obs = steering.refresh_obs(venv)
    st = prod._steer
    onset, z_min = None, 9.9
    dists = []
    for i in range(steps):
        # поводок: pin = робот + dir(psi)*margin, каждый шаг
        with torch.no_grad():
            a = policy(obs)
        obs, rew, done, _ = venv.step(a)
        p = pose(venv)
        st["p_pin"][0] = p[0] + math.cos(st["psi"]) * margin
        st["p_pin"][1] = p[1] + math.sin(st["psi"]) * margin
        z_min = min(z_min, p[2])
        d = math.hypot(p[0], p[1])
        dists.append(d)
        if onset is None and d > 0.3:
            onset = i
        if i % 100 == 0:
            print(f"    {tag} шаг {i:3d}: d={d:5.2f} z={p[2]:5.2f} "
                  f"кадр={int(prod.time_steps[0].item())}", flush=True)
    cruise = dists[-1] - dists[-200]
    print(f"  {tag}: t={t_start} поводок={margin} -> старт {onset}, итог "
          f"{dists[-1]:.2f} м, круиз(4с) {cruise:.2f} м ({cruise/4:.2f} м/с), "
          f"min z {z_min:.2f}", flush=True)

game.steering.install()
venv, policy = game.load_skill("walk")
prod = steering.attach(venv)
venv.unwrapped.task_layer._termination_checkers = []

print("=== A: t=530 (S3-150), поводок 0.15 ===", flush=True)
trial(530, 0.15, 800, "A")
print("=== B: t=530, поводок 0.25 ===", flush=True)
trial(530, 0.25, 800, "B")
print("=== C: t=620 (S3-60), поводок 0.15 ===", flush=True)
trial(620, 0.15, 800, "C")
print("=== D: t=680 (сразу походка), поводок 0.15 ===", flush=True)
trial(680, 0.15, 800, "D")
print("E8_DONE", flush=True)
PYEOF
echo "E8_EXIT=$?"
