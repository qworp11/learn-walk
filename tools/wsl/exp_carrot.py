#!/bin/bash
# E7: морковка - pin двигается с ФИКСИРОВАННОЙ скоростью (а не со скоростью
# клипа). Робот в тренировке догонял якорь с рассогласованием - используем
# это: pin ahead -> робот идёт. Пробуем скорость и точку старта.
source /root/hlc-venv/bin/activate
python - <<'PYEOF'
import math, sys, torch
sys.path.insert(0, "/mnt/g/LEARN WALK/src/hlc")
import game, steering
import numpy as np

def pose(venv):
    q = venv.unwrapped.device_layer.sim.data.qpos
    return q[0, :3].detach().cpu().numpy().copy()

def trial(t_start, carrot, cap, steps, tag):
    venv.reset()
    steering.begin(venv, (0.0, 0.0), 0.0, t_start=t_start, blend_steps=50)
    obs = steering.refresh_obs(venv)
    p0 = pose(venv)
    st = prod._steer
    onset, z_min = None, 9.9
    dists = []
    for i in range(steps):
        # морковка: pin += psi_dir * carrot * dt; заморозка на бленде;
        # зазор не больше cap (не уводить политику в OOD)
        t_cur = int(prod.time_steps[0].item())
        k = (t_cur - t_start) / 50.0
        if k >= 1.0 and carrot > 0:
            gap = math.hypot(st["p_pin"][0] - p0[0], st["p_pin"][1] - p0[1])
            if gap < cap:
                st["p_pin"][0] += carrot * steering.STEP_DT
        with torch.no_grad():
            a = policy(obs)
        obs, rew, done, _ = venv.step(a)
        p = pose(venv)
        z_min = min(z_min, p[2])
        d = math.hypot(p[0], p[1])
        dists.append(d)
        if onset is None and d > 0.3:
            onset = i
        if i % 100 == 0:
            gap = math.hypot(st["p_pin"][0] - p[0], st["p_pin"][1] - p[1])
            print(f"    {tag} шаг {i:3d}: d={d:5.2f} z={p[2]:5.2f} gap={gap:4.2f}",
                  flush=True)
    cruise = dists[-1] - dists[-200]
    print(f"  {tag}: t_start={t_start} морковка={carrot} -> старт на шаге {onset}, "
          f"итог {dists[-1]:.2f} м, круиз(4 с) {cruise:.2f} м "
          f"({cruise/4.0:.2f} м/с), min z {z_min:.2f}", flush=True)

game.steering.install()
venv, policy = game.load_skill("walk")
prod = steering.attach(venv)
venv.unwrapped.task_layer._termination_checkers = []
S3 = 680   # устойчиво >0.3 м/с - настоящая походка walk3

print("=== B: t=530 (за 3 с), морковка 0.4 cap 0.4 ===", flush=True)
trial(530, 0.4, 0.4, 800, "B")
print("=== C: t=530, морковка 0.55 cap 0.55 ===", flush=True)
trial(530, 0.55, 0.55, 800, "C")
print("=== E: t=680 (сразу походка), морковка 0.4 cap 0.4 ===", flush=True)
trial(680, 0.4, 0.4, 800, "E")
print("E7_DONE", flush=True)
PYEOF
echo "E7_EXIT=$?"
