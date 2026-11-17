#!/bin/bash
# Эксперимент 2: где ставить t_start, чтобы робот трогался?
# E4: t=430 (за 5 с до первого шага walk3)  E5: t=530 (за 3 с)
# E6: t=680 (сразу шаг) + длинный бленд 150
source /root/hlc-venv/bin/activate
python - <<'PYEOF'
import math, sys, torch
sys.path.insert(0, "/mnt/g/LEARN WALK/src/hlc")
import game, steering

def pose(venv):
    q = venv.unwrapped.device_layer.sim.data.qpos
    return q[0, :3].detach().cpu().numpy().copy()

game.steering.install()
venv, policy = game.load_skill("walk")
prod = steering.attach(venv)
venv.unwrapped.task_layer._termination_checkers = []

def trial(t_start, blend, steps, tag):
    venv.reset()
    obs = steering.refresh_obs(venv) if False else venv.reset()[0]
    steering.begin(venv, (0.0, 0.0), 0.0, t_start=t_start, blend_steps=blend)
    obs = steering.refresh_obs(venv)
    p0 = pose(venv)
    walk_start = None
    z_min = 9.9
    dist_end = 0.0
    for i in range(steps):
        steering.update(prod, 0.0, steering.MAX_RATE_WALK)
        steering.advance_pin(prod)
        with torch.no_grad():
            a = policy(obs)
        obs, rew, done, _ = venv.step(a)
        p = pose(venv)
        z_min = min(z_min, p[2])
        d = float(((p[:2] - p0[:2]) ** 2).sum()) ** 0.5
        dist_end = d
        if walk_start is None and d > 0.25:
            walk_start = i
        if i % 100 == 0:
            print(f"    {tag} шаг {i:4d}: d={d:5.2f} z={p[2]:5.2f}", flush=True)
    st = walk_start if walk_start is not None else -1
    print(f"  {tag}: t_start={t_start} бленд={blend} -> пошёл на шаге {st} "
          f"(кадр клипа {t_start + st if st >= 0 else '-'}), "
          f"дистанция {dist_end:.2f} м, min z {z_min:.2f}", flush=True)

print("=== E4: t=430 (5 с до шага), бленд 50, 900 шагов ===", flush=True)
trial(430, 50, 900, "E4")
print("=== E5: t=530 (3 с до шага), бленд 50, 700 шагов ===", flush=True)
trial(530, 50, 700, "E5")
print("=== E6: t=680 (сразу шаг), длинный бленд 150, 700 шагов ===", flush=True)
trial(680, 150, 700, "E6")
print("EXP2_DONE", flush=True)
PYEOF
echo "EXP2_EXIT=$?"
