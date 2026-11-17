#!/bin/bash
# Решающий эксперимент: политика walk БЕЗ стирринга, перемотка клипа.
# E1: t=680 (mid-walk), терминаторы ВЫКЛ — робот ходит?
# E2: t=0, 1200 шагов (24 с) — начнёт ли ходить, когда клип тронется (~14 с)?
# E3: t=680, терминаторы вкл — что сделает done?
source /root/hlc-venv/bin/activate
python - <<'PYEOF'
import math, sys, torch
sys.path.insert(0, "/mnt/g/LEARN WALK/src/hlc")
import game, steering

def pose(venv):
    q = venv.unwrapped.device_layer.sim.data.qpos
    return q[0, :3].detach().cpu().numpy().copy()

def rollout(venv, policy, steps, tag, with_term=False):
    obs, _ = venv.reset()
    if not with_term:
        venv.unwrapped.task_layer._termination_checkers = []
    z_min, moved_max = 9.9, 0.0
    p0 = pose(venv)
    dones = 0
    for i in range(steps):
        with torch.no_grad():
            a = policy(obs)
        obs, rew, done, _ = venv.step(a)
        if bool(done[0]):
            dones += 1
            obs, _ = venv.reset()  # как обёртка
        p = pose(venv)
        z_min = min(z_min, p[2])
        moved_max = max(moved_max, float(((p[:2] - p0[:2]) ** 2).sum()) ** 0.5)
        if i % 100 == 0:
            print(f"    {tag} шаг {i:4d}: xy=({p[0]:6.2f},{p[1]:6.2f}) "
                  f"z={p[2]:5.2f} done_count={dones}", flush=True)
    print(f"    {tag}: итог дистанция {moved_max:.2f} м, min z {z_min:.2f}, "
          f"done_count {dones}", flush=True)
    return moved_max

game.steering.install()
venv, policy = game.load_skill("walk")
prod = steering.attach(venv)   # steering НЕ включаем (on=False)

print("=== E1: t=680 mid-walk, без стирринга, терминаторы выкл ===", flush=True)
venv.reset()
prod.time_steps[:] = 680
d1 = rollout(venv, policy, 300, "E1")

print("=== E2: t=0, 1200 шагов (24 с), без стирринга ===", flush=True)
prod.time_steps[:] = 0
d2 = rollout(venv, policy, 1200, "E2")

print("=== E3: t=680 mid-walk, терминаторы вкл (как было) ===", flush=True)
import reaf.core.environment as reaf_env_mod
# вернуть чекеры нельзя просто так - они были очищены в load_skill;
# проверим через новый env
venv2, policy2 = game.load_skill("walk")
prod2 = steering.attach(venv2)
print(f"  чекеров в новом env: "
      f"{len(venv2.unwrapped.task_layer._termination_checkers)}", flush=True)
venv2.reset()
prod2.time_steps[:] = 680
d3 = rollout(venv2, policy2, 300, "E3", with_term=True)
print("EXP_DONE", flush=True)
PYEOF
echo "EXP_EXIT=$?"
