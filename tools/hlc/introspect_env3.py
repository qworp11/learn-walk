#!/bin/bash
source /root/hlc-venv/bin/activate
python - <<'PYEOF' 2>&1 | tail -40
import sys, glob, re, os, torch
sys.path.insert(0, "/mnt/g/LEARN WALK/src/hlc")
import game

# 1) producer доступен по имени?
venv, policy = game.load_skill("stand")
env = venv.unwrapped
prod = env.task_layer.get_component_by_name("motion")
print("producer:", type(prod).__name__, "| T =", prod.motion.time_step_total)

# 2) шагаем 50 шагов политикой, запоминаем qpos
obs, _ = venv.reset()
for i in range(50):
    with torch.no_grad():
        a = policy(obs)
    obs, rew, done, _ = venv.step(a)
sim = env.device_layer.sim.data
q_saved = sim.qpos.clone()
root_before = q_saved[0, :3].tolist()
print("root pos after 50 steps:", [round(x, 3) for x in root_before])

# 3) reset (телепорт в кадр 0) -> проверяем
obs, _ = venv.reset()
root_reset = sim.qpos[0, :3].tolist()
print("root pos after reset:", [round(x, 3) for x in root_reset])

# 4) ТРАНСПЛАНТ: вернуть qpos/qvel без reset
sim.qpos[:] = q_saved
print("root pos after transplant:", [round(x, 3) for x in sim.qpos[0, :3].tolist()])
with torch.no_grad():
    a = policy(obs)
obs2, rew, done, _ = venv.step(a)
print("after 1 step: obs finite:", torch.isfinite(obs2["policy"]).all().item(),
      "| done:", done.item() if hasattr(done, "item") else done,
      "| root:", [round(x, 3) for x in sim.qpos[0, :3].tolist()])

# 5) ещё 100 шагов — стабильность после транспланта
for i in range(100):
    with torch.no_grad():
        a = policy(obs2)
    obs2, rew, done, _ = venv.step(a)
print("after 100 more: done:", done.item(), "| root:",
      [round(x, 3) for x in sim.qpos[0, :3].tolist()],
      "| pelvis z:", round(sim.qpos[0, 2].item(), 3))
print("INTROSPECT3_OK")
PYEOF
