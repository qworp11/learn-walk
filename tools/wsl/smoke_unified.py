#!/bin/bash
# Смоук: unified_all.npz грузится через game.build_env_and_policy, 60 шагов.
source /root/hlc-venv/bin/activate
python - <<'PYEOF'
import sys, torch
sys.path.insert(0, "/mnt/g/LEARN WALK/src/hlc")
import game

NPZ = "/mnt/g/LEARN WALK/data/motions/npz_g1/unified_all.npz"
CK = "/root/hlc-logs/s3_run/run_0/model_5999.pt"
venv, policy = game.build_env_and_policy(CK, NPZ)
venv.unwrapped.task_layer._termination_checkers = []
obs = venv.reset()[0]
for i in range(60):
    with torch.no_grad():
        a = policy(obs)
    obs, rew, done, _ = venv.step(a)
    if i % 20 == 0 or bool(done[0]):
        q = venv.unwrapped.device_layer.sim.data.qpos
        print(f"шаг {i}: rew={float(rew[0]):.3f} done={bool(done[0])} "
              f"z={float(q[0,2]):.2f}", flush=True)
print("SMOKE_OK: unified npz работает")
PYEOF
echo "SMOKE_EXIT=$?"