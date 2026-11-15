#!/bin/bash
# Полная проверка лестницы: текущий этап + все будущие этапы
source /root/hlc-venv/bin/activate
cd "/mnt/g/LEARN WALK"

echo "=== 1. ПРОЦЕССЫ ==="
pgrep -af "stage_orchestrator|run_stage|watchdog.sh|preview_worker|live_stream" | grep -v "pgrep"

echo
echo "=== 2. ДИСК / RAM / GPU ==="
df -h / /mnt/g 2>/dev/null | tail -2
free -g | sed -n '2p'
nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null

echo
echo "=== 3. СТАТИКА: все этапы лестницы ==="
python - <<'EOF'
import json, pathlib
cfg = json.loads(pathlib.Path("configs/stages.json").read_text())
ladder = ["s2_walk", "s2_walk_run_a", "s3_run", "s3_sprint",
          "s4_jump", "s6_fall", "s6_fall2", "s8_dance"]
LOGROOT = pathlib.Path("/root/hlc-logs")
idx = {s: i for i, s in enumerate(ladder)}

def latest_ckpt(stage):
    best = None
    d = LOGROOT / stage
    if d.exists():
        for run in sorted(d.glob("run_*")):
            for p in run.glob("model_*.pt"):
                try:
                    n = int(p.stem.split("_")[1])
                except ValueError:
                    continue
                if best is None or n > best[0]:
                    best = (n, p)
    return best

done = {s: any((LOGROOT/s).glob("run_*/TRAIN_DONE")) if (LOGROOT/s).exists() else False
        for s in ladder}

for s in ladder:
    c = cfg[s]
    npz = pathlib.Path(f"data/motions/npz_g1/{c['motion']}.npz")
    mark = "✓" if npz.exists() else "ОШИБКА: НЕТ NPZ"
    line = f"{s}: ref={c['motion']} [{mark}] iter={c['max_iterations']} TRAIN_DONE={'да' if done[s] else 'нет'}"
    init = c.get("init_from")
    if init:
        src = init["stage"]
        b = latest_ckpt(src)
        if b:
            w = f"warm-start от {src}: чекпоинт есть ({b[1].name})"
        else:
            w = f"warm-start от {src}: чекпоинт появится после {src} (порядок {'OK' if idx[src] < idx[s] else 'ОШИБКА ПОРЯДКА!'})"
        line += f" | {w}"
    else:
        line += " | старт с нуля"
    print(line)

# текущее состояние run_a
b = latest_ckpt("s2_walk_run_a")
print(f"\nrun_a сейчас: последний чекпоинт {b[1].name if b else 'нет'}")
EOF

echo
echo "=== 4. stage_motion.py для будущих этапов (оркестратор его вызывает) ==="
for s in s3_run s3_sprint s4_jump s6_fall s6_fall2 s8_dance; do
  printf "%s -> " "$s"
  python tools/hlc/stage_motion.py "$s" 2>&1 | head -1
done

echo
echo "=== 5. ЖИВОЙ ТЕСТ warm-start на CPU (точный путь s3_run) ==="
HLC_WARM_START=1 python - <<'EOF'
import os, sys, dataclasses, pathlib
sys.path.insert(0, "/mnt/g/LEARN WALK/src/hlc")
import hlc_train  # применяет monkey-patch OnPolicyRunner.load

import torch
from mujoco_playground._src.rl import env_registry
from mujoco_playground._src.rl.rsl_rl import vecenv_wrapper
from rsl_rl import runners as rsl_runners

# источник warm-start: последний чекпоинт s2_walk
LOGROOT = pathlib.Path("/root/hlc-logs")
best = None
for run in sorted((LOGROOT/"s2_walk").glob("run_*")):
    for p in run.glob("model_*.pt"):
        n = int(p.stem.split("_")[1])
        if best is None or n > best[0]:
            best = (n, p)
ckpt = str(best[1])
print("источник:", ckpt)

# env будущего s3_run с его реальным референсом run2_subject1
npz = "/mnt/g/LEARN WALK/data/motions/npz_g1/run2_subject1.npz"
entry = env_registry.get("G1WholeBodyTracking")
agent_cfg = entry.runner_cfg
agent_cfg.nworld = 1
env = entry.env_builder(nworld=1, device="cpu", trajectory_path=npz, play=True)
venv = vecenv_wrapper.RslRlVecEnvWrapper(
    env, clip_actions=agent_cfg.clip_actions,
    cfg=entry.runner_cfg.vecenv_wrapper_cfg)

runner = rsl_runners.OnPolicyRunner(
    venv, dataclasses.asdict(agent_cfg), device="cpu")
runner.load(ckpt)  # должен напечатать "WARM-START load: ..."

policy = runner.get_inference_policy(device="cpu")
obs_td, _ = venv.reset()
rew_sum = 0.0
nan = False
for i in range(30):
    with torch.no_grad():
        a = policy(obs_td)
    obs_td, rew, done, extras = venv.step(a)
    r = float(rew.mean().item())
    rew_sum += r
    if r != r or abs(r) > 1e6:
        nan = True
print(f"WARMTEST: 30 шагов на run2, средняя награда {rew_sum/30:.2f}, nan={nan}")
print("WARMTEST_OK" if not nan else "WARMTEST_FAIL")
EOF
