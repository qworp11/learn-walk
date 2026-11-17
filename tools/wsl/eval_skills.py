#!/bin/bash
# Сравнение моделей по скиллам: специалисты vs единая модель.
# Протокол: reset -> playhead в кадр сегмента -> телепорт в позу референса
# (как в тренировке) -> 400 шагов. Метрики: reward, падения, путь робота/клипа.
# usage: eval_skills.sh [specialists | unified <путь_к_ckpt>]
source /root/hlc-venv/bin/activate
python - "$1" "$2" <<'PYEOF'
import json, math, sys, torch
sys.path.insert(0, "/mnt/g/LEARN WALK/src/hlc")
import game

MODE = sys.argv[1] if len(sys.argv) > 1 else "specialists"
CK_UNIFIED = sys.argv[2] if len(sys.argv) > 2 else None
NPZ = "/mnt/g/LEARN WALK/data/motions/npz_g1/unified_all.npz"
SEGM = json.load(open(NPZ.replace("unified_all.npz", "unified_segments.json")))
MASTER = {s["skill"]: s for s in SEGM["segments"] if s.get("master")}

# локальный офсет кадра внутри клипа (куда телепортируем)
OFF = {"stand": 1000, "walk": 680, "run": 1000, "sprint": 1000,
       "jump": 1000, "fall": 1000, "dance": 1000, "sit": 100}
# чекпоинт специалиста + его npz (кадр тот же локальный офсет)
SPEC = {"stand": "/root/hlc-logs/s1_balance/run_0/model_3499.pt"}
import glob, os, re
def ck_of(stage):
    runs = sorted(glob.glob(f"/root/hlc-logs/{stage}/run_*"))
    done = [r for r in runs if os.path.exists(r + "/TRAIN_DONE")]
    run = done[-1] if done else runs[-1]
    pts = sorted(glob.glob(run + "/model_*.pt"),
                 key=lambda p: int(p.rsplit("_", 1)[1][:-3]))
    cmd = json.load(open(run + "/cmd.json"))
    return pts[-1], cmd["env_overrides"]["trajectory_path"]

SPEC_TRAJ = {}
for sk, st in [("stand", "s1_balance"), ("walk", "s2_walk"), ("run", "s3_run"),
               ("sprint", "s3_sprint"), ("jump", "s4_jump"),
               ("fall", "s6_fall2"), ("dance", "s8_dance"), ("sit", "s9_sit")]:
    ck, traj = ck_of(st)
    SPEC_TRAJ[sk] = (ck, traj)

def pose(venv):
    q = venv.unwrapped.device_layer.sim.data.qpos
    return q[0, :3].detach().cpu().numpy().copy()

def run_skill(venv, policy, frame, steps=400):
    tl = venv.unwrapped.task_layer
    prod = tl.get_component_by_name("motion")
    mr = next(o for o in tl._resettable_objects
              if hasattr(o, "_teleport_to_reference"))
    rews, zs, dists = [], [], []
    for ep in range(2):
        venv.reset()
        prod.time_steps[:] = frame
        mr._teleport_to_reference(torch.ones(1, dtype=torch.bool))
        try:
            venv.unwrapped.device_layer.forward()
        except Exception:
            pass
        obs = game.steering.refresh_obs(venv)
        p0 = pose(venv)
        fall = 0
        for i in range(steps):
            with torch.no_grad():
                a = policy(obs)
            obs, rew, done, _ = venv.step(a)
            p = pose(venv)
            if p[2] < 0.45:
                fall += 1
            rews.append(float(rew[0]))
            zs.append(float(p[2]))
            dists.append(math.hypot(p[0] - p0[0], p[1] - p0[1]))
        # путь клипа за те же кадры
        T = prod.motion.time_step_total
        t = int(prod.time_steps[0].item()) % T
        ax = prod.motion.body_pos_w[:, prod._anchor_index, :2]
        clip_d = torch.norm(ax[t % T] - ax[(t - steps) % T]).item()
    n = len(rews)
    return {"rew": sum(rews) / n, "fall": fall / steps,
            "dist": dists[-1], "clip": clip_d, "z": sum(zs) / n}

print(f"=== {MODE} ===", flush=True)
if MODE == "specialists":
    for sk in ["stand", "walk", "run", "sprint", "jump", "fall", "dance",
               "sit"]:
        ck, traj = SPEC_TRAJ[sk]
        try:
            venv, policy = game.build_env_and_policy(ck, traj)
        except Exception as e:
            print(f"{sk:7s}: ОШИБКА загрузки {e}", flush=True)
            continue
        m = run_skill(venv, policy, OFF[sk])
        print(f"{sk:7s}: rew={m['rew']:.3f} падения={m['fall']*100:.0f}% "
              f"робот {m['dist']:.2f} м / клип {m['clip']:.2f} м z={m['z']:.2f}",
              flush=True)
        del venv, policy
else:
    venv, policy = game.build_env_and_policy(CK_UNIFIED, NPZ)
    for sk in MASTER:
        off = OFF.get(sk, 1000)
        m = run_skill(venv, policy, MASTER[sk]["start"] + off)
        print(f"{sk:7s}: rew={m['rew']:.3f} падения={m['fall']*100:.0f}% "
              f"робот {m['dist']:.2f} м / клип {m['clip']:.2f} м z={m['z']:.2f}",
              flush=True)
print("EVAL_DONE", flush=True)
PYEOF
echo "EVAL_EXIT=$?"