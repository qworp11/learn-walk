#!/bin/bash
# Диагностика s9_sit: rollout play=False (как на тренировке), nworld=1.
# Перехватываем 3 терминации: печатаем refz/robz/anchor-z-err/ori-err/body-z-err
# каждый шаг и в момент срабатывания. Политики: s1_balance (warm-start)
# и последний чекпоинт s9_sit.
source /root/hlc-venv/bin/activate
python - <<'EOF'
import dataclasses, glob, os, re
import numpy as np
import torch

import mujoco_playground.tasks.whole_body.tracking.mdp as wmdp
import mujoco_playground as _mp  # noqa: E402
import warp as wp
wp.config.kernel_cache_dir = "/root/wp_kernel_cache"

from mujoco_playground._src.rl import env_registry
from mujoco_playground._src.rl.rsl_rl import checkpoint, vecenv_wrapper
from rsl_rl import runners as rsl_runners

# --- перехват терминаций ---
FIRST = "termination_bad_anchor_pos_z_only"
_orig = {n: getattr(wmdp, n) for n in
         ("termination_bad_anchor_pos_z_only",
          "termination_bad_anchor_ori",
          "termination_bad_motion_body_pos_z_only")}
pg = wmdp.pg_math


def make_wrap(name):
    orig = _orig[name]

    def wrapped(f, *a, **k):
        out = orig(f, *a, **k)
        try:
            az = (f["motion/anchor_pos_ref"][:, -1]
                  - f["motion/anchor_pos_robot"][:, -1]).abs()
            g = torch.zeros_like(f["motion/anchor_pos_ref"])
            g[:, 2] = -1.0
            pr = pg.quat_rotate_inverse(f["motion/anchor_quat_ref"], g)[:, 2]
            po = pg.quat_rotate_inverse(f["motion/anchor_quat_robot"], g)[:, 2]
            ar = (pr - po).abs()
            bp = (f["motion/body_pos_relative"][:, :, -1]
                  - f["motion/body_pos_robot"][:, :, -1]).abs().amax(dim=-1)
            i = int(az.argmax())
            if name == FIRST:
                print(f"      az={float(az[i]):.3f} ori={float(ar[i]):.3f} "
                      f"bodyz={float(bp[i]):.3f} "
                      f"refz={float(f['motion/anchor_pos_ref'][i, -1]):+.3f} "
                      f"robz={float(f['motion/anchor_pos_robot'][i, -1]):+.3f}",
                      flush=True)
            if float(az[i]) > 0.25 or float(ar[i]) > 0.8 or float(bp[i]) > 0.25:
                print(f"      [TERM {name}] az={float(az[i]):.3f} "
                      f"ori={float(ar[i]):.3f} bodyz={float(bp[i]):.3f} "
                      f"refz={float(f['motion/anchor_pos_ref'][i, -1]):+.3f} "
                      f"robz={float(f['motion/anchor_pos_robot'][i, -1]):+.3f}",
                      flush=True)
        except Exception:
            pass
        return out

    return wrapped


for n in _orig:
    setattr(wmdp, n, make_wrap(n))

TRAJ = "/mnt/g/LEARN WALK/data/motions/npz_g1/sit_synth.npz"
CKPTS = [("s1_balance(warm-start)",
          "/root/hlc-logs/s1_balance/run_1/model_3499.pt")]
cands = sorted(glob.glob("/root/hlc-logs/s9_sit/run_0/model_*.pt"),
               key=lambda p: int(re.search(r"model_(\d+)", p).group(1)))
if cands:
    CKPTS.append(("s9_sit latest", cands[-1]))

entry = env_registry.get("G1WholeBodyTracking")


def rollout(label, ckpt):
    print(f"\n########## политика {label}: {ckpt}", flush=True)
    agent_cfg = entry.runner_cfg
    agent_cfg.nworld = 1
    env = entry.env_builder(nworld=1, device="cpu",
                            trajectory_path=TRAJ, play=False)
    venv = vecenv_wrapper.RslRlVecEnvWrapper(
        env, clip_actions=agent_cfg.clip_actions,
        cfg=agent_cfg.vecenv_wrapper_cfg)
    runner = rsl_runners.OnPolicyRunner(
        venv, dataclasses.asdict(agent_cfg), device="cpu")
    runner.load(checkpoint.read_checkpoint(ckpt),
                load_cfg={"actor": True, "critic": False,
                          "optimizer": False, "iteration": False},
                map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")

    def qz():
        q = venv.unwrapped.device_layer.sim.data.qpos
        return float(q.detach().cpu().numpy()[0, 2])

    for ep in range(3):
        obs, extra = venv.reset()
        print(f"  EP{ep}: спавн z={qz():+.3f} reset-extra: "
              f"{list(extra.keys()) if isinstance(extra, dict) else type(extra)}",
              flush=True)
        for step in range(60):
            with torch.no_grad():
                a = policy(obs)
            obs, rew, done, info = venv.step(a)
            r = float(rew.mean())
            print(f"    шаг {step:2d} rew={r:+.3f} z={qz():+.3f}", flush=True)
            if bool(done[0]) if hasattr(done, "__getitem__") else bool(done):
                print(f"    DONE на шаге {step} "
                      f"(время {step * 0.02:.2f}с)", flush=True)
                if isinstance(info, dict):
                    for k, v in list(info.items())[:25]:
                        try:
                            vv = v.item() if hasattr(v, "item") else v
                        except Exception:
                            vv = "?"
                        print(f"      info {k} = {vv}", flush=True)
                break
        if ep < 2:
            continue
    try:
        venv.close()
    except Exception:
        pass


for label, ckpt in CKPTS:
    rollout(label, ckpt)
print("\ndiag_sit завершён")
EOF
