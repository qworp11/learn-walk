#!/bin/bash
# Бенч v2: 1 мир на CPU vs GPU, с конфигом как в тренировке (mathdx off)
source /root/hlc-venv/bin/activate
python - <<'EOF'
import time, dataclasses
import warp as wp

# конфиг как в hlc_train — ДО импорта playground
wp.config.enable_mathdx_solver = False
wp.config.enable_mathdx_gemm = False
wp.config.kernel_cache_dir = "/root/wp_kernel_cache"

import torch
from mujoco_playground._src.rl import env_registry
from mujoco_playground._src.rl.rsl_rl import vecenv_wrapper

npz = "/mnt/g/LEARN WALK/data/motions/npz_g1/walk1_subject1.npz"
entry = env_registry.get("G1WholeBodyTracking")

def bench(device, n=300):
    agent_cfg = entry.runner_cfg
    agent_cfg.nworld = 1
    env = entry.env_builder(nworld=1, device=device,
                            trajectory_path=npz, play=True)
    venv = vecenv_wrapper.RslRlVecEnvWrapper(
        env, clip_actions=agent_cfg.clip_actions,
        cfg=entry.runner_cfg.vecenv_wrapper_cfg)
    obs_td, _ = venv.reset()
    dev = obs_td["policy"].device
    a = torch.zeros(1, 29, device=dev)
    for _ in range(20):
        venv.step(a)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        obs_td, rew, done, _ = venv.step(a)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / n * 1000
    print(f"device={device}: {dt:.1f} мс/шаг -> {1000/dt:.0f} шаг/с", flush=True)
    try:
        venv.close()
    except Exception:
        pass

bench("cpu", n=200)
bench("cuda", n=300)
EOF
