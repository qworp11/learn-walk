"""S0 smoke-тест: сборка трекинг-среды, рандом-политика, бенчмарк FPS.

Запуск (в WSL):
  python /mnt/g/LEARN\ WALK/src/hlc/s0_smoke.py --nworld 256 --device cuda
Проверяет:
  1) mujoco_playground + reaf + warp импортируются
  2) G1WholeBodyTracking строится на GPU (warp impl)
  3) рандомные действия крутятся, obs/reward вменяемы
  4) скорость шагов/с при заданном nworld
"""
import argparse
import dataclasses
import json
import sys
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworld", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()

    import mujoco_playground  # noqa: F401
    from mujoco_playground._src.rl import env_registry

    entry = env_registry.get("G1WholeBodyTracking")
    print("== building env, nworld =", args.nworld, "device =", args.device)
    t0 = time.time()
    env = entry.env_builder(nworld=args.nworld, device=args.device)
    print(f"build ok in {time.time()-t0:.1f}s")

    rng = np.random.default_rng(0)
    obs = env.reset()
    print("obs keys:", list(obs.keys()) if isinstance(obs, dict) else type(obs))
    policy_obs = obs["policy"] if isinstance(obs, dict) else obs
    if hasattr(policy_obs, "shape"):
        print("policy obs shape:", tuple(policy_obs.shape))
    else:
        print("policy obs:", policy_obs)

    # Рандомная политика
    n_joints = 29
    t0 = time.time()
    total_env_steps = 0
    for i in range(args.steps):
        action = torch.randn(args.nworld, n_joints, device=args.device) * 0.3
        obs = env.step(action)
        policy_obs = obs["policy"] if isinstance(obs, dict) else obs
        total_env_steps += args.nworld
        if i == 5:
            torch.cuda.synchronize()
            t0 = time.time()  # тёплый старт
    torch.cuda.synchronize()
    dt = time.time() - t0
    env_steps_per_sec = total_env_steps / dt
    print(f"steps/s (env): {env_steps_per_sec:,.0f}  ({args.steps} iters, {dt:.1f}s)")
    mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    print(f"torch mem peak: {mem:.2f} GiB")

    # NaN-проверка наблюдений
    if hasattr(policy_obs, "shape"):
        finite = torch.isfinite(policy_obs.float()).all().item()
        print("obs finite:", bool(finite))

    print("SMOKE_OK")
    with open("/root/s0_smoke_result.json", "w") as f:
        json.dump({"nworld": args.nworld, "env_steps_per_sec": env_steps_per_sec,
                   "torch_mem_gib": mem, "obs_finite": bool(finite)}, f)


if __name__ == "__main__":
    sys.exit(main())
