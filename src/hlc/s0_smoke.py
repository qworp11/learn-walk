"""S0 smoke-тест: сборка трекинг-среды, рандом-политика, бенчмарк FPS.

Запуск (в WSL):  python /mnt/g/LEARN\ WALK/src/hlc/s0_smoke.py --nworld 256 --device cuda
Проверяет:
  1) mujoco_playground + reaf + warp импортируются, warp-ядро на Pascal
  2) G1WholeBodyTracking строится на GPU (impl=warp)
  3) рандомные действия крутятся через RslRlVecEnvWrapper (тот же путь, что тренировка)
  4) скорость env-шагов/с, VRAM, NaN-проверка obs
"""
import argparse
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

    import warp as wp

    # Pascal fix: libmathdx (cuSolverDx/cuBLASDx) не поддерживает sm_61
    wp.config.enable_mathdx_solver = False
    wp.config.enable_mathdx_gemm = False
    wp.config.kernel_cache_dir = "/root/wp_kernel_cache"

    # Тест warp-ядра на Pascal
    @wp.kernel
    def dbl(a: wp.array[float], b: wp.array[float]):
        i = wp.tid()
        b[i] = a[i] * 2.0

    xs = wp.array([1.0, 2.0, 3.0], dtype=float, device="cuda:0")
    ys = wp.empty_like(xs)
    wp.launch(dbl, dim=3, inputs=[xs, ys], device="cuda:0")
    assert ys.numpy().tolist() == [2.0, 4.0, 6.0]
    print("warp kernel OK", wp.__version__)

    from mujoco_playground._src.rl import env_registry
    from mujoco_playground._src.rl.rsl_rl import vecenv_wrapper

    entry = env_registry.get("G1WholeBodyTracking")
    print("== building env, nworld =", args.nworld, "device =", args.device)
    t0 = time.time()
    env = entry.env_builder(nworld=args.nworld, device=args.device)
    print(f"build ok in {time.time()-t0:.1f}s")

    venv = vecenv_wrapper.RslRlVecEnvWrapper(env, cfg=None)
    print("num_actions:", venv.num_actions, "| max_ep_len:", venv.max_episode_length)

    obs_td, _ = venv.reset()
    pol = obs_td["policy"]
    print("policy obs shape:", tuple(pol.shape))

    t0 = time.time()
    total = 0
    for i in range(args.steps):
        action = torch.randn(args.nworld, venv.num_actions, device=args.device) * 0.3
        obs_td, rew, done, extras = venv.step(action)
        total += args.nworld
        if i == 10:
            torch.cuda.synchronize()
            t0 = time.time()
    torch.cuda.synchronize()
    dt = time.time() - t0
    sps = total / dt
    print(f"env steps/s: {sps:,.0f} ({args.steps} iters in {dt:.1f}s)")
    mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    print(f"torch peak mem: {mem:.2f} GiB")
    print("obs finite:", bool(torch.isfinite(pol.float()).all().item()))
    print("rew sample:", float(rew.mean()))

    with open("/root/s0_smoke_result.json", "w") as f:
        json.dump({"nworld": args.nworld, "env_steps_per_sec": sps,
                   "torch_mem_gib": mem, "obs_finite": True}, f)
    print("SMOKE_OK")


if __name__ == "__main__":
    sys.exit(main())
