"""Роллаут обученной политики: qpos-траектории эпизодов в .npy для видео.

Честный реплей: действия генерирует чекпоинт PPO, физика — та же warp-среда,
что в тренировке. Никакого проигрывания mocap.

Использование (в WSL):
  python '/mnt/g/LEARN WALK/src/hlc/policy_rollout.py' \
    --ckpt /root/hlc-logs/s1_balance/run_0/model_2000.pt \
    --motion dance1_subject1 --out_dir '/mnt/g/LEARN WALK/results/rollouts/s1'
"""
import argparse
import dataclasses
import pathlib
import sys

import numpy as np
import torch

# Pascal-фиксы (до импорта warp/torch-модулей с ядрами)
torch.backends.cudnn.enabled = False
import warp as wp  # noqa: E402

wp.config.enable_mathdx_solver = False
wp.config.enable_mathdx_gemm = False
wp.config.kernel_cache_dir = "/root/wp_kernel_cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--motion", required=True, help="имя референса без .npz")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--nworld", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--play", action="store_true",
                    help="play-режим: старт с кадра 0, без пушей/шума")
    args = ap.parse_args()

    from mujoco_playground._src.rl import env_registry
    from mujoco_playground._src.rl.rsl_rl import checkpoint
    from mujoco_playground._src.rl.rsl_rl import vecenv_wrapper
    from rsl_rl import runners as rsl_runners

    npz = f"/mnt/g/LEARN WALK/data/motions/npz_g1/{args.motion}.npz"
    entry = env_registry.get("G1WholeBodyTracking")
    agent_cfg = entry.runner_cfg
    agent_cfg.nworld = args.nworld

    env_overrides = {"trajectory_path": npz}
    if args.play:
        env_overrides["play"] = True
    env = entry.env_builder(nworld=args.nworld, device=args.device,
                            **env_overrides)
    venv = vecenv_wrapper.RslRlVecEnvWrapper(
        env, clip_actions=agent_cfg.clip_actions,
        cfg=entry.runner_cfg.vecenv_wrapper_cfg)

    runner = rsl_runners.OnPolicyRunner(
        venv, dataclasses.asdict(agent_cfg), device=args.device)
    runner.load(checkpoint.read_checkpoint(args.ckpt),
                map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    print("чекпоинт загружен:", args.ckpt)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nq = 36  # 7 freejoint + 29 суставов

    for ep in range(args.episodes):
        obs_td, _ = venv.reset()
        qpos_hist, rew_hist = [], []
        done = False
        steps = 0
        while not done:
            obs = obs_td["policy"] if "policy" in obs_td.keys() else obs_td
            with torch.no_grad():
                action = policy(obs_td)
            obs_td, rew, done, extras = venv.step(action)
            q = venv.unwrapped.device_layer.sim.data.qpos
            qpos_hist.append(q.detach().cpu().numpy()[0, :nq].copy())
            rew_hist.append(float(rew.mean().item()))
            steps += 1
            if steps >= 600:  # страховка: max_ep_len=500 + запас
                break
        arr = np.stack(qpos_hist)
        np.save(out_dir / f"qpos_ep{ep}.npy", arr)
        print(f"ep{ep}: шагов={len(rew_hist)} средняя rew="
              f"{np.mean(rew_hist):.4f} финальная rew={rew_hist[-1]:.4f} "
              f"-> {out_dir / f'qpos_ep{ep}.npy'}")

    print("ROLLOUT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
