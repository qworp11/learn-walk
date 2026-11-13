"""Проба: как достать qpos из reaf/mujoco_warp симуляции."""
import sys

import torch

torch.backends.cudnn.enabled = False
import warp as wp  # noqa: E402

wp.config.enable_mathdx_solver = False
wp.config.enable_mathdx_gemm = False
wp.config.kernel_cache_dir = "/root/wp_kernel_cache"

from mujoco_playground._src.rl import env_registry  # noqa: E402
from mujoco_playground._src.rl.rsl_rl import vecenv_wrapper  # noqa: E402

entry = env_registry.get("G1WholeBodyTracking")
env = entry.env_builder(nworld=2, device="cuda",
                        trajectory_path="/mnt/g/LEARN WALK/data/motions/npz_g1/dance1_subject1.npz")
venv = vecenv_wrapper.RslRlVecEnvWrapper(env, cfg=entry.runner_cfg.vecenv_wrapper_cfg)
obs_td, _ = venv.reset()
a = torch.zeros(2, venv.num_actions, device="cuda")
venv.step(a)

reaf_env = venv.unwrapped
print("env type:", type(reaf_env).__name__)
dl = getattr(reaf_env, "device_layer", None)
print("device_layer:", type(dl).__name__ if dl else None)
sim = getattr(dl, "sim", None)
print("sim:", type(sim).__name__ if sim else None)
if sim is not None:
    print("sim attrs:", [x for x in dir(sim) if not x.startswith("_")][:30])
    data = getattr(sim, "data", None)
    if data is not None:
        print("data attrs:", [x for x in dir(data) if not x.startswith("_")][:40])
        q = getattr(data, "qpos", None)
        if q is not None:
            print("qpos type:", type(q).__name__, "shape:", q.shape)
            print("qpos[0][:40]:", q.numpy()[0][:40])
