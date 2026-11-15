#!/bin/bash
source /root/hlc-venv/bin/activate
python - <<'EOF'
import gc, dataclasses, json, os, time
os.environ.setdefault("MUJOCO_GL", "egl")

import torch
torch.backends.cudnn.enabled = False
import warp as wp
wp.config.enable_mathdx_solver = False
wp.config.enable_mathdx_gemm = False
wp.config.kernel_cache_dir = "/root/wp_kernel_cache"

from mujoco_playground._src.rl import env_registry
from mujoco_playground._src.rl.rsl_rl import vecenv_wrapper
from mujoco_playground.tasks.whole_body.tracking import motion_command

entry = env_registry.get("G1WholeBodyTracking")
agent_cfg = entry.runner_cfg
agent_cfg.nworld = 1
npz = "/mnt/g/LEARN WALK/data/motions/npz_g1/walk1_subject1.npz"
env = entry.env_builder(nworld=1, device="cpu", trajectory_path=npz, play=True)
venv = vecenv_wrapper.RslRlVecEnvWrapper(
    env, clip_actions=agent_cfg.clip_actions,
    cfg=entry.runner_cfg.vecenv_wrapper_cfg)

# найти MotionReferenceProducer и включить uniform-семплирование
producers = [o for o in gc.get_objects()
             if isinstance(o, motion_command.MotionReferenceProducer)]
print("producers found:", len(producers))
for p in producers:
    print("  был режим:", p._sampling_mode)
    p._sampling_mode = "uniform"
    print("  стал: uniform, клип:", p._num_time_steps if hasattr(p, "_num_time_steps") else "?")

nq = 36
for ep in range(5):
    obs_td, _ = venv.reset()
    for p in producers:
        ts = p.time_steps
        print(f"ep{ep}: старт-кадр={int(ts.flatten()[0])}")
    rew_sum = 0.0
    for i in range(50):
        with torch.no_grad():
            action = torch.zeros(1, 29)
        obs_td, rew, done, extras = venv.step(action)
        rew_sum += float(rew.mean().item())
    q = venv.unwrapped.device_layer.sim.data.qpos
    root = q.detach().cpu().numpy()[0, :3]
    print(f"    50 шагов нулевых действий: rew_sum={rew_sum:.2f}, корень={root.round(2)}")
print("TEST_OK")
EOF
