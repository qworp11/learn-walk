#!/bin/bash
source /root/hlc-venv/bin/activate
python - <<'EOF'
import sys
sys.path.insert(0, '/mnt/g/LEARN WALK/src/hlc')
# как в live_stream: тот же порядок импортов
import glob, io, json, os, dataclasses, threading, time, traceback
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
import torch
torch.backends.cudnn.enabled = False
import warp as wp
wp.config.enable_mathdx_solver = False
wp.config.enable_mathdx_gemm = False
wp.config.kernel_cache_dir = "/root/wp_kernel_cache"
import mujoco
from mujoco import Renderer

# проверить объект Renderer из mujoco
print("Renderer:", Renderer)
import inspect
sig = inspect.signature(Renderer.__init__)
print("sig:", sig)

from mujoco_playground.robots.g1.sim import scene as g1_scene
model = g1_scene.g1_scene_spec().compile()
print("offwidth/offheight:", model.vis.global_.offwidth, model.vis.global_.offheight)
r = Renderer(model, height=270, width=480)
print("Renderer OK после torch+warp (egl)")
EOF
