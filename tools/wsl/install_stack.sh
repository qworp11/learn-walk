#!/bin/bash
# Пересоздание venv на Python 3.12 + полный стек
set -e
export UV_CACHE_DIR=/root/.cache/uv
uv python install 3.12 2>&1 | tail -1
rm -rf /root/hlc-venv
uv venv /root/hlc-venv --python 3.12
source /root/hlc-venv/bin/activate
echo "== torch 2.9.1+cu126 =="
uv pip install "torch==2.9.1+cu126" --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -2
echo "== playground[rl] + reaf + warp =="
uv pip install "playground[rl]" "reaf-sim>=1.0.1" "mujoco_warp>=4.0.0" "warp-lang>=1.18.0,<1.18.1" "mujoco[sysid]>=4.0.0" 2>&1 | tail -3
uv pip install numpy scipy imageio imageio-ffmpeg 2>&1 | tail -1
python - <<'EOF'
import torch, mujoco, warp, numpy
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("archs:", torch.cuda.get_arch_list())
print("mujoco", mujoco.__version__, "| warp", warp.__version__)
import mujoco_warp; print("mujoco_warp OK")
import reaf; print("reaf OK")
import playground; print("playground OK")
x = torch.randn(2048, 2048, device="cuda"); import time
t0=time.time()
for _ in range(10): y = x @ x
torch.cuda.synchronize()
print(f"matmul avg {(time.time()-t0)*100:.1f} ms x10")
EOF
