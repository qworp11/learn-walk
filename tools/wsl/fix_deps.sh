#!/bin/bash
# Фикс зависимостей: torch cu126 (sm_61), warp cu12, всё остальное по пинам reaf
set -e
source /root/hlc-venv/bin/activate
export UV_CACHE_DIR=/root/.cache/uv

echo "== 1. torch 2.9.1+cu126 =="
uv pip install "torch==2.9.1+cu126" --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -1

echo "== 2. warp-lang 1.18.0+cu12 (Pascal ok) =="
uv pip install "https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl" 2>&1 | tail -1

echo "== 3. constraints-защита от апгрейда torch/warp при достановке прочего =="
cat > /root/constraints.txt <<'EOF'
torch==2.9.1+cu126
warp-lang==1.18.0+cu12
rsl-rl-lib==5.4.2
EOF
uv pip install --constraint /root/constraints.txt --extra-index-url https://download.pytorch.org/whl/cu126 \
  "playground[rl]" "reaf-sim>=1.0.1" "mujoco_warp>=4.0.0" "mujoco[sysid]>=4.0.0" \
  numpy scipy imageio imageio-ffmpeg 2>&1 | tail -2

echo "== 4. Верификация =="
python - <<'PYEOF'
import torch, time
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| archs:", torch.cuda.get_arch_list())
x = torch.randn(2048, 2048, device="cuda")
t0 = time.time()
for _ in range(10): y = x @ x
torch.cuda.synchronize()
print(f"matmul {(time.time()-t0)*100:.1f} ms x10")
import warp as wp
print("warp", wp.__version__)
@wp.kernel
def dbl(a: wp.array[float], b: wp.array[float]):
    i = wp.tid(); b[i] = a[i] * 2.0
xs = wp.array([1., 2., 3.], dtype=float, device="cuda:0")
ys = wp.empty_like(xs)
wp.launch(dbl, dim=3, inputs=[xs, ys], device="cuda:0")
print("warp result:", ys.numpy().tolist(), "(pascal kernel OK)")
import mujoco, mujoco_warp, reaf, mujoco_playground, rsl_rl
print("mujoco", mujoco.__version__, "| mujoco_warp OK | reaf OK | playground OK | rsl_rl", rsl_rl.__version__ if hasattr(rsl_rl, "__version__") else "ok")
import numpy, scipy, imageio
print("numpy", numpy.__version__, "scipy", scipy.__version__, "imageio OK")
print("STACK_OK")
PYEOF
