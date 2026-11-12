#!/bin/bash
# Замена warp-lang на +cu12 колесо (CUDA 12, поддержка Pascal sm_61)
set -e
source /root/hlc-venv/bin/activate
uv pip install "https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl" 2>&1 | tail -2
python - <<'EOF'
import warp as wp
import torch
print("warp", wp.__version__)
# прямой тест warp-ядра на Pascal
dev = "cuda:0"
@wp.kernel
def double(xs: wp.array[float], ys: wp.array[float]):
    i = wp.tid()
    ys[i] = xs[i] * 2.0

xs = wp.array([1.0, 2.0, 3.0], dtype=float, device=dev)
ys = wp.empty_like(xs)
wp.launch(double, dim=3, inputs=[xs, ys], device=dev)
print("warp kernel result:", ys.numpy())
print("WARP_PASCAL_OK")
EOF
