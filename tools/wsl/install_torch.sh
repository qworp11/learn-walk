#!/bin/bash
# venv + torch cu126 (проверка sm_61/Pascal)
set -e
uv venv /root/hlc-venv --python 3.10 --allow-existing 2>/dev/null || uv venv /root/hlc-venv --python 3.10
source /root/hlc-venv/bin/activate
export UV_CACHE_DIR=/root/.cache/uv
uv pip install torch --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -2
python - <<'EOF'
import torch, time
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("compiled archs:", torch.cuda.get_arch_list())
x = torch.randn(2048, 2048, device="cuda")
t0 = time.time(); y = x @ x; torch.cuda.synchronize()
print(f"matmul 2048x2048 fp32: {(time.time()-t0)*1000:.1f} ms (первый вызов с JIT)")
t0 = time.time()
for _ in range(10): y = x @ x
torch.cuda.synchronize()
print(f"matmul avg: {(time.time()-t0)*100:.1f} ms x10")
EOF
