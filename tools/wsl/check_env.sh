#!/bin/bash
# Проверка окружения WSL2 (CUDA passthrough, инструменты)
export PATH=/usr/lib/wsl/lib:$PATH
echo "=== nvidia-smi ==="
nvidia-smi 2>&1 | head -14
echo "=== nvcc ==="
nvcc --version 2>/dev/null | tail -1 || echo "nvcc нет — норм, ставится позже при необходимости"
echo "=== python ==="
which python3 && python3 --version
echo "=== uv/pip ==="
which uv pip pip3 2>/dev/null
echo "=== df ==="
df -h / /mnt/d /mnt/g 2>/dev/null
echo "=== cpu ==="
nproc
echo "=== MEM ==="
free -h | head -2
