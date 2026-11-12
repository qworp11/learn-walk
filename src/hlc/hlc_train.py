"""Обёртка над mujoco_playground.rl.rsl_rl.train с фиксами для Pascal (sm_61).

1) Отключает libmathdx (cuSolverDx/cuBLASDx) — их LTO не поддерживает sm_61;
   warp использует нативные тайл-реализации Cholesky/GEMM.
2) Ставит кэш warp-ядер в /root/wp_kernel_cache.

Запуск: python /mnt/g/LEARN WALK/src/hlc/hlc_train.py <аргументы train>
"""
import sys

import torch

# Pascal (sm_61): cuDNN 9 не даёт движки для свёрток → нативный fallback.
# Сеть — MLP, cuDNN не нужен.
torch.backends.cudnn.enabled = False

import warp as wp

wp.config.enable_mathdx_solver = False
wp.config.enable_mathdx_gemm = False
wp.config.kernel_cache_dir = "/root/wp_kernel_cache"

from mujoco_playground.rl.rsl_rl import train as pg_train  # noqa: E402

if __name__ == "__main__":
    # подменяем argv на то, что идёт после имени скрипта, и зовём main напрямую
    # (multiproc не нужен: 1080 Ti одна)
    argv = sys.argv[1:]
    pg_train.main(argv)
