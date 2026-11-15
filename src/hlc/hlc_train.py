"""Обёртка над mujoco_playground.rl.rsl_rl.train с фиксами для Pascal (sm_61).

1) Отключает libmathdx (cuSolverDx/cuBLASDx) — их LTO не поддерживает sm_61;
   warp использует нативные тайл-реализации Cholesky/GEMM.
2) Ставит кэш warp-ядер в /root/wp_kernel_cache.
3) HLC_WARM_START=1: resume-чекпоинт грузится как warm-start
   (actor переносится, optimizer и счётчик итераций — свежие).

Запуск: python /mnt/g/LEARN WALK/src/hlc/hlc_train.py <аргументы train>
"""
import os

import torch

# Pascal (sm_61): cuDNN 9 не даёт движков для свёрток → нативный fallback.
# Сеть — MLP, cuDNN не нужен.
torch.backends.cudnn.enabled = False

import warp as wp

wp.config.enable_mathdx_solver = False
wp.config.enable_mathdx_gemm = False
wp.config.kernel_cache_dir = "/root/wp_kernel_cache"

from absl import app  # noqa: E402

from mujoco_playground.rl.rsl_rl import train as pg_train  # noqa: E402

_WARM_START = os.environ.get("HLC_WARM_START") == "1"

if _WARM_START:
    from rsl_rl import runners as rsl_runners

    _orig_load = rsl_runners.OnPolicyRunner.load

    def _warm_load(self, path, load_cfg=None, strict=True,
                   map_location=None):
        cfg = {"actor": True, "critic": False,
               "optimizer": False, "iteration": False}
        print("WARM-START load:", path, cfg, flush=True)
        return _orig_load(self, path, load_cfg=cfg, strict=strict,
                          map_location=map_location)

    rsl_runners.OnPolicyRunner.load = _warm_load


def main(argv):
    # флаги уже распарсены app.run; playground main принимает argv
    pg_train.main(argv)


if __name__ == "__main__":
    app.run(main)
