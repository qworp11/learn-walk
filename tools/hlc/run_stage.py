"""Запуск обучения этапа по конфигу configs/stages.json.

Использование: python run_stage.py <stage> [ключ=значение overrides...]

Собирает команду mujoco_playground.rl.rsl_rl.train:
  --env G1WholeBodyTracking --env_overrides '{"trajectory_path": ...}'
  --overrides '{"nworld": ..., "max_iterations": ...}'
  --logdir /root/hlc-logs/<stage>/run_<N>
  --resume_path (если есть последний чекпоинт)

Отдельные скиллы = отдельные запуски с разными trajectory_path.
"""
import json
import os
import pathlib
import subprocess
import sys

PROJ = pathlib.Path("/mnt/g/LEARN WALK")
LOGROOT = pathlib.Path("/root/hlc-logs")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    stage = sys.argv[1]
    manual_overrides = dict(kv.split("=", 1) for kv in sys.argv[2:])
    for k, v in manual_overrides.items():
        try:
            manual_overrides[k] = json.loads(v)
        except json.JSONDecodeError:
            pass

    cfg = json.loads((PROJ / "configs" / "stages.json").read_text())
    if stage not in cfg:
        print(f"Неизвестный этап {stage}. Есть: {list(cfg)}")
        return 1
    c = cfg[stage]
    motion = c["motion"]
    npz = f'/mnt/g/LEARN WALK/data/motions/npz_g1/{motion}.npz'
    if not pathlib.Path(npz).exists():
        print(f"Нет NPZ {npz}. Сначала convert_motions.sh")
        return 1

    overrides = {
        "nworld": c.get("nworld", 1024),
        "max_iterations": c.get("max_iterations", 5000),
        **c.get("overrides", {}),
        **manual_overrides,
    }
    env_overrides = {"trajectory_path": npz, **c.get("env_overrides", {})}

    # выбор стартовой точки:
    # 1) свои чекпоинты → обычный resume (продолжить)
    # 2) cfg.init_from = {"stage": ...} → warm-start от чекпоинта другого этапа
    stage_dir = LOGROOT / stage
    resume = None
    runs = sorted(stage_dir.glob("run_*"), key=lambda p: p.name) if stage_dir.exists() else []
    for cand in reversed(runs):
        pts = sorted(cand.glob("model_*.pt"),
                     key=lambda p: int(p.stem.split("_")[1]) if p.stem[6:].isdigit() else -1)
        if pts:
            run_dir = cand
            resume = str(pts[-1])
            print(f"RESUME: {run_dir.name} <- {resume}", flush=True)
            break
    warm_start = False
    if resume is None:
        init = c.get("init_from")
        if init:
            src_dir = LOGROOT / init["stage"]
            src_runs = sorted(src_dir.glob("run_*"), key=lambda p: p.name) if src_dir.exists() else []
            for cand in reversed(src_runs):
                if (cand / "TRAIN_DONE").exists() or True:  # берём самый свежий чекпоинт
                    pts = sorted(cand.glob("model_*.pt"),
                                 key=lambda p: int(p.stem.split("_")[1]) if p.stem[6:].isdigit() else -1)
                    if pts:
                        resume = str(pts[-1])
                        warm_start = True
                        print(f"WARM-START: {init['stage']}/{cand.name} <- {resume}", flush=True)
                        break
    if resume is None:
        run_dir = stage_dir / f"run_{next_run(stage_dir)}"
        run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", str(PROJ / "src" / "hlc" / "hlc_train.py"),
        "--env", c.get("env", "G1WholeBodyTracking"),
        "--seed", str(c.get("seed", 0)),
        "--logdir", str(run_dir),
        "--env_overrides", json.dumps(env_overrides),
        "--overrides", json.dumps(overrides),
        "--wp_kernel_cache_dir", "/root/wp_kernel_cache",
    ]
    if resume:
        cmd += ["--resume_path", resume]
    print("RUN:", " ".join(cmd), flush=True)
    (run_dir / "cmd.json").write_text(json.dumps({
        "stage": stage, "cmd": cmd, "overrides": overrides,
        "env_overrides": env_overrides, "warm_start": warm_start}, indent=2))
    env = dict(os.environ, HLC_WARM_START="1" if warm_start else "0")
    with (run_dir / "train_stdout.log").open("w") as logf:
        rc = subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT,
                             env=env)
    if rc == 0:
        (run_dir / "TRAIN_DONE").touch()  # маркер штатного завершения
    return rc


def next_run(stage_dir: pathlib.Path) -> int:
    if not stage_dir.exists():
        return 0
    runs = [int(d.name.split("_")[1]) for d in stage_dir.iterdir()
            if d.name.startswith("run_") and d.name.split("_")[1].isdigit()]
    return (max(runs) + 1) if runs else 0


if __name__ == "__main__":
    sys.exit(main())
