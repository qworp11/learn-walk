# ENVIRONMENT.md — зафиксированное окружение

Дата фиксации: 2026-11-13

## Железо (хост Windows 10, build 19045)
- GPU: NVIDIA GeForce GTX 1080 Ti, 11 264 MiB VRAM, Pascal (sm_61, cc 6.1)
  - Driver 582.78 (Windows) → внутри WSL виден как 580.178.01
  - CUDA capability драйвера: 13.0 (совместим со стеками CUDA ≤ 13)
  - **Ограничения sm_61:** нет Tensor cores, нет bf16, fp16 без ускорения → только fp32, AMP не использовать
- CPU: Intel i5-12400F (6P/12T, 12 потоков)
- RAM: 16 ГБ (10 ГБ выделено WSL2 через .wslconfig)
- Диски: C: 2.7 ГБ свободно (КРИТИЧНО — ничего системного туда не ставить),
  D: 418 ГБ свободно (WSL vhdx здесь), E: 74 ГБ, G: 45 ГБ свободно (репо проекта)

## ОС-обёртка
- Windows 10 Pro 19045 + WSL2 (WSL 3.0.5, kernel 6.18.5.8, WSLg 1.0.79)
- Дистрибутив: Ubuntu 22.04.5 LTS, установлен в `D:\wsl\Ubuntu-22.04`
  (команда: `wsl --install -d Ubuntu-22.04 --location D:\wsl\Ubuntu-22.04`)
- Сеть: DNS-туннелирование на Win10 НЕ поддерживается → статический DNS
  (8.8.8.8/1.1.1.1), `/etc/resolv.conf` зафиксирован chattr +i,
  в `/etc/wsl.conf`: `generateResolvConf = false`
- `.wslconfig` (C:\Users\user\.wslconfig): memory=10GB, processors=12, swap=4GB
- Работа в WSL от root (compute-VM, интерфейсы sudo не нужны)
- Проброс GPU: `/usr/lib/wsl/lib/libcuda.so` + nvidia-smi работают из WSL ✅

## Проверка CUDA (2026-11-13)
- `nvidia-smi` внутри WSL: GTX 1080 Ti видна, 11264 MiB, драйвер 580.178.01 ✅
- nvcc не установлен (не нужен: warp-lang использует собственный NVRTC-стек)
- Python 3.10.12 системный, uv 0.13.8, venv: /root/hlc-venv

## Программный стек (план)
- torch 2.7.x cu126 (sm_61 в списке arch — проверить при установке, иначе cu118)
- mujoco >= 4.0, mujoco_warp >= 4.0 (батчированная GPU-симуляция)
- reaf-sim >= 1.0.1 (REAF — torch-фреймворк от Google DeepMind Robotics)
- mujoco_playground (последний, с задачей G1WholeBodyTracking)
- rsl-rl-lib 5.4.2 (PPO asymmetric actor-critic)
- warp-lang 1.18.0
- ffmpeg (видеореплеи), OSMesa/EGL GL (headless-рендер)

## Морф ( Unitree G1, 29 dof )
- MJCF: `github:google-deepmind/mujoco_menagerie/unitree_g1/g1_mjlab.xml`
- 29 суставов: 2×6 ноги, 3 талия, 2×7 руки; моторы реальные Unitree
  (5020/7520/4010, усилия 5–139 Н·м, отражённая инерция планетарных редукторов)
- PD-актуаторы позиционного типа: Kp/Kd из natural freq 10 Гц, damping 2.0
- dt 0.005 (200 Гц), control 50 Гц, интегратор implicitfast, Newton solver

## Данные движений
- LAFAN1 Retargeting Dataset → G1 (HuggingFace: lvhaidong/LAFAN1_Retargeting_Dataset, 314 МБ)
- 40 файлов в `data/motions/lafan1_g1/*.csv` (30 FPS):
  walk×12, run×4, sprint×2, jumps×3, fallAndGetUp×6, dance×9, fight×6
- **Лицензия LAFAN1: CC BY-NC-ND 4.0 (некоммерческое, no-derivatives на исходные файлы) — проект research-only**
- Конверсия CSV→NPZ: `csv_to_npz.py` (интерполяция 30→50 FPS, FK-разворот, мировые позы/скорости)

## Мониторинг
- TensorBoard на порту 6006 (WSL2 пробрасывает localhost автоматически)
- Watchdog-цикл: CPU/GPU load, VRAM, температура (nvidia-smi), reward/KL/entropy/grad-norm, NaN
- GPU-лимиты: при >83°C — снизить nworld / power limit (тек. 250 W cap)
