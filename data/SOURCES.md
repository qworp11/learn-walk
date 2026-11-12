# SOURCES.md — источники данных и лицензии

## LAFAN1 Retargeting Dataset (основные референсы)
- Источник: https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
- Скачано: `data/motions/lafan1_g1/*.csv` (40 файлов, 30 FPS, формат:
  base_pos(3), base_quat_xyzw(4), joint_pos(29))
- Датасет: ретаргетинг LAFAN1 → Unitree G1 (interaction mesh + IK,
  kinematic-only, от lvhaidong). Код ретаргетинга — MIT.
- **Лицензия исходного LAFAN1: CC BY-NC-ND 4.0** (Ubisoft La Forge).
  → Использование: только некоммерческое. Нельзя продавать продукт с этими
  данными, нельзя распространять производные исходных файлов.
  → Наша трассировка/обучение политики: research, ок. Передал Виталику.
- Датасет содержит: walk, run, sprint, jumps, fallAndGetUp, dance, fight.

## LAFAN1 (сырые данные, если понадобятся)
- https://github.com/ubisoft/ubisoft-laforge-animation-dataset — CC BY-NC-ND.

## MuJoCo Playground / REAF / mujoco_warp / rsl_rl / mujoco
- Apache 2.0 / MIT — можно свободно форкать и модифицировать.
- MJCF G1: google-deepmind/mujoco_menagerie (Apache 2.0 / BSD-like per-model).

## Планируемые дополнительные источники (S7, акробатика)
- Mixamo (Adobe): перекаты/cartwheel/сальто — лицензия Mixamo content
  (бесплатно для использования в проектах, нужен аккаунт Adobe).
  Требуется согласие Виталика на создание аккаунта — СНАЧАЛА СПРОСИТЬ.
- Если Mixamo недоступен: генерация референсов траекторной оптимизацией
  (CMA-ES/iLQR по нашему же MJCF) — полностью свой контент.

## Что НЕ используется и почему
- AMASS/SMPL: нужна регистрация с академической почтой + research-лицензия,
  предупреждение заказчику дано; заменён готовым LAFAN1→G1.
- Isaac Gym Preview 4: официальный доступ удалён NVIDIA (deprecated), зеркало
  CodeOcean 403; мёртвый стек Python 3.8/torch 1.13. Выбран MuJoCo Warp.
