# LEARN WALK — гуманоид учится ходить (в прямом эфире)

Обучение политики whole-body tracking для гуманоида **Unitree G1** на одной потребительской
видеокарте (**GTX 1080 Ti**): стояние → ходьба → бег → спринт → прыжки → падения и вставания → танец.

Весь процесс обучения идёт в прямом эфире на YouTube («ИИ младенец: 0 шагов → уверенный бег»):
LIVE-окно показывает робота, выполняющего **самую свежую** политику — каждые 500 итераций чекпоинт
подменяется на лету, и зрители видят, как навык растёт.

## Стек

- **MuJoCo Playground** — whole-body tracking (подход BeyondMimic), env G1
- **rsl_rl** — PPO
- **MuJoCo Warp** — симуляция: 1024 параллельных мира на одном GPU
- Тренировка в WSL2 Ubuntu 22.04 + CUDA 12.9; стрим/дашборд — CPU-роллауты, GPU не трогают

## Лестница этапов

Каждый этап **warm-start'ится** от предыдущего (перенос весов актёра, `init_from` в
`configs/stages.json`) — поэтому бег учится за час-полтора, а не с нуля.

| Этап | Навык | Референс | Статус |
|---|---|---|---|
| s1_balance | стояние | stand | ✅ rew −2.08 → +2.5 |
| s2_walk | ходьба | walk1 | ✅ rew 0.34 → 15.0 |
| s2_walk_run_a | доводка ходьбы (resume +5000) | walk1 | 🔄 в эфире |
| s3_run | бег | run1 | ⏳ |
| s3_sprint | спринт | sprint1 | ⏳ |
| s4_jump | прыжки | jumps1 | ⏳ |
| s6_fall / s6_fall2 | падения и вставания | fallAndGetUp | ⏳ |
| s8_dance | танец | dance1 | ⏳ |

## Структура

```
src/hlc/            обучение (hlc_train), live-стрим, дашборд, рендер видео, роллауты
tools/hlc/          run_stage.py — запуск этапа: train/resume/warm-start
tools/wsl/          оркестратор лестницы, watchdog, автозапуск после ребута, стрим-утилиты
tools/win/          скачивание LAFAN1→G1, старт конвейера
configs/stages.json параметры всех этапов + init_from
data/SOURCES.md     источники данных и лицензии
PROGRESS.md         честный журнал: включая инциденты (OOM, рестарты) и выводы
```

Ключевые решения:
- **Стрим :8081** — MJPEG, CPU-физика + EGL, uniform-старты эпизодов (случайное место клипа),
  плашка с этапом/итерацией/наградой из тренировочного лога (честные цифры)
- **Оркестратор** сам ведёт лестницу этапов, дозавершает обрывы, пишет отчёт
- **Watchdog** считает прогресс по итерациям и убивает зависшие тренировки
- **boot_pipeline.sh** — автозапуск всего конвейера после перезагрузки Windows

## Данные (LAFAN1)

Референс-моции — [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
(LAFAN1 → Unitree G1, 40 клипов).

**Моции в репозиторий не входят**: лицензия исходного LAFAN1 — CC BY-NC-ND 4.0
(Ubisoft La Forge), распространение производных запрещено. Обучение/исследование — можно.
Скачать и конвертировать: `data/SOURCES.md`, `tools/win/download_lafan_g1.ps1`, `tools/wsl/convert_motions.sh`.

## Быстрый старт (WSL2)

```bash
bash tools/wsl/install_stack.sh      # venv + стек (py3.12, mujoco_playground[rl])
bash tools/wsl/convert_motions.sh    # CSV → NPZ (нужны скачанные LAFAN1→G1)
bash tools/wsl/boot_pipeline.sh      # оркестратор + стрим + дашборд + watchdog
```

## Лицензия

Код репозитория — MIT (см. LICENSE). Данные LAFAN1 — под их собственной лицензией
(CC BY-NC-ND 4.0), в репозиторий не входят.
