"""Живой стрим обучения: MJPEG http://localhost:8081/stream

Отдельный процесс. Крутит непрерывный роллаут свежайшего чекпоинта
(физика warp на CPU — тренировку на GPU не трогает; рендер EGL 480x270).
Между эпизодами проверяет новые чекпоинты и горячо подменяет политику:
робот в окне улучшается на глазах. Внизу кадра — плашка с этапом,
номером чекпоинта и текущей наградой.

Рецепт env/политики — 1:1 как в policy_rollout.py (он проверен демо-съёмкой).
"""
import glob
import io
import json
import os
import dataclasses
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import torch  # noqa: E402

torch.backends.cudnn.enabled = False
import warp as wp  # noqa: E402

wp.config.enable_mathdx_solver = False
wp.config.enable_mathdx_gemm = False
wp.config.kernel_cache_dir = "/root/wp_kernel_cache"

import mujoco  # noqa: E402
from mujoco import Renderer  # noqa: E402

LOGROOT = "/root/hlc-logs"
W, H = 640, 360
JPEGQ = 80
MAX_EP_STEPS = 600  # страховка (max_ep_len=500 + запас)

# человеческие названия этапов для зрителей стрима
STAGE_RU = {
    "s1_balance": "СТОЯНИЕ",
    "s2_walk": "ХОДЬБА",
    "s2_walk_run_a": "БЫСТРАЯ ХОДЬБА",
    "s3_run": "БЕГ",
    "s3_sprint": "СПРИНТ",
    "s4_jump": "ПРЫЖКИ",
    "s6_fall": "ПАДЕНИЯ И ВСТАВАНИЯ 1",
    "s6_fall2": "ПАДЕНИЯ И ВСТАВАНИЯ 2",
    "s8_dance": "ТАНЕЦ",
}

_stats_cache = {"path": None, "mtime": 0, "stats": {}}


def train_stats(run_dir):
    """(итерация, всего, награда) из хвоста тренировочного лога — с кэшем mtime."""
    log = os.path.join(run_dir, "train_stdout.log")
    try:
        m = os.path.getmtime(log)
    except OSError:
        return None
    if _stats_cache["path"] == log and _stats_cache["mtime"] == m:
        return _stats_cache["stats"]
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8000))
            txt = f.read().decode("utf-8", "ignore")
    except OSError:
        return None
    import re

    st = {}
    it = re.findall(r"Learning iteration (\d+)/(\d+)", txt)
    if it:
        st["iter"], st["total"] = int(it[-1][0]), int(it[-1][1])
    rew = re.findall(r"Mean reward: ([-0-9.e]+)", txt)
    if rew:
        st["rew"] = float(rew[-1])
    _stats_cache.update(path=log, mtime=m, stats=st)
    return st

_state = {"jpg": None, "lock": threading.Lock()}


def latest_checkpoint():
    """(ckpt_path, run_dir) свежайшего run-каталога по mtime, где есть model_*.pt."""
    runs = sorted(glob.glob(f"{LOGROOT}/*/run_*"), key=os.path.getmtime, reverse=True)
    for run in runs:
        pts = glob.glob(f"{run}/model_*.pt")
        if not pts:
            continue
        pts.sort(key=lambda p: int(os.path.basename(p)[6:-3])
                 if os.path.basename(p)[6:-3].isdigit() else -1)
        return pts[-1], run
    return None, None


def build_scene():
    from mujoco_playground.robots.g1.sim import scene as g1_scene

    spec = g1_scene.g1_scene_spec()
    return spec.compile()


def enable_uniform_starts():
    """Стрим: стартовать эпизоды со случайного места клипа.

    play-режим всегда начинает с кадра 0 — у walk1 это пауза актёра в первые
    4 секунды (зритель видит «2 шага и стоит»). uniform-семплирование
    разбрасывает старты по всему клипу.
    """
    import gc

    from mujoco_playground.tasks.whole_body.tracking import motion_command

    n = 0
    for o in gc.get_objects():
        if isinstance(o, motion_command.MotionReferenceProducer):
            o._sampling_mode = "uniform"
            n += 1
    print(f"stream: uniform-старты включены ({n} producer)", flush=True)
    return n


_FONTS = {}


def _font_at(size):
    from PIL import ImageFont

    if size not in _FONTS:
        for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            try:
                _FONTS[size] = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
        else:
            try:
                _FONTS[size] = ImageFont.load_default(size=size)
            except Exception:
                _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


FONT = None


def encode_frame(img, caption):
    """numpy HxWx3 → JPEG bytes с нижней плашкой."""
    global FONT
    from PIL import Image, ImageDraw

    if FONT is None:
        FONT = _font_at(20)
    im = Image.fromarray(np.asarray(img))
    d = ImageDraw.Draw(im)
    wpx = im.width
    d.rectangle([0, im.height - 32, wpx, im.height], fill=(0, 0, 0))
    font = FONT
    for size in (20, 18, 16, 14, 12):
        cand = _font_at(size)
        font = cand  # безотказно: каждый проход уменьшает шрифт
        if d.textlength(caption, font=cand) <= wpx - 16:
            break
    d.text((8, im.height - 28), caption, fill=(255, 255, 255), font=font)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=JPEGQ)
    return b.getvalue()


def rollout_loop():
    global FONT
    model = build_scene()
    data = mujoco.MjData(model)
    nq = model.nq  # 36: 7 freejoint + 29 суставов
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    cam = mujoco.MjvCamera()
    print("HDBG H=", repr(H), type(H).__name__, "W=", repr(W), type(W).__name__,
          flush=True)
    try:
        renderer = Renderer(model, height=H, width=W)
    except Exception as e:
        print("EGL failed:", e, "→ osmesa", flush=True)
        os.environ["MUJOCO_GL"] = "osmesa"
        renderer = Renderer(model, height=H, width=W)

    from mujoco_playground._src.rl import env_registry
    from mujoco_playground._src.rl.rsl_rl import checkpoint
    from mujoco_playground._src.rl.rsl_rl import vecenv_wrapper
    from rsl_rl import runners as rsl_runners

    cur_ckpt = None
    cur_stage = None
    policy = None
    venv = None
    ep_rew, ep_steps = 0.0, 0

    while True:
        try:
            ckpt, run_dir = latest_checkpoint()
            if ckpt is None:
                with _state["lock"]:
                    _state["jpg"] = encode_frame(
                        np.zeros((H, W, 3), np.uint8), "нет чекпоинтов — жду…")
                time.sleep(10)
                continue

            stage = os.path.basename(os.path.dirname(run_dir))
            new_stage = stage != cur_stage

            # смена этапа → пересобрать env под новый референс
            if venv is None or new_stage:
                motion = ""
                try:
                    cmd = json.load(open(os.path.join(run_dir, "cmd.json")))
                    motion = cmd.get("env_overrides", {}).get("trajectory_path", "")
                except Exception:
                    pass
                with _state["lock"]:
                    _state["jpg"] = encode_frame(
                        np.zeros((H, W, 3), np.uint8),
                        f"пересборка env: {STAGE_RU.get(stage, stage)}…")
                if venv is not None:
                    try:
                        venv.close()
                    except Exception:
                        pass
                    venv = None
                    policy = None
                try:
                    entry = env_registry.get("G1WholeBodyTracking")
                    agent_cfg = entry.runner_cfg
                    agent_cfg.nworld = 1
                    env = entry.env_builder(
                        nworld=1, device="cpu", trajectory_path=motion or None,
                        play=True)
                    venv = vecenv_wrapper.RslRlVecEnvWrapper(
                        env, clip_actions=agent_cfg.clip_actions,
                        cfg=entry.runner_cfg.vecenv_wrapper_cfg)
                    enable_uniform_starts()
                    cur_stage = stage
                    ep_rew, ep_steps = 0.0, 0
                    print("stream: env готов,", stage, motion, flush=True)
                except Exception:
                    print("env build fail:", flush=True)
                    traceback.print_exc()
                    time.sleep(15)
                    continue

            # горячая подмена политики при новом чекпоинте
            if policy is None or ckpt != cur_ckpt:
                try:
                    entry = env_registry.get("G1WholeBodyTracking")
                    agent_cfg = entry.runner_cfg
                    runner = rsl_runners.OnPolicyRunner(
                        venv, dataclasses.asdict(agent_cfg), device="cpu")
                    runner.load(checkpoint.read_checkpoint(ckpt),
                                load_cfg={"actor": True, "critic": False,
                                          "optimizer": False, "iteration": False},
                                map_location="cpu")
                    policy = runner.get_inference_policy(device="cpu")
                    cur_ckpt = ckpt
                    ep_rew, ep_steps = 0.0, 0
                    print("stream: policy ->", ckpt, flush=True)
                except Exception:
                    print("policy load fail:", ckpt, flush=True)
                    traceback.print_exc()
                    policy = None
                    time.sleep(10)
                    continue

            # эпизод
            obs_td, _ = venv.reset()
            ep_rew, ep_steps = 0.0, 0
            trn = {}
            t0 = time.time()
            while True:
                step_t = time.time()
                # чекпоинт обновился? проверяем раз в ~1с (каждые 50 шагов)
                if ep_steps % 50 == 0:
                    nc, _ = latest_checkpoint()
                    if nc and nc != cur_ckpt:
                        break
                with torch.no_grad():
                    action = policy(obs_td)
                obs_td, rew, done, extras = venv.step(action)
                ep_rew += float(rew.mean().item())
                ep_steps += 1

                q = venv.unwrapped.device_layer.sim.data.qpos
                data.qpos[:] = q.detach().cpu().numpy()[0, :nq]
                mujoco.mj_forward(model, data)
                cam.lookat[:] = [data.xpos[pelvis_id][0],
                                 data.xpos[pelvis_id][1],
                                 data.xpos[pelvis_id][2] + 0.2]
                cam.distance = 3.0
                cam.elevation = -10
                cam.azimuth = -90 + ((ep_steps // 50) * 4) % 360
                renderer.update_scene(data, camera=cam)
                img = renderer.render()

                if ep_steps % 25 == 0:
                    trn = train_stats(run_dir) or {}
                name = STAGE_RU.get(stage, stage)
                parts = [name]
                if trn.get("iter") is not None:
                    parts.append(f"{trn['iter']}/{trn['total']}")
                if trn.get("rew") is not None:
                    parts.append(f"награда {trn['rew']:.1f}")
                parts.append(f"шаг {ep_steps / 50:.1f}/10с")
                cap = " · ".join(parts)
                with _state["lock"]:
                    _state["jpg"] = encode_frame(img, cap)

                # реальное время: 1 шаг = 20 мс (50 Гц)
                dt = time.time() - step_t
                if dt < 0.02:
                    time.sleep(0.02 - dt)
                if done or ep_steps >= MAX_EP_STEPS:
                    break
            fps = ep_steps / max(time.time() - t0, 1e-6)
            print(f"stream: эпизод завершён, {ep_steps} шагов, {fps:.0f} шаг/с",
                  flush=True)
            time.sleep(0.2)
        except Exception:
            print("rollout loop fail:", flush=True)
            traceback.print_exc()
            time.sleep(10)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = None
            try:
                while True:
                    with _state["lock"]:
                        buf = _state["jpg"]
                    if buf is None or buf == last:
                        time.sleep(0.05)
                        continue
                    last = buf
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(buf)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=rollout_loop, daemon=True).start()
    print("live stream: http://localhost:8081/stream", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8081), Handler).serve_forever()
