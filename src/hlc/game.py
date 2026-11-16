"""Игра-полигон: тестируй выученные скиллы G1 своими руками.

Запуск (WSL): tools/game.bat или
  wsl -d Ubuntu-22.04 -u root -- bash "/mnt/g/LEARN WALK/tools/wsl/run_game.sh"

Управление:
  W — ходьба            SHIFT+W — спринт        E — бег
  ПРОБЕЛ — прыжок       CTRL — сесть (после дообучения s9_sit)
  F — упасть            G — встать              Q — танец
  A / D — камера влево/вправо   колесо — зум    T — сменить ландшафт
  H — помощь            ESC — выход

Ландшафты: РОВНАЯ / УХАБЫ / ЯМЫ / СКАЛЫ / НЕРОВНАЯ / СТЕНА.
Ландшафт патчится и в ФИЗИКУ env (через g1_scene_spec), и в картинку —
никакой подделки: робот реально сталкивается с рельефом.

Рецепт env/политики 1:1 из live_stream.py / policy_rollout.py.
Autotest: GAME_AUTOTEST=1 python game.py — гоняет матрицу скилл×ландшафт
без окна (SDL dummy) и печатает PASS/FAIL по каждому пункту.
"""
import dataclasses
import glob
import json
import os
import re
import sys
import time
import traceback

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import torch  # noqa: E402

torch.backends.cudnn.enabled = False
import warp as wp  # noqa: E402

wp.config.enable_mathdx_solver = False
wp.config.enable_mathdx_gemm = False
wp.config.kernel_cache_dir = "/root/wp_kernel_cache"

import mujoco_warp as _mjw  # noqa: E402

_orig_put_data = _mjw.put_data


def _put_data_bigger(mjm, mjd, **kw):
    """reaf выделяет буферы контактов впритык, а hfield генерирует сотни
    контактов (робот × ячейки сетки) — поднимаем минимум, иначе
    nconmax overflow на этапе сборки env."""
    kw["nconmax"] = max(kw.get("nconmax") or 0, 2048)
    kw["njmax"] = max(kw.get("njmax") or 0, 2048)
    return _orig_put_data(mjm, mjd, **kw)


_mjw.put_data = _put_data_bigger

import mujoco  # noqa: E402
from mujoco import Renderer  # noqa: E402

LOGROOT = "/root/hlc-logs"
WIN_W, WIN_H = 960, 540      # окно игры
RENDER_W, RENDER_H = 800, 450  # рендер EGL (растягиваем в окно)
MAX_EP_STEPS = 600
AUTOTEST = os.environ.get("GAME_AUTOTEST") == "1"
if AUTOTEST:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# ---------------------------------------------------------------- скиллы ---
# (каталог этапа, русское имя, залипающий, лимит шагов)
SKILLS = {
    "stand": ("s1_balance", "СТОЯНИЕ", False, 0),
    "walk": ("s2_walk", "ХОДЬБА", False, 0),
    "run": ("s3_run", "БЕГ", False, 0),
    "sprint": ("s3_sprint", "СПРИНТ", False, 0),
    "jump": ("s4_jump", "ПРЫЖОК", True, 400),
    "fall": ("s6_fall", "ПАДЕНИЕ", True, 400),
    "getup": ("s6_fall2", "ВСТАТЬ", True, 600),
    "dance": ("s8_dance", "ТАНЕЦ", True, 1200),
    "sit": ("s9_sit", "СЕСТЬ", True, 600),  # появится после дообучения
}
STICKY = {k for k, v in SKILLS.items() if v[2]}


def stage_run_dir(stage):
    runs = sorted(glob.glob(f"{LOGROOT}/{stage}/run_*"))
    if not runs:
        return None
    done = [r for r in runs if os.path.exists(os.path.join(r, "TRAIN_DONE"))]
    return done[-1] if done else runs[-1]


def stage_ckpt(run_dir):
    if run_dir is None:
        return None
    pts = glob.glob(run_dir + "/model_*.pt")
    if not pts:
        return None

    def it(p):
        m = re.search(r"model_(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1

    pts.sort(key=it)
    return pts[-1]


def stage_traj(run_dir):
    try:
        cmd = json.load(open(os.path.join(run_dir, "cmd.json")))
        return cmd.get("env_overrides", {}).get("trajectory_path", "")
    except Exception:
        return ""


def skill_ready(key):
    stage, _, _, _ = SKILLS[key]
    return stage_ckpt(stage_run_dir(stage)) is not None


# ------------------------------------------------------------- ландшафты ---
CURRENT_TERRAIN = "flat"

TERRAIN_RU = {
    "flat": "РОВНАЯ ДОРОГА",
    "bumps": "ДОРОГА С УХАБАМИ",
    "holes": "ДОРОГА С ЯМАМИ",
    "rock": "СКАЛЫ",
    "rough": "НЕРОВНАЯ ПОВЕРХНОСТЬ",
    "wall": "СТЕНА",
}
TERRAIN_ORDER = ["flat", "bumps", "holes", "rock", "rough", "wall"]


def _terrain_geoms(spec):
    """Добавить геомы текущего ландшафта в MjSpec (физика + картинка)."""
    if CURRENT_TERRAIN == "flat":
        return
    if CURRENT_TERRAIN in ("bumps", "holes", "rough"):
        # MuJoCo нормирует данные hfield min-max в [0..1], warp считает
        # высоту = data * size[2]. Проектируем высоты в МЕТРАХ с минимумом
        # ровно 0 и zmax = максимум — тогда нормировка тождественна и
        # картинка = физике.
        nrow, ncol = 140, 70          # сетка высот
        sx, sy = 7.0, 3.5             # полные размеры поля
        xs = np.linspace(-sx / 2, sx / 2, nrow)
        ys = np.linspace(-sy / 2, sy / 2, ncol)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        if CURRENT_TERRAIN == "bumps":      # ухабы: волны 0..12 см
            h = 0.06 * np.sin(2 * np.pi * X / 0.9) * \
                np.cos(2 * np.pi * Y / 1.4)
        elif CURRENT_TERRAIN == "rough":    # неровная: сглаженный шум 0..10 см
            rng = np.random.default_rng(7)
            h = rng.normal(0, 1, (nrow, ncol))
            k = np.ones((9, 9)) / 81.0
            for _ in range(2):
                h = np.abs(np.fft.irfft2(np.fft.rfft2(h) *
                                         np.fft.rfft2(k, h.shape)))
            h = (h - h.min()) / (np.ptp(h) + 1e-9) * 0.10
        else:  # holes: дорога поднята на 10 см, ямы уходят до нуля
            h = np.full((nrow, ncol), 0.10)
            hole = (np.abs(np.mod(X + 0.7, 1.4) - 0.7) < 0.35) & \
                   (np.abs(Y) < 0.55) & (X > 0.5)
            h[hole] = 0.0
        h = h - h.min()                      # минимум ровно 0
        zmax = max(float(h.max()), 1e-3)
        data = (h / zmax).ravel().astype(np.float32)
        hf = spec.add_hfield(name="hf_terrain",
                             size=[sx / 2, sy / 2, zmax, 0.5],
                             nrow=nrow, ncol=ncol)
        hf.userdata = list(data)
        g = spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_HFIELD, pos=[0.0, 0.0, 0.0],
            name="terrain_hf", rgba=[0.55, 0.45, 0.35, 1.0])
        g.hfieldname = "hf_terrain"
        return
    if CURRENT_TERRAIN == "rock":
        boulders = [(2.0, 0.0, 0.55), (3.4, 0.7, 0.75), (3.5, -0.8, 0.65),
                    (5.0, 0.0, 0.9), (4.9, 1.0, 0.6), (6.2, -0.6, 0.8)]
        for i, (x, y, s) in enumerate(boulders):
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[s, s, s],
                pos=[x, y, s * 0.92], name=f"boulder_{i}",
                rgba=[0.42, 0.40, 0.38, 1.0])
        for i, (x, y, s) in enumerate([(2.6, -0.6, 0.25), (4.2, 0.9, 0.2),
                                       (5.7, 0.4, 0.3)]):
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX, size=[s, s, s * 0.6],
                pos=[x, y, s * 0.6], name=f"rock_{i}",
                rgba=[0.5, 0.47, 0.44, 1.0])
        return
    if CURRENT_TERRAIN == "wall":
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.15, 2.0, 0.6],
            pos=[4.0, 0.0, 0.6], name="wall",
            rgba=[0.7, 0.5, 0.35, 1.0])


def _install_scene_patch():
    import mujoco_playground.robots.g1.sim.scene as scene_mod

    orig = scene_mod.g1_scene_spec

    def patched(xml_uri=None):
        spec = orig(xml_uri)
        _terrain_geoms(spec)
        return spec

    scene_mod.g1_scene_spec = patched
    return patched


_build_visual_model = None


def visual_model():
    """Визуальная модель: та же сцена, что и у физики env."""
    import mujoco_playground.robots.g1.sim.scene as scene_mod

    spec = scene_mod.g1_scene_spec()
    return spec.compile()


# --------------------------------------------------------------- env/мод ---
def build_env_and_policy(ckpt, traj):
    from mujoco_playground._src.rl import env_registry
    from mujoco_playground._src.rl.rsl_rl import checkpoint, vecenv_wrapper
    from rsl_rl import runners as rsl_runners

    entry = env_registry.get("G1WholeBodyTracking")
    agent_cfg = entry.runner_cfg
    agent_cfg.nworld = 1
    env = entry.env_builder(nworld=1, device="cpu",
                            trajectory_path=traj or None, play=True)
    venv = vecenv_wrapper.RslRlVecEnvWrapper(
        env, clip_actions=agent_cfg.clip_actions,
        cfg=agent_cfg.vecenv_wrapper_cfg)
    runner = rsl_runners.OnPolicyRunner(
        venv, dataclasses.asdict(agent_cfg), device="cpu")
    runner.load(checkpoint.read_checkpoint(ckpt),
                load_cfg={"actor": True, "critic": False,
                          "optimizer": False, "iteration": False},
                map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")
    return venv, policy


def load_skill(key):
    """venv+policy скилла. Возвращает (venv, policy) или None (нет чекпоинта)."""
    stage, _, _, _ = SKILLS[key]
    run = stage_run_dir(stage)
    ckpt = stage_ckpt(run)
    if ckpt is None:
        return None
    print(f"game: загрузка скилла {key} ({ckpt})", flush=True)
    t0 = time.time()
    venv, policy = build_env_and_policy(ckpt, stage_traj(run))
    print(f"game: скилл {key} готов за {time.time() - t0:.0f}с", flush=True)
    return venv, policy


class SkillManager:
    """LRU-кэш venv (2 шт) + кэш политик (все, они маленькие)."""

    def __init__(self, max_live=2):
        self.max_live = max_live
        self.live = {}       # key -> (venv, policy)
        self.order = []      # LRU-порядок живых venv

    def get(self, key, protect=None):
        if key in self.live:
            self.order.remove(key)
            self.order.append(key)
            return self.live[key]
        try:
            built = load_skill(key)
        except Exception:
            traceback.print_exc()
            built = None
        if built is None:
            return None
        self.live[key] = built
        self.order.append(key)
        # выгружаем лишние, но НИКОГДА не трогаем активный (protect) и новый
        while True:
            evictable = [k for k in self.order
                         if k in self.live and k != key and k != protect]
            if len(self.live) <= self.max_live or not evictable:
                break
            old = evictable[0]
            self.order.remove(old)
            try:
                self.live[old][0].close()
            except Exception:
                pass
            del self.live[old]
            print(f"game: скилл {old} выгружен (LRU)", flush=True)
        return self.live[key]

    def drop_all(self):
        for k, (venv, _) in self.live.items():
            try:
                venv.close()
            except Exception:
                pass
        self.live.clear()
        self.order.clear()


# ------------------------------------------------------------ автотест ----
def _finite(obs):
    return bool(torch.isfinite(obs["policy"]).all().item())


def autotest():
    """Матрица скилл×ландшафт без окна. Печатает PASS/FAIL, exit 0/1."""
    _install_scene_patch()

    results = []
    terr_full = ["flat", "bumps", "wall"]
    terr_half = ["holes", "rock", "rough"]
    skills_full = ["stand", "walk", "sprint", "run", "jump", "fall",
                   "getup", "dance"]
    skills_half = ["stand", "walk", "jump"]
    plan = [(t, skills_full) for t in terr_full] + \
           [(t, skills_half) for t in terr_half]

    mgr = SkillManager(max_live=2)
    global CURRENT_TERRAIN
    for terr, skills in plan:
        CURRENT_TERRAIN = terr
        mgr.drop_all()
        for key in skills:
            if not skill_ready(key):
                results.append((terr, key, "SKIP", "чекпоинта нет"))
                continue
            try:
                got = mgr.get(key)
                if got is None:
                    results.append((terr, key, "FAIL", "загрузка вернула None"))
                    continue
                venv, policy = got
                obs_td, _ = venv.reset()
                ok = True
                msg = ""
                for step in range(100):
                    with torch.no_grad():
                        a = policy(obs_td)
                    if not torch.isfinite(a).all().item():
                        ok, msg = False, "NaN в действии"
                        break
                    obs_td, rew, done, _ = venv.step(a)
                    r = float(rew.mean().item())
                    if not np.isfinite(r) or not _finite(obs_td):
                        ok, msg = False, "NaN в obs/награде"
                        break
                    if done and key not in ("jump", "fall", "getup"):
                        obs_td, _ = venv.reset()
                results.append((terr, key, "PASS" if ok else "FAIL", msg))
            except Exception as e:
                results.append((terr, key, "FAIL", f"{type(e).__name__}: {e}"))
                traceback.print_exc()

    n_pass = sum(1 for r in results if r[2] == "PASS")
    n_fail = sum(1 for r in results if r[2] == "FAIL")
    print("=== АВТОТЕСТ ИГРЫ ===", flush=True)
    for terr, key, verdict, msg in results:
        print(f"  [{verdict}] {terr:6s} × {key:7s} {msg}", flush=True)
    print(f"=== итог: PASS={n_pass} FAIL={n_fail} "
          f"SKIP={len(results) - n_pass - n_fail} ===", flush=True)
    sys.exit(1 if n_fail else 0)


# ---------------------------------------------------------------- камера ---
class Camera:
    def __init__(self):
        self.azimuth = -90.0
        self.distance = 3.2
        self.elevation = -12.0

    def update(self, keys, dt):
        if keys.get("a"):
            self.azimuth += 60 * dt
        if keys.get("d"):
            self.azimuth -= 60 * dt


# ---------------------------------------------------------------- main ----
def desired_skill(events, keys, sit_ready):
    if "jump" in events:
        return "jump"
    if "fall" in events:
        return "fall"
    if "getup" in events:
        return "getup"
    if keys.get("ctrl") and sit_ready:
        return "sit"
    if keys.get("q"):
        return "dance"
    if keys.get("w") and keys.get("shift"):
        return "sprint"
    if keys.get("w"):
        return "walk"
    if keys.get("e"):
        return "run"
    return "stand"


def main():
    _install_scene_patch()
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("G1 TEST — полигон скиллов")
    clock = pygame.time.Clock()
    try:
        hud_font = pygame.font.Font(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        small_font = pygame.font.Font(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except Exception:
        hud_font = pygame.font.Font(None, 26)
        small_font = pygame.font.Font(None, 20)
    HINT = ("W ходьба · SHIFT+W спринт · E бег · ПРОБЕЛ прыжок · CTRL сесть · "
            "F упасть · G встать · Q танец · A/D камера · T ландшафт · "
            "H помощь · ESC выход")

    model = visual_model()
    data = mujoco.MjData(model)
    nq = model.nq
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                      "torso_link")
    renderer = Renderer(model, height=RENDER_H, width=RENDER_W)
    cam = mujoco.MjvCamera()
    camera = Camera()

    mgr = SkillManager()
    cur = None            # активный скилл
    sticky_left = 0       # остаток шагов залипающего скилла
    ep_rew = ep_steps = 0
    obs_td = None
    fps_show = 0.0
    show_help = False
    sit_ready = skill_ready("sit")

    keymap = {
        pygame.K_w: "w", pygame.K_a: "a", pygame.K_d: "d",
        pygame.K_e: "e", pygame.K_q: "q",
        pygame.K_LSHIFT: "shift", pygame.K_RSHIFT: "shift",
        pygame.K_LCTRL: "ctrl", pygame.K_RCTRL: "ctrl",
    }

    running = True
    _last_sit_check = 0.0
    while running:
        if time.time() - _last_sit_check > 10:
            _last_sit_check = time.time()
            sit_ready = sit_ready or skill_ready("sit")
        events, pressed = [], set()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    events.append("jump")
                elif ev.key == pygame.K_f:
                    events.append("fall")
                elif ev.key == pygame.K_g:
                    events.append("getup")
                elif ev.key == pygame.K_t:
                    i = TERRAIN_ORDER.index(CURRENT_TERRAIN)
                    newt = TERRAIN_ORDER[(i + 1) % len(TERRAIN_ORDER)]
                    _switch_terrain(newt, mgr)
                    model, data, renderer, pelvis_id = _rebuild_visual()
                    cur, obs_td, sticky_left = None, None, 0
                    ep_rew = ep_steps = 0
                elif ev.key == pygame.K_h:
                    show_help = not show_help
                elif ev.key in keymap:
                    events.append(keymap[ev.key])
                if ev.key in (pygame.K_a, pygame.K_d):
                    pressed.add(keymap[ev.key])
            elif ev.type == pygame.MOUSEWHEEL:
                camera.distance = min(8.0, max(
                    1.6, camera.distance - 0.4 * ev.y))
        ks = pygame.key.get_pressed()
        keys = {}
        for code, name in keymap.items():
            if ks[code]:
                keys[name] = True
        for name in ("a", "d"):
            if name in pressed:
                keys[name] = True
        if ks[pygame.K_w]:
            keys["w"] = True
        if ks[pygame.K_e]:
            keys["e"] = True
        if ks[pygame.K_q]:
            keys["q"] = True
        if ks[pygame.K_LSHIFT] or ks[pygame.K_RSHIFT]:
            keys["shift"] = True
        if ks[pygame.K_LCTRL] or ks[pygame.K_RCTRL]:
            keys["ctrl"] = True

        # скилл: залипающий не меняем, пока нет события/нажатой клавиши управления
        want = desired_skill(events, keys, sit_ready)
        held_move = any(keys.get(k) for k in ("w", "e", "q")) or \
            (keys.get("ctrl") and sit_ready)
        if cur in STICKY and sticky_left > 0 and \
                not any(k in SKILLS for k in events) and not held_move:
            want = cur
        if want != cur or obs_td is None:
            if want not in mgr.live:
                screen.fill((10, 12, 18))
                t = hud_font.render(f"загрузка скилла: {SKILLS[want][1]}…",
                                    True, (255, 200, 80))
                screen.blit(t, (WIN_W // 2 - t.get_width() // 2, WIN_H // 2))
                pygame.display.flip()
            got = mgr.get(want, protect=cur)
            if got is None:
                cur, sticky_left, obs_td = None, 0, None
                ep_rew = ep_steps = 0
            else:
                venv, _ = got
                obs_td, _ = venv.reset()
                cur, sticky_left = want, SKILLS[want][3]
                ep_rew = ep_steps = 0
                print(f"game: скилл -> {want}", flush=True)

        # шаг физики + рендер
        t0 = time.time()
        if obs_td is not None:
            venv, policy = mgr.live[cur]
            with torch.no_grad():
                action = policy(obs_td)
            obs_td, rew, done, _ = venv.step(action)
            ep_rew += float(rew.mean().item())
            ep_steps += 1
            if cur in STICKY:
                sticky_left -= 1
            q = venv.unwrapped.device_layer.sim.data.qpos
            data.qpos[:] = q.detach().cpu().numpy()[0, :nq]
            mujoco.mj_forward(model, data)
            if done or ep_steps >= MAX_EP_STEPS or \
                    (cur in STICKY and sticky_left <= 0):
                obs_td, _ = venv.reset()
                sticky_left = SKILLS[cur][3] if cur in STICKY else 0
                ep_rew = ep_steps = 0
            camera.update(keys, 1 / 50)
            cam.lookat[:] = [data.xpos[pelvis_id][0],
                             data.xpos[pelvis_id][1],
                             data.xpos[pelvis_id][2] + 0.2]
            cam.azimuth = camera.azimuth
            cam.distance = camera.distance
            cam.elevation = camera.elevation
            renderer.update_scene(data, camera=cam)
            img = renderer.render()
            surf = pygame.surfarray.make_surface(
                np.asarray(img).transpose(1, 0, 2))
            surf = pygame.transform.scale(surf, (WIN_W, WIN_H))
            screen.blit(surf, (0, 0))
        else:
            screen.fill((10, 12, 18))
            msg = "скилл недоступен (нет чекпоинта)" if cur else "загрузка…"
            t = hud_font.render(msg, True, (255, 200, 80))
            screen.blit(t, (WIN_W // 2 - t.get_width() // 2, WIN_H // 2))

        # HUD
        dt = max(time.time() - t0, 1e-6)
        fps_show = 0.9 * fps_show + 0.1 / dt if obs_td is not None else 0
        name = SKILLS.get(cur, (None, "—", None, None))[1] if cur else "—"
        fell = (obs_td is not None and
                data.xpos[pelvis_id][2] < 0.45 and cur not in ("fall", "getup"))
        line = f"{name} · {TERRAIN_RU[CURRENT_TERRAIN]} · {fps_show:.0f} fps"
        if obs_td is not None:
            line += f" · награда {ep_rew:.1f}"
        if fell:
            line += " · УПАЛ (G — встать)"
        if cur is None and want == "sit":
            line = "СЕСТЬ: дообучение s9_sit идёт — CTRL заработает автоматически"
        screen.blit(hud_font.render(line, True, (255, 255, 255)), (12, 10))
        screen.blit(small_font.render(HINT, True, (200, 200, 200)),
                    (12, WIN_H - 26))
        if show_help:
            help_lines = [
                "W — ходьба (удерживай), отпусти — стой",
                "SHIFT+W — спринт, E — бег",
                "ПРОБЕЛ — прыжок, F — упасть, G — встать",
                "CTRL — сесть (когда s9_sit дообучится), Q — танец",
                "A/D — поворот камеры, колесо мыши — зум",
                "T — сменить ландшафт (6 видов), H — эта помощь",
                "Скиллы обучены на РОВНОМ полу — ухабы/скалы это настоящий тест!",
            ]
            panel = pygame.Surface((560, 200), pygame.SRCALPHA)
            panel.fill((0, 0, 0, 190))
            screen.blit(panel, (12, 44))
            for i, h in enumerate(help_lines):
                screen.blit(small_font.render(h, True, (255, 255, 255)),
                            (24, 52 + i * 26))
        pygame.display.flip()
        if not AUTOTEST:
            clock.tick(50)
    mgr.drop_all()
    pygame.quit()


def _switch_terrain(newt, mgr):
    global CURRENT_TERRAIN
    CURRENT_TERRAIN = newt
    mgr.drop_all()
    print(f"game: ландшафт -> {TERRAIN_RU[newt]}", flush=True)


def _rebuild_visual():
    """Пересобрать визуальную модель под новый ландшафт."""
    global _VISUAL
    model = visual_model()
    data = mujoco.MjData(model)
    renderer = Renderer(model, height=RENDER_H, width=RENDER_W)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                      "torso_link")
    return model, data, renderer, pelvis_id


if __name__ == "__main__":
    if AUTOTEST:
        autotest()
    else:
        main()
