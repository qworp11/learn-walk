#!/bin/bash
# E11: ВАНИЛЬНЫЙ трекинг - сырой клип без стрирринга (без pin/psi/бленда).
# Вопрос: ходит ли политика непрерывно за клипом на этой чекпоинте?
# Если да - меряем естественный зазор робот-якорь и воспроизводим его.
source /root/hlc-venv/bin/activate
python - <<'PYEOF'
import math, sys, torch
sys.path.insert(0, "/mnt/g/LEARN WALK/src/hlc")
import game, steering

def pose(venv):
    q = venv.unwrapped.device_layer.sim.data.qpos
    return q[0, :3].detach().cpu().numpy().copy()

def anchor_xy(prod, t):
    T = prod.motion.time_step_total
    return prod.motion.body_pos_w[t % T, prod._anchor_index, :2] \
        .detach().cpu().numpy().copy()

game.steering.install()
venv, policy = game.load_skill("walk")
prod = steering.attach(venv)
venv.unwrapped.task_layer._termination_checkers = []

def trial(mode, t0, steps, tag, psi_rate=0.0):
    venv.reset()
    if mode == "vanilla":
        st = prod._steer
        st["on"] = False
        obs = steering.refresh_obs(venv)
        a0 = anchor_xy(prod, int(prod.time_steps[0].item()))
    else:
        steering.begin(venv, (0.0, 0.0), 0.0, t_start=t0, blend_steps=50)
        obs = steering.refresh_obs(venv)
        st = prod._steer
        a0 = anchor_xy(prod, t0)
        st["prev_frame"] = t0
    p0 = pose(venv)
    dists, gaps = [], []
    for i in range(steps):
        if mode == "path" and psi_rate:
            steering.update(prod, st["psi"] + psi_rate * 5.0,
                            steering.MAX_RATE_WALK)
        with torch.no_grad():
            a = policy(obs)
        obs, rew, done, _ = venv.step(a)
        p = pose(venv)
        t = int(prod.time_steps[0].item())
        if mode == "path":
            if st.get("blend_from") is not None:
                k = (t - st.get("blend_t0", t)) / max(st.get("blend_len", 1), 1)
            else:
                k = 1.0
            if k < 1.0:
                # бленд - путь ждёт робота
                st["p_pin"] = [float(p[0]), float(p[1])]
                st["prev_frame"] = t
            else:
                d = anchor_xy(prod, t) - anchor_xy(prod, st["prev_frame"])
                if psi_rate:
                    c, s = math.cos(st["psi"]), math.sin(st["psi"])
                    dx = c * d[0] - s * d[1]
                    dy = s * d[0] + c * d[1]
                else:
                    dx, dy = d[0], d[1]
                st["p_pin"][0] += dx
                st["p_pin"][1] += dy
                st["prev_frame"] = t
        d = math.hypot(p[0] - p0[0], p[1] - p0[1])
        if mode == "vanilla":
            gap = math.hypot(*(anchor_xy(prod, t) - p[:2]))
        else:
            gap = math.hypot(st["p_pin"][0] - p[0], st["p_pin"][1] - p[1])
        dists.append(d)
        gaps.append(gap)
        if i % 100 == 0:
            v_cl = math.hypot(*(anchor_xy(prod, t) - anchor_xy(
                prod, max(t - 50, 0)))) / 1.0
            print(f"    {tag} шаг {i:3d}: робот {d:5.2f} м, зазор {gap:5.2f} м, "
                  f"v_клипа(1с) {v_cl:4.2f} кадр={t}", flush=True)
    cruise_r = dists[-1] - dists[-200]
    print(f"  {tag}: итог {dists[-1]:.2f} м, круиз(4с) {cruise_r/4:.2f} м/с, "
          f"зазор медиана {sorted(gaps)[len(gaps)//2]:.2f} м, "
          f"зазор конец {gaps[-1]:.2f} м", flush=True)

print("=== V1: ВАНИЛА (клип с кадра 0, без стрирринга), 1000 шагов ===",
      flush=True)
trial("vanilla", 0, 1000, "V1")
print("=== V2: путь клипа (pin=дельты клипа), t=530, psi=0, 900 шагов ===",
      flush=True)
trial("path", 530, 900, "V2")
print("=== V3: путь клипа + psi вращается 0.1 рад/с, t=530, 900 шагов ===",
      flush=True)
trial("path", 530, 900, "V3", psi_rate=0.1)
print("E11_DONE", flush=True)
PYEOF
echo "E11_EXIT=$?"