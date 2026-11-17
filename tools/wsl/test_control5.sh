#!/bin/bash
# Тест управления v5: pre-roll старт клипов (E4/E5), бесшовные переходы,
# родной подъём из sit (CTRL-release). Порог - под реальные скорости
# робота (~0.25-0.3 м/с против клипа 0.33 м/с).
source /root/hlc-venv/bin/activate
python - <<'PYEOF'
import math, sys, torch
sys.path.insert(0, "/mnt/g/LEARN WALK/src/hlc")
import game, steering

fails = []
def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}", flush=True)
    if not cond:
        fails.append(name)

def pose(venv):
    q = venv.unwrapped.device_layer.sim.data.qpos
    p = q[0, :3].detach().cpu().numpy().copy()
    quat = q[0, 3:7].detach().cpu().numpy().copy()
    yaw = math.atan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                     1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
    return p, yaw

venv_obs = [None]
z_min = [9.9]

def run(venv, policy, prod, steps, psi_target=None, rate=None, tag="",
        clamp_frame=None):
    """Вернуть (шаг_старта_ходьбы, история_дистанции)."""
    p0, _ = pose(venv)
    onset, hist = None, []
    for i in range(steps):
        if psi_target is not None:
            steering.update(prod, psi_target, rate or steering.MAX_RATE_WALK)
        if clamp_frame is not None:
            prod.time_steps[:] = clamp_frame  # эмуляция CTRL-hold
        steering.advance_pin(prod)
        with torch.no_grad():
            a = policy(venv_obs[0])
        venv_obs[0], rew, done, _ = venv.step(a)
        if torch.isnan(venv_obs[0]["policy"]).any():
            print(f"  !! NaN в obs на шаге {i}", flush=True)
            return onset, hist
        p, _ = pose(venv)
        z_min[0] = min(z_min[0], p[2])
        d = math.hypot(p[0] - p0[0], p[1] - p0[1])
        hist.append(d)
        if onset is None and d > 0.25:
            onset = i
        if i % 100 == 0:
            st = prod._steer
            print(f"    {tag} шаг {i:3d}: d={d:5.2f} z={p[2]:5.2f} "
                  f"psi={math.degrees(st['psi']):6.1f}", flush=True)
    return onset, hist

game.steering.install()

print("=== 1. ХОДЬБА: pre-roll старт ===", flush=True)
venv, policy = game.load_skill("walk")
prod = steering.attach(venv)
t_start = steering.pick_start_frame(prod)
print(f"  t_start={t_start}", flush=True)
venv_obs[0], _ = venv.reset()
steering.begin(venv, (0.0, 0.0), 0.0, t_start=t_start)
venv_obs[0] = steering.refresh_obs(venv)
z_min[0] = 9.9
onset, hist = run(venv, policy, prod, 800, 0.0, tag="ход")
cruise = hist[-1] - hist[min(onset + 150, len(hist) - 1)] \
    if onset is not None else 0.0
check("робот трогается (onset < 450)", onset is not None and onset < 450,
      f"шаг старта {onset}")
check("прошёл >= 0.6 м за 16 с", hist[-1] >= 0.6, f"{hist[-1]:.2f} м")
check("круиз >= 0.5 м после старта", cruise >= 0.5, f"{cruise:.2f} м")
check("не упал (min z > 0.5)", z_min[0] > 0.5, f"min z={z_min[0]:.3f}")

print("=== 2. ПОВОРОТ на ходу: psi += 90° ===", flush=True)
p1, yaw1 = pose(venv)
z_min[0] = 9.9
onset2, _ = run(venv, policy, prod, 250, yaw1 + math.pi / 2, tag="пов")
p2, yaw2 = pose(venv)
dyaw = abs(math.degrees(((yaw2 - yaw1 + math.pi) % (2 * math.pi)) - math.pi))
seg = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
check("развернулся >= 25°", dyaw >= 25, f"{dyaw:.0f}°")
check("сегмент >= 0.3 м", seg >= 0.3, f"{seg:.2f} м")
check("не упал (min z > 0.5)", z_min[0] > 0.5, f"min z={z_min[0]:.3f}")

print("=== 3. БЕСШОВНЫЙ переход walk -> stand -> walk ===", flush=True)
pA, _ = pose(venv)
venv2, policy2 = game.load_skill("stand")
prod2 = steering.attach(venv2)
venv_obs[0], _ = venv2.reset()
steering.transplant(venv2, venv)
pB, yawB = pose(venv2)
jump = math.hypot(pB[0] - pA[0], pB[1] - pA[1]) + abs(pB[2] - pA[2])
check("скачок позы < 2 см при пересадке", jump < 0.02, f"{jump*100:.1f} см")
steering.begin(venv2, (pB[0], pB[1]), yawB)
venv_obs[0] = steering.refresh_obs(venv2)
z_min[0] = 9.9
_, _ = run(venv2, policy2, prod2, 100, tag="стой")
pC, _ = pose(venv2)
check("стоит стабильно (z > 0.7)", pC[2] > 0.7, f"z={pC[2]:.3f}")
check("не упал (min z > 0.5)", z_min[0] > 0.5, f"min z={z_min[0]:.3f}")
venv_obs[0], _ = venv.reset()
steering.transplant(venv, venv2)
pD, yawD = pose(venv)
jump2 = math.hypot(pD[0] - pC[0], pD[1] - pC[1]) + abs(pD[2] - pC[2])
check("обратный переход < 2 см", jump2 < 0.02, f"{jump2*100:.1f} см")
steering.begin(venv, (pD[0], pD[1]), yawD, t_start=steering.pick_start_frame(prod))
venv_obs[0] = steering.refresh_obs(venv)
z_min[0] = 9.9
onset3, hist3 = run(venv, policy, prod, 800, yawD, tag="ход2")
check("после перехода снова ходит (>= 0.5 м)", hist3[-1] >= 0.5,
      f"{hist3[-1]:.2f} м")
check("не упал (min z > 0.5)", z_min[0] > 0.5, f"min z={z_min[0]:.3f}")

print("=== 4. СЕСТЬ (CTRL) -> подъём родным клипом -> ходьба ===", flush=True)
venv3, policy3 = game.load_skill("sit")
prod3 = steering.attach(venv3)
venv_obs[0], _ = venv3.reset()
steering.transplant(venv3, venv)
pE, yawE = pose(venv3)
jump3 = math.hypot(pE[0] - pD[0], pE[1] - pD[1]) + abs(pE[2] - pD[2])
check("переход walk->sit < 2 см", jump3 < 0.02, f"{jump3*100:.1f} см")
steering.begin(venv3, (pE[0], pE[1]), yawE, t_start=60)
venv_obs[0] = steering.refresh_obs(venv3)
z_min[0] = 9.9
_, _ = run(venv3, policy3, prod3, 150, tag="спуск")  # 3 с на спуск
pF, _ = pose(venv3)
check("сел (z в диапазоне 0.40-0.62)", 0.40 <= pF[2] <= 0.62, f"z={pF[2]:.3f}")
# CTRL-hold: зажим на 225, сидит 2 с
_, _ = run(venv3, policy3, prod3, 100, tag="сидит",
           clamp_frame=game.SIT_HOLD_FRAME)
pG, _ = pose(venv3)
check("сидит на зажиме (z стабильно 0.40-0.62)", 0.40 <= pG[2] <= 0.62,
      f"z={pG[2]:.3f}")
# отпустили: клип доигрывает подъём 225->310
_, _ = run(venv3, policy3, prod3, 200, tag="подъём")
pH, _ = pose(venv3)
check("встал родным клипом (z > 0.7)", pH[2] > 0.7, f"z={pH[2]:.3f}")
check("не упал за весь sit-цикл (min z > 0.4)", z_min[0] > 0.4,
      f"min z={z_min[0]:.3f}")
venv_obs[0], _ = venv.reset()
steering.transplant(venv, venv3)
pI, yawI = pose(venv)
jump4 = math.hypot(pI[0] - pH[0], pI[1] - pH[1]) + abs(pI[2] - pH[2])
check("переход sit->walk < 2 см", jump4 < 0.02, f"{jump4*100:.1f} см")
steering.begin(venv, (pI[0], pI[1]), yawI, t_start=steering.pick_start_frame(prod))
venv_obs[0] = steering.refresh_obs(venv)
z_min[0] = 9.9
onset4, hist4 = run(venv, policy, prod, 800, yawI, tag="ход3")
check("после вставания снова ходит (>= 0.5 м)", hist4[-1] >= 0.5,
      f"{hist4[-1]:.2f} м")
check("не упал (min z > 0.5)", z_min[0] > 0.5, f"min z={z_min[0]:.3f}")

print("=== ИТОГ ===", flush=True)
if fails:
    print("FAIL:", ", ".join(fails), flush=True)
    sys.exit(1)
print("CONTROL5_OK", flush=True)
PYEOF
echo "CONTROL5_EXIT=$?"
