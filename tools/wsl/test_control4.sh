#!/bin/bash
# Тест управления v4: фазовый старт + fresh obs + без скрытых сбросов.
# done больше не стреляет (терминаторы отключены в игровом процессе) -
# вместо "без терминации" проверяем живучесть: z > 0.5 на каждом шаге.
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

def run(venv, policy, prod, steps, psi_target=None, rate=None, tag=""):
    for i in range(steps):
        if psi_target is not None:
            steering.update(prod, psi_target, rate or steering.MAX_RATE_WALK)
        steering.advance_pin(prod)
        with torch.no_grad():
            a = policy(venv_obs[0])
        venv_obs[0], rew, done, _ = venv.step(a)
        if torch.isnan(venv_obs[0]["policy"]).any():
            print(f"  !! NaN в obs на шаге {i}", flush=True)
            return i
        p, _ = pose(venv)
        z_min[0] = min(z_min[0], p[2])
        if i % 40 == 0:
            st = prod._steer
            aerr = math.hypot(st["p_pin"][0] - p[0], st["p_pin"][1] - p[1])
            print(f"    {tag} шаг {i:3d}: xy=({p[0]:6.2f},{p[1]:6.2f}) z={p[2]:5.2f} "
                  f"|pin|={aerr:4.2f} psi={math.degrees(st['psi']):6.1f}", flush=True)
    return None

game.steering.install()

print("=== 1. ХОДЬБА: фазовый старт + fresh obs ===", flush=True)
venv, policy = game.load_skill("walk")
prod = steering.attach(venv)
t_start = steering.pick_start_frame(prod)
print(f"  pick_start_frame -> кадр {t_start}", flush=True)
venv_obs[0], _ = venv.reset()
steering.begin(venv, (0.0, 0.0), 0.0, t_start=t_start)
venv_obs[0] = steering.refresh_obs(venv)
p0, _ = pose(venv)
z_min[0] = 9.9
run(venv, policy, prod, 150, 0.0, tag="ход")  # 3 с
p1, yaw1 = pose(venv)
dist = float(((p1[:2] - p0[:2]) ** 2).sum()) ** 0.5
check("прошёл >= 0.6 м за 3 с", dist >= 0.6, f"{dist:.2f} м")
check("не упал в процессе (min z > 0.5)", z_min[0] > 0.5, f"min z={z_min[0]:.3f}")

print("=== 2. ПОВОРОТ на ходу: psi += 90° (0.5 рад/с) ===", flush=True)
p1b, yaw1b = pose(venv)
z_min[0] = 9.9
run(venv, policy, prod, 200, yaw1b + math.pi / 2, tag="пов")  # 4 с
p2, yaw2 = pose(venv)
dyaw = abs(math.degrees(((yaw2 - yaw1b + math.pi) % (2 * math.pi)) - math.pi))
seg = p2[:2] - p1b[:2]
moved = float((seg ** 2).sum()) ** 0.5
check("развернулся >= 25°", dyaw >= 25, f"{dyaw:.0f}°")
check("сегмент >= 0.4 м", moved >= 0.4, f"{moved:.2f} м")
check("не упал в процессе (min z > 0.5)", z_min[0] > 0.5, f"min z={z_min[0]:.3f}")

print("=== 3. БЕСШОВНЫЙ переход walk -> stand -> walk ===", flush=True)
pA, _ = pose(venv)
venv2, policy2 = game.load_skill("stand")
prod2 = steering.attach(venv2)
venv_obs[0], _ = venv2.reset()
steering.transplant(venv2, venv)
pB, yawB = pose(venv2)
jump = float(((pB[:3] - pA[:3]) ** 2).sum()) ** 0.5
check("скачок позы < 2 см при пересадке", jump < 0.02, f"{jump*100:.1f} см")
steering.begin(venv2, (pB[0], pB[1]), yawB)
venv_obs[0] = steering.refresh_obs(venv2)
z_min[0] = 9.9
run(venv2, policy2, prod2, 100, tag="стой")  # 2 с стойки
pC, _ = pose(venv2)
check("стоит стабильно (z > 0.7)", pC[2] > 0.7, f"z={pC[2]:.3f}")
check("не упал (min z > 0.5)", z_min[0] > 0.5, f"min z={z_min[0]:.3f}")
venv_obs[0], _ = venv.reset()
steering.transplant(venv, venv2)
pD, yawD = pose(venv)
jump2 = float(((pD[:3] - pC[:3]) ** 2).sum()) ** 0.5
check("обратный переход < 2 см", jump2 < 0.02, f"{jump2*100:.1f} см")
steering.begin(venv, (pD[0], pD[1]), yawD, t_start=steering.pick_start_frame(prod))
venv_obs[0] = steering.refresh_obs(venv)
z_min[0] = 9.9
run(venv, policy, prod, 150, yawD, tag="ход2")
pE, _ = pose(venv)
moved2 = float(((pE[:2] - pD[:2]) ** 2).sum()) ** 0.5
check("после перехода снова ходит (>= 0.5 м)", moved2 >= 0.5, f"{moved2:.2f} м")
check("не упал (min z > 0.5)", z_min[0] > 0.5, f"min z={z_min[0]:.3f}")

print("=== 4. СЕСТЬ (CTRL) -> отпустить -> stand -> walk ===", flush=True)
venv3, policy3 = game.load_skill("sit")
prod3 = steering.attach(venv3)
venv_obs[0], _ = venv3.reset()
steering.transplant(venv3, venv)
pF, yawF = pose(venv3)
jump3 = float(((pF[:3] - pE[:3]) ** 2).sum()) ** 0.5
check("переход walk->sit < 2 см", jump3 < 0.02, f"{jump3*100:.1f} см")
steering.begin(venv3, (pF[0], pF[1]), yawF)
venv_obs[0] = steering.refresh_obs(venv3)
z_min[0] = 9.9
run(venv3, policy3, prod3, 150, tag="сид")  # 3 с
pG, _ = pose(venv3)
check("сел (z в диапазоне 0.40-0.62)", 0.40 <= pG[2] <= 0.62, f"z={pG[2]:.3f}")
check("не упал (min z > 0.4)", z_min[0] > 0.4, f"min z={z_min[0]:.3f}")
# отпустили CTRL -> stand: робот должен подняться
venv_obs[0], _ = venv2.reset()
steering.transplant(venv2, venv3)
pH, yawH = pose(venv2)
jump4 = float(((pH[:3] - pG[:3]) ** 2).sum()) ** 0.5
check("переход sit->stand < 2 см", jump4 < 0.02, f"{jump4*100:.1f} см")
steering.begin(venv2, (pH[0], pH[1]), yawH)
venv_obs[0] = steering.refresh_obs(venv2)
z_min[0] = 9.9
run(venv2, policy2, prod2, 200, tag="подъём")  # 4 с на подъём
pI, _ = pose(venv2)
check("встал после sit->stand (z > 0.7)", pI[2] > 0.7, f"z={pI[2]:.3f}")
check("не упал (min z > 0.4)", z_min[0] > 0.4, f"min z={z_min[0]:.3f}")
venv_obs[0], _ = venv.reset()
steering.transplant(venv, venv2)
pJ, yawJ = pose(venv)
steering.begin(venv, (pJ[0], pJ[1]), yawJ, t_start=steering.pick_start_frame(prod))
venv_obs[0] = steering.refresh_obs(venv)
z_min[0] = 9.9
run(venv, policy, prod, 150, yawJ, tag="ход3")
pK, _ = pose(venv)
moved3 = float(((pK[:2] - pJ[:2]) ** 2).sum()) ** 0.5
check("после вставания снова ходит (>= 0.4 м)", moved3 >= 0.4, f"{moved3:.2f} м")
check("не упал (min z > 0.5)", z_min[0] > 0.5, f"min z={z_min[0]:.3f}")

print("=== ИТОГ ===", flush=True)
if fails:
    print("FAIL:", ", ".join(fails), flush=True)
    sys.exit(1)
print("CONTROL4_OK", flush=True)
PYEOF
echo "CONTROL4_EXIT=$?"
