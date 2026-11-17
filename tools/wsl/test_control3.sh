#!/bin/bash
# Тест управления v3: фазовый старт клипов + стирринг + трансплант + сесть/встать
# ПОЛНЫЙ вывод без обрезки. Пороги под реальные скорости клипов (walk3 ~0.33 м/с).
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

def run(venv, policy, prod, steps, psi_target=None, rate=None, tag=""):
    term = None
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
        if i % 40 == 0 or done:
            st = prod._steer
            aerr = math.hypot(st["p_pin"][0] - p[0], st["p_pin"][1] - p[1])
            print(f"    {tag} шаг {i:3d}: xy=({p[0]:6.2f},{p[1]:6.2f}) z={p[2]:5.2f} "
                  f"|pin|={aerr:4.2f} psi={math.degrees(st['psi']):6.1f}"
                  f"{' DONE!' if done else ''}", flush=True)
        if done:
            term = i
            break
    return term

game.steering.install()
vmin_fallback = None

print("=== 1. ХОДЬБА: фазовый старт + стирринг ===", flush=True)
venv, policy = game.load_skill("walk")
prod = steering.attach(venv)
t_start = steering.pick_start_frame(prod)
print(f"  pick_start_frame -> кадр {t_start} (клип T={prod.motion.time_step_total})",
      flush=True)
check("фазовый старт нашёл шаг (кадр > 100)", t_start > 100, f"t={t_start}")
venv_obs[0], _ = venv.reset()
steering.begin(venv, (0.0, 0.0), 0.0, t_start=t_start)
p0, _ = pose(venv)
t1 = run(venv, policy, prod, 150, 0.0, tag="ход")  # 3 с
p1, yaw1 = pose(venv)
dist = float(((p1[:2] - p0[:2]) ** 2).sum()) ** 0.5
# walk3 идёт 0.33 м/с -> за 3 с честные ~1 м; допускаем >= 0.6 м
check("прошёл >= 0.6 м за 3 с", dist >= 0.6, f"{dist:.2f} м")
check("клип не заглушка (шёл, а не топтался)", dist >= 0.3, f"{dist:.2f} м")
check("без терминации", t1 is None, f"терминация на {t1}")
seg_yaw = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])) if dist > 0.05 else 0.0
check("движение по курсу (ошибка < 35°)", abs(seg_yaw) < 35 or dist < 0.3,
      f"курс сегмента {seg_yaw:.0f}°")

print("=== 2. ПОВОРОТ на ходу: psi += 90° (0.5 рад/с) ===", flush=True)
p1b, yaw1b = pose(venv)
t2 = run(venv, policy, prod, 200, yaw1b + math.pi / 2, tag="пов")  # 4 с
p2, yaw2 = pose(venv)
dyaw = abs(math.degrees(((yaw2 - yaw1b + math.pi) % (2 * math.pi)) - math.pi))
seg = p2[:2] - p1b[:2]
moved = float((seg ** 2).sum()) ** 0.5
check("пережил поворот 200 шагов", t2 is None, f"терминация на {t2}")
check("развернулся >= 25°", dyaw >= 25, f"{dyaw:.0f}°")
check("сегмент >= 0.4 м", moved >= 0.4, f"{moved:.2f} м")

print("=== 3. БЕСШОВНЫЙ переход walk -> stand -> walk ===", flush=True)
pA, _ = pose(venv)
venv2, policy2 = game.load_skill("stand")
prod2 = steering.attach(venv2)
venv_obs[0], _ = venv2.reset()
steering.transplant(venv2, venv)
pB, _ = pose(venv2)
jump = float(((pB[:3] - pA[:3]) ** 2).sum()) ** 0.5
check("скачок позы < 2 см при пересадке", jump < 0.02, f"{jump*100:.1f} см")
steering.begin(venv2, (pB[0], pB[1]), 0.0)
t3 = run(venv2, policy2, prod2, 100, tag="стой")  # 2 с стойки
pC, _ = pose(venv2)
check("стоит стабильно (z > 0.7)", pC[2] > 0.7, f"z={pC[2]:.3f}")
check("без терминации", t3 is None, f"терминация на {t3}")
venv_obs[0], _ = venv.reset()
steering.transplant(venv, venv2)
pD, _ = pose(venv)
jump2 = float(((pD[:3] - pC[:3]) ** 2).sum()) ** 0.5
check("обратный переход < 2 см", jump2 < 0.02, f"{jump2*100:.1f} см")
steering.begin(venv, (pD[0], pD[1]), 0.0, t_start=steering.pick_start_frame(prod))
t4 = run(venv, policy, prod, 150, 0.0, tag="ход2")
pE, _ = pose(venv)
moved2 = float(((pE[:2] - pD[:2]) ** 2).sum()) ** 0.5
check("после перехода снова ходит (>= 0.5 м)", moved2 >= 0.5, f"{moved2:.2f} м")
check("без терминации", t4 is None, f"терминация на {t4}")

print("=== 4. СЕСТЬ (CTRL) и встать - бесшовно ===", flush=True)
venv3, policy3 = game.load_skill("sit")
prod3 = steering.attach(venv3)
venv_obs[0], _ = venv3.reset()
steering.transplant(venv3, venv)
pF, _ = pose(venv3)
jump3 = float(((pF[:3] - pE[:3]) ** 2).sum()) ** 0.5
check("переход walk->sit < 2 см", jump3 < 0.02, f"{jump3*100:.1f} см")
steering.begin(venv3, (pF[0], pF[1]), 0.0)
t5 = run(venv3, policy3, prod3, 150, tag="сид")  # 3 с
pG, _ = pose(venv3)
check("сел (z в диапазоне 0.40-0.62)", 0.40 <= pG[2] <= 0.62, f"z={pG[2]:.3f}")
check("без терминации", t5 is None, f"терминация на {t5}")
venv_obs[0], _ = venv.reset()
steering.transplant(venv, venv3)
pH, _ = pose(venv)
steering.begin(venv, (pH[0], pH[1]), 0.0, t_start=steering.pick_start_frame(prod))
t6 = run(venv, policy, prod, 200, 0.0, tag="вст")  # 4 с: встать+шаг
pI, _ = pose(venv)
check("встал и пошёл (z > 0.7)", pI[2] > 0.7, f"z={pI[2]:.3f}")
check("без терминации", t6 is None, f"терминация на {t6}")

print("=== ИТОГ ===", flush=True)
if fails:
    print("FAIL:", ", ".join(fails), flush=True)
    if vmin_fallback is not None:
        print(f"подсказка: попробовать fallback t_start с v_min={vmin_fallback}",
              flush=True)
    sys.exit(1)
print("CONTROL3_OK", flush=True)
PYEOF
echo "CONTROL3_EXIT=$?"
