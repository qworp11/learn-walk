#!/bin/bash
# Headless-тест бесшовного управления: стирринг + трансплант + сесть/встать
source /root/hlc-venv/bin/activate
python - <<'PYEOF' 2>&1 | tail -30
import math, sys, torch
sys.path.insert(0, "/mnt/g/LEARN WALK/src/hlc")
import game, steering

def pose(venv):
    q = venv.unwrapped.device_layer.sim.data.qpos
    return q[0, :3].detach().cpu().numpy().copy(), q[0, 3:7].detach().cpu().numpy().copy()

def run(venv, policy, prod, steps, psi_target=None, rate=steering.MAX_RATE_WALK):
    for _ in range(steps):
        if psi_target is not None:
            steering.update(prod, psi_target, rate)
        steering.advance_pin(prod)
        with torch.no_grad():
            a = policy(venv_obs[0])
        venv_obs[0], rew, done, _ = venv.step(a)
        assert not torch.isnan(venv_obs[0]["policy"]).any(), "NaN в obs!"
        if done:
            return True  # terminated
    return False

fails = []
def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        fails.append(name)

game.steering.install()

print("=== 1. Ходьба вперёд (psi = 0) ===")
venv, policy = game.load_skill("walk")
prod = steering.attach(venv)
venv_obs = [None]
venv_obs[0], _ = venv.reset()
p0, _ = pose(venv)
steering.begin(venv, (p0[0], p0[1]), 0.0)
done = run(venv, policy, prod, 150, psi_target=0.0)   # 3 с
p1, _ = pose(venv)
dist = float(((p1[:2] - p0[:2]) ** 2).sum()) ** 0.5
yaw_err = abs(math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])))
check("прошёл >= 0.8 м за 3 с", dist >= 0.8, f"{dist:.2f} м")
check("движение по курсу (ошибка < 30°)", yaw_err < 30 or dist < 0.3,
      f"{yaw_err:.0f}°")
check("без терминации", not done)

print("=== 2. Поворот на ходу (psi -> 90°) ===")
done = run(venv, policy, prod, 150, psi_target=math.pi / 2)  # ещё 3 с
p2, q2 = pose(venv)
seg = p2[:2] - p1[:2]
seg_yaw = math.degrees(math.atan2(seg[1], seg[0]))
moved = float((seg ** 2).sum()) ** 0.5
check("сегмент >= 0.5 м", moved >= 0.5, f"{moved:.2f} м")
check("курс сменился на ~90°", abs(seg_yaw - 90) < 45 or moved < 0.3,
      f"{seg_yaw:.0f}°")
check("без терминации", not done)

print("=== 3. Бесшовный переход walk -> stand -> walk ===")
pA, _ = pose(venv)
venv2, policy2 = game.load_skill("stand")
prod2 = steering.attach(venv2)
venv2_obs = [None]
venv2_obs[0], _ = venv2.reset()
steering.transplant(venv2, venv)
pB, _ = pose(venv2)
jump = float(((pB[:3] - pA[:3]) ** 2).sum()) ** 0.5
check("скачок позы < 2 см при пересадке", jump < 0.02, f"{jump*100:.1f} см")
steering.begin(venv2, (pB[0], pB[1]), 0.0)
done = run(venv2, policy2, prod2, 100)
pC, _ = pose(venv2)
check("стоит стабильно (z > 0.7)", pC[2] > 0.7, f"z={pC[2]:.3f}")
check("без терминации", not done)
# назад в walk с той же точки
venv_obs[0], _ = venv.reset()
steering.transplant(venv, venv2)
pD, _ = pose(venv)
jump2 = float(((pD[:3] - pC[:3]) ** 2).sum()) ** 0.5
check("обратный переход < 2 см", jump2 < 0.02, f"{jump2*100:.1f} см")
steering.begin(venv, (pD[0], pD[1]), 0.0)
done = run(venv, policy, prod, 100, psi_target=0.0)
pE, _ = pose(venv)
moved2 = float(((pE[:2] - pD[:2]) ** 2).sum()) ** 0.5
check("после перехода снова ходит (>= 0.4 м)", moved2 >= 0.4, f"{moved2:.2f} м")
check("без терминации", not done)

print("=== 4. CTRL сесть и обратно (бесшовно) ===")
venv3, policy3 = game.load_skill("sit")
prod3 = steering.attach(venv3)
venv3_obs = [None]
venv3_obs[0], _ = venv3.reset()
steering.transplant(venv3, venv)
pF, _ = pose(venv3)
steering.begin(venv3, (pF[0], pF[1]), 0.0)
done = run(venv3, policy3, prod3, 150)   # 3 с: присел и держит
pG, _ = pose(venv3)
check("сел (z ~ 0.45-0.62)", 0.40 <= pG[2] <= 0.62, f"z={pG[2]:.3f}")
check("без терминации", not done)
# встаём: walk с той же позы
venv_obs[0], _ = venv.reset()
steering.transplant(venv, venv3)
pH, _ = pose(venv)
steering.begin(venv, (pH[0], pH[1]), 0.0)
done = run(venv, policy, prod, 200, psi_target=0.0)  # 4 с на подъём+шаг
pI, _ = pose(venv)
check("встал и пошёл (z > 0.7)", pI[2] > 0.7, f"z={pI[2]:.3f}")
check("без терминации", not done)

print("=== ИТОГ ===")
if fails:
    print("FAIL:", ", ".join(fails))
    sys.exit(1)
print("CONTROL_TEST_OK")
PYEOF
echo "CONTROL_EXIT=$?"
