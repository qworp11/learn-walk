#!/bin/bash
# Headless-тест управления v2: полный вывод + диагностика + базлайн без стирринга
source /root/hlc-venv/bin/activate
python - <<'PYEOF'
import math, sys, torch
sys.path.insert(0, "/mnt/g/LEARN WALK/src/hlc")
import game, steering

def pose(venv):
    q = venv.unwrapped.device_layer.sim.data.qpos
    p = q[0, :3].detach().cpu().numpy().copy()
    quat = q[0, 3:7].detach().cpu().numpy().copy()  # pelvis (w,x,y,z)
    yaw = math.atan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                     1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
    return p, yaw

fails = []
def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}", flush=True)
    if not cond:
        fails.append(name)

def run(venv, policy, prod, steps, psi_target=None, tag=""):
    term_step = None
    for i in range(steps):
        if psi_target is not None:
            steering.update(prod, psi_target, steering.MAX_RATE_WALK)
        steering.advance_pin(prod)
        with torch.no_grad():
            a = policy(venv_obs[0])
        venv_obs[0], rew, done, _ = venv.step(a)
        if torch.isnan(venv_obs[0]["policy"]).any():
            print(f"  !! NaN в obs на шаге {i}", flush=True)
            return i
        p, _ = pose(venv)
        if i % 25 == 0 or done:
            st = prod._steer
            aerr = math.hypot(st["p_pin"][0] - p[0], st["p_pin"][1] - p[1])
            print(f"    {tag} шаг {i:3d}: xyz=({p[0]:6.2f},{p[1]:6.2f},{p[2]:5.2f}) "
                  f"|pin-робот|={aerr:5.2f} psi={math.degrees(st['psi']):6.1f}°"
                  f"{' DONE!' if done else ''}", flush=True)
        if done:
            term_step = i
            break
    return term_step

game.steering.install()

print("=== 0. БАЗЛАЙН: ходьба БЕЗ стирринга (как старая игра) ===")
venv, policy = game.load_skill("walk")
prod = steering.attach(venv)
venv_obs = [None]
venv_obs[0], _ = venv.reset()
p0, yaw0 = pose(venv)
print(f"  старт yaw робота = {math.degrees(yaw0):.1f}°", flush=True)
t0 = run(venv, policy, prod, 150, None, tag="базл")
p1, _ = pose(venv)
dist0 = float(((p1[:2] - p0[:2]) ** 2).sum()) ** 0.5
check("базлайн живёт 150 шагов", t0 is None, f"терминация на {t0}")
check("базлайн прошёл >= 0.8 м", dist0 >= 0.8, f"{dist0:.2f} м")

print("=== 1. Ходьба СО стиррингом, курс = курс робота ===")
venv_obs[0], _ = venv.reset()
p0, yaw0 = pose(venv)
steering.begin(venv, (p0[0], p0[1]), yaw0)
t1 = run(venv, policy, prod, 150, yaw0, tag="стир")
p1, _ = pose(venv)
dist1 = float(((p1[:2] - p0[:2]) ** 2).sum()) ** 0.5
check("стирринг живёт 150 шагов", t1 is None, f"терминация на {t1}")
check("стирринг прошёл >= 0.8 м", dist1 >= 0.8, f"{dist1:.2f} м")

print("=== 2. Поворот на ходу: курс += 90° ===")
p1, yaw1 = pose(venv)
t2 = run(venv, policy, prod, 200, yaw1 + math.pi / 2, tag="пов90")
p2, yaw2 = pose(venv)
seg = p2[:2] - p1[:2]
moved = float((seg ** 2).sum()) ** 0.5
seg_yaw = math.degrees(math.atan2(seg[1], seg[0])) if moved > 1e-4 else 0.0
check("пережил поворот 200 шагов", t2 is None, f"терминация на {t2}")
check("сегмент >= 0.5 м", moved >= 0.5, f"{moved:.2f} м")
check("направление сегмента ~ +90° от старого",
      abs(((seg_yaw - math.degrees(yaw1) + 90 + 180) % 360) - 180) < 45,
      f"yaw робота {math.degrees(yaw1):.0f}° -> сегмент {seg_yaw:.0f}°")

print("=== ИТОГ ===")
if fails:
    print("FAIL:", ", ".join(fails))
    sys.exit(1)
print("CONTROL2_OK")
PYEOF
echo "CONTROL2_EXIT=$?"
