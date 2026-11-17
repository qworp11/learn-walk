#!/bin/bash
# E10: ускорение плейхеда (референс играет 1.5x/2x быстрее) как способ
# разогнать круиз. Морковка на скорости клипа со СНАПОМ (как в E4,
# старый advance_pin) vs поводок. Политика уже детерминированная
# (MLPModel.forward по умолчанию = mean) - стохастика отпала.
source /root/hlc-venv/bin/activate
python - <<'PYEOF'
import math, sys, torch
sys.path.insert(0, "/mnt/g/LEARN WALK/src/hlc")
import game, steering
import mujoco_playground.tasks.whole_body.tracking.motion_command as mc

def pose(venv):
    q = venv.unwrapped.device_layer.sim.data.qpos
    return q[0, :3].detach().cpu().numpy().copy()

def anchor_speed(prod):
    """Скорость якоря клипа в ТЕКУЩЕМ кадре, м/с (1 кадр = 1 шаг 0.02 с)."""
    T = prod.motion.time_step_total
    t = int(prod.time_steps[0].item()) % T
    p = prod.motion.body_pos_w[:, prod._anchor_index, :2]  # (T,2)
    v = torch.norm(p[(t + 1) % T] - p[t]).item()
    return v / 0.02

# --- ускоритель плейхеда: поверх advance_looped из steering.install() ---
adv1 = mc.MotionReferenceProducer.advance_time_steps

def make_fast(mult):
    acc = [0.0]
    def adv(self):
        acc[0] += mult
        n = int(acc[0])
        acc[0] -= n
        out = None
        for _ in range(n):
            out = adv1(self)
        return out
    return adv

def trial(mult, pin_mode, margin, steps, tag):
    if mult != 1.0:
        mc.MotionReferenceProducer.advance_time_steps = make_fast(mult)
    else:
        mc.MotionReferenceProducer.advance_time_steps = adv1
    venv.reset()
    steering.begin(venv, (0.0, 0.0), 0.0, t_start=530, blend_steps=50)
    obs = steering.refresh_obs(venv)
    st = prod._steer
    onset, z_min = None, 9.9
    dists = []
    for i in range(steps):
        p = pose(venv)
        if pin_mode == "leash":
            st["p_pin"][0] = p[0] + math.cos(st["psi"]) * margin
            st["p_pin"][1] = p[1] + math.sin(st["psi"]) * margin
        else:  # carrot-snap: pin шагает по курсу на скорости клипа*mult,
               # убежал дальше 0.7 м - снап назад к роботу (чуть впереди)
            bf = st.get("blend_from")
            if bf is None:
                v = anchor_speed(prod) * mult
                st["p_pin"][0] += math.cos(st["psi"]) * v * 0.02
                st["p_pin"][1] += math.sin(st["psi"]) * v * 0.02
                if math.hypot(st["p_pin"][0] - p[0],
                              st["p_pin"][1] - p[1]) > 0.7:
                    st["p_pin"][0] = p[0] + math.cos(st["psi"]) * 0.3
                    st["p_pin"][1] = p[1] + math.sin(st["psi"]) * 0.3
        with torch.no_grad():
            a = policy(obs)
        obs, rew, done, _ = venv.step(a)
        p = pose(venv)
        z_min = min(z_min, p[2])
        d = math.hypot(p[0], p[1])
        dists.append(d)
        if onset is None and d > 0.3:
            onset = i
        if i % 200 == 0:
            print(f"    {tag} шаг {i:3d}: d={d:5.2f} z={p[2]:5.2f} "
                  f"v_клипа={anchor_speed(prod):4.2f} "
                  f"кадр={int(prod.time_steps[0].item())}", flush=True)
    cruise = dists[-1] - dists[-200]
    print(f"  {tag}: mult={mult} {pin_mode}{margin} -> старт {onset}, итог "
          f"{dists[-1]:.2f} м, круиз(4с) {cruise:.2f} м ({cruise/4:.2f} м/с), "
          f"min z {z_min:.2f}", flush=True)

game.steering.install()
venv, policy = game.load_skill("walk")
prod = steering.attach(venv)
venv.unwrapped.task_layer._termination_checkers = []

print("=== A: 1.0x, морковка-снап (база E4) ===", flush=True)
trial(1.0, "carrot", 0, 700, "A")
print("=== B: 1.5x, морковка-снап ===", flush=True)
trial(1.5, "carrot", 0, 700, "B")
print("=== C: 2.0x, морковка-снап ===", flush=True)
trial(2.0, "carrot", 0, 700, "C")
print("=== D: 1.5x, поводок 0.15 ===", flush=True)
trial(1.5, "leash", 0.15, 700, "D")
print("E10_DONE", flush=True)
PYEOF
echo "E10_EXIT=$?"