#!/bin/bash
# Знак ankle для плоской стопы: z_sit для обеих конвенций
source /root/hlc-venv/bin/activate
python - <<'EOF'
import numpy as np
import mujoco
from mujoco_playground.robots.g1.sim import scene as g1_scene

spec = g1_scene.g1_robot_spec()
model = spec.compile()
data = mujoco.MjData(model)
nj = model.nq - 7
names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
         for j in range(model.njnt)]
ranges = model.jnt_range
ank_j = [j for j, nm in enumerate(names) if "ankle_pitch" in nm][0]
print("ankle_pitch range:", ranges[ank_j])

def make_pose(hip, knee, ankle):
    p = np.zeros(nj)
    for j, nm in enumerate(names):
        v = 0.0
        if "hip_pitch" in nm: v = hip
        elif "knee" in nm: v = knee
        elif "ankle_pitch" in nm: v = ankle
        elif "waist_pitch" in nm: v = 0.15
        if v != 0.0:
            lo, hi = ranges[j]
            p[j] = float(np.clip(v, lo, hi))
    return p

def fk(z, joints):
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = [0.0, 0.0, z]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:] = joints
    mujoco.mj_kinematics(model, data)
    return data.xpos.copy()

ank_ids = [i for i in range(1, model.nbody)
           if "ankle_roll" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "")]
z0 = float(model.qpos0[2])
X0 = fk(z0, np.zeros(nj))
h_ank = min(X0[i][2] for i in ank_ids)

for hip in (-0.7, -1.0, -1.3):
    for knee in (0.9, 1.5, 2.1):
        for sign, lab in ((-1.0, "A=-(h+k)"), (+1.0, "A=+(h+k)")):
            a = sign * (hip + knee)
            lo, hi = ranges[ank_j]
            if not (lo <= a <= hi):
                print(f"hip={hip:+.1f} knee={knee:.1f} {lab} ankle={a:+.2f}: ВНЕ диапазона")
                continue
            p = make_pose(hip, knee, a)
            X = fk(z0, p)
            la = X[ank_ids[0]][2] - z0
            ra = X[ank_ids[1]][2] - z0
            z_sit = h_ank - min(la, ra)
            print(f"hip={hip:+.1f} knee={knee:.1f} {lab} ankle={a:+.2f}: "
                  f"z_sit={z_sit:.3f} asym={abs(la-ra):.3f}")
EOF
