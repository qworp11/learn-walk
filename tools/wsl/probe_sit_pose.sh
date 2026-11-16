#!/bin/bash
# Выбор позы сидя: KNEES_BENT (нулевой оффсет действий), FK z, контакты корпуса
source /root/hlc-venv/bin/activate
python - <<'EOF'
import numpy as np
import torch
import mujoco
from mujoco_playground.robots.g1.sim import scene as g1_scene
from mujoco_playground.robots.g1.sim import device as g1_device

# оффсет действий (нулевое действие = эта поза)
try:
    kb = g1_device.KNEES_BENT_JOINT_POS
    print("KNEES_BENT =", np.round(np.asarray(kb), 2).tolist())
except Exception as e:
    print("KNEES_BENT: нет в device:", e)

spec = g1_scene.g1_robot_spec().compile()
m = spec
d = mujoco.MjData(m)
names = [m.joint(i).name for i in range(m.njnt)]
print("суставы:", names)
pelvis = m.body("pelvis").id


def probe(jdict, label):
    mujoco.mj_resetData(m, d)
    q = np.zeros(m.nq)
    q[0:3] = [0, 0, 0.8]
    q[3:7] = [1, 0, 0, 0]
    for i, nm in enumerate(names):
        if nm in jdict:
            q[7 + i] = jdict[nm]
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    z = float(d.xpos[pelvis][2])
    # контакты корпуса
    rc = []
    for i in range(d.ncon):
        c = d.contact[i]
        b1 = m.geom_bodyid[c.geom1]
        b2 = m.geom_bodyid[c.geom2]
        if b1 == pelvis or b2 == pelvis:
            other = b2 if b1 == pelvis else b1
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f6)
            rc.append((m.body(other).name, round(float(np.linalg.norm(f6[:3])), 1)))
    print(f"{label}: pelvis z={z:.3f} корневые контакты={rc if rc else 'НЕТ'}")


# candidates: глубокий (текущий), стул-высокий, средний
probe({"left_hip_pitch_joint": -1.4, "right_hip_pitch_joint": -1.4,
       "left_knee_joint": 2.2, "right_knee_joint": 2.2}, "глубокий hip-1.4/knee2.2")
for hip, knee in ((-1.0, 1.5), (-1.1, 1.4), (-1.2, 1.4), (-1.0, 1.3),
                  (-0.9, 1.2), (-0.8, 1.1)):
    probe({"left_hip_pitch_joint": hip, "right_hip_pitch_joint": hip,
           "left_knee_joint": knee, "right_knee_joint": knee},
          f"стул hip{hip}/knee{knee}")
EOF
