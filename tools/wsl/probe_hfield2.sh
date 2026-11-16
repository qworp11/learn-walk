#!/bin/bash
# Семантика данных hfield: метры или нормированные [0..1]*zmax?
# Шар радиуса 0.05 падает на hfield: если данные в метрах — отдых на 0.1+0.05=0.15;
# если нормированные (×zmax=0.6) — на 0.06+0.05=0.11.
source /root/hlc-venv/bin/activate
python - <<'EOF'
import mujoco
import numpy as np

def run(Z, label):
    spec = mujoco.MjSpec.from_string(
        "<mujoco><worldbody><body name='b'><freejoint/>"
        "<geom name='ball' type='sphere' size='0.05' mass='1'/></body>"
        "</worldbody></mujoco>")
    hf = spec.add_hfield(name="h", size=[1.0, 1.0, 0.6, 0.5],
                         nrow=5, ncol=5, userdata=[float(v) for v in Z])
    g = spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, name="hg")
    g.hfieldname = "h"
    m = spec.compile()
    d = mujoco.MjData(m)
    d.qpos[2] = 1.0
    for _ in range(600):
        mujoco.mj_step(m, d)
    print(label, "rest z =", round(float(d.qpos[2]), 4),
          "xy =", round(float(d.qpos[0]), 3), round(float(d.qpos[1]), 3))

Z1 = np.full(25, 0.1, dtype=np.float32)             # плоскость +0.1 м
run(Z1, "плоско +0.1:")
Z2 = np.full(25, 0.1, dtype=np.float32)
Z2[12] = -0.3                                        # яма -0.3 в центре
run(Z2, "яма -0.3 в центре:")
EOF
