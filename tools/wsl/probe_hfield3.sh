#!/bin/bash
# Калибровка hfield: данные в метрах или [0..1]*zmax? + как их читает warp
source /root/hlc-venv/bin/activate
python - <<'EOF'
import mujoco
import numpy as np

def build(Z):
    spec = mujoco.MjSpec.from_string(
        "<mujoco><worldbody><body name='b'><freejoint/>"
        "<geom name='ball' type='sphere' size='0.05' mass='1'/></body>"
        "</worldbody></mujoco>")
    spec.add_hfield(name="h", size=[1.0, 1.0, 0.6, 0.5],
                    nrow=5, ncol=5, userdata=[float(v) for v in Z])
    g = spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, name="hg")
    g.hfieldname = "h"
    m = spec.compile()
    return m

for level in (0.0, 0.1, 0.25, 0.5, 1.0):
    Z = np.full(25, level, dtype=np.float32)
    m = build(Z)
    d = mujoco.MjData(m)
    d.qpos[2] = 1.0
    for _ in range(600):
        mujoco.mj_step(m, d)
    print(f"level={level:+.2f} rest={float(d.qpos[2]):+.4f} "
          f"hfield_data[:3]={np.asarray(m.hfield_data[:3]).round(3)} "
          f"size={np.asarray(m.hfield_size).round(3)}")

m = build(np.full(25, 0.1, dtype=np.float32))
print("nrow ncol:", m.hfield_nrow[0], m.hfield_ncol[0])
EOF
echo "=== warp: как читаются высоты ==="
grep -rn "hfield_data" /root/hlc-venv/lib/python3.12/site-packages/mujoco_warp/_src/*.py | head -10
echo "=== reaf: hfield ==="
grep -rln "hfield" /root/hlc-venv/lib/python3.12/site-packages/reaf/ 2>/dev/null | head -5
