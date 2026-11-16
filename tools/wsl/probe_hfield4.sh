#!/bin/bash
# Как правильно присвоить userdata hfield + терпит ли компилятор отрицательные
source /root/hlc-venv/bin/activate
python - <<'EOF'
import mujoco
import numpy as np

spec = mujoco.MjSpec.from_string("<mujoco><worldbody/></mujoco>")
hf = spec.add_hfield(name="h", size=[1.0, 1.0, 0.6, 0.5], nrow=5, ncol=5)
print("тип hf:", type(hf))
print("атрибуты:", [a for a in dir(hf) if not a.startswith("_")])
Z = np.linspace(0.0, 1.0, 25, dtype=np.float32)
for how in ("assign_list", "assign_slice", "item_assignment"):
    try:
        h2 = spec.add_hfield(name="h_"+how, size=[1.0, 1.0, 0.6, 0.5],
                             nrow=5, ncol=5)
        if how == "assign_list":
            h2.userdata = list(Z)
        elif how == "assign_slice":
            h2.userdata[:] = Z
        else:
            h2.userdata = Z
        m = spec.compile()
        print(f"{how}: OK, hfield_data[:3]={np.asarray(m.hfield_data[:3]).round(3)} "
              f"n_data={m.nhfield} data_len={len(m.hfield_data)}")
        spec = mujoco.MjSpec.from_string("<mujoco><worldbody/></mujoco>")
    except Exception as e:
        print(f"{how}: FAIL {type(e).__name__}: {e}")

# отрицательные данные
try:
    spec = mujoco.MjSpec.from_string("<mujoco><worldbody/></mujoco>")
    h3 = spec.add_hfield(name="neg", size=[1.0, 1.0, 0.6, 0.5], nrow=5, ncol=5)
    Zn = np.full(25, 0.5, dtype=np.float32)
    Zn[12] = -0.3
    h3.userdata = list(Zn)
    m = spec.compile()
    print("отрицательные: hfield_data min/max =",
          float(np.asarray(m.hfield_data[:25]).min()),
          float(np.asarray(m.hfield_data[:25]).max()))
except Exception as e:
    print("отрицательные: FAIL", type(e).__name__, e)
EOF
