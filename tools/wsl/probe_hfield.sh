#!/bin/bash
source /root/hlc-venv/bin/activate
python - <<'EOF'
import mujoco
spec = mujoco.MjSpec.from_string("<mujoco><worldbody/></mujoco>")
g = spec.worldbody.add_geom(name="g")
attrs = [a for a in dir(g) if not a.startswith("__")]
print("MjsGeom attrs:", attrs)
EOF
