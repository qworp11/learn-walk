#!/bin/bash
source /root/hlc-venv/bin/activate
export MUJOCO_GL=egl
python - <<'PYEOF'
import numpy as np
import mujoco
from mujoco import Renderer
from mujoco_playground.robots.g1.sim import scene as g1_scene

spec = g1_scene.g1_scene_spec()
spec.visual.global_.offwidth = 800
spec.visual.global_.offheight = 450
model = spec.compile()
data = mujoco.MjData(model)
r = Renderer(model, height=450, width=800)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, 0.8]
cam.distance = 3.0
cam.elevation = -12.0
for azim in (0.0, -90.0, 90.0, 180.0):
    cam.azimuth = azim
    r.update_scene(data, camera=cam)
    pos = np.array(r.scene.camera[0].pos)
    fwd = np.array(cam.lookat) - pos
    fwd[2] = 0
    yaw = np.degrees(np.arctan2(fwd[1], fwd[0]))
    print(f"azimuth={azim:6.1f}  campos={pos.round(2)}  forward_yaw={yaw:7.1f}")
PYEOF
