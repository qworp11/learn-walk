#!/bin/bash
source /root/hlc-venv/bin/activate
python - <<'EOF'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
from mujoco import Renderer
from mujoco_playground.robots.g1.sim import scene as g1_scene

spec = g1_scene.g1_scene_spec()
model = spec.compile()
print("model type:", type(model))
print("offwidth:", model.vis.global_.offwidth, type(model.vis.global_.offwidth))
print("offheight:", model.vis.global_.offheight, type(model.vis.global_.offheight))
try:
    r = Renderer(model, height=270, width=480)
    print("Renderer OK (egl)")
except Exception as e:
    print("Renderer fail:", type(e).__name__, e)
    os.environ["MUJOCO_GL"] = "osmesa"
    try:
        r = Renderer(model, height=270, width=480)
        print("Renderer OK (osmesa)")
    except Exception as e2:
        print("Renderer fail2:", type(e2).__name__, e2)
        import inspect
        src = inspect.getsource(type(r) if False else mujoco.rendering.classic.renderer.Renderer.__init__)
        print(src[:800])
EOF
