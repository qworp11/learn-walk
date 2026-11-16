#!/bin/bash
# Проба для игры: как патчить сцену (terrain), API MjSpec, конфиг env
source /root/hlc-venv/bin/activate
python - <<'EOF'
import dataclasses, glob, os, re

# 1) конфиг env: какие поля есть
from mujoco_playground._src.rl import env_registry
entry = env_registry.get("G1WholeBodyTracking")
cfg = entry.runner_cfg
print("=== runner_cfg поля ===")
for f in dataclasses.fields(cfg):
    print(" ", f.name, "=", repr(getattr(cfg, f.name))[:80])

# 2) где env строит сцену: ищем g1_scene в пакете
import mujoco_playground, pathlib
root = pathlib.Path(mujoco_playground.__file__).parent
print("=== использование g1_scene / scene_spec ===")
for p in root.rglob("*.py"):
    txt = p.read_text(errors="ignore")
    if "g1_scene" in txt or "g1_scene_spec" in txt:
        for i, line in enumerate(txt.splitlines(), 1):
            if "g1_scene" in line or "scene_spec" in line:
                print(f"  {p.relative_to(root.parent)}:{i}: {line.strip()[:100]}")
EOF
echo "=== MjSpec API тест (бокс + hfield) ==="
python - <<'EOF'
import mujoco
spec = mujoco.MjSpec.from_string(
    "<mujoco><worldbody><body name='a'><geom type='sphere' size='0.1'/>"
    "</body></worldbody></mujoco>")
try:
    g = spec.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.3, 0.1, 0.05],
                      pos=[1.0, 0.0, 0.05], name="wall_test")
    print("add_geom BOX ok:", g.name)
except Exception as e:
    print("add_geom FAIL:", e)
try:
    hf = spec.add_hfield(name="hf_test", nrow=64, ncol=64, size=[4.0, 4.0, 0.5],
                         nvalue=256)
    print("add_hfield ok:", hf.name)
except Exception as e:
    print("add_hfield FAIL:", e)
m = spec.compile()
print("compile ok, ngeom:", m.ngeom, "nhfield:", m.nhfield)
EOF
echo "=== чекпоинты (числовая сортировка) ==="
for st in s1_balance s2_walk s3_run s3_sprint s4_jump s6_fall s6_fall2 s8_dance; do
  last=$(ls /root/hlc-logs/$st/run_*/model_*.pt 2>/dev/null | sed 's/.*model_\([0-9]*\)\.pt/\1 \0/' | sort -n | tail -1 | cut -d' ' -f2)
  echo "  $st: $last"
done
