#!/bin/bash
# Откуда reaf берёт nconmax и как его поднять
echo "=== reaf simulator.py: compile ==="
sed -n '90,135p' /root/hlc-venv/lib/python3.12/site-packages/reaf/simulation/torch/_src/core/simulator.py
echo "=== reaf build.py: build_device_layer ==="
sed -n '210,260p' /root/hlc-venv/lib/python3.12/site-packages/reaf/simulation/torch/_src/builders/flat_scene/build.py
echo "=== mujoco_warp io.py: put_data сигнатура ==="
sed -n '2470,2545p' /root/hlc-venv/lib/python3.12/site-packages/mujoco_warp/_src/io.py
echo "=== кто вызывает put_data с nconmax ==="
grep -rn "put_data(" /root/hlc-venv/lib/python3.12/site-packages/reaf/ | head -5
