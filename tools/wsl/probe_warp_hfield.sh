#!/bin/bash
# Поддерживает ли mujoco_warp hfield-коллизии?
grep -rn "hfield" /root/hlc-venv/lib/python3.12/site-packages/mujoco_warp/ --include="*.py" | head -20
echo "---"
grep -c "hfield" /root/hlc-venv/lib/python3.12/site-packages/mujoco_warp/_src/*.py 2>/dev/null | grep -v ":0" | head
