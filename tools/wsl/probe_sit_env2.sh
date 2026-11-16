#!/bin/bash
# Настоящий модуль G1WholeBodyTracking: builder/cfg -> класс env
source /root/hlc-venv/bin/activate
python - <<'EOF'
import inspect, re, dataclasses
from mujoco_playground._src.rl import env_registry
e = env_registry.get("G1WholeBodyTracking")
print("=== env_cfg_fn source ===")
print(inspect.getsource(e.env_cfg_fn))
print("=== env_builder source ===")
print(inspect.getsource(e.env_builder))
EOF
