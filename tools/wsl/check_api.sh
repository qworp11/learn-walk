#!/bin/bash
source /root/hlc-venv/bin/activate
python - <<'EOF'
import PIL
print('PIL ok')
from mujoco_playground._src.rl import env_registry
e = env_registry.get('G1WholeBodyTracking')
print('env_builder callable:', callable(getattr(e, 'env_builder', None)))
print('nworld:', e.runner_cfg.nworld)
from mujoco_playground._src.rl.rsl_rl import checkpoint
import inspect
print(inspect.getsource(checkpoint.read_checkpoint))
EOF
echo '--- cmd.json head ---'
python - <<'EOF'
import json
c = json.load(open('/root/hlc-logs/s2_walk_run_a/run_0/cmd.json'))
print('stage:', c['stage'])
print('trajectory:', c['env_overrides'].get('trajectory_path'))
print('warm_start:', c.get('warm_start'))
EOF
ls /root/hlc-logs/s2_walk_run_a/run_0/ | head -5
