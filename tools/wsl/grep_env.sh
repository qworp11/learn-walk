#!/bin/bash
sed -n 92,135p /root/hlc-venv/lib/python3.12/site-packages/mujoco_playground/tasks/whole_body/tracking/task.py
echo '===== component ====='
sed -n 195,215p /root/hlc-venv/lib/python3.12/site-packages/mujoco_playground/tasks/whole_body/tracking/task.py
echo '===== env_builder ====='
grep -n 'def env_builder\|class EnvEntry\|env_cfg_fn\|entry\.' /root/hlc-venv/lib/python3.12/site-packages/mujoco_playground/_src/rl/env_registry.py | head -10
