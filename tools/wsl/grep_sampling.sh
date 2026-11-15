#!/bin/bash
grep -n 'sampling_mode' /root/hlc-venv/lib/python3.12/site-packages/mujoco_playground/tasks/whole_body/tracking/motion_command.py
echo '---'
grep -n 'def sample\|adaptive\|start\b\|random\|uniform' /root/hlc-venv/lib/python3.12/site-packages/mujoco_playground/tasks/whole_body/tracking/motion_command.py | head -30
