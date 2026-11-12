#!/bin/bash
# S0: системные зависимости WSL Ubuntu 22.04
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential git curl wget ffmpeg libosmesa6-dev libgl1-mesa-dev libegl1-mesa-dev libglfw3 unzip zip ca-certificates python3-venv python3-pip >/dev/null
echo "APT OK"
pip3 install -q uv 2>&1 | tail -1
uv --version
