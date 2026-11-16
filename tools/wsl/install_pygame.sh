#!/bin/bash
source /root/hlc-venv/bin/activate
if which uv >/dev/null 2>&1; then
  echo "ставлю через uv"
  uv pip install --python /root/hlc-venv/bin/python pygame 2>&1 | tail -2
else
  echo "uv нет, пробую python -m pip"
  /root/hlc-venv/bin/python -m pip install pygame 2>&1 | tail -2
fi
python - <<'EOF'
try:
    import pygame
    print("import OK:", pygame.version.ver)
except Exception as e:
    print("import FAIL:", e)
EOF
