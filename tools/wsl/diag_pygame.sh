#!/bin/bash
source /root/hlc-venv/bin/activate
echo "which python: $(which python)"
echo "which pip:    $(which pip)"
python --version
pip --version
pip show pygame 2>/dev/null | head -4 || echo "pip show: пусто"
python - <<'EOF'
import sys, site
print("sys.prefix:", sys.prefix)
print("site-packages:")
for p in sys.path:
    print("  ", p)
try:
    import pygame
    print("import OK:", pygame.version.ver)
except Exception as e:
    print("import FAIL:", e)
EOF
