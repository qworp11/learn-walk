#!/bin/bash
# Автотест игры: матрица скилл×ландшафт без окна (GAME_AUTOTEST=1)
cp '/mnt/g/LEARN WALK/src/hlc/game.py' /root/game.py
cp '/mnt/g/LEARN WALK/src/hlc/steering.py' /root/steering.py
md5sum /root/game.py '/mnt/g/LEARN WALK/src/hlc/game.py'
source /root/hlc-venv/bin/activate
GAME_AUTOTEST=1 python /root/game.py
echo "AUTOTEST_EXIT=$?"
