#!/bin/bash
# Запуск игры для Виталика (WSLg — окно появится прямо в Windows)
cp '/mnt/g/LEARN WALK/src/hlc/game.py' /root/game.py
cp '/mnt/g/LEARN WALK/src/hlc/steering.py' /root/steering.py
md5sum /root/game.py '/mnt/g/LEARN WALK/src/hlc/game.py'
source /root/hlc-venv/bin/activate
python /root/game.py
