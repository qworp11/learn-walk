#!/bin/bash
# диагностика/перезапуск live_stream
# 1) убить старый процесс по явному PID
PIDS=$(pgrep -f 'live_stream.py' || true)
echo "pids: $PIDS"
for p in $PIDS; do kill "$p" 2>/dev/null && echo "killed $p"; done
sleep 1
# 2) свежая копия в нативную ФС (обход кэша DrvFS)
cp '/mnt/g/LEARN WALK/src/hlc/live_stream.py' /root/live_stream.py
md5sum /root/live_stream.py '/mnt/g/LEARN WALK/src/hlc/live_stream.py'
# 3) добавить диагностику перед Renderer, если её нет
grep -n 'HDBG' /root/live_stream.py || true
# 4) запустить
source /root/hlc-venv/bin/activate
nohup python /root/live_stream.py > /root/live_stream.out 2>&1 &
echo "launched pid $!"
sleep 35
echo '--- log ---'
cat /root/live_stream.out
echo '--- stream bytes ---'
curl -s --max-time 4 http://localhost:8081/stream 2>/dev/null | head -c 80
echo
