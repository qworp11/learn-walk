#!/bin/bash
# перезапуск дашборда (подхватить LIVE-карточку) + проверка прокси стрима
PIDS=$(pgrep -f 'dashboard.py' || true)
echo "dashboard pids: $PIDS"
for p in $PIDS; do kill "$p" 2>/dev/null && echo "killed $p"; done
sleep 1
source /root/hlc-venv/bin/activate
nohup python '/mnt/g/LEARN WALK/src/hlc/dashboard.py' > /root/dashboard.out 2>&1 &
echo "relaunched $!"
sleep 4
curl -s -o /dev/null -w 'dashboard HTTP %{http_code}\n' http://localhost:8080/
curl -s -o /dev/null -w 'api HTTP %{http_code}\n' http://localhost:8080/api/status
curl -s --max-time 4 http://localhost:8080/livestream 2>/dev/null | head -c 40
echo
