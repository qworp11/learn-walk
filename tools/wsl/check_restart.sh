#!/bin/bash
date -u '+now UTC: %FT%TZ'
echo '--- watchdog ---'
tail -3 /root/hlc-logs/s2_walk_run_a/watchdog.log 2>/dev/null
echo '--- процесс ---'
pgrep -af 'hlc_train.py' | grep -v 'bash -c' | head -1
echo '--- итерации/награда ---'
grep -E 'Learning iteration|Mean reward' /root/hlc-logs/s2_walk_run_a/run_0/train_stdout.log | tail -4
echo '--- cmd (resume?) ---'
python - <<'EOF'
import json
c = json.load(open('/root/hlc-logs/s2_walk_run_a/run_0/cmd.json'))
cmd = c['cmd']
print('resume:', cmd[-1] if 'resume_path' in ' '.join(cmd) else 'НЕТ')
print('warm_start flag в cmd.json:', c.get('warm_start'))
EOF
echo '--- GPU ---'
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
