#!/bin/bash
source /root/hlc-venv/bin/activate
python - <<'EOF'
import urllib.request

req = urllib.request.urlopen("http://127.0.0.1:8081/stream", timeout=10)
buf = b""
jpeg_start = None
jpeg_end = None
for _ in range(200):
    chunk = req.read(4096)
    buf += chunk
    if jpeg_start is None:
        i = buf.find(b"\xff\xd8")
        if i >= 0:
            jpeg_start = i
    if jpeg_start is not None:
        j = buf.find(b"\xff\xd9", jpeg_start)
        if j >= 0:
            jpeg_end = j + 2
            break
jpg = buf[jpeg_start:jpeg_end]
open("/mnt/g/LEARN WALK/results/live_frame_test.png", "wb").write(jpg)
print("frame bytes:", len(jpg))
EOF
