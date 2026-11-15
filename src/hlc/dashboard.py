"""Дашборд обучения: http://localhost:8080

Запуск (в WSL): python '/mnt/g/LEARN WALK/src/hlc/dashboard.py'
Порт 8080, с Windows доступен как http://localhost:8080

Эндпоинты:
  /            — HTML-дашборд (автообновление данных каждые 15 с)
  /api/status  — JSON: этапы, вердикты, спарклайны награды, видео, GPU
  /videos/*    — mp4/кадры из results/videos
"""
import glob
import html
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

LOGROOT = "/root/hlc-logs"
VIDEOS = "/mnt/g/LEARN WALK/results/videos"
REPORT = "/mnt/g/LEARN WALK/results/orchestrator_report.txt"
HEARTBEAT = "/mnt/g/LEARN WALK/heartbeat.json"
CACHE = {}  # path -> (mtime, parsed)

PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>Конвейер обучения G1</title>
<style>
:root { --bg:#0e1116; --card:#161b22; --line:#30363d; --txt:#c9d1d9; --dim:#8b949e; --ok:#3fb950; --warn:#d29922; --bad:#f85149; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--txt); font:14px/1.5 'Segoe UI',system-ui,sans-serif; }
header { padding:14px 20px; border-bottom:1px solid var(--line); display:flex; gap:16px; align-items:baseline; flex-wrap:wrap;}
header h1 { font-size:18px; margin:0; }
header .gpu { color:var(--dim); font-size:12px; }
main { padding:16px 20px; max-width:1400px; margin:0 auto; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:14px; }
.stages { display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:12px; }
.stage { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
.stage h3 { margin:0 0 6px; font-size:15px; }
.badge { font-size:11px; padding:2px 8px; border-radius:10px; display:inline-block; margin-left:6px; vertical-align:2px;}
.done { background:#1a3a1e; color:var(--ok); }
.run  { background:#3a2e1a; color:var(--warn); }
.wait { background:#22272e; color:var(--dim); }
.mono { font-family:Consolas,monospace; font-size:12px; color:var(--dim); }
.spark { width:100%; height:60px; margin-top:6px; }
.vids { display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:12px; }
video { width:100%; border-radius:8px; background:#000; }
.cap { font-size:12px; color:var(--dim); margin-top:4px; }
pre.report { font:12px/1.45 Consolas,monospace; color:var(--dim); white-space:pre-wrap; max-height:260px; overflow:auto; margin:0; }
</style></head><body>
<header><h1>🏃 Конвейер обучения G1</h1><span class="gpu" id="gpu"></span><span class="gpu" id="upd"></span></header>
<main>
  <div class="card" id="stagesCard"><b>Этапы лестницы</b><div class="stages" id="stages"></div></div>
  <div class="card"><b>Финальные видео</b><div class="vids" id="finals"></div></div>
  <div class="card"><b>Свежие превью (по чекпоинтам)</b><div class="vids" id="previews"></div></div>
  <div class="card"><b>Отчёт оркестратора</b><pre class="report" id="report"></pre></div>
</main>
<script>
async function tick() {
  try {
    const r = await fetch('/api/status'); const d = await r.json();
    document.getElementById('gpu').textContent = d.gpu;
    document.getElementById('upd').textContent = 'обновлено ' + new Date().toLocaleTimeString() + ' · авто через 15с';
    const st = document.getElementById('stages'); st.innerHTML = '';
    for (const s of d.stages) {
      const cls = s.state==='done'?'done':(s.state==='run'?'run':'wait');
      const label = s.state==='done'?'готово':(s.state==='run'?'идёт':'ожидает');
      const div = document.createElement('div'); div.className='stage';
      const spark = s.reward ? `<canvas class="spark" data-vals="${s.reward.join(',')}"></canvas>` : '';
      div.innerHTML = `<h3>${s.name}<span class="badge ${cls}">${label}</span></h3>` +
        `<div class="mono">${s.info||''}</div>${spark}`;
      st.appendChild(div);
    }
    document.querySelectorAll('canvas.spark').forEach(drawSpark);
    const fin = document.getElementById('finals'); fin.innerHTML='';
    for (const v of d.finals) fin.appendChild(videoCard(v));
    const pre = document.getElementById('previews'); pre.innerHTML='';
    for (const v of d.previews) pre.appendChild(videoCard(v));
    document.getElementById('report').textContent = d.report;
  } catch(e) { document.getElementById('upd').textContent = 'ошибка связи: '+e; }
}
function videoCard(v) {
  const d = document.createElement('div');
  d.innerHTML = `<video controls preload="metadata" src="${v.url}"></video><div class="cap">${v.cap}</div>`;
  return d;
}
function drawSpark(cv) {
  const vals = cv.dataset.vals.split(',').map(Number).filter(v=>!isNaN(v));
  if (vals.length<2) return;
  const w=cv.width=cv.clientWidth*2, h=cv.height=120, pad=8;
  const min=Math.min(...vals), max=Math.max(...vals), rg=(max-min)||1;
  const g=cv.getContext('2d');
  g.strokeStyle='#58a6ff'; g.lineWidth=2.5; g.beginPath();
  vals.forEach((v,i)=>{ const x=pad+i*(w-2*pad)/(vals.length-1), y=h-pad-(v-min)/rg*(h-2*pad); i?g.lineTo(x,y):g.moveTo(x,y);});
  g.stroke();
  g.strokeStyle='#30363d'; g.beginPath(); g.moveTo(pad,h-pad); g.lineTo(w-pad,h-pad); g.stroke();
  g.fillStyle='#8b949e'; g.font='20px monospace'; g.fillText(max.toFixed(1), pad, pad+16);
  g.fillText(min.toFixed(1), pad, h-6);
}
tick(); setInterval(tick, 15000);
</script></body></html>"""


def parse_run_log(path):
    """(reward list, eplen list, elapsed, iteration time, nan?) — с кэшем по mtime."""
    try:
        m = os.path.getmtime(path)
    except OSError:
        return None
    if CACHE.get(path, (0,))[0] != m:
        with open(path, "rb") as f:
            txt = f.read().decode("utf-8", "ignore")
        rew = [float(x) for x in re.findall(r"Mean reward: ([-0-9.e]+)", txt)]
        epl = [int(x) for x in re.findall(r"Mean episode length: (\d+)", txt)]
        el = re.findall(r"Time elapsed: (\d+:\d+:\d+)", txt)
        nan = bool(re.search(r"nan|NaN", txt[-200000:]))
        CACHE[path] = (m, {"rew": rew[-400:], "epl": epl, "el": el[-1] if el else None, "nan": nan})
    return CACHE[path][1]


def trend(vals, n=12):
    if len(vals) < 2 * n:
        return "RISING" if len(vals) >= 4 else "…"
    old, new = sum(vals[-2*n:-n]) / n, sum(vals[-n:]) / n
    p = (new - old) / (abs(old) + 1e-9)
    return "RISING" if p > 0.02 else ("FALLING" if p < -0.05 else "STAGNANT")


def collect():
    hb = {}
    try:
        hb = json.load(open(HEARTBEAT))
    except Exception:
        pass
    gpu = hb.get("gpu", {})
    gpu_s = (f"GPU {gpu.get('temp_c','?')}°C · {gpu.get('vram_mb','?')} МБ · "
             f"{gpu.get('util','?')}% · {gpu.get('power_w','?')} Вт")

    stages = []
    for d in sorted(glob.glob(f"{LOGROOT}/*/")):
        stage = os.path.basename(d.rstrip("/"))
        runs = sorted(glob.glob(d + "run_*"))
        if not runs:
            continue
        run = runs[-1]
        log = os.path.join(run, "train_stdout.log")
        info = parse_run_log(log)
        done = any(os.path.exists(os.path.join(r, "TRAIN_DONE")) for r in runs)
        if info and info["rew"]:
            rew, epl = info["rew"], info["epl"]
            stages.append({
                "name": stage,
                "state": "done" if done else "run",
                "info": (f"rew {rew[-1]:.2f} · ep {epl[-1] if epl else '?'} шагов · "
                         f"{info['el'] or ''} · {trend(rew)}"
                         + (" · NaN!" if info["nan"] else "")),
                "reward": rew[-200:],
            })
        else:
            stages.append({"name": stage, "state": "done" if done else "run",
                           "info": "лог пуст", "reward": []})
    # порядок: run сверху, потом done
    stages.sort(key=lambda s: (s["state"] != "run", s["name"]))

    finals, previews = [], []
    for p in sorted(glob.glob(f"{VIDEOS}/final/*.mp4"), key=os.path.getmtime, reverse=True):
        finals.append({"url": "/videos/final/" + os.path.basename(p),
                       "cap": os.path.basename(p) + " · " +
                              time.strftime("%H:%M %d.%m", time.localtime(os.path.getmtime(p)))})
    prev_all = sorted(glob.glob(f"{VIDEOS}/previews/*/*.mp4"), key=os.path.getmtime, reverse=True)
    for p in prev_all[:8]:
        rel = os.path.relpath(p, VIDEOS).replace("\\", "/")
        previews.append({"url": "/videos/" + rel,
                         "cap": rel + " · " + time.strftime("%H:%M", time.localtime(os.path.getmtime(p)))})

    try:
        rep = open(REPORT, encoding="utf-8", errors="ignore").read()[-2500:]
    except Exception:
        rep = "нет отчёта"
    return {"gpu": gpu_s, "stages": stages, "finals": finals,
            "previews": previews, "report": rep}


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        path = unquote(self.path)
        if path == "/" or path.startswith("/index"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/status":
            body = json.dumps(collect(), ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path.startswith("/videos/"):
            rel = path[len("/videos/"):]
            fp = os.path.normpath(os.path.join(VIDEOS, rel))
            if not fp.startswith(VIDEOS) or not os.path.isfile(fp):
                self.send_response(404); self.end_headers(); return
            ext = os.path.splitext(fp)[1]
            ctype = {".mp4": "video/mp4", ".png": "image/png", ".jpg": "image/jpeg"}.get(ext, "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(os.path.getsize(fp)))
            self.send_header("Accept-Ranges", "none")
            self.end_headers()
            with open(fp, "rb") as f:
                while chunk := f.read(65536):
                    self.wfile.write(chunk)
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", 8080), H)
    print("dashboard: http://localhost:8080", flush=True)
    srv.serve_forever()
