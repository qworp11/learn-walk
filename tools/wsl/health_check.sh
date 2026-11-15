#!/bin/bash
# Health-check этапа: вердикт по динамике награды и эпизодов.
# Usage: health_check.sh <stage>
STAGE="${1:-s2_walk}"
source /root/hlc-venv/bin/activate
python - "$STAGE" <<'PYEOF'
import glob, os, re, sys

stage = sys.argv[1]
root = f"/root/hlc-logs/{stage}"
runs = sorted(glob.glob(f"{root}/run_*"))
if not runs:
    print(f"{stage}: НЕТ ЗАПУСКОВ"); sys.exit(1)
run = runs[-1]
log = os.path.join(run, "train_stdout.log")
txt = open(log, "rb").read().decode("utf-8", "ignore")

rew = [float(x) for x in re.findall(r"Mean reward: ([-0-9.e]+)", txt)]
epl = [int(x) for x in re.findall(r"Mean episode length: (\d+)", txt)]
elapsed = re.findall(r"Time elapsed: (\d+:\d+:\d+)", txt)

def trend(vals, n=12):
    if len(vals) < 2*n: return None, None, 0
    old = sum(vals[-2*n:-n])/n
    new = sum(vals[-n:])/n
    return old, new, (new-old)/(abs(old)+1e-9)

ro, rn, rpct = trend(rew)
eo, en, epct = trend(epl)
nan = "nan" in txt.lower()[-200000:]
last5 = sum(rew[-5:])/max(1,len(rew[-5:]))

verdict = "RISING"
if rpct is not None:
    if rpct < 0.02: verdict = "STAGNANT"
    if rpct < -0.05: verdict = "FALLING"

print(f"{stage} | {run.split('/')[-1]} | elapsed={elapsed[-1] if elapsed else '?'}")
print(f"reward: сейчас ~{last5:.2f}, было ~{ro:.2f} -> {rn:.2f} ({rpct*100:+.0f}%) | ep_len: {eo:.0f} -> {en:.0f}")
print(f"nan_suspect={nan} | вердикт: {verdict}")
PYEOF
