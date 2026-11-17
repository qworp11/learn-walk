#!/bin/bash
source /root/hlc-venv/bin/activate
python - <<'PYEOF'
import json
for st, run in [("s2_walk", "run_0"), ("s1_balance", "run_1"),
                ("s9_sit", "run_2"), ("s3_run", "run_0")]:
    d = json.load(open(f"/root/hlc-logs/{st}/{run}/cmd.json"))
    ov = d.get("env_overrides", {})
    print(st, "->", json.dumps(ov, ensure_ascii=False)[:240])
PYEOF
