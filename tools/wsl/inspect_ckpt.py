#!/bin/bash
source /root/hlc-venv/bin/activate
python - <<'PYEOF'
import torch
d = torch.load("/root/hlc-logs/s2_walk/run_0/model_5499.pt",
               weights_only=False, map_location="cpu")
print("TOP:", list(d.keys()))
for grp in ("actor_state_dict", "critic_state_dict"):
    print(grp, ":", list(d[grp].keys()))
inf = d.get("infos", {})
print("infos:", {k: (v if not hasattr(v, "shape") else tuple(v.shape))
                 for k, v in inf.items()} if isinstance(inf, dict) else inf)
PYEOF
echo "KEYS_EXIT=$?"