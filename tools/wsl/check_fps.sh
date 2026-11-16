#!/bin/bash
source /root/hlc-venv/bin/activate
python - <<'EOF'
import numpy as np
for n in ["walk3_subject1", "run2_subject1", "jumps1_subject1",
          "fallAndGetUp1_subject1", "dance2_subject1", "sit_synth"]:
    d = np.load(f"/mnt/g/LEARN WALK/data/motions/npz_g1/{n}.npz")
    print(n, "fps =", d["fps"], "T =", d["joint_pos"].shape[0])
EOF
