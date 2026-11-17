import json
import numpy as np

U = np.load("/mnt/g/LEARN WALK/data/motions/npz_g1/unified_all.npz")
print("keys:", sorted(U.keys()))
SEGM = json.load(open("/mnt/g/LEARN WALK/data/motions/npz_g1/unified_segments.json"))
starts = {}
for s in SEGM["segments"]:
    starts.setdefault(s["skill"], s["start"])

# какой ключ - мировые позы тел
key_pos = None
for k in U.keys():
    if "body_pos" in k:
        print("кандидат поз:", k, U[k].shape)
        key_pos = key_pos or k

# оригинальный танцевальный клип (из cmd.json s8_dance)
cmd = json.load(open("/root/hlc-logs/s8_dance/run_0/cmd.json"))
ORIG = np.load(cmd["env_overrides"]["trajectory_path"])
print("оригинал dance:", cmd["env_overrides"]["trajectory_path"])

def rootz(arr, frames, tag):
    a = arr[key_pos]
    out = []
    for f in frames:
        f2 = f % a.shape[0]
        out.append(round(float(a[f2, 0, 2]), 3))
    print(tag, "z тела0 по кадрам", frames, "=", out)

F = [900, 1000, 1100, 1300, 1600, 2000]
rootz(ORIG, F, "dance ОРИГИНАЛ ")
ds = starts["dance"]
rootz(U, [ds + f for f in F], f"dance UNIFIED+{ds} ")

# контроль: walk (он в оценке нормальный)
cmdw = json.load(open("/root/hlc-logs/s2_walk/run_0/cmd.json"))
ORW = np.load(cmdw["env_overrides"]["trajectory_path"])
rootz(ORW, [680, 800, 1000], "walk ОРИГИНАЛ  ")
ws = starts["walk"]
rootz(U, [ws + 680, ws + 800, ws + 1000], f"walk UNIFIED+{ws} ")
print("CHECK_DONE")
