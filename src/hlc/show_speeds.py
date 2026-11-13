import json

d = json.load(open(r"G:\LEARN WALK\data\motions\speeds.json"))
keys = ["sprint1_subject2", "sprint1_subject4", "run2_subject1", "run1_subject2",
        "jumps1_subject1", "walk1_subject1", "fallAndGetUp1_subject1",
        "dance2_subject1", "fight1_subject2"]
for k in keys:
    v = d[k]
    print("%-26s mean %.2f  p95 %.2f  dur %ss  net %sm" % (k, v["mean"], v["p95"], v["dur_s"], v["net_m"]))
