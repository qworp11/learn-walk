"""Печать motion-имени этапа: python stage_motion.py <stage>"""
import json
import sys

cfg = json.load(open("/mnt/g/LEARN WALK/configs/stages.json"))
print(cfg.get(sys.argv[1], {}).get("motion", ""), end="")
