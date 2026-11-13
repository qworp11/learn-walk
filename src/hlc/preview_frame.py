"""Кадр из превью для просмотра: preview_frame.py <mp4> <n>"""
import sys

import imageio

mp4, idx = sys.argv[1], int(sys.argv[2])
out = mp4.replace(".mp4", f"_frame{idx}.png")
r = imageio.get_reader(mp4)
imageio.imwrite(out, r.get_data(idx))
print("saved:", out)
