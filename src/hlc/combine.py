"""Склейка npy-эпизодов и mp4-видео.
python combine_npy.py <out.npy> <npy1> <npy2> ...
python combine_npy.py <out.mp4> <mp4_1> <mp4_2> ...  (по расширению)
"""
import sys

import imageio
import numpy as np


def main():
    args = sys.argv[1:]
    out, inputs = args[0], args[1:]
    if out.endswith(".npy"):
        arrs = [np.load(p) for p in inputs]
        n = min(a.shape[0] for a in arrs)
        np.save(out, np.concatenate([a[:n] for a in arrs], axis=0))
        print("npy combined:", out, n * len(arrs), "frames")
    else:
        frames_written = 0
        with imageio.get_writer(out, fps=50, codec="libx264", quality=8) as w:
            for p in inputs:
                for fr in imageio.get_reader(p):
                    w.append_data(fr)
                    frames_written += 1
        print("mp4 merged:", out, frames_written, "frames")


if __name__ == "__main__":
    main()
