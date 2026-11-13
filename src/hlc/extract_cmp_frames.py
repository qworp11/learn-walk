"""Извлечение кадров из сравнительного видео."""
import imageio

out = "/mnt/g/LEARN WALK/results/videos"
r = imageio.get_reader(f"{out}/test_compare.mp4")
for i in range(2):
    imageio.imwrite(f"{out}/cmp_frame_{i}.png", r.get_data(i * 15))
print("frames saved")
