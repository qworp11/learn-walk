"""Извлечение кадров из mp4 для визуальной проверки."""
import imageio

out = "/mnt/g/LEARN WALK/results/videos"
r = imageio.get_reader(f"{out}/test_ref.mp4")
for i in range(3):
    imageio.imwrite(f"{out}/frame_{i}.png", r.get_data(i * 40))
print("frames saved")
