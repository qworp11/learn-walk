"""Проверка warp-ядра на Pascal (sm_61)."""
import warp as wp

print("warp", wp.__version__)


@wp.kernel
def dbl(a: wp.array[float], b: wp.array[float]):
    i = wp.tid()
    b[i] = a[i] * 2.0


xs = wp.array([1.0, 2.0, 3.0], dtype=float, device="cuda:0")
ys = wp.empty_like(xs)
wp.launch(dbl, dim=3, inputs=[xs, ys], device="cuda:0")
res = ys.numpy().tolist()
print("result:", res)
assert res == [2.0, 4.0, 6.0], res
print("WARP_PASCAL_OK")
