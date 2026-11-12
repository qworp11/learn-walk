"""Тест: conv1d на cuda с выключенным cuDNN (Pascal)."""
import torch

print("cudnn version:", torch.backends.cudnn.version())
torch.backends.cudnn.enabled = False

x = torch.randn(4, 1, 50, device="cuda")
w = torch.ones(1, 1, 1, device="cuda")
y = torch.nn.functional.conv1d(x, w)
print("conv1d ok:", y.shape, float(y.sum()))

# большой тест как в motion_command: kernel_size=1
big = torch.randn(1024, 1, 300, device="cuda")
y2 = torch.nn.functional.conv1d(big, w)
torch.cuda.synchronize()
print("CONV_NO_CUDNN_OK", float(y2.mean()))
