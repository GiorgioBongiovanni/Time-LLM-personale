import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

    # semplice test con tensori su GPU
    x = torch.rand(3, 3).cuda()
    y = torch.rand(3, 3).cuda()
    z = x @ y
    print("Matrix product result:\n", z)
