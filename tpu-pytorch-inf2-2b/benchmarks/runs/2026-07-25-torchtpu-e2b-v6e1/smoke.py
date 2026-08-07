import torch

x = torch.randn(128, 128, dtype=torch.bfloat16, device="tpu")
y = torch.randn(128, 128, dtype=torch.bfloat16, device="tpu")
matmul = torch.compile(torch.matmul, backend="tpu")
out = matmul(x, y).cpu()
print("compiled matmul OK:", tuple(out.shape), out.dtype)
