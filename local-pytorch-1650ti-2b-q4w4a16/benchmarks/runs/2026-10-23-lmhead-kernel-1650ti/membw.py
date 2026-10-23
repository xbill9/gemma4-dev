"""Measured device memory bandwidth: device-to-device copy of a 512 MiB buffer (reads + writes counted)."""
import json, time, torch
a = torch.empty(512 * 2**20, dtype=torch.uint8, device="cuda"); b = torch.empty_like(a)
for _ in range(3): b.copy_(a)
torch.cuda.synchronize(); s = time.perf_counter(); n = 20
for _ in range(n): b.copy_(a)
torch.cuda.synchronize(); t = (time.perf_counter() - s) / n
x = torch.empty(256 * 2**20, dtype=torch.float32, device="cuda"); x.sum(); torch.cuda.synchronize()
s = time.perf_counter()
for _ in range(n): x.sum()
torch.cuda.synchronize(); t2 = (time.perf_counter() - s) / n
print(json.dumps({"copy_GBps_read_plus_write": round(2 * a.numel() / t / 1e9, 1), "sum_GBps_read": round(x.numel() * 4 / t2 / 1e9, 1)}))
