import json, time, torch
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from compressed_tensors.quantization.lifecycle.forward import dequantize
torch.manual_seed(0)
O, I = 12288, 1536  # largest MLP shape in E2B
shape = torch.tensor([O, I])
packed = torch.randint(-2**31, 2**31 - 1, (O, I // 8), dtype=torch.int32, device="cuda")
scale = torch.rand(O, I // 32, dtype=torch.bfloat16, device="cuda")
x = torch.randn(1, 1, I, dtype=torch.bfloat16, device="cuda")
def t(f, n=20):
    f(); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return round((time.perf_counter() - s) / n * 1e3, 2)
w = dequantize(x_q=unpack_from_int32(packed, 4, shape), scale=scale)
wq = unpack_from_int32(packed, 4, shape)
r = {"unpack_ms": t(lambda: unpack_from_int32(packed, 4, shape)),
     "dequantize_ms": t(lambda: dequantize(x_q=wq, scale=scale)),
     "matvec_bf16_ms": t(lambda: torch.nn.functional.linear(x, w)),
     "matvec_fp16_ms": t(lambda: torch.nn.functional.linear(x.half(), w.half())),
     "simple_dequant_ms": t(lambda: (((packed.unsqueeze(-1) >> torch.arange(0, 32, 4, device="cuda", dtype=torch.int32)) & 15).view(O, I // 32, 32).to(torch.bfloat16) - 8).mul_(scale.unsqueeze(-1)).view(O, I))}
print(json.dumps(r))
