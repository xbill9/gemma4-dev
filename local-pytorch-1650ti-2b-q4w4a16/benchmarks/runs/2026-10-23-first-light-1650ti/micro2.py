import json, time, torch
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from compressed_tensors.quantization.lifecycle.forward import dequantize
torch.manual_seed(0)
O, I = 12288, 1536
packed = torch.randint(-2**31, 2**31 - 1, (O, I // 8), dtype=torch.int32, device="cuda")
scale = torch.rand(O, I // 32, dtype=torch.float16, device="cuda")
x = torch.randn(1, 1, I, dtype=torch.float16, device="cuda")
SHIFTS = torch.arange(0, 32, 4, device="cuda", dtype=torch.int32)
def w4a16(x, packed, scale):
    o, g = scale.shape
    q = ((packed.unsqueeze(-1) >> SHIFTS) & 15).view(o, g, 32).to(x.dtype) - 8
    return torch.nn.functional.linear(x, (q * scale.unsqueeze(-1)).view(o, g * 32))
ref = dequantize(x_q=unpack_from_int32(packed, 4, torch.tensor([O, I])), scale=scale)
mine = ((packed.unsqueeze(-1) >> SHIFTS) & 15).view(O, I // 32, 32).to(torch.float16).sub(8).mul(scale.unsqueeze(-1)).view(O, I)
fused = torch.compile(w4a16, dynamic=False)
def t(f, n=50):
    f(); f(); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return round((time.perf_counter() - s) / n * 1e3, 3)
y_ref = torch.nn.functional.linear(x, ref.half())
print(json.dumps({"dequant_bit_exact": bool(torch.equal(ref, mine)),
    "compiled_ms": t(lambda: fused(x, packed, scale)),
    "compiled_max_abs_err": float((fused(x, packed, scale) - y_ref).abs().max()), "ref_max_abs": float(y_ref.abs().max())}))
