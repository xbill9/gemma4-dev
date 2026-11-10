"""SGLang's Marlin path on sm_75 (patched to vLLM's ops) against fp16 matmul on the dequantized weights."""
import torch
from vllm.model_executor.layers.quantization.utils.quant_utils import gptq_quantize_weights, gptq_pack
from vllm.scalar_type import scalar_types as vst
from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
from sglang.kernels.ops.gemm.gptq_marlin import gptq_marlin_gemm
from sglang.srt.layers.quantization.marlin_utils import (marlin_permute_scales, marlin_make_workspace,
                                                        query_marlin_supported_quant_types, scalar_types)
dev = torch.device("cuda")
print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())
print("supported:", [str(t) for t in query_marlin_supported_quant_types(False)])
torch.manual_seed(0)
G = 32
for k, n in [(1536, 6144), (6144, 1536), (1536, 2048), (2048, 1536)]:
    w = torch.randn(k, n, dtype=torch.float16, device=dev) * 0.02
    w_ref, q_w, s = gptq_quantize_weights(w, vst.uint4b8, G)
    packed = gptq_pack(q_w, 4, k, n)
    mq = gptq_marlin_repack(packed, torch.empty(0, dtype=torch.int, device=dev), k, n, 4)
    ms = marlin_permute_scales(s, k, n, G)
    ws = marlin_make_workspace(dev)
    for m in (1, 7, 16, 64, 256):
        a = torch.randn(m, k, dtype=torch.float16, device=dev)
        out = gptq_marlin_gemm(a, None, mq, ms, None, None, None, None, ws, scalar_types.uint4b8, m, n, k)
        ref = a @ w_ref
        rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        print(f"k={k} n={n} m={m:3d} rel_err={rel:.2e} {'OK' if rel < 1e-2 else 'BAD'}")
