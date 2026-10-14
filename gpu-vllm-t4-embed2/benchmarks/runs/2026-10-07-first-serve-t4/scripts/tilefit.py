import torch, sys
import vllm.v1.attention.ops.triton_prefill_attention as m
torch.manual_seed(0)
for D, H, KVH in ((256, 4, 2), (512, 4, 1)):
    for blk in (32, 16):
        m.get_block_size = lambda dtype, b=blk: b
        L = 300
        q = torch.randn(L, H, D, device="cuda", dtype=torch.float32)
        k = torch.randn(L, KVH, D, device="cuda", dtype=torch.float32)
        v = torch.randn(L, KVH, D, device="cuda", dtype=torch.float32)
        o = torch.empty_like(q)
        try:
            m.context_attention_fwd(q, k, v, o, torch.tensor([0], device="cuda", dtype=torch.int32),
                                    torch.tensor([L], device="cuda", dtype=torch.int32), L, is_causal=False, softmax_scale=1.0)
            torch.cuda.synchronize()
            kk = k.repeat_interleave(H // KVH, 1); vv = v.repeat_interleave(H // KVH, 1)
            ref = torch.softmax(torch.einsum("qhd,khd->hqk", q, kk), -1)
            ref = torch.einsum("hqk,khd->qhd", ref, vv)
            print(f"D={D} BLOCK={blk}: OK max|err|={(o-ref).abs().max().item():.2e}")
        except Exception as e:
            print(f"D={D} BLOCK={blk}: {type(e).__name__}: {str(e)[:110]}")
