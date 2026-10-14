import torch
import vllm.v1.attention.ops.triton_prefill_attention as m
torch.manual_seed(0)
D, H, KVH, L = 512, 4, 1, 1500
for dt, blk in ((torch.float16, 32), (torch.float16, 16)):
    m.get_block_size = lambda dtype, b=blk: b
    q = torch.randn(L, H, D, device="cuda"); k = torch.randn(L, KVH, D, device="cuda"); v = torch.randn(L, KVH, D, device="cuda")
    o = torch.empty(L, H, D, device="cuda", dtype=dt)
    try:
        m.context_attention_fwd(q.to(dt), k.to(dt), v.to(dt), o, torch.tensor([0], device="cuda", dtype=torch.int32),
                                torch.tensor([L], device="cuda", dtype=torch.int32), L, is_causal=False, softmax_scale=D**-0.5)
        torch.cuda.synchronize()
        ref = torch.einsum("hqk,khd->qhd", torch.softmax(torch.einsum("qhd,khd->hqk", q, k.expand(L,H,D)) * D**-0.5, -1), v.expand(L,H,D))
        print(f"{dt} BLOCK={blk}: OK max|err|={(o.float()-ref).abs().max().item():.2e} rel={((o.float()-ref).norm()/ref.norm()).item():.2e}")
    except Exception as e:
        print(f"{dt} BLOCK={blk}: {type(e).__name__}: {str(e)[:100]}")
