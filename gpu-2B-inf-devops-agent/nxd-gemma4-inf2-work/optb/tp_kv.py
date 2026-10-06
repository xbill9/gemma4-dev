"""TP=2 two-graph KV-cache decode for Gemma4-E2B. Shards q/o/gate/up/down across 2 cores
(k/v replicated, MQA), traces prefill + static-KV decode via NxD parallel_model_trace,
runs device greedy, compares to CPU fp32 reference, and measures decode tok/s."""
import sys, os, types, time
m = types.ModuleType("transformers.utils.fx"); m.HFTracer=object; m.symbolic_trace=None
sys.modules["transformers.utils.fx"] = m
import multiprocessing

MP = "/workspace/real-gemma4-E2B-it"
TP = 2
MAX = int(os.environ.get("KV_MAX", "2048"))
BUCKET = int(os.environ.get("KV_BUCKET", "512"))
MAXNEW = int(os.environ.get("MAXNEW", "30"))
CARGS = ["--model-type","transformer","--auto-cast","all","--auto-cast-type","bf16"]

def _build_shared():
    """Load model, gelu patch, discover nonshared layers. Returns (mm, lang, lm_head, softcap, NONSHARED, LINFO, SW)."""
    import torch
    from transformers import Gemma4ForConditionalGeneration
    mm = Gemma4ForConditionalGeneration.from_pretrained(MP, torch_dtype=torch.float32, attn_implementation="eager"); mm.eval()
    lang = mm.model.language_model; cfg = lang.config
    class GeluTanh(torch.nn.Module):
        def forward(s, x): return 0.5*x*(1.0+torch.tanh(0.7978845608028654*(x+0.044715*x*x*x)))
    for mod in lang.modules():
        if hasattr(mod, "act_fn"): mod.act_fn = GeluTanh()
    NONSHARED, LINFO = [], {}
    for i, lyr in enumerate(lang.layers[:cfg.num_hidden_layers]):
        a = lyr.self_attn
        if not a.is_kv_shared_layer:
            hd = a.head_dim; NONSHARED.append(i); LINFO[i] = (a.k_proj.out_features // hd, hd)
    softcap = getattr(mm.config.text_config, "final_logit_softcapping", None)
    return mm, lang, mm.lm_head, softcap, NONSHARED, LINFO, cfg.sliding_window

def _shard(lang, cfg_nlayers):
    import torch
    from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
    from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_rank
    rank = get_tensor_model_parallel_rank()
    def col(orig):
        out, inf = orig.weight.shape
        l = ColumnParallelLinear(inf, out, bias=False, gather_output=False); c = out // TP
        l.weight.data.copy_(orig.weight.data[rank*c:(rank+1)*c, :]); return l
    def row(orig):
        out, inf = orig.weight.shape
        l = RowParallelLinear(inf, out, bias=False, input_is_parallel=True); c = inf // TP
        l.weight.data.copy_(orig.weight.data[:, rank*c:(rank+1)*c]); return l
    for lyr in lang.layers[:cfg_nlayers]:
        a = lyr.self_attn
        a.q_proj = col(a.q_proj); a.o_proj = row(a.o_proj)
        a.num_key_value_groups = a.num_key_value_groups // TP
        mlp = lyr.mlp
        mlp.gate_proj = col(mlp.gate_proj); mlp.up_proj = col(mlp.up_proj); mlp.down_proj = row(mlp.down_proj)

NEG = None
def _sc(softcap, lg):
    import torch
    return softcap*torch.tanh(lg/softcap) if softcap else lg

def get_pre():
    import torch
    from transformers import DynamicCache
    mm, lang, lm_head, softcap, NONSHARED, LINFO, SW = _build_shared()
    _shard(lang, lang.config.num_hidden_layers)
    class PreWrap(torch.nn.Module):
        def __init__(s): super().__init__(); s.lang=lang; s.head=lm_head
        def forward(s, ie, am, ple):
            cache = DynamicCache()
            out = s.lang(inputs_embeds=ie, per_layer_inputs=ple, attention_mask=am, use_cache=True, past_key_values=cache)
            lg = _sc(softcap, s.head(out.last_hidden_state))
            ks = [cache.layers[i].keys for i in NONSHARED]; vs = [cache.layers[i].values for i in NONSHARED]
            return (lg, ks, vs)
    return PreWrap().eval(), {}

def get_dec():
    import torch
    mm, lang, lm_head, softcap, NONSHARED, LINFO, SW = _build_shared()
    _shard(lang, lang.config.num_hidden_layers)
    class StaticKV:
        is_compileable = False
        def __init__(s, kb, vb, oh):
            s.key={i:kb[j] for j,i in enumerate(NONSHARED)}; s.val={i:vb[j] for j,i in enumerate(NONSHARED)}; s.oh=oh
        def update(s,k,v,idx,*a,**kw):
            s.key[idx]=s.key[idx]*(1.0-s.oh)+k*s.oh; s.val[idx]=s.val[idx]*(1.0-s.oh)+v*s.oh; return s.key[idx],s.val[idx]
        def get_seq_length(s,*a,**k): return 0
        def export(s): return [s.key[i] for i in NONSHARED],[s.val[i] for i in NONSHARED]
    class DecWrap(torch.nn.Module):
        def __init__(s): super().__init__(); s.lang=lang; s.head=lm_head
        def forward(s, ie, ple, position_ids, onehot, full_mask, slide_mask, key_bufs, val_bufs):
            cache = StaticKV(key_bufs, val_bufs, onehot)
            masks = {"full_attention": full_mask, "sliding_attention": slide_mask}
            out = s.lang(inputs_embeds=ie, per_layer_inputs=ple, position_ids=position_ids, attention_mask=masks, use_cache=True, past_key_values=cache)
            lg = _sc(softcap, s.head(out.last_hidden_state))
            ks, vs = cache.export(); return (lg, ks, vs)
    return DecWrap().eval(), {}

def main():
    import torch
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    import neuronx_distributed
    global NEG; NEG = torch.finfo(torch.float32).min
    tok = AutoTokenizer.from_pretrained(MP)
    ref_mm, rlang, rhead, softcap, NONSHARED, LINFO, SW = _build_shared()
    ec = ref_mm.generation_config.eos_token_id
    EOS = set(ec) if isinstance(ec,(list,tuple)) else {ec}

    def embed_ids(idl):
        ids=torch.tensor([idl])
        with torch.no_grad(): ie=rlang.embed_tokens(ids); ple=rlang.get_per_layer_inputs(ids, ie)
        return ie, ple
    def host_pos(pos):
        ar=torch.arange(MAX)
        onehot=(ar==pos).view(1,1,MAX,1).to(torch.float32); valid=ar<=pos
        full=torch.where(valid,0.0,NEG).view(1,1,1,MAX)
        slide=torch.where(valid&(ar>pos-SW),0.0,NEG).view(1,1,1,MAX)
        return torch.tensor([[pos]],dtype=torch.long), onehot, full, slide

    enc = tok.apply_chat_template([{"role":"user","content":"What is the capital of France?"}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
    prompt = enc["input_ids"][0].tolist(); n0 = len(prompt)
    assert n0 <= BUCKET
    print("prompt", n0, "MAX", MAX, "BUCKET", BUCKET, flush=True)

    print("tracing TP prefill ...", flush=True)
    pre = neuronx_distributed.trace.parallel_model_trace(get_pre, (embed_ids(prompt+[0]*(BUCKET-n0))[0], torch.tensor([[1]*n0+[0]*(BUCKET-n0)]), embed_ids(prompt+[0]*(BUCKET-n0))[1]), tp_degree=TP, compiler_args=CARGS)
    print("PREFILL_TRACED", flush=True)
    ie1, ple1 = embed_ids([prompt[-1]])
    kb0=[torch.zeros(1,LINFO[i][0],MAX,LINFO[i][1]) for i in NONSHARED]; vb0=[torch.zeros(1,LINFO[i][0],MAX,LINFO[i][1]) for i in NONSHARED]
    pid,oh,fm,sm = host_pos(n0)
    print("tracing TP decode ...", flush=True)
    dec = neuronx_distributed.trace.parallel_model_trace(get_dec, (ie1, ple1, pid, oh, fm, sm, kb0, vb0), tp_degree=TP, compiler_args=CARGS)
    print("DECODE_TRACED", flush=True)

    def run_greedy(pre_fn, dec_fn, maxnew):
        pad = prompt+[0]*(BUCKET-n0); ie,ple=embed_ids(pad); am=torch.tensor([[1]*n0+[0]*(BUCKET-n0)])
        lg,ks,vs = pre_fn(ie,am,ple); first=int(lg[0,n0-1].argmax())
        kb=[torch.zeros(1,LINFO[i][0],MAX,LINFO[i][1]) for i in NONSHARED]; vb=[torch.zeros(1,LINFO[i][0],MAX,LINFO[i][1]) for i in NONSHARED]
        for j in range(len(NONSHARED)):
            kb[j][:,:,:n0,:]=ks[j][:,:,:n0,:]; vb[j][:,:,:n0,:]=vs[j][:,:,:n0,:]
        seq=[first]; cur=n0; t0=time.time(); steps=0
        for _ in range(maxnew):
            if seq[-1] in EOS: break
            e1,p1=embed_ids([seq[-1]]); pi,o,f,s=host_pos(cur)
            l1,kb,vb=dec_fn(e1,p1,pi,o,f,s,kb,vb); seq.append(int(l1[0,0].argmax())); cur+=1; steps+=1
        return seq, (time.time()-t0), steps

    # CPU reference (eager, same static-KV logic) for correctness
    print("=== CPU ref greedy ===", flush=True)
    def cpu_pre(ie,am,ple):
        from transformers import DynamicCache
        cache=DynamicCache()
        out=rlang(inputs_embeds=ie,per_layer_inputs=ple,attention_mask=am,use_cache=True,past_key_values=cache)
        lg=_sc(softcap, rhead(out.last_hidden_state))
        return lg,[cache.layers[i].keys for i in NONSHARED],[cache.layers[i].values for i in NONSHARED]
    class SKV:
        is_compileable=False
        def __init__(s,kb,vb,oh): s.key={i:kb[j] for j,i in enumerate(NONSHARED)}; s.val={i:vb[j] for j,i in enumerate(NONSHARED)}; s.oh=oh
        def update(s,k,v,idx,*a,**kw): s.key[idx]=s.key[idx]*(1.0-s.oh)+k*s.oh; s.val[idx]=s.val[idx]*(1.0-s.oh)+v*s.oh; return s.key[idx],s.val[idx]
        def get_seq_length(s,*a,**k): return 0
    def cpu_dec(ie,ple,pi,oh,fm,sm,kb,vb):
        cache=SKV(kb,vb,oh)
        out=rlang(inputs_embeds=ie,per_layer_inputs=ple,position_ids=pi,attention_mask={"full_attention":fm,"sliding_attention":sm},use_cache=True,past_key_values=cache)
        return _sc(softcap,rhead(out.last_hidden_state)),[cache.key[i] for i in NONSHARED],[cache.val[i] for i in NONSHARED]
    cpu_seq,_,_ = run_greedy(cpu_pre, cpu_dec, MAXNEW)
    print("=== DEVICE TP greedy ===", flush=True)
    dev_seq, secs, steps = run_greedy(pre, dec, MAXNEW)
    print("CPU GEN:", repr(tok.decode([s for s in cpu_seq if s not in EOS], skip_special_tokens=True)), flush=True)
    print("DEV GEN:", repr(tok.decode([s for s in dev_seq if s not in EOS], skip_special_tokens=True)), flush=True)
    print("SEQ_MATCH", cpu_seq == dev_seq, flush=True)
    print("DECODE tok/s (post-load):", round(steps/secs,1), "|", round(secs/max(steps,1)*1000), "ms/tok |", steps, "steps", flush=True)
    print("TP_KV_OK", flush=True)

if __name__ == "__main__":
    if os.environ.get("_TP_CHILD") != "1":
        os.environ["_TP_CHILD"]="1"; multiprocessing.set_start_method("spawn", force=True); main()
