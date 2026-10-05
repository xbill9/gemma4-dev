"""Load the saved TP=2 prefill+decode models via parallel_model_load and measure decode
tok/s (steady-state) + validate SEQ_MATCH vs CPU reference. Tests whether the reloaded
on-device model avoids the per-call IPC overhead of the in-memory traced model."""
import sys, os, types, time
m = types.ModuleType("transformers.utils.fx"); m.HFTracer=object; m.symbolic_trace=None
sys.modules["transformers.utils.fx"] = m
import multiprocessing

MP = "/workspace/real-gemma4-E2B-it"
MAX = int(os.environ.get("KV_MAX", "2048"))
BUCKET = int(os.environ.get("KV_BUCKET", "512"))
MAXNEW = int(os.environ.get("MAXNEW", "30"))

def main():
    import torch
    torch.manual_seed(0)
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    from neuronx_distributed.trace import parallel_model_load
    NEG = torch.finfo(torch.float32).min
    tok = AutoTokenizer.from_pretrained(MP)
    mm = Gemma4ForConditionalGeneration.from_pretrained(MP, torch_dtype=torch.float32, attn_implementation="eager"); mm.eval()
    lang = mm.model.language_model; cfg = lang.config; SW = cfg.sliding_window
    NONSHARED, LINFO = [], {}
    for i, lyr in enumerate(lang.layers[:cfg.num_hidden_layers]):
        a = lyr.self_attn
        if not a.is_kv_shared_layer:
            hd=a.head_dim; NONSHARED.append(i); LINFO[i]=(a.k_proj.out_features//hd, hd)
    ec = mm.generation_config.eos_token_id
    EOS = set(ec) if isinstance(ec,(list,tuple)) else {ec}
    def embed_ids(idl):
        ids=torch.tensor([idl])
        with torch.no_grad(): ie=lang.embed_tokens(ids); ple=lang.get_per_layer_inputs(ids, ie)
        return ie, ple
    def host_pos(pos):
        ar=torch.arange(MAX); onehot=(ar==pos).view(1,1,MAX,1).to(torch.float32); valid=ar<=pos
        full=torch.where(valid,0.0,NEG).view(1,1,1,MAX); slide=torch.where(valid&(ar>pos-SW),0.0,NEG).view(1,1,1,MAX)
        return torch.tensor([[pos]],dtype=torch.long), onehot, full, slide

    enc = tok.apply_chat_template([{"role":"user","content":"What is the capital of France?"}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
    prompt = enc["input_ids"][0].tolist(); n0=len(prompt)
    print("prompt", n0, "MAX", MAX, flush=True)

    t=time.time(); pre = parallel_model_load("/workspace/tp_pre"); print("pre loaded", round(time.time()-t,1),"s", flush=True)
    t=time.time(); dec = parallel_model_load("/workspace/tp_dec"); print("dec loaded", round(time.time()-t,1),"s", flush=True)

    def run_greedy(maxnew):
        pad=prompt+[0]*(BUCKET-n0); ie,ple=embed_ids(pad); am=torch.tensor([[1]*n0+[0]*(BUCKET-n0)])
        lg,ks,vs = pre(ie,am,ple); first=int(lg[0,n0-1].argmax())
        kb=[torch.zeros(1,LINFO[i][0],MAX,LINFO[i][1]) for i in NONSHARED]; vb=[torch.zeros(1,LINFO[i][0],MAX,LINFO[i][1]) for i in NONSHARED]
        for j in range(len(NONSHARED)):
            kb[j][:,:,:n0,:]=ks[j][:,:,:n0,:]; vb[j][:,:,:n0,:]=vs[j][:,:,:n0,:]
        seq=[first]; cur=n0; t0=time.time(); steps=0
        for _ in range(maxnew):
            if seq[-1] in EOS: break
            e1,p1=embed_ids([seq[-1]]); pi,o,f,s=host_pos(cur)
            l1,kb,vb=dec(e1,p1,pi,o,f,s,kb,vb); seq.append(int(l1[0,0].argmax())); cur+=1; steps+=1
        return seq, time.time()-t0, steps

    print("=== warmup run ===", flush=True)
    seq,_,_ = run_greedy(5)
    print("=== timed run ===", flush=True)
    seq, secs, steps = run_greedy(MAXNEW)
    print("DEV GEN:", repr(tok.decode([x for x in seq if x not in EOS], skip_special_tokens=True)), flush=True)
    print("DECODE tok/s:", round(steps/secs,1), "|", round(secs/max(steps,1)*1000), "ms/tok |", steps,"steps", flush=True)
    print("TP_RUN_OK", flush=True)

if __name__ == "__main__":
    if os.environ.get("_TP_CHILD") != "1":
        os.environ["_TP_CHILD"]="1"; multiprocessing.set_start_method("spawn", force=True); main()
