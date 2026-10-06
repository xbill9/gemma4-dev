"""Load TP+alias prefill/decode (parallel_model_load), seed per-rank device-resident KV,
run greedy, validate + measure decode tok/s. Target: ~65-80 tok/s (both cores, aliased)."""
import sys, os, types, time
m = types.ModuleType("transformers.utils.fx"); m.HFTracer=object; m.symbolic_trace=None
sys.modules["transformers.utils.fx"] = m
import multiprocessing
MP="/workspace/real-gemma4-E2B-it"; TP=2
MAX=int(os.environ.get("KV_MAX","2048")); BUCKET=int(os.environ.get("KV_BUCKET","512")); MAXNEW=int(os.environ.get("MAXNEW","30"))

def main():
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    from neuronx_distributed.trace import parallel_model_load
    NEG=torch.finfo(torch.float32).min
    tok=AutoTokenizer.from_pretrained(MP)
    mm=Gemma4ForConditionalGeneration.from_pretrained(MP,torch_dtype=torch.float32,attn_implementation="eager"); mm.eval()
    lang=mm.model.language_model; cfg=lang.config; SW=cfg.sliding_window
    NONSHARED,LINFO=[],{}
    for i,lyr in enumerate(lang.layers[:cfg.num_hidden_layers]):
        a=lyr.self_attn
        if not a.is_kv_shared_layer:
            hd=a.head_dim; NONSHARED.append(i); LINFO[i]=(a.k_proj.out_features//hd,hd)
    NK=len(NONSHARED)
    ec=mm.generation_config.eos_token_id; EOS=set(ec) if isinstance(ec,(list,tuple)) else {ec}
    def embed_ids(idl):
        ids=torch.tensor([idl])
        with torch.no_grad(): ie=lang.embed_tokens(ids); ple=lang.get_per_layer_inputs(ids,ie)
        return ie,ple
    def host_pos(pos):
        ar=torch.arange(MAX); oh=(ar==pos).view(1,1,MAX,1).to(torch.float32); valid=ar<=pos
        fm=torch.where(valid,0.0,NEG).view(1,1,1,MAX); sm=torch.where(valid&(ar>pos-SW),0.0,NEG).view(1,1,1,MAX)
        return torch.tensor([[pos]],dtype=torch.long),oh,fm,sm
    enc=tok.apply_chat_template([{"role":"user","content":"What is the capital of France?"}],add_generation_prompt=True,return_tensors="pt",return_dict=True)
    prompt=enc["input_ids"][0].tolist(); n0=len(prompt)
    print("prompt",n0,"MAX",MAX,flush=True)
    t=time.time(); pre=parallel_model_load("/workspace/tpa_pre"); print("pre loaded",round(time.time()-t,1),flush=True)
    t=time.time(); dec=parallel_model_load("/workspace/tpa_dec"); print("dec loaded",round(time.time()-t,1),flush=True)
    # locate per-rank state params (jit-scripted; access via named_parameters 'states.N')
    def state_params(r):
        sp=[(int(''.join(c for c in n if c.isdigit())),p) for n,p in dec.models[r].named_parameters() if n.startswith("states.")]
        return [p for _,p in sorted(sp,key=lambda x:x[0])]
    SP_RANK=[state_params(r) for r in range(TP)]
    print("per-rank states:",[len(s) for s in SP_RANK],"(expect",2*NK,"each)",flush=True)

    def seed(ks,vs):
        for r in range(TP):
            st=SP_RANK[r]
            for j in range(NK):
                sk=torch.zeros(1,LINFO[NONSHARED[j]][0],MAX,LINFO[NONSHARED[j]][1]); sk[:,:,:n0,:]=ks[j][:,:,:n0,:]
                sv=torch.zeros(1,LINFO[NONSHARED[j]][0],MAX,LINFO[NONSHARED[j]][1]); sv[:,:,:n0,:]=vs[j][:,:,:n0,:]
                st[j].data.copy_(sk); st[NK+j].data.copy_(sv)
    def run(maxnew):
        pad=prompt+[0]*(BUCKET-n0); ie,ple=embed_ids(pad); am=torch.tensor([[1]*n0+[0]*(BUCKET-n0)])
        lg,ks,vs=pre(ie,am,ple); first=int(lg[0,n0-1].argmax()); seed(ks,vs)
        seq=[first]; cur=n0; td=0.0; steps=0
        for _ in range(maxnew):
            if seq[-1] in EOS: break
            e1,p1=embed_ids([seq[-1]]); pi,o,f,s=host_pos(cur); a0=time.time()
            r=dec(e1,p1,pi,o,f,s); lg1=r[0] if isinstance(r,(tuple,list)) else r
            td+=time.time()-a0; seq.append(int(lg1[0,0].argmax())); cur+=1; steps+=1
        return seq,td,steps
    print("=== warmup ===",flush=True); run(5)
    print("=== timed ===",flush=True); seq,secs,steps=run(MAXNEW)
    txt=tok.decode([x for x in seq if x not in EOS],skip_special_tokens=True)
    print("DEV GEN:",repr(txt),flush=True); print("CORRECT(Paris):","Paris" in txt,flush=True)
    print("TP+ALIAS DECODE tok/s:",round(steps/secs,1),"|",round(secs/max(steps,1)*1000),"ms/tok |",steps,"steps",flush=True)
    print("TPA_RUN_OK",flush=True)

if __name__ == "__main__":
    if os.environ.get("_TP_CHILD") != "1":
        os.environ["_TP_CHILD"]="1"; multiprocessing.set_start_method("spawn", force=True); main()
