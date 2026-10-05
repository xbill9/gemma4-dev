import os, sys, torch, transformers, time
torch.manual_seed(0)
MP="/workspace/real-gemma4-E2B-it"
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration, DynamicCache

tok=AutoTokenizer.from_pretrained(MP)
m=Gemma4ForConditionalGeneration.from_pretrained(MP, torch_dtype=torch.float32, attn_implementation="eager"); m.eval()
lang=m.model.language_model; lm_head=m.lm_head
softcap=getattr(m.config.text_config,"final_logit_softcapping",None)

class GeluTanh(torch.nn.Module):
    def forward(self,x): return 0.5*x*(1.0+torch.tanh(0.7978845608028654*(x+0.044715*x*x*x)))
for mod in lang.modules():
    if hasattr(mod,"act_fn"): mod.act_fn=GeluTanh()

class Wrap(torch.nn.Module):
    def __init__(s,lang,head,sc): super().__init__(); s.lang=lang; s.head=head; s.sc=sc
    def forward(s, inputs_embeds, attention_mask, per_layer_inputs):
        out=s.lang(inputs_embeds=inputs_embeds, per_layer_inputs=per_layer_inputs,
                   attention_mask=attention_mask, use_cache=True, past_key_values=DynamicCache())
        h=out.last_hidden_state; lg=s.head(h)
        if s.sc: lg=s.sc*torch.tanh(lg/s.sc)
        return lg
w=Wrap(lang,lm_head,softcap).eval()

L=40
PAD=0
ec=m.generation_config.eos_token_id
EOS=set(ec) if isinstance(ec,(list,tuple)) else {ec}
print("EOS ids",EOS,"PAD",PAD,"L",L, flush=True)

msgs=[{"role":"user","content":"What is the capital of France?"}]
enc=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
ids0=enc["input_ids"][0].tolist(); n0=len(ids0)
print("prompt tokens",n0, flush=True); assert n0 < L

def build(cur):
    n=len(cur); padded=cur+[PAD]*(L-n)
    ids=torch.tensor([padded]); am=torch.tensor([[1]*n+[0]*(L-n)])
    with torch.no_grad():
        ie=lang.embed_tokens(ids); ple=lang.get_per_layer_inputs(ids, ie)
    return ie, am, ple, n

def greedy(fn, tag):
    cur=list(ids0)
    for step in range(L-n0):
        ie,am,ple,n=build(cur)
        with torch.no_grad(): lg=fn(ie,am,ple)
        nt=int(lg[0,n-1].argmax()); cur.append(nt)
        if nt in EOS: break
    txt=tok.decode(cur[n0:], skip_special_tokens=True)
    print(tag,"GEN:",repr(txt), flush=True)
    return cur

print("=== CPU greedy ===", flush=True)
t=time.time(); cpu_seq=greedy(w,"CPU"); print("cpu secs",round(time.time()-t,1), flush=True)

print("=== trace at L ===", flush=True)
import torch_neuronx
os.environ["NEURON_RT_VISIBLE_CORES"]="0"
ie,am,ple,_=build(list(ids0)); t=time.time()
neff=torch_neuronx.trace(w,(ie,am,ple),compiler_workdir="/workspace/optb_gen_wd",
    compiler_args=["--model-type","transformer","--auto-cast","none"])
torch.jit.save(neff,"/workspace/optb_gen_neff.pt")
print("TRACE_DONE secs",round(time.time()-t,1), flush=True)

print("=== DEVICE greedy ===", flush=True)
t=time.time(); dev_seq=greedy(neff,"DEVICE"); print("dev secs",round(time.time()-t,1), flush=True)

print("SEQ_MATCH", cpu_seq==dev_seq, flush=True)
print("CPU_IDS", cpu_seq[n0:], flush=True)
print("DEV_IDS", dev_seq[n0:], flush=True)
print("ALL_DONE", flush=True)
