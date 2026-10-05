"""Single-core Option B decode with KV I/O ALIASING: KV buffers become device-resident
state (input_output_aliases), so they are NOT transferred host<->device each token.
Target: 2048 decode 25 -> ~40+ tok/s. Validates SEQ_MATCH + measures tok/s."""
import os, sys, time, torch
torch.manual_seed(0)
MODE = sys.argv[1] if len(sys.argv) > 1 else "trace"
MP = "/workspace/real-gemma4-E2B-it"
MAX = int(os.environ.get("KV_MAX", "2048"))
BUCKET = int(os.environ.get("KV_BUCKET", "512"))
MAXNEW = int(os.environ.get("MAXNEW", "30"))
NEG = torch.finfo(torch.float32).min

from transformers import AutoTokenizer, Gemma4ForConditionalGeneration, DynamicCache
tok = AutoTokenizer.from_pretrained(MP)
m = Gemma4ForConditionalGeneration.from_pretrained(MP, torch_dtype=torch.float32, attn_implementation="eager"); m.eval()
lang = m.model.language_model; lm_head = m.lm_head
softcap = getattr(m.config.text_config, "final_logit_softcapping", None)
cfg = lang.config; SW = cfg.sliding_window
class GeluTanh(torch.nn.Module):
    def forward(s, x): return 0.5*x*(1.0+torch.tanh(0.7978845608028654*(x+0.044715*x*x*x)))
for mod in lang.modules():
    if hasattr(mod, "act_fn"): mod.act_fn = GeluTanh()
NONSHARED, LINFO = [], {}
for i, lyr in enumerate(lang.layers[:cfg.num_hidden_layers]):
    a = lyr.self_attn
    if not a.is_kv_shared_layer:
        hd = a.head_dim; NONSHARED.append(i); LINFO[i] = (a.k_proj.out_features // hd, hd)
NK = len(NONSHARED)
def sc(lg): return softcap*torch.tanh(lg/softcap) if softcap else lg

class PreWrap(torch.nn.Module):
    def __init__(s): super().__init__(); s.lang=lang; s.head=lm_head
    def forward(s, ie, am, ple):
        cache=DynamicCache()
        out=s.lang(inputs_embeds=ie, per_layer_inputs=ple, attention_mask=am, use_cache=True, past_key_values=cache)
        lg=sc(s.head(out.last_hidden_state))
        return (lg,[cache.layers[i].keys for i in NONSHARED],[cache.layers[i].values for i in NONSHARED])
class StaticKV:
    is_compileable=False
    def __init__(s,kb,vb,oh): s.key={i:kb[j] for j,i in enumerate(NONSHARED)}; s.val={i:vb[j] for j,i in enumerate(NONSHARED)}; s.oh=oh
    def update(s,k,v,idx,*a,**kw): s.key[idx]=s.key[idx]*(1.0-s.oh)+k*s.oh; s.val[idx]=s.val[idx]*(1.0-s.oh)+v*s.oh; return s.key[idx],s.val[idx]
    def get_seq_length(s,*a,**k): return 0
    def export(s): return [s.key[i] for i in NONSHARED],[s.val[i] for i in NONSHARED]
class DecWrap(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.lang=lang; s.head=lm_head
        s.kbuf=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1,LINFO[i][0],MAX,LINFO[i][1]),requires_grad=False) for i in NONSHARED])
        s.vbuf=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1,LINFO[i][0],MAX,LINFO[i][1]),requires_grad=False) for i in NONSHARED])
    def forward(s, ie, ple, position_ids, onehot, full_mask, slide_mask):
        cache=StaticKV(list(s.kbuf),list(s.vbuf),onehot)
        out=s.lang(inputs_embeds=ie, per_layer_inputs=ple, position_ids=position_ids, attention_mask={"full_attention":full_mask,"sliding_attention":slide_mask}, use_cache=True, past_key_values=cache)
        lg=sc(s.head(out.last_hidden_state)); ks,vs=cache.export(); return (lg,ks,vs)

ec=m.generation_config.eos_token_id; EOS=set(ec) if isinstance(ec,(list,tuple)) else {ec}
def embed_ids(idl):
    ids=torch.tensor([idl])
    with torch.no_grad(): ie=lang.embed_tokens(ids); ple=lang.get_per_layer_inputs(ids, ie)
    return ie,ple
def host_pos(pos):
    ar=torch.arange(MAX); onehot=(ar==pos).view(1,1,MAX,1).to(torch.float32); valid=ar<=pos
    full=torch.where(valid,0.0,NEG).view(1,1,1,MAX); slide=torch.where(valid&(ar>pos-SW),0.0,NEG).view(1,1,1,MAX)
    return torch.tensor([[pos]],dtype=torch.long), onehot, full, slide

enc=tok.apply_chat_template([{"role":"user","content":"What is the capital of France?"}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
prompt=enc["input_ids"][0].tolist(); n0=len(prompt); assert n0<=BUCKET
print("prompt",n0,"MAX",MAX,"NONSHARED",NK,flush=True)

import torch_neuronx
os.environ["NEURON_RT_VISIBLE_CORES"]="0,1"
CARGS=["--model-type","transformer","--auto-cast","all","--auto-cast-type","bf16"]
pre=PreWrap().eval(); dec=DecWrap().eval()

# trace prefill
pad=prompt+[0]*(BUCKET-n0); ie,ple=embed_ids(pad); am=torch.tensor([[1]*n0+[0]*(BUCKET-n0)])
t=time.time(); pre_n=torch_neuronx.trace(pre,(ie,am,ple),compiler_args=CARGS); print("PREFILL_TRACED",round(time.time()-t,1),flush=True)

# trace decode WITH aliasing: KV state params -> KV outputs (device-resident state)
ie1,ple1=embed_ids([prompt[-1]]); pid,oh,fm,sm=host_pos(n0)
aliases={}
for j in range(NK): aliases[dec.kbuf[j]]=1+j          # output 0=lg, 1..NK=ks
for j in range(NK): aliases[dec.vbuf[j]]=1+NK+j        # NK+1..2NK=vs
t=time.time(); dec_n=torch_neuronx.trace(dec,(ie1,ple1,pid,oh,fm,sm),input_output_aliases=aliases,compiler_args=CARGS)
print("DECODE_TRACED(aliased)",round(time.time()-t,1),flush=True)
# MOVE aliased state (+weights) to the neuron device so KV persists in-place across calls
from torch_neuronx.xla_impl.trace import move_trace_to_device
DEC_CORE=int(os.environ.get("DEC_CORE","1"))
move_trace_to_device(dec_n, DEC_CORE)
print("MOVED_TO_DEVICE core",DEC_CORE,flush=True)
# collect device-resident state params (named 'states.N'), numeric order
_sp = [(int(''.join(ch for ch in n if ch.isdigit())), p) for n,p in dec_n.states._parameters.items()]
STATE_PARAMS = [p for _,p in sorted(_sp, key=lambda x:x[0])]
print("state params found:", len(STATE_PARAMS), "(expect", 2*NK, ")", flush=True)

def run_aliased(maxnew):
    pad=prompt+[0]*(BUCKET-n0); ie,ple=embed_ids(pad); am=torch.tensor([[1]*n0+[0]*(BUCKET-n0)])
    lg,ks,vs=pre_n(ie,am,ple); first=int(lg[0,n0-1].argmax())
    # seed device-resident KV state with prompt K/V at [:n0]
    for j in range(NK):
        sk=torch.zeros(1,LINFO[NONSHARED[j]][0],MAX,LINFO[NONSHARED[j]][1]); sk[:,:,:n0,:]=ks[j][:,:,:n0,:]
        sv=torch.zeros(1,LINFO[NONSHARED[j]][0],MAX,LINFO[NONSHARED[j]][1]); sv[:,:,:n0,:]=vs[j][:,:,:n0,:]
        STATE_PARAMS[j].data.copy_(sk); STATE_PARAMS[NK+j].data.copy_(sv)
    seq=[first]; cur=n0; th=td=0.0; steps=0
    for _ in range(maxnew):
        if seq[-1] in EOS: break
        a0=time.time(); e1,p1=embed_ids([seq[-1]]); pi,o,f,s=host_pos(cur); b0=time.time()
        l1=dec_n(e1,p1,pi,o,f,s); lg1=l1[0] if isinstance(l1,(tuple,list)) else l1
        c0=time.time(); seq.append(int(lg1[0,0].argmax())); cur+=1; steps+=1; th+=b0-a0; td+=c0-b0
    return seq, td, steps, th

print("=== DEVICE aliased warmup ===",flush=True)
_=run_aliased(5)
print("=== DEVICE aliased timed ===",flush=True)
dev_seq, secs, steps, hprep = run_aliased(MAXNEW)
txt=tok.decode([x for x in dev_seq if x not in EOS],skip_special_tokens=True)
print("DEV GEN:",repr(txt),flush=True)
print("CORRECT(Paris):", "Paris" in txt, flush=True)
print("HOST_PREP ms/tok:", round(hprep/max(steps,1)*1000,2), flush=True)
print("ALIASED DECODE tok/s:",round(steps/secs,1),"|",round(secs/max(steps,1)*1000),"ms/tok |",steps,"steps",flush=True)
print("ALIAS_OK",flush=True)
