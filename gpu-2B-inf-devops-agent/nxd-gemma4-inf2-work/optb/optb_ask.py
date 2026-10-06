import os, sys, torch, time
torch.manual_seed(0)
MP="/workspace/real-gemma4-E2B-it"
PROMPT=sys.argv[1] if len(sys.argv)>1 else "What is Gemma?"
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
import torch_neuronx

tok=AutoTokenizer.from_pretrained(MP)
m=Gemma4ForConditionalGeneration.from_pretrained(MP, torch_dtype=torch.float32, attn_implementation="eager"); m.eval()
lang=m.model.language_model
L=40; PAD=0
ec=m.generation_config.eos_token_id
EOS=set(ec) if isinstance(ec,(list,tuple)) else {ec}

os.environ["NEURON_RT_VISIBLE_CORES"]="0"
neff=torch.jit.load("/workspace/optb_gen_neff.pt")   # reuse compiled device graph, no recompile

msgs=[{"role":"user","content":PROMPT}]
enc=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
ids0=enc["input_ids"][0].tolist(); n0=len(ids0)
print("PROMPT:",repr(PROMPT),"| prompt tokens",n0,"| L",L, flush=True)
assert n0 < L, "prompt too long for L=40 graph"

def build(cur):
    n=len(cur); padded=cur+[PAD]*(L-n)
    ids=torch.tensor([padded]); am=torch.tensor([[1]*n+[0]*(L-n)])
    with torch.no_grad():
        ie=lang.embed_tokens(ids); ple=lang.get_per_layer_inputs(ids, ie)
    return ie, am, ple, n

cur=list(ids0); t=time.time()
for step in range(L-n0):
    ie,am,ple,n=build(cur)
    with torch.no_grad(): lg=neff(ie,am,ple)
    nt=int(lg[0,n-1].argmax()); cur.append(nt)
    if nt in EOS: break
print("DEVICE ANSWER:",repr(tok.decode(cur[n0:], skip_special_tokens=True)), flush=True)
print("secs",round(time.time()-t,1),"| tokens",len(cur)-n0, flush=True)
print("ALL_DONE", flush=True)
