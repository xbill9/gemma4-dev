
import os, torch, transformers
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

msgs=[{"role":"user","content":"What is the capital of France?"}]
enc=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
ids=enc["input_ids"]; am=enc["attention_mask"]
with torch.no_grad():
    ie=lang.embed_tokens(ids)
    ple=lang.get_per_layer_inputs(ids, ie)
print("inputs_embeds",tuple(ie.shape),"per_layer_inputs",tuple(ple.shape), flush=True)
with torch.no_grad(): cpu=w(ie,am,ple)
tid=int(cpu[0,-1].argmax()); print("CPU argmax",tid,repr(tok.decode([tid])), flush=True)

print("=== trace (host-embeddings) ===", flush=True)
import torch_neuronx
os.environ["NEURON_RT_VISIBLE_CORES"]="0"
neff=torch_neuronx.trace(w,(ie,am,ple),compiler_workdir="/workspace/optb_wd3",
        compiler_args=["--model-type","transformer","--auto-cast","none"])
torch.jit.save(neff,"/workspace/optb_neff.pt"); print("TRACE_DONE", flush=True)
with torch.no_grad(): dev=neff(ie,am,ple)
dtid=int(dev[0,-1].argmax()); print("DEVICE argmax",dtid,repr(tok.decode([dtid])), flush=True)
print("MATCH",dtid==tid,"| maxabs diff",float((dev[0,-1]-cpu[0,-1]).abs().max()), flush=True)
print("ALL_DONE", flush=True)
