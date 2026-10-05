"""Minimal NxDI compile+run wrapper (replaces the missing autoport skill scripts).
Run in the neuron container after reconstruct_kvshare_stock.py + building /workspace/real4.
MODE=cpu -> to_cpu() eager (validate "Paris"). MODE=device -> compile()+load() on /dev/neuron0 (tp=2).
Needs /workspace/real4_cfg.py (RealCfg subclass of Gemma3InferenceConfig w/ from_pretrained).
"""
import torch, sys, os
sys.path.insert(0,"/workspace")
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.gemma3.modeling_gemma3 import NeuronGemma3ForCausalLM
from real4_cfg import RealCfg
HFA=None
for mod in ["neuronx_distributed_inference.utils.hf_adapter","neuronx_distributed_inference.utils.generation"]:
    try:
        m=__import__(mod, fromlist=["HuggingFaceGenerationAdapter"]); HFA=getattr(m,"HuggingFaceGenerationAdapter"); break
    except Exception: pass
MP="/workspace/real4"; CP="/workspace/real4_neuron"
MODE=os.environ.get("MODE","cpu"); CPU=(MODE=="cpu")
nc=NeuronConfig(tp_degree=(1 if CPU else 2), batch_size=1, seq_len=128, torch_dtype=torch.float32, on_cpu=CPU)
cfg=RealCfg.from_pretrained(MP, neuron_config=nc)
model=NeuronGemma3ForCausalLM(MP, cfg)
if CPU: model.to_cpu()
else: model.compile(CP); model.load(CP)
tok=AutoTokenizer.from_pretrained(MP); gc=GenerationConfig.from_pretrained(MP)
ids=tok("The capital of France is", return_tensors="pt").input_ids
out=HFA(model).generate(ids, generation_config=gc, max_new_tokens=8, do_sample=False)
r=tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
open("/workspace/recon_%s.txt"%MODE,"w").write("%s: %r"%(MODE,r)); print("RESULT:", repr(r))
