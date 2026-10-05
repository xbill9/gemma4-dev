
import os, sys
SKILL="/root/.claude/skills/neuron-framework-autoport"
sys.path.insert(0, os.path.join(SKILL,"scripts")); sys.path.insert(0,"/workspace")
os.environ.setdefault("BASE_COMPILE_WORK_DIR","/workspace/real4_neff")
import torch
from model_compiler import DirectModelCompiler, CompilationConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.gemma3.modeling_gemma3 import NeuronGemma3ForCausalLM
from real4_cfg import RealCfg
cfg=CompilationConfig(
    model_class=NeuronGemma3ForCausalLM, config_class=RealCfg, neuron_config_class=NeuronConfig,
    model_path="/workspace/real4", output_path="/workspace/real4_neuron",
    batch_size=1, seq_len=128, tp_degree=2, use_fp16=False)
print("=== Compiling native NeuronGemma3ForCausalLM (real weights, fp32, tp=2, seq=128) ===", flush=True)
ok=DirectModelCompiler(cfg).compile()
print("COMPILE_RESULT:", "SUCCESS" if ok else "FAILURE", flush=True)
open("/workspace/real4_compile_done.txt","w").write("SUCCESS" if ok else "FAILURE")
sys.exit(0 if ok else 1)
