"""Smoke test: Gemma 4 E2B QAT W4A16 repack on a 4 GiB CUDA card with plain PyTorch + transformers.

Packed int4 decoder layers on the GPU, vocabulary tables (embed_tokens, per-layer embeddings) in CPU
RAM. Prints one JSON line per step.
"""

import collections
import json
import sys
import time

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
KEY_MAPPING = {r"^model\.language_model\.": "model."}
PROMPT = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
NEW_TOKENS = 64
CPU_PARTS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["model.embed_tokens", "model.embed_tokens_per_layer"]


def out(**kw):
    print(json.dumps(kw), flush=True)


out(
    step="versions",
    torch=torch.__version__,
    transformers=transformers.__version__,
    gpu=torch.cuda.get_device_name(0),
    total_mib=torch.cuda.get_device_properties(0).total_memory // 2**20,
)

# Build a device map from the model's own top-level modules.
with torch.device("meta"):
    meta = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(MODEL))
names = [n for n, _ in meta.model.named_children()]
out(step="modules", top=names)
device_map = {f"model.{n}": ("cpu" if f"model.{n}" in CPU_PARTS else 0) for n in names}
device_map["lm_head"] = "cpu" if "lm_head" in CPU_PARTS else 0
del meta

start = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    key_mapping=KEY_MAPPING,
    device_map=device_map,
    quantization_config=transformers.CompressedTensorsConfig(run_compressed=True),
).eval()
torch.cuda.synchronize()
load_s = time.perf_counter() - start

count, gib = collections.Counter(), collections.Counter()
dtypes = collections.Counter()
for _, t in list(model.named_parameters()) + list(model.named_buffers()):
    count[t.device.type] += 1
    gib[t.device.type] += t.numel() * t.element_size() / 2**30
    dtypes[f"{t.device.type}:{t.dtype}"] += 1
lin = collections.Counter(type(m).__name__ for m in model.modules() if "Linear" in type(m).__name__)
out(
    step="load",
    seconds=round(load_s, 1),
    tensors=dict(count),
    gib={k: round(v, 3) for k, v in gib.items()},
    dtypes=dict(dtypes),
    linear_types=dict(lin),
    cuda_alloc_mib=torch.cuda.memory_allocated() // 2**20,
    tied=model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(),
    lm_head_device=str(model.lm_head.weight.device),
)

# transformers forces a full unpack to bf16 at the first forward (3.58 GiB of layers, too big for this card).
# Drop that hook and instead unpack each linear's int4 weight inside its own forward, then discard it.
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from compressed_tensors.quantization.lifecycle.forward import dequantize

model.ct_decompress_hook.remove()


def packed_forward(mod):
    def forward(x):
        w = dequantize(x_q=unpack_from_int32(mod.weight_packed, 4, mod.weight_shape), scale=mod.weight_scale)
        return torch.nn.functional.linear(x, w.to(x.dtype), mod.bias)

    return forward


patched = 0
for m in model.modules():
    if hasattr(m, "weight_packed"):
        m.forward = packed_forward(m)
        patched += 1
# Accelerate's CPU offload keeps the per-layer table on meta and copies all 4.38 GiB to the GPU every forward.
# Load it into CPU RAM ourselves, look rows up on the CPU and send only those rows to the GPU.
import json as _json
import pathlib

from accelerate.hooks import remove_hook_from_module
from huggingface_hub import snapshot_download
from safetensors import safe_open

ple_mod = model.model.embed_tokens_per_layer
remove_hook_from_module(ple_mod)
snap = pathlib.Path(snapshot_download(MODEL))
key = "model.language_model.embed_tokens_per_layer.weight"
shard = _json.load(open(snap / "model.safetensors.index.json"))["weight_map"][key]
with safe_open(str(snap / shard), "pt", device="cpu") as f:
    ple_mod.weight = torch.nn.Parameter(f.get_tensor(key).to(torch.bfloat16), requires_grad=False)
ple_cpu_forward = ple_mod.forward
ple_mod.forward = lambda ids: ple_cpu_forward(ids.cpu()).to(ids.device, non_blocking=True)
ple = ple_mod.weight
out(step="patch", patched=patched, ple_device=str(ple.device), ple_gib=round(ple.numel() * 2 / 2**30, 3),
    scale_dtype=str(next(m.weight_scale.dtype for m in model.modules() if hasattr(m, "weight_scale"))))

tok = AutoTokenizer.from_pretrained(MODEL)
ids = tok.apply_chat_template(PROMPT, add_generation_prompt=True, return_tensors="pt", return_dict=True)
ids = {k: v.to(0) for k, v in ids.items()}
for run in ("cold", "warm"):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        gen = model.generate(**ids, max_new_tokens=NEW_TOKENS, do_sample=False)
    torch.cuda.synchronize()
    s = time.perf_counter() - start
    new = gen[0, ids["input_ids"].shape[1] :]
    out(
        step=f"generate_{run}",
        prompt_tokens=ids["input_ids"].shape[1],
        new_tokens=len(new),
        seconds=round(s, 2),
        tok_per_s=round(len(new) / s, 2),
        peak_cuda_mib=torch.cuda.max_memory_allocated() // 2**20,
        token_ids=new.tolist()[:16],
        text=tok.decode(new, skip_special_tokens=True),
    )
