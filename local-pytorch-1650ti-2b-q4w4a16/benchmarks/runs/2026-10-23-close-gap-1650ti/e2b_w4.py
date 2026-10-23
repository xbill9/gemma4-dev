"""Gemma 4 E2B QAT W4A16 on a 4 GiB Turing card: greedy decode with a captured CUDA graph per step.

    python3 e2b_w4.py text OUT.jsonl    # vocabulary tables as in -text: fp16 embed_tokens/lm_head (tied) on the GPU,
                                        # 4.38 GiB per-layer table in CPU RAM, looked up outside the graph
    python3 e2b_w4.py emb4 OUT.jsonl    # vocabulary tables from -text-emb4: embed_tokens, per-layer table and an
                                        # untied lm_head all int4 on the GPU, lookups inside the graph

Decoder linears are the -text repack's int4 in both modes. fp16 compute (Inductor skips bf16 on sm_75).
Prints and appends one JSON line per prompt.
"""

import json
import os
import pathlib
import sys
import time

import torch
import transformers
import transformers.cache_utils as cu
from accelerate.hooks import remove_hook_from_module
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, StaticCache

# KERNEL picks the int4 GEMV: v2 (lmhead-kernel run), v3 (2^23 bit-trick conversion), v4 (wide autotune), v5 (v4 pinned).
KERNEL = os.environ.get("KERNEL", "v5")
w4_gemv = __import__(f"w4gemv_{KERNEL}").w4_gemv
import w4attn

TEXT = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
EMB4 = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4"
MODE, OUT = sys.argv[1], pathlib.Path(sys.argv[2])
ATTN = os.environ.get("ATTN", "sdpa")  # sdpa, eager, or w4decode (Triton decode kernel, SDPA for prompts)
if ATTN == "w4decode":
    w4attn.register("w4decode")
# Decode-step attention backend under sdpa: "default" lets PyTorch choose (memory-efficient kernel on sm_75),
# "math" forces plain matmul + softmax, which suits one query token.
DECODE_SDPA = os.environ.get("DECODE_SDPA", "default")
DTYPE = torch.float16
MAX_CACHE = 384
NEW_TOKENS = 128
PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Write a Python function that reverses a singly linked list.",
    "A train leaves at 9:40 and arrives at 13:15. How long is the trip in minutes? Show the steps.",
    "Translate into French: The meeting has been moved to Thursday afternoon because the room is booked.",
    "List three facts about Jupiter.",
    "Write a haiku about a laptop fan.",
    'Extract the name and the city as JSON: "Maria Lopez moved from Lisbon to Toronto in 2019."',
    "What is the difference between a process and a thread? Answer briefly.",
]
SHIFTS = torch.arange(0, 32, 4, device="cuda", dtype=torch.int32)


def out(**kw):
    line = json.dumps(kw)
    print(line, flush=True)
    with OUT.open("a") as f:
        f.write(line + "\n")


def tensor(repo, key, patterns):
    d = pathlib.Path(snapshot_download(repo, allow_patterns=patterns))
    for p in sorted(d.glob("*.safetensors")):
        with safe_open(str(p), "pt") as f:
            if key in f.keys():
                return f.get_tensor(key)
    raise KeyError(key)


# --- int4 kernels: one-row decode reads packed weights directly; multi-row unpacks then multiplies ---------------
def _dequant(packed, scale):
    o, g = scale.shape
    q = ((packed.unsqueeze(-1) >> SHIFTS) & 15).view(o, g, 32).to(DTYPE) - 8
    return (q * scale.unsqueeze(-1).to(DTYPE)).view(o, g * 32)


# Compile only the weight unpack: its shapes are fixed per layer, so prompt length never triggers a recompile.
torch._dynamo.config.cache_size_limit = 64
_dequant = torch.compile(_dequant, dynamic=False)


def _deq_mm(x, packed, scale):
    return torch.nn.functional.linear(x, _dequant(packed, scale))


def w4a16(x, packed, scale):
    if x.numel() == x.shape[-1]:
        return w4_gemv(x, packed, scale)
    return _deq_mm(x, packed, scale)


class PackedLinear(torch.nn.Module):
    def __init__(self, packed, scale):
        super().__init__()
        self.packed, self.scale = torch.nn.Buffer(packed), torch.nn.Buffer(scale)

    def forward(self, x):
        return w4a16(x, self.packed, self.scale)


class PackedEmbedding(torch.nn.Module):
    """Gemma4TextScaledWordEmbedding over int4 rows: gather packed rows, unpack, multiply by the embed scale."""

    def __init__(self, packed, scale, embed_scale):
        super().__init__()
        self.packed, self.scale = torch.nn.Buffer(packed), torch.nn.Buffer(scale)
        self.embed_scale = torch.nn.Buffer(torch.tensor(embed_scale, dtype=DTYPE))

    def forward(self, ids):
        p, s = self.packed[ids], self.scale[ids]
        q = ((p.unsqueeze(-1) >> SHIFTS) & 15).to(DTYPE) - 8
        w = q.view(*ids.shape, s.shape[-1], 32) * s.unsqueeze(-1).to(DTYPE)
        return w.view(*ids.shape, -1) * self.embed_scale


# --- load ----------------------------------------------------------------------------------------------------------
with torch.device("meta"):
    meta = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(TEXT))
device_map = {f"model.{n}": 0 for n, _ in meta.model.named_children()}
device_map["model.embed_tokens_per_layer"] = "cpu"
device_map["lm_head"] = 0
del meta
model = AutoModelForCausalLM.from_pretrained(
    TEXT,
    dtype=DTYPE,
    key_mapping={r"^model\.language_model\.": "model."},
    device_map=device_map,
    quantization_config=transformers.CompressedTensorsConfig(run_compressed=True),
    attn_implementation=ATTN,
).eval()
model.ct_decompress_hook.remove()
def patch_linears():
    """Route every packed linear through the int4 kernels (re-run after anything that restores forwards)."""
    for m in model.modules():
        if hasattr(m, "weight_packed"):
            assert m.bias is None
            m.forward = (lambda mod: lambda x: w4a16(x, mod.weight_packed, mod.weight_scale))(m)


patch_linears()
tm = model.model
remove_hook_from_module(tm.embed_tokens_per_layer)

if MODE == "text":
    key = "model.language_model.embed_tokens_per_layer.weight"
    tm.embed_tokens_per_layer.weight = torch.nn.Parameter(
        tensor(TEXT, key, ["*.safetensors"]).to(DTYPE), requires_grad=False
    )
    ple_cpu = tm.embed_tokens_per_layer.forward
    tm.embed_tokens_per_layer.forward = lambda ids: ple_cpu(ids.cpu()).to(ids.device)
    model.to(DTYPE)
elif MODE == "emb4":
    pats = ["model-rest-00[01].safetensors"]
    pre = "model.language_model."

    def packed(name):
        return (tensor(EMB4, name + ".weight_packed", pats).cuda(), tensor(EMB4, name + ".weight_scale", pats).cuda())

    ple_scale, emb_scale = tm.embed_tokens_per_layer.scalar_embed_scale, tm.embed_tokens.scalar_embed_scale
    # The fp16 embed_tokens (tied lm_head) stays referenced after the modules are replaced; free its storage outright.
    tm.embed_tokens.weight.data = torch.empty(0, dtype=DTYPE, device=0)
    tm.embed_tokens_per_layer = PackedEmbedding(*packed(pre + "embed_tokens_per_layer"), ple_scale)
    tm.embed_tokens = PackedEmbedding(*packed(pre + "embed_tokens"), emb_scale)
    model.lm_head = PackedLinear(*packed("lm_head"))
    model.config.tie_word_embeddings = False
    # Everything is on the GPU now, so accelerate's device-placement hooks do nothing but break torch.compile graphs.
    remove_hook_from_module(model, recurse=True)
    patch_linears()
    model.to(DTYPE)
    torch.cuda.empty_cache()
else:
    raise SystemExit(f"unknown mode {MODE}")


def _merge(mods):
    """Concatenate packed int4 linears along the output dim and free the originals' storage."""
    packed = torch.cat([m.weight_packed for m in mods])
    scale = torch.cat([m.weight_scale for m in mods])
    sizes = [m.weight_packed.shape[0] for m in mods]
    for m in mods:
        m.weight_packed.data = torch.empty(0, dtype=torch.int32, device=0)
        m.weight_scale.data = torch.empty(0, dtype=DTYPE, device=0)
    return packed, scale, sizes


def merge_projections():
    """One int4 GEMV for gate+up in every MLP, and for q+k+v in every attention layer that computes its own K/V."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention, Gemma4TextMLP

    n_mlp = n_qkv = 0
    for mod in list(model.modules()):
        if isinstance(mod, Gemma4TextMLP):
            packed, scale, sizes = _merge([mod.gate_proj, mod.up_proj])

            def mlp_forward(x, mod=mod, packed=packed, scale=scale, sizes=sizes):
                g, u = w4a16(x, packed, scale).split(sizes, dim=-1)
                return mod.down_proj(mod.act_fn(g) * u)

            mod.forward = mlp_forward
            n_mlp += 1
        elif isinstance(mod, Gemma4TextAttention) and not mod.is_kv_shared_layer:
            projs = [mod.q_proj, mod.k_proj] + ([mod.v_proj] if mod.v_proj is not None else [])
            packed, scale, sizes = _merge(projs)
            stash = []  # q_proj runs first and leaves k (and v) for the calls that follow

            def q_forward(x, packed=packed, scale=scale, sizes=sizes, stash=stash):
                parts = w4a16(x, packed, scale).split(sizes, dim=-1)
                stash[:] = parts[1:]
                return parts[0]

            mod.q_proj.forward = q_forward
            mod.k_proj.forward = lambda x, stash=stash: stash[0]
            if mod.v_proj is not None:
                mod.v_proj.forward = lambda x, stash=stash: stash[1]
            n_qkv += 1
    torch.cuda.empty_cache()
    return n_mlp, n_qkv


MERGE = os.environ.get("MERGE") == "1"
merged_counts = merge_projections() if MERGE else (0, 0)

floats = sorted(
    {str(t.dtype) for t in list(model.parameters()) + list(model.buffers()) if t.is_floating_point() and t.is_cuda}
)
gpu_gib = (
    sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()) if t.is_cuda) / 2**30
)
out(
    step="load",
    mode=MODE,
    merged_mlp_qkv=list(merged_counts),
    kernel=KERNEL,
    attn=model.config._attn_implementation,
    gpu_tensors_gib=round(gpu_gib, 3),
    cuda_alloc_mib=torch.cuda.memory_allocated() // 2**20,
    float_dtypes_on_gpu=floats,
    torch=torch.__version__,
    transformers=transformers.__version__,
)

# --- CUDA graph for one decode step --------------------------------------------------------------------------------
# StaticSlidingWindowLayer keeps a Python-int length: a CUDA graph freezes it (the sliding mask's query position
# goes stale) and torch.compile guards on it (a recompile every step). While the cache is no longer than the
# sliding window it never rolls, so it behaves exactly like StaticLayer, whose length is an on-device counter.
assert model.config.sliding_window >= MAX_CACHE
for _name in ("update", "get_seq_length", "get_mask_sizes"):
    setattr(cu.StaticSlidingWindowLayer, _name, getattr(cu.StaticLayer, _name))
cache = StaticCache(config=model.config, max_cache_len=MAX_CACHE)
eos = model.generation_config.eos_token_id
eos = set(eos if isinstance(eos, list) else [eos])
tok = AutoTokenizer.from_pretrained(TEXT)


def prefill(ids):
    cache.reset()
    n = ids.shape[1]
    return model(
        input_ids=ids, cache_position=torch.arange(n, device=0), past_key_values=cache, use_cache=True, logits_to_keep=1
    ).logits[:, -1]


from contextlib import nullcontext

from torch.nn.attention import SDPBackend, sdpa_kernel


def decode_attention():
    return sdpa_kernel(SDPBackend.MATH) if DECODE_SDPA == "math" else nullcontext()


tok_buf = torch.zeros(1, 1, dtype=torch.long, device=0)
pos_buf = torch.zeros(1, dtype=torch.long, device=0)
emb_buf = ple_buf = None
if MODE == "text":
    with torch.inference_mode():
        emb_buf = tm.embed_tokens(tok_buf).clone()
        ple_buf = tm.get_per_layer_inputs(tok_buf, None).clone()

    def stage(t, pos):
        tt = torch.tensor([[t]], device=0)
        emb_buf.copy_(tm.embed_tokens(tt))
        ple_buf.copy_(tm.get_per_layer_inputs(tt, None))
        pos_buf.fill_(pos)

    def step():
        with decode_attention():
            return (
                model(
                    inputs_embeds=emb_buf,
                    per_layer_inputs=ple_buf,
                    position_ids=pos_buf.view(1, 1),
                    cache_position=pos_buf,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                .logits[:, -1]
                .argmax(-1)
            )
else:

    def stage(t, pos):
        tok_buf.fill_(t)
        pos_buf.fill_(pos)

    def step():
        with decode_attention():
            return (
                model(
                    input_ids=tok_buf,
                    position_ids=pos_buf.view(1, 1),
                    cache_position=pos_buf,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                .logits[:, -1]
                .argmax(-1)
            )


# STEP_COMPILE=1: compile the decode step with Inductor (fuses norms and elementwise ops), then capture the
# compiled step in the same hand-made CUDA graph. Inductor's own cudagraphs stay off.
STEP_COMPILE = os.environ.get("STEP_COMPILE") == "1"
if STEP_COMPILE:
    step = torch.compile(step, mode="default", dynamic=False)

with torch.inference_mode():
    warm = tok.apply_chat_template(
        [{"role": "user", "content": "Hi"}], add_generation_prompt=True, return_tensors="pt", return_dict=True
    )["input_ids"].to(0)
    first = int(prefill(warm).argmax(-1))
    stage(first, warm.shape[1])
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):  # Triton autotuning and cuBLAS workspaces happen here, not inside the capture
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.empty_cache()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        next_buf = step()
out(
    step="captured",
    mode=MODE,
    cuda_alloc_mib=torch.cuda.memory_allocated() // 2**20,
    cuda_reserved_mib=torch.cuda.memory_reserved() // 2**20,
)

# --- greedy decode over the prompt set -----------------------------------------------------------------------------
with torch.inference_mode():
    for i, text in enumerate(PROMPTS):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, return_tensors="pt", return_dict=True
        )["input_ids"].to(0)
        n = ids.shape[1]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        toks = [int(prefill(ids).argmax(-1))]
        t1 = time.perf_counter()
        while len(toks) < NEW_TOKENS and toks[-1] not in eos:
            stage(toks[-1], n + len(toks) - 1)
            graph.replay()
            toks.append(int(next_buf))
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        out(
            step="prompt",
            mode=MODE,
            attn=ATTN,
            decode_sdpa=DECODE_SDPA,
            step_compile=STEP_COMPILE,
            prompt=i,
            prompt_tokens=n,
            new_tokens=len(toks),
            prefill_ms=round((t1 - t0) * 1e3, 1),
            decode_tok_per_s=round((len(toks) - 1) / (t2 - t1), 2),
            peak_cuda_mib=torch.cuda.max_memory_allocated() // 2**20,
            token_ids=toks,
            text=tok.decode(toks, skip_special_tokens=True),
        )

# --- optional: where one decode step's GPU time goes (plain step, not the graph) ------------------------------------
if len(sys.argv) > 3 and sys.argv[3] == "profile":
    from torch.profiler import ProfilerActivity, profile

    n_steps = 16
    with torch.inference_mode():
        ids = tok.apply_chat_template(
            [{"role": "user", "content": PROMPTS[0]}], add_generation_prompt=True, return_tensors="pt", return_dict=True
        )["input_ids"].to(0)
        t = int(prefill(ids).argmax(-1))
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for k in range(n_steps):
                stage(t, ids.shape[1] + k)
                t = int(step())
            torch.cuda.synchronize()
        graph_ms = []
        for k in range(20):
            stage(t, ids.shape[1] + n_steps + k)
            torch.cuda.synchronize()
            s = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            graph_ms.append((time.perf_counter() - s) * 1e3)
    kern = [e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
    total = sum(e.self_device_time_total for e in kern)
    buckets = {}
    for e in kern:
        k = e.key
        b = (
            "int4 gemv"
            if "_w4_gemv" in k
            else "gemv/gemm (cuBLAS)"
            if ("gemv" in k or "gemm" in k)
            else "attention"
            if ("attention" in k.lower() or "flash" in k.lower() or "softmax" in k.lower())
            else "elementwise/reduce/other"
        )
        buckets.setdefault(b, [0.0, 0])
        buckets[b][0] += e.self_device_time_total / 1e3 / n_steps
        buckets[b][1] += e.count // n_steps
    out(
        step="profile",
        mode=MODE,
        attn=ATTN,
        decode_sdpa=DECODE_SDPA,
        kernels_per_token=sum(e.count for e in kern) // n_steps,
        gpu_kernel_ms_per_token=round(total / 1e3 / n_steps, 2),
        graph_replay_ms=round(sorted(graph_ms)[10], 2),
        buckets={k: [round(v[0], 2), v[1]] for k, v in sorted(buckets.items(), key=lambda kv: -kv[1][0])},
        top=[
            (e.key[:70], round(e.self_device_time_total / 1e3 / n_steps, 3), e.count // n_steps)
            for e in sorted(kern, key=lambda e: -e.self_device_time_total)[:12]
        ],
    )
