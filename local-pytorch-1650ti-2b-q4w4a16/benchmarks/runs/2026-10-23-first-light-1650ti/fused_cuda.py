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
CPU_PARTS = ["model.embed_tokens_per_layer"]
# DTYPE: bf16 or fp16. KERNEL: fused (one dequant + matmul function, torch.compile'd) or eager (same function, uncompiled).
DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}[sys.argv[1]]
KERNEL = sys.argv[2]
TAG = f"{sys.argv[1]}-{KERNEL}"


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
    dtype=DTYPE,
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
# The dequant below is bit-exact against compressed-tensors' unpack_from_int32 + dequantize (checked in micro2.py):
# symmetric int4, 8 nibbles per int32, low nibble first, offset 8, one scale per group of 32.
model.ct_decompress_hook.remove()
SHIFTS = torch.arange(0, 32, 4, device="cuda", dtype=torch.int32)


def w4a16(x, packed, scale):
    o, g = scale.shape
    q = ((packed.unsqueeze(-1) >> SHIFTS) & 15).view(o, g, 32).to(x.dtype) - 8
    return torch.nn.functional.linear(x, (q * scale.unsqueeze(-1).to(x.dtype)).view(o, g * 32))


if KERNEL in ("fused", "triton"):
    torch._dynamo.config.cache_size_limit = 64
    w4a16 = torch.compile(w4a16, dynamic=False)
if KERNEL == "triton":
    # Decode (one row): the Triton kernel that reads packed int4 directly. Prefill (many rows): dequant + matmul.
    from w4gemv import w4_gemv

    _deq_mm = w4a16

    def w4a16(x, packed, scale):
        if x.numel() == x.shape[-1]:
            return w4_gemv(x, packed, scale)
        return _deq_mm(x, packed, scale)


def packed_forward(mod):
    assert mod.bias is None

    def forward(x):
        return w4a16(x, mod.weight_packed, mod.weight_scale)

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
    ple_mod.weight = torch.nn.Parameter(f.get_tensor(key).to(DTYPE), requires_grad=False)
ple_cpu_forward = ple_mod.forward
ple_mod.forward = lambda ids: ple_cpu_forward(ids.cpu()).to(ids.device, non_blocking=True)
ple = ple_mod.weight
# The compressed-tensors loader keeps float tensors in bf16 whatever dtype= says; cast every float tensor.
model.to(DTYPE)
ple = ple_mod.weight
out(step="cast", float_dtypes=sorted({str(t.dtype) for t in list(model.parameters()) + list(model.buffers()) if t.is_floating_point()}))
out(step="patch", patched=patched, ple_device=str(ple.device), ple_gib=round(ple.numel() * 2 / 2**30, 3),
    scale_dtype=str(next(m.weight_scale.dtype for m in model.modules() if hasattr(m, "weight_scale"))))

# Overflow watch: largest |activation| and any non-finite value leaving each decoder layer.
act = {"max_abs": 0.0, "nonfinite_layers": set()}


def watch(i):
    def hook(_m, _a, o):
        h = (o[0] if isinstance(o, tuple) else o).detach()
        act["max_abs"] = max(act["max_abs"], float(h.abs().max()))
        if not torch.isfinite(h).all():
            act["nonfinite_layers"].add(i)

    return hook


tok = AutoTokenizer.from_pretrained(MODEL)
ids = tok.apply_chat_template(PROMPT, add_generation_prompt=True, return_tensors="pt", return_dict=True)
ids = {k: v.to(0) for k, v in ids.items()}
handles = [layer.register_forward_hook(watch(i)) for i, layer in enumerate(model.model.layers)]
with torch.inference_mode():
    model.generate(**ids, max_new_tokens=NEW_TOKENS, do_sample=False)
for h in handles:
    h.remove()
out(step="overflow_watch", fp16_max=65504, max_abs=act["max_abs"], nonfinite_layers=sorted(act["nonfinite_layers"]))
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
        token_ids=new.tolist(),
        text=tok.decode(new, skip_special_tokens=True),
    )

if len(sys.argv) > 3 and sys.argv[3] == "profile":
    from torch.profiler import ProfilerActivity, profile

    n = 16
    with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        torch.cuda.synchronize()
        start = time.perf_counter()
        model.generate(**ids, max_new_tokens=n, min_new_tokens=n, do_sample=False)
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
    ev = prof.key_averages()
    gpu_us = sum(e.self_device_time_total for e in ev)
    kernels = sum(e.count for e in ev if e.self_device_time_total > 0)
    top_gpu = sorted(ev, key=lambda e: -e.self_device_time_total)[:8]
    top_cpu = sorted(ev, key=lambda e: -e.self_cpu_time_total)[:10]
    out(step="profile", tokens=n, wall_s=round(wall, 2), gpu_busy_s=round(gpu_us / 1e6, 3),
        gpu_busy_pct=round(100 * gpu_us / 1e6 / wall, 1), gpu_launches=kernels, launches_per_token=kernels // n,
        top_gpu=[(e.key[:60], round(e.self_device_time_total / 1e3, 1), e.count) for e in top_gpu],
        top_cpu=[(e.key[:60], round(e.self_cpu_time_total / 1e3, 1), e.count) for e in top_cpu])

if len(sys.argv) > 3 and sys.argv[3] in ("cudagraph", "graphdiff"):
    # StaticSlidingWindowLayer reports its length as a Python int, which the sliding-window mask reads as the query
    # position; a CUDA graph freezes that int at capture. Report the on-device counter instead. Same value while the
    # cache (max_cache_len=256) is shorter than the 512-token window, so these layers never roll.
    import transformers.cache_utils as cu

    assert model.config.sliding_window >= 256
    cu.StaticSlidingWindowLayer.get_seq_length = lambda self: self.cumulative_length if self.is_initialized else 0

if len(sys.argv) > 3 and sys.argv[3] == "cudagraph":
    # Same greedy loop, but the decode step is captured once with torch.cuda.CUDAGraph (no torch.compile):
    # one KV cache reused across runs, fixed input buffers, one replay per token.
    from transformers import StaticCache

    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])
    text_model = model.model
    cache = StaticCache(config=model.config, max_cache_len=256)
    prompt = ids["input_ids"]
    n = prompt.shape[1]

    def step(embeds, ple, pos):
        return model(inputs_embeds=embeds, per_layer_inputs=ple, position_ids=pos.view(1, 1), cache_position=pos,
                     past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1]

    def prefill():
        cache.reset()
        return model(input_ids=prompt, cache_position=torch.arange(n, device=0), past_key_values=cache,
                     use_cache=True, logits_to_keep=1).logits[:, -1]

    def inputs(tok, pos):
        t = torch.tensor([[tok]], device=0)
        return text_model.embed_tokens(t), text_model.get_per_layer_inputs(t, None), torch.tensor([pos], device=0)

    with torch.inference_mode():
        first = int(prefill().argmax(-1))
        emb_buf, ple_buf, pos_buf = inputs(first, n)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):  # warm-up: Triton autotuning and cuBLAS workspaces happen here, not in the capture
                step(emb_buf, ple_buf, pos_buf)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.empty_cache()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits_buf = step(emb_buf, ple_buf, pos_buf)
        out(step="captured", cuda_alloc_mib=torch.cuda.memory_allocated() // 2**20,
            cuda_reserved_mib=torch.cuda.memory_reserved() // 2**20)

        def greedy(n_new):
            toks = [int(prefill().argmax(-1))]
            while len(toks) < n_new and toks[-1] not in eos:
                e, p, q = inputs(toks[-1], n + len(toks) - 1)
                emb_buf.copy_(e), ple_buf.copy_(p), pos_buf.copy_(q)
                graph.replay()
                toks.append(int(logits_buf.argmax(-1)))
            return toks

        ref = json.loads(open("run-bf16-eager.jsonl").read().splitlines()[-1])["token_ids"]
        for label in ("cudagraph", "cudagraph_again"):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            toks = greedy(NEW_TOKENS)
            torch.cuda.synchronize()
            s = time.perf_counter() - start
            out(step=label, new_tokens=len(toks), seconds=round(s, 2), tok_per_s=round(len(toks) / s, 2),
                peak_cuda_mib=torch.cuda.max_memory_allocated() // 2**20, identical_to_bf16=toks == ref,
                first_diff=next((i for i, (a, b) in enumerate(zip(toks, ref)) if a != b), None))

if len(sys.argv) > 3 and sys.argv[3] == "maskprobe":
    import transformers.models.gemma4.modeling_gemma4 as g4
    from transformers import StaticCache

    seen = []
    for fname in ("create_causal_mask", "create_sliding_window_causal_mask"):
        orig = getattr(g4, fname)

        def wrap(*a, _o=orig, _n=fname, **k):
            m = _o(*a, **k)
            seen.append((_n, None if m is None else (type(m).__name__, list(getattr(m, "shape", [])), str(getattr(m, "dtype", "")))))
            return m

        setattr(g4, fname, wrap)
    cache = StaticCache(config=model.config, max_cache_len=256)
    prompt = ids["input_ids"]
    n = prompt.shape[1]
    with torch.inference_mode():
        first = int(model(input_ids=prompt, cache_position=torch.arange(n, device=0), past_key_values=cache, use_cache=True,
                          logits_to_keep=1).logits[:, -1].argmax(-1))
        seen.clear()
        t = torch.tensor([[first]], device=0)
        model(inputs_embeds=model.model.embed_tokens(t), per_layer_inputs=model.model.get_per_layer_inputs(t, None),
              position_ids=torch.tensor([[n]], device=0), cache_position=torch.tensor([n], device=0), past_key_values=cache,
              use_cache=True, logits_to_keep=1)
    out(step="maskprobe", attn_impl=model.config._attn_implementation, masks=seen)

if len(sys.argv) > 3 and sys.argv[3] == "graphdiff":
    # Teacher-forced on the bf16 reference tokens: logits from the plain step vs the captured graph, per position.
    from transformers import StaticCache

    ref = json.loads(open("run-bf16-eager.jsonl").read().splitlines()[-1])["token_ids"]
    text_model = model.model
    cache = StaticCache(config=model.config, max_cache_len=256)
    prompt = ids["input_ids"]
    n = prompt.shape[1]

    def step(embeds, ple, pos):
        return model(inputs_embeds=embeds, per_layer_inputs=ple, position_ids=pos.view(1, 1), cache_position=pos,
                     past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1]

    def prefill():
        cache.reset()
        model(input_ids=prompt, cache_position=torch.arange(n, device=0), past_key_values=cache, use_cache=True, logits_to_keep=1)

    def inputs(tok, pos):
        t = torch.tensor([[tok]], device=0)
        return text_model.embed_tokens(t), text_model.get_per_layer_inputs(t, None), torch.tensor([pos], device=0)

    with torch.inference_mode():
        prefill()
        plain = []
        for i, tk in enumerate(ref[:-1]):
            plain.append(step(*inputs(tk, n + i)).float().clone())
        prefill()
        emb_buf, ple_buf, pos_buf = inputs(ref[0], n)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step(emb_buf, ple_buf, pos_buf)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits_buf = step(emb_buf, ple_buf, pos_buf)
        prefill()
        rows = []
        for i, tk in enumerate(ref[:-1]):
            e, p, q = inputs(tk, n + i)
            emb_buf.copy_(e), ple_buf.copy_(p), pos_buf.copy_(q)
            graph.replay()
            g = logits_buf.float()
            rows.append([i, round(float((g - plain[i]).abs().max()), 3), int(g.argmax()) == int(plain[i].argmax())])
    out(step="graphdiff", per_position=[r for r in rows if r[0] < 10 or not r[2]][:20])
