"""Does quantizing the PLE table buy concurrency, and does it cost quality?

E2B's shipped W4A16 checkpoint compressed 1.06 GB of transformer weights and
left 5.50 GB of lookup tables in BF16. `embed_tokens_per_layer` alone is 4.70 GB
— 72% of resident weights. It is read by a gather, never a matmul, so it is the
largest and lowest-risk quantization target left in the model.

An earlier measurement called int8 PLE "0.95x, no capacity unlock". That was
tested against the PREFILL OOM wall, which is set by activation temporaries and
which weight savings cannot move. The decode budget is the term weight savings
actually free, and this run measures against that instead.

Two questions:
  1. capacity — resident KV tokens at ple_bits in {0, 8, 4} x cache in {bf16, int8}
  2. quality  — real checkpoint, real tokenizer, greedy, at each ple_bits
"""
import os, sys, time, json, statistics, gc
sys.path.insert(0, os.path.expanduser("~/gemma"))
import jax, jax.numpy as jnp
from ports.gemma4.jax_e_model import (Gemma4EConfig, Gemma4EModelJAX, init_kv_cache,
                                      make_cached_decode_step, quantize_ple_table)
from ports.gemma4.jax_e_benchmark_sweep import build_benchmark_params
from ports.gemma4.jax_e_benchmark_sweep_v2 import E2B_CONFIG

OUT = []
cfg = Gemma4EConfig(**dict(E2B_CONFIG, sliding_window=512))
model = Gemma4EModelJAX(cfg)


def param_bytes(p):
    tot = 0
    stack = [p]
    while stack:
        n = stack.pop()
        if isinstance(n, dict):
            stack.extend(n.values())
        elif hasattr(n, "size"):
            tot += n.size * n.dtype.itemsize
    return tot


# ---- 1. capacity: does freeing weight bytes buy KV tokens? ----------------
# BISECTION, not a doubling ladder. int8 PLE frees 2.35 GB, which at 18 KiB/token
# is ~127k KV tokens — enough to move max B at ctx 8192 from 64 to ~79. A ladder
# stepping 64 -> 96 reports that real gain as "no change", which is exactly how
# the earlier "int8 PLE buys no capacity" conclusion was reached. Resolve to 2%.
def fits(params, cache_dtype, ctx, B, step):
    try:
        total = ctx + 16
        caches = init_kv_cache(cfg, batch_size=B, max_seq_len=total, dtype=cache_dtype)
        valid = jnp.zeros((B, total), dtype=jnp.bool_).at[:, :ctx].set(True)
        tok = jnp.ones((B, 1), dtype=jnp.int32)
        lens = jnp.full((B,), ctx, dtype=jnp.int32)
        args = (params, caches, valid, tok, lens, jnp.int32(ctx))
        jax.block_until_ready(step(*args))
        ts = []
        for _ in range(3):
            t0 = time.perf_counter(); jax.block_until_ready(step(*args))
            ts.append((time.perf_counter() - t0) * 1000)
        del caches
        return statistics.median(ts)
    except Exception:
        gc.collect()
        return None


def max_batch(params, label, cache_dtype, ctx, lo_seed):
    """Bracket by doubling from a known-good seed, then bisect to ~2%."""
    step = jax.jit(make_cached_decode_step(model, quant_mode="w4a16", window_kv=False))
    lo, lo_ms = None, None
    B = lo_seed
    while True:                              # bracket
        ms = fits(params, cache_dtype, ctx, B, step)
        if ms is None:
            hi = B
            break
        lo, lo_ms = B, ms
        B *= 2
        if B > 1 << 16:
            hi = B
            break
    if lo is None:
        print(f"  -> {label} ctx={ctx}: OOM even at seed {lo_seed}", flush=True)
        OUT.append(dict(kind="capacity", label=label, ctx=ctx, max_B=None, kv_tokens=0))
        return None
    while hi - lo > max(1, lo // 50):        # bisect to 2%
        mid = (lo + hi) // 2
        ms = fits(params, cache_dtype, ctx, mid, step)
        if ms is None:
            hi = mid
        else:
            lo, lo_ms = mid, ms
    print(f"  -> {label} ctx={ctx}: max B {lo}, KV tokens {ctx*lo:,}, "
          f"step {lo_ms:.2f} ms", flush=True)
    OUT.append(dict(kind="capacity", label=label, ctx=ctx, max_B=lo,
                    kv_tokens=ctx*lo, step_ms=lo_ms))
    return lo


SEEDS = {512: 256, 8192: 16}

print(f"synthetic params: "
      f"{param_bytes(build_benchmark_params(cfg))/1e9:.2f} GB", flush=True)

for bits in (0, 8, 4):
    # Rebuild from scratch each time. Sharing one `base` keeps the 4.70 GB BF16
    # table resident alongside every quantized copy, which costs the quantized
    # configs exactly the headroom this run exists to measure.
    src = build_benchmark_params(cfg)
    p = src if bits == 0 else quantize_ple_table(
        src, bits=bits, group_size=cfg.hidden_size_per_layer_input)
    if bits:
        del src
    gc.collect()
    gb = param_bytes(p) / 1e9
    for cache_name, cache_dt in (("bf16", jnp.bfloat16), ("int8", jnp.int8)):
        label = f"ple-{bits or 'bf16'}/kv-{cache_name}"
        print(f"\n### {label}  (weights {gb:.2f} GB) ###", flush=True)
        for ctx, seed in SEEDS.items():
            max_batch(p, label, cache_dt, ctx, seed)
    del p
    gc.collect()

# ---- 2. quality on the real checkpoint -----------------------------------
print("\n########## real-checkpoint quality ##########", flush=True)
try:
    from jax_engine import JaxGemmaEngine
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    MID = "google/gemma-4-E2B-it-qat-w4a16-ct"
    snap = snapshot_download(MID, allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"])
    tok = AutoTokenizer.from_pretrained(snap)
    eos = [t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>"))
           if isinstance(t, int) and t >= 0]
    QS = ["What is 2+2?", "The capital of France is", "Name three colours.",
          "Explain gravity in one sentence.", "Write a haiku about TPUs."]
    for bits in (0, 8, 4):
        print(f"\n--- ple_bits={bits or 'bf16'} ---", flush=True)
        eng = JaxGemmaEngine(MID, kv_cache_dtype="int8", quant_mode="w4a16",
                             max_model_len=512, ple_bits=bits)
        eng.load(local_dir=snap)
        eng.bos_token_id = tok.bos_token_id
        print(f"    weights {eng.weight_bytes/1e9:.2f} GB", flush=True)
        for q in QS:
            ids = tok(f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n",
                      add_special_tokens=False)["input_ids"]
            o, st = eng.generate(ids, max_new_tokens=24, temperature=0.0, eos_token_ids=eos)
            txt = tok.decode(o, skip_special_tokens=True)
            print(f"    {q!r} -> {txt!r}  [{st.decode_tok_per_s:.0f} tok/s]", flush=True)
            OUT.append(dict(kind="quality", ple_bits=bits, q=q, a=txt,
                            weights_gb=round(eng.weight_bytes/1e9, 2)))
        del eng
        gc.collect()
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"quality section FAILED: {e}", flush=True)

with open(os.path.expanduser("~/ple_run.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print("\nPLE_RUN_DONE", flush=True)
