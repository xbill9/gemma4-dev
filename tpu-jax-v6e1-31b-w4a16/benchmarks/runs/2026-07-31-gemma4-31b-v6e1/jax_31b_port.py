"""Port + verify + benchmark gemma-4-31B-it-qat-w4a16-ct on the E-series JAX engine.

One process, four stages, because the load is the expensive part at this size and
running smoke / parity / sweep as separate scripts would pay it three times.
Every stage prints as it completes and appends to --json-out, so a flex-start VM
that expires mid-sweep still leaves the earlier stages on disk.

What the 31B is, from its config.json (same `gemma4_unified` family as E2B/12B):

    hidden_size                5376   num_hidden_layers            60
    intermediate_size         21504   layer_types      50 sliding / 10 full
    num_attention_heads          32   num_key_value_heads          16
    num_global_key_value_heads    4   global_head_dim             512
    hidden_size_per_layer_input   0   -> no Per-Layer Embeddings
    num_kv_shared_layers          0   -> every layer owns its KV
    attention_k_eq_v           true   -> full-attention layers ship k_proj, no v_proj

So, like the 12B, a strict simplification of what the engine already implements.
The one thing it is NOT is small: ~19.3 GB of W4A16 text weights against a v6e-1's
33.55 GB, versus the 12B's 8.15 GB. Headroom, not architecture, is what is under
test here.

    python3.13 ports/gemma4/jax_31b_port.py --stages config,load,parity,sweep
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp

from jax_engine import JaxGemmaEngine, _is_non_text_tensor, config_from_hf
from ports.gemma4.jax_e_loader import convert_safetensors_to_jax_params
from ports.gemma4.jax_e_model import (
    Gemma4EModelJAX,
    make_cached_decode_step,
    pad_to_tpu_v6e_bucket,
    prefill_with_kv_cache,
    set_w4a16_impl,
)

DEFAULT_MODEL = "google/gemma-4-31B-it-qat-w4a16-ct"
PARITY_PROMPT = "What is the capital of France?"

# safetensors dtype string -> bytes, for accounting the towers we never read.
_ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1,
             "I16": 2, "I32": 4, "I64": 8, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}

# Fields worth asserting rather than eyeballing: each one is a place where a
# silently-wrong default would still load and still generate, just incorrectly.
EXPECTED_CONFIG = {
    "hidden_size": 5376,
    "intermediate_size": 21504,
    "num_hidden_layers": 60,
    "num_attention_heads": 32,
    "num_key_value_heads": 16,
    "num_global_key_value_heads": 4,
    "head_dim": 256,
    "global_head_dim": 512,
    "sliding_window": 1024,
    "num_kv_shared_layers": 0,
    "hidden_size_per_layer_input": 0,
    "attention_k_eq_v": True,
    "logit_softcapping": 30.0,
    "rope_theta": 10000.0,
    "global_rope_theta": 1000000.0,
}


class Streaming31BEngine(JaxGemmaEngine):
    """JaxGemmaEngine that streams the checkpoint through host memory.

    The base ``load()`` calls ``safetensors.flax.load_file`` per shard, which
    materializes every tensor in the file as a jnp array — on the default device —
    and only then drops the vision/audio towers. At E2B/12B scale the spike is
    affordable. Here the single shard is 23.3 GB against 33.55 GB of HBM, and the
    towers are ~3.9 GB of it, so the filter has to happen before anything reaches
    the device. Reading with ``framework="np"`` keeps the raw tensors in host RAM
    (the VM has 172 GB) and lets the converter place only text weights on device.

    Numerically identical to the base loader — same converter, same arguments.
    """

    def load(self, local_dir: str | None = None) -> None:
        from safetensors import safe_open

        path = local_dir or self._download()
        with open(os.path.join(path, "config.json")) as fh:
            hf_config = json.load(fh)
        self.config = config_from_hf(hf_config)

        shards = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
        if not shards:
            raise FileNotFoundError(f"No .safetensors found in {path}")

        raw: Dict[str, Any] = {}
        skipped_bytes = kept_bytes = 0
        for shard in shards:
            with safe_open(os.path.join(path, shard), framework="np", device="cpu") as fh:
                for key in fh.keys():
                    if _is_non_text_tensor(key):
                        # Read the header-declared shape instead of the tensor: no
                        # point paying host bandwidth for weights we discard. The
                        # byte count is advisory, so an unrecognized dtype falls
                        # back to 2 rather than failing the load.
                        sl = fh.get_slice(key)
                        n = 1
                        for d in sl.get_shape():
                            n *= d
                        skipped_bytes += n * _ITEMSIZE.get(sl.get_dtype().upper(), 2)
                        continue
                    arr = fh.get_tensor(key)
                    kept_bytes += arr.size * arr.dtype.itemsize
                    raw[key] = arr
        print(f"  host read: {kept_bytes / 1e9:.2f} GB text, "
              f"skipped ~{skipped_bytes / 1e9:.2f} GB of vision/audio towers", flush=True)

        self.params = convert_safetensors_to_jax_params(
            raw,
            num_layers=self.config.num_hidden_layers,
            first_kv_shared_idx=self.config.first_kv_shared_layer_idx,
            attention_k_eq_v=self.config.attention_k_eq_v,
        )
        raw.clear()
        gc.collect()

        set_w4a16_impl(self.w4a16_impl, self.w4a16_layout)

        self.device = jax.devices()[0]
        self.params = jax.device_put(self.params, self.device)
        jax.block_until_ready(self.params)
        self.weight_bytes = sum(
            int(x.size) * int(x.dtype.itemsize)
            for x in jax.tree_util.tree_leaves(self.params)
        )

        self.model = Gemma4EModelJAX(self.config)
        if self.window_kv is None:
            win = self.config.sliding_window
            self.window_kv = bool(win) and self.max_model_len > int(win)
        self._decode_step = jax.jit(
            make_cached_decode_step(self.model, quant_mode=self.quant_mode,
                                    window_kv=self.window_kv),
            **({"donate_argnums": (1, 2)} if self.donate_cache else {}),
        )
        self._jit_prefill = jax.jit(
            prefill_with_kv_cache,
            static_argnames=("model", "max_new_tokens", "quant_mode", "cache_dtype", "window_kv"),
        )


# --------------------------------------------------------------------- stages


def stage_config(model_id: str, out: Dict[str, Any]) -> Any:
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(model_id, "config.json")) as fh:
        cfg = config_from_hf(json.load(fh))

    print("\n--- resolved config -------------------------------------------")
    mismatches = []
    for field, expected in EXPECTED_CONFIG.items():
        got = getattr(cfg, field)
        ok = got == expected
        if not ok:
            mismatches.append(f"{field}: got {got!r}, config.json says {expected!r}")
        print(f"  {field:30s} {str(got):>12s}   {'ok' if ok else 'MISMATCH'}")

    sliding = cfg.layer_types.count("sliding_attention")
    full = cfg.layer_types.count("full_attention")
    print(f"  {'layer_types':30s} {sliding:>4d} sliding / {full} full")
    print(f"  {'first_kv_shared_layer_idx':30s} {cfg.first_kv_shared_layer_idx:>12d}")

    out["config"] = {
        "resolved": {f: getattr(cfg, f) for f in EXPECTED_CONFIG},
        "sliding_layers": sliding,
        "full_layers": full,
        "mismatches": mismatches,
    }
    if mismatches:
        print("  !! config resolution disagrees with the checkpoint:")
        for m in mismatches:
            print(f"     {m}")
    return cfg


def stage_load(args, out: Dict[str, Any]) -> Streaming31BEngine:
    print("\n--- load ------------------------------------------------------", flush=True)
    eng = Streaming31BEngine(
        model_id=args.model,
        kv_cache_dtype=args.kv_cache_dtype,
        quant_mode="w4a16",
        max_model_len=args.max_model_len,
        w4a16_impl=args.w4a16_impl,
    )
    t0 = time.perf_counter()
    eng.load()
    t_load = time.perf_counter() - t0

    mem = eng.memory_stats()
    limit = mem["hbm_bytes_limit"] or 33.55e9
    print(f"  load time            {t_load:8.1f} s")
    print(f"  weight bytes         {mem['weight_bytes'] / 1e9:8.2f} GB")
    print(f"  HBM in use           {mem['hbm_bytes_in_use'] / 1e9:8.2f} GB")
    print(f"  HBM limit            {limit / 1e9:8.2f} GB")
    print(f"  headroom             {(limit - mem['hbm_bytes_in_use']) / 1e9:8.2f} GB "
          f"({100.0 * (limit - mem['hbm_bytes_in_use']) / limit:.1f}%)")

    out["load"] = {
        "load_s": round(t_load, 1),
        "weight_gb": round(mem["weight_bytes"] / 1e9, 3),
        "hbm_in_use_gb": round(mem["hbm_bytes_in_use"] / 1e9, 3),
        "hbm_limit_gb": round(limit / 1e9, 3),
        "headroom_gb": round((limit - mem["hbm_bytes_in_use"]) / 1e9, 3),
        "w4a16_impl": args.w4a16_impl,
    }
    return eng


def stage_parity(eng, args, out: Dict[str, Any]) -> None:
    """Greedy-decode the chat-templated parity prompt.

    The 12B port established that this family's IT QAT checkpoints emit digit
    strings for a bare prompt on BOTH the HF PyTorch reference and this engine —
    the chat template plus <bos> is a correctness requirement, not a nicety. So
    the prompt goes through apply_chat_template, and an unformatted control runs
    alongside it: matching digits on the control is expected and is what tells us
    a wrong answer here would be formatting rather than engine math.
    """
    from transformers import AutoTokenizer

    print("\n--- parity ----------------------------------------------------", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    eng.bos_token_id = tok.bos_token_id

    formatted = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    print(f"  formatted prompt: {formatted!r}")

    # Without these the run never stops early and the transcript shows the model
    # answering correctly and then rambling to the token limit, which reads like a
    # defect and is not one. <turn|> ends a turn; <eos> ends the sequence.
    eos_ids = {i for i in (tok.eos_token_id,) if i is not None}
    for name in ("<turn|>", "<end_of_turn>", "<eos>"):
        tid = tok.convert_tokens_to_ids(name)
        if tid is not None and tid != tok.unk_token_id:
            eos_ids.add(tid)
    print(f"  stop tokens: {sorted(eos_ids)}")

    results = {}
    for label, text in (("templated", formatted), ("bare", args.prompt)):
        ids = tok(text, add_special_tokens=False, return_tensors="np")["input_ids"][0].tolist()
        t0 = time.perf_counter()
        toks, stats = eng.generate(ids, max_new_tokens=args.parity_tokens, temperature=0.0,
                                   eos_token_ids=sorted(eos_ids))
        wall = time.perf_counter() - t0
        gen = tok.decode(toks)
        print(f"  [{label:9s}] {len(ids):4d} prompt tok -> {gen!r}")
        print(f"  {'':11s} {wall:.1f}s wall (incl. compile), "
              f"prefill {getattr(stats, 'prefill_ms', float('nan')):.1f} ms, "
              f"decode {stats.decode_tok_per_s:.1f} tok/s")
        results[label] = {
            "prompt_tokens": len(ids),
            "output": gen,
            "finish_reason": getattr(stats, "finish_reason", None),
            "prefill_ms": round(getattr(stats, "prefill_ms", 0.0), 1),
            "decode_tok_s": round(stats.decode_tok_per_s, 2),
            "wall_s": round(wall, 1),
        }

    out["parity"] = results


def _time_median_ms(fn, repeats: int = 3, warmup: int = 1) -> float:
    for _ in range(warmup):
        jax.block_until_ready(fn())
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def _bench_cell(model, params, B: int, S: int, decode_steps: int, repeats: int) -> Dict[str, Any]:
    raw_ids = jnp.ones((B, S), dtype=jnp.int32)
    padded_ids, valid_mask = pad_to_tpu_v6e_bucket(raw_ids)
    bucket_s = int(padded_ids.shape[1])

    jit_prefill = jax.jit(
        prefill_with_kv_cache,
        static_argnames=("model", "max_new_tokens", "quant_mode", "cache_dtype", "window_kv"),
    )

    def run_prefill():
        return jit_prefill(
            model=model, prompt_ids=padded_ids, prompt_valid=valid_mask, params=params,
            max_new_tokens=decode_steps, quant_mode="w4a16", window_kv=True,
        )

    prefill_ms = _time_median_ms(run_prefill, repeats=repeats)

    last_logits, caches, valid = jax.block_until_ready(run_prefill())
    # No donation here: run_step is called repeatedly with the same caches, which
    # a donated buffer would invalidate after the first call.
    step = jax.jit(make_cached_decode_step(model, quant_mode="w4a16", window_kv=True))
    prompt_lens = valid_mask.sum(axis=1).astype(jnp.int32)
    tok = jnp.argmax(last_logits, axis=-1, keepdims=True)

    def run_step():
        return step(params, caches, valid, tok, prompt_lens, jnp.int32(bucket_s))

    step_ms = _time_median_ms(run_step, repeats=repeats)

    del last_logits, caches, valid
    gc.collect()

    return {
        "B": B, "S": S, "bucket_S": bucket_s,
        "prefill_ms": round(prefill_ms, 2),
        "decode_step_ms": round(step_ms, 3),
        "agg_decode_tok_s": round(B * 1000.0 / step_ms, 1),
        "per_user_tok_s": round(1000.0 / step_ms, 1),
        "status": "OK",
    }


def stage_sweep(eng, args, out: Dict[str, Any], json_out: str) -> None:
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    contexts = [int(x) for x in args.contexts.split(",")]

    print("\n--- sweep -----------------------------------------------------", flush=True)
    results: List[Dict[str, Any]] = []
    for B in batch_sizes:
        for S in contexts:
            label = f"{S // 1024}K" if S >= 1024 else str(S)
            print(f"  B={B:2d} | ctx={label:>5s} ... ", end="", flush=True)
            try:
                cell = _bench_cell(eng.model, eng.params, B, S,
                                   decode_steps=args.decode_steps, repeats=args.repeats)
                print(f"prefill {cell['prefill_ms']:9.1f} ms | step {cell['decode_step_ms']:8.3f} ms | "
                      f"agg {cell['agg_decode_tok_s']:7.1f} tok/s | "
                      f"per-user {cell['per_user_tok_s']:6.1f} tok/s")
            except Exception as exc:
                msg = str(exc).split("\n")[0][:70]
                print(f"FAILED: {msg}")
                cell = {"B": B, "S": S, "status": "FAILED", "error": msg}
                gc.collect()
            results.append(cell)
            out["sweep"] = results
            _write(json_out, out)     # checkpoint after every cell

    print("\n| Users (B) | Context (S) | Prefill (TTFT) | Decode Step | Aggregate | Per-User |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        label = f"{r['S'] // 1024}K" if r["S"] >= 1024 else str(r["S"])
        if r["status"] == "OK":
            print(f"| {r['B']} | {label} | {r['prefill_ms']:.1f} ms | {r['decode_step_ms']:.2f} ms | "
                  f"{r['agg_decode_tok_s']:.1f} tok/s | {r['per_user_tok_s']:.1f} tok/s |")
        else:
            print(f"| {r['B']} | {label} | {r['status']} | — | — | — |")


def stage_chunked(args, out: Dict[str, Any], json_out: str) -> None:
    """Break the one-shot prefill ceiling with chunked prefill.

    The reference sweep OOMs above ~4K total prompt tokens because one-shot prefill
    materializes temporaries linear in the tokens admitted per pass. Chunking caps
    that per pass, at the cost of requiring window_kv=False — a chunk writes
    contiguous slots at an arbitrary offset that a shorter ring buffer would wrap.

    Turning windowing off is not free at this size. Unwindowed, the 50 sliding
    layers allocate max_seq_len slots at 800 KiB/token instead of capping at 1024,
    so KV becomes 880 KiB/token overall: ~7.2 GB at 8K against 14.25 GB of headroom,
    and ~14.4 GB at 16K, which is over. So 8K is the honest target here and 16K is
    expected to fail — the run records both rather than only the one that works.
    """
    print("\n--- chunked prefill -------------------------------------------", flush=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    results = []
    for ctx in [int(x) for x in args.chunked_contexts.split(",")]:
        chunk = args.prefill_chunk_size
        print(f"  ctx={ctx:6d} chunk={chunk:5d} ... ", end="", flush=True)
        eng = None
        try:
            eng = Streaming31BEngine(
                model_id=args.model, kv_cache_dtype=args.kv_cache_dtype,
                quant_mode="w4a16", max_model_len=ctx + 64,
                w4a16_impl=args.w4a16_impl,
                window_kv=False, prefill_chunk_size=chunk,
            )
            eng.load()
            eng.bos_token_id = tok.bos_token_id
            # A synthetic prompt of exactly ctx tokens: what is under test is the
            # prefill memory profile, not what the tokens say.
            ids = [tok.bos_token_id or 2] + [1000] * (ctx - 1)
            t0 = time.perf_counter()
            toks, stats = eng.generate(ids, max_new_tokens=8, temperature=0.0)
            wall = time.perf_counter() - t0
            mem = eng.memory_stats()
            print(f"OK  prefill {stats.prefill_ms:9.1f} ms | decode {stats.decode_tok_per_s:5.2f} tok/s "
                  f"| HBM {mem['hbm_bytes_in_use'] / 1e9:5.2f} GB")
            results.append({
                "ctx": ctx, "chunk": chunk, "status": "OK",
                "prefill_ms": round(stats.prefill_ms, 1),
                "decode_tok_s": round(stats.decode_tok_per_s, 2),
                "hbm_in_use_gb": round(mem["hbm_bytes_in_use"] / 1e9, 3),
                "wall_s": round(wall, 1),
            })
        except Exception as exc:
            msg = str(exc).split("\n")[0][:80]
            print(f"FAILED: {msg}")
            results.append({"ctx": ctx, "chunk": chunk, "status": "FAILED", "error": msg})
        finally:
            # Each context needs its own cache shapes, so the engine is rebuilt per
            # point; drop the previous one's 19.3 GB before the next load.
            del eng
            gc.collect()
        out["chunked"] = results
        _write(json_out, out)


def _write(path: str, out: Dict[str, Any]) -> None:
    if not path:
        return
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--stages", default="config,load,parity,sweep")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--kv-cache-dtype", default="bf16")
    ap.add_argument("--w4a16-impl", default="reference",
                    help="reference matches the 12B report's methodology; auto tries the Pallas kernel")
    ap.add_argument("--prompt", default=PARITY_PROMPT)
    ap.add_argument("--parity-tokens", type=int, default=32)
    ap.add_argument("--batch-sizes", default="1,2,4")
    ap.add_argument("--contexts", default="1024,2048,4096,8192")
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--chunked-contexts", default="8192,16384")
    ap.add_argument("--prefill-chunk-size", type=int, default=1024)
    ap.add_argument("--json-out", default="gemma4_31b_results.json")
    args = ap.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    out: Dict[str, Any] = {
        "model": args.model,
        "devices": [str(d) for d in jax.devices()],
        "jax_version": jax.__version__,
        "stages_requested": stages,
    }

    print("=" * 92)
    print(f"GEMMA 4 31B QAT — pure-JAX E-series engine on {jax.devices()}")
    print("=" * 92)

    eng = None
    if "config" in stages:
        stage_config(args.model, out)
        _write(args.json_out, out)
    if "load" in stages:
        eng = stage_load(args, out)
        _write(args.json_out, out)
    if "parity" in stages and eng is not None:
        stage_parity(eng, args, out)
        _write(args.json_out, out)
    if "sweep" in stages and eng is not None:
        stage_sweep(eng, args, out, args.json_out)
        _write(args.json_out, out)
    if "chunked" in stages:
        # Builds its own engines (one per context, different cache shapes), so it
        # must not share the chip with the one above.
        if eng is not None:
            del eng
            eng = None
            gc.collect()
        stage_chunked(args, out, args.json_out)
        _write(args.json_out, out)

    print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
