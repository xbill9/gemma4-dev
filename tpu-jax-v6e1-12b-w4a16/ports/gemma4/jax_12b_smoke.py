"""Exploratory smoke test: gemma-4-12B-it-qat-w4a16-ct on the E-series JAX engine.

The 12B is the same `gemma4_unified` architecture as E2B with every MatFormer
feature switched off in config.json:

    hidden_size_per_layer_input: 0   -> no Per-Layer Embeddings
    num_kv_shared_layers:        0   -> no KV sharing (all 48 layers own their KV)
    use_double_wide_mlp:     false   -> plain intermediate_size MLP
    attention_k_eq_v:         true   -> full-attention layers ship k_proj, no v_proj

so it should load through `config_from_hf` + the existing loader unchanged. This
script checks that claim on device: load, report resident bytes, greedy-decode a
short prompt, and print decode throughput.

    python3.13 ports/gemma4/jax_12b_smoke.py [--max-new-tokens 48]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DEFAULT_MODEL = "google/gemma-4-12B-it-qat-w4a16-ct"


def build_tokenizer(model_id: str):
    """Prefer transformers (gives the chat template); fall back to raw tokenizers."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_id), "transformers"
    except Exception:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        return Tokenizer.from_file(hf_hub_download(model_id, "tokenizer.json")), "tokenizers"


def encode(tok, kind: str, prompt: str):
    chat = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    if kind == "transformers":
        return tok(chat, add_special_tokens=False)["input_ids"]
    return tok.encode(chat, add_special_tokens=False).ids


def decode(tok, kind: str, ids):
    return tok.decode(ids) if kind == "transformers" else tok.decode(ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--kv-cache-dtype", default="bf16")
    ap.add_argument("--prompt", default="Explain in two sentences why TPUs suit JAX workloads.")
    args = ap.parse_args()

    import jax
    from jax_engine import JaxGemmaEngine, config_from_hf  # noqa: F401

    print(f"jax {jax.__version__}  devices={jax.devices()}")

    eng = JaxGemmaEngine(
        model_id=args.model,
        kv_cache_dtype=args.kv_cache_dtype,
        quant_mode="w4a16",
        max_model_len=args.max_model_len,
    )

    t0 = time.perf_counter()
    eng.load()
    t_load = time.perf_counter() - t0

    c = eng.config
    print("\n--- resolved config -------------------------------------------")
    for f in ("hidden_size", "intermediate_size", "num_hidden_layers",
              "num_attention_heads", "num_key_value_heads", "head_dim",
              "num_global_key_value_heads", "global_head_dim", "sliding_window",
              "num_kv_shared_layers", "hidden_size_per_layer_input",
              "attention_k_eq_v", "rope_theta", "global_rope_theta",
              "logit_softcapping"):
        print(f"  {f:32s} {getattr(c, f)}")
    print(f"  {'first_kv_shared_layer_idx':32s} {c.first_kv_shared_layer_idx}")
    print(f"  {'layer_types':32s} {c.layer_types.count('sliding_attention')} sliding / "
          f"{c.layer_types.count('full_attention')} full")

    mem = eng.memory_stats()
    print("\n--- memory ----------------------------------------------------")
    print(f"  load time            {t_load:8.1f} s")
    print(f"  weight bytes         {mem['weight_bytes'] / 1e9:8.2f} GB")
    print(f"  HBM in use           {mem['hbm_bytes_in_use'] / 1e9:8.2f} GB")
    print(f"  HBM limit            {mem['hbm_bytes_limit'] / 1e9:8.2f} GB")

    tok, kind = build_tokenizer(args.model)
    ids = encode(tok, kind, args.prompt)
    print(f"\n--- generate ({kind}, {len(ids)} prompt tokens) ---------------")

    t0 = time.perf_counter()
    out, stats = eng.generate(ids, max_new_tokens=args.max_new_tokens, temperature=0.0)
    wall = time.perf_counter() - t0

    print(decode(tok, kind, out))
    print(f"\n  {len(out)} tokens in {wall:.1f}s wall (includes compile)")
    if stats is not None:
        print(f"  prefill {getattr(stats, 'prefill_ms', float('nan')):.1f} ms   "
              f"decode {stats.decode_tok_per_s:.1f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
