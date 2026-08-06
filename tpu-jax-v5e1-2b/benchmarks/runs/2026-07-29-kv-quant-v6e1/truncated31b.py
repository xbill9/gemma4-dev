"""Run the 31B architecture on CPU with only its first 6 layers.

A full 60-layer forward pass OOM-killed a 125 GB host at 130.5 GB: the reference
W4A16 path dequantizes each packed weight to dense bf16 inside the forward, and
XLA:CPU keeps those temporaries alive across layers. That is a property of the
CPU backend, not of the model — on TPU the dequant fuses into the matmul.

The loader paths already passed on the real 60-layer checkpoint (stage 05). What
is still unproven is that the resulting parameter tree actually *runs*. Six real
layers of real weights answer that, and six is the smallest prefix that covers
every structural case:

    layers 0-4  sliding attention, 16 KV heads x 256 head_dim, own k/v_proj
    layer  5    FULL attention, 4 KV heads x 512 global_head_dim, and
                attention_k_eq_v -> no v_proj at all, V aliased to K

Plus the two branches the 31B is here to test: no PLE, and no KV sharing.
"""
import os, sys, time, json, gc
sys.path.insert(0, os.path.expanduser("~/gemma"))
import jax, jax.numpy as jnp
from safetensors.flax import load_file
from jax_engine import config_from_hf, _is_non_text_tensor
from ports.gemma4.jax_e_loader import convert_safetensors_to_jax_params
from ports.gemma4.jax_e_model import (Gemma4EModelJAX, prefill_with_kv_cache,
                                      make_cached_decode_step, pad_to_tpu_v6e_bucket)

N_LAYERS = 6
SNAP = [p for p in __import__("glob").glob(
    os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-31B*/snapshots/*"))][0]

cfg_full = config_from_hf(json.load(open(os.path.join(SNAP, "config.json"))))
print(f"full model: {cfg_full.num_hidden_layers} layers, k_eq_v={cfg_full.attention_k_eq_v}, "
      f"ple={cfg_full.hidden_size_per_layer_input}, shared={cfg_full.num_kv_shared_layers}", flush=True)
print(f"layer_types[:6] = {cfg_full.layer_types[:6]}", flush=True)

import dataclasses
cfg = dataclasses.replace(cfg_full, num_hidden_layers=N_LAYERS,
                          layer_types=cfg_full.layer_types[:N_LAYERS])
assert "full_attention" in cfg.layer_types, "prefix must include a full-attention layer"
assert cfg.first_kv_shared_layer_idx == N_LAYERS, "no KV sharing expected"


def wanted(key: str) -> bool:
    if _is_non_text_tensor(key):
        return False
    if ".layers." in key:
        return int(key.split(".layers.")[1].split(".")[0]) < N_LAYERS
    return True


raw = {}
for f in sorted(os.listdir(SNAP)):
    if not f.endswith(".safetensors"):
        continue
    part = load_file(os.path.join(SNAP, f))
    raw.update({k: v for k, v in part.items() if wanted(k)})
    part.clear(); del part; gc.collect()
nbytes = sum(a.size * a.dtype.itemsize for a in raw.values())
print(f"kept {len(raw)} tensors, {nbytes/1e9:.2f} GB", flush=True)

params = convert_safetensors_to_jax_params(
    raw, num_layers=N_LAYERS, first_kv_shared_idx=N_LAYERS,
    attention_k_eq_v=cfg.attention_k_eq_v)

# The alias is the whole point: layer 5 ships no v_proj.
a5 = params["layer_5"]["attn"]
assert a5["v_proj_packed"] is a5["k_proj_packed"], "layer 5 V should BE K"
a0 = params["layer_0"]["attn"]
assert a0["v_proj_packed"] is not a0["k_proj_packed"], "layer 0 has its own v_proj"
print("alias check: layer 5 V is K; layer 0 V is independent", flush=True)

model = Gemma4EModelJAX(cfg)
ids = jnp.array([[2, 818, 5279, 529, 6081, 563]], dtype=jnp.int32)   # arbitrary real ids
padded, valid = pad_to_tpu_v6e_bucket(ids)
print(f"prefill: ids {ids.shape} -> bucket {padded.shape}", flush=True)

t0 = time.time()
last, caches, vmask = jax.block_until_ready(prefill_with_kv_cache(
    model, padded, valid, params, 4, quant_mode="w4a16",
    cache_dtype=jnp.int8, window_kv=False))
print(f"prefill OK in {time.time()-t0:.1f}s, logits {last.shape}", flush=True)
assert jnp.all(jnp.isfinite(last)), "prefill produced non-finite logits"

step = jax.jit(make_cached_decode_step(model, quant_mode="w4a16"))
tok = jnp.argmax(last, axis=-1, keepdims=True)
lens = jnp.array([padded.shape[1]], dtype=jnp.int32)
toks = [int(tok[0, 0])]
for t in range(3):
    t0 = time.time()
    caches, vmask, last = jax.block_until_ready(
        step(params, caches, vmask, tok, lens + t, jnp.int32(padded.shape[1] + t)))
    assert jnp.all(jnp.isfinite(last)), f"step {t} produced non-finite logits"
    tok = jnp.argmax(last, axis=-1, keepdims=True)
    toks.append(int(tok[0, 0]))
    print(f"  decode step {t}: {time.time()-t0:.1f}s -> token {toks[-1]}", flush=True)

print(f"\ntokens: {toks}")
print("(a 6-layer truncation is not a coherent model; finite logits and distinct "
      "tokens are the assertion, not the text)")
print("TRUNC31B_OK", flush=True)
