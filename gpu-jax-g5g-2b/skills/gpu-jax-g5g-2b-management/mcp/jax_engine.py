"""Serving engine wrapper around the pure-JAX Gemma 4 E2B model.

Loads QAT safetensors checkpoints straight into JAX (no PyTorch, no torch_xla)
and exposes streaming generation built on the cache-correct decode path in
``ports.gemma4.jax_e_model`` — the one verified token-for-token against a
full-sequence re-forward in ``tests/test_kv_cache_parity.py``.

Used by ``jax_openai_server.py``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from ports.gemma4.jax_e_loader import convert_safetensors_to_jax_params
from ports.gemma4.jax_e_model import (
    COMPUTE_DTYPE,
    HARDWARE,
    IS_PRE_AMPERE,
    Gemma4EConfig,
    Gemma4EModelJAX,
    chunked_prefill_with_kv_cache,
    dequantize_params_to_dense,
    make_cached_decode_step,
    pad_to_bucket,
    prefill_with_kv_cache,
    quantize_lm_head,
    quantize_ple_table,
    set_w4a16_impl,
)

# One-byte entries halve the KV cache, which is the only term that grows with
# load — on a v6e-1 the decode ceiling is a fixed budget of resident KV tokens,
# so halving bytes/token doubles concurrent capacity at any context length.
# All of these carry per-token scales; see quantize_kv in jax_e_model.
_CACHE_DTYPES = {
    "bf16": jnp.bfloat16,
    "bfloat16": jnp.bfloat16,
    "fp16": jnp.float16,
    "float16": jnp.float16,
    # int8 is the accurate one: a per-row scale resolves 1/127 of that row's max,
    # where e4m3 spends bits on an exponent the scale already provides.
    "int8": jnp.int8,
    "fp8": jnp.float8_e4m3fn,
    "float8": jnp.float8_e4m3fn,
    "fp8_e4m3": jnp.float8_e4m3fn,
    # DEGRADED. Same capacity as int8 and e4m3 (1,048,576 KV tokens) but only 2
    # mantissa bits, and it visibly truncates output on the real checkpoint where
    # the other two match bf16. Kept for comparison; do not serve with it.
    "fp8_e5m2": jnp.float8_e5m2,
}

# Formats with no pre-Ampere datapath. bfloat16 is deliberately NOT in here:
# XLA emulates it on Turing and the numbers come out right, so it is a
# performance mistake rather than an error, and it gets a warning instead.
_PRE_AMPERE_UNSUPPORTED = {"fp8", "float8", "fp8_e4m3", "fp8_e5m2"}


logger = logging.getLogger(__name__)

# Substrings identifying weights that belong to a multimodal tower rather than the
# text decoder. Matched on a substring rather than a prefix because exports vary in
# whether they nest under `model.` — E2B ships `model.vision_tower.*` while other
# revisions use a bare `vision_tower.*`.
_NON_TEXT_MARKERS = (
    "vision_tower",
    "audio_tower",
    "vision_model",
    "audio_model",
    "multi_modal_projector",
    "embed_vision",
    "embed_audio",
)


def _is_non_text_tensor(key: str) -> bool:
    """True for tensors the text decoder never reads."""
    return any(marker in key for marker in _NON_TEXT_MARKERS)


def resolve_quant_mode(mode: str, model_id: str) -> str:
    """Resolve --quant-mode, defaulting to whatever the checkpoint actually is.

    quant_mode must match the checkpoint: a `-w4a16-` export carries packed int4
    weights and a dense export does not, so this is a property of the weights,
    not of the chip. "auto" reads it off the checkpoint name rather than
    guessing, because getting it wrong loads garbage rather than failing loudly.
    """
    key = (mode or "auto").strip().lower()
    if key == "auto":
        key = "w4a16" if "w4a16" in model_id.lower() else "fp16"
        logger.info("quant_mode=auto resolved to %r from checkpoint %s", key, model_id)
    if key not in ("w4a16", "fp16"):
        raise ValueError(f"quant_mode must be auto|w4a16|fp16, got {mode!r}")
    return key


def resolve_cache_dtype(name: str):
    """Map a KV-cache dtype name to a JAX dtype, refusing ones this chip lacks.

    "auto" follows the device's compute dtype, which is how tpu.env spells the
    default: float16 on Turing, bfloat16 on TPU and Ampere+.
    """
    key = (name or "auto").strip().lower()
    if key == "auto":
        return COMPUTE_DTYPE
    if key not in _CACHE_DTYPES:
        raise ValueError(f"Unsupported kv cache dtype {name!r}; choose from {[*sorted(_CACHE_DTYPES), 'auto']}")
    if IS_PRE_AMPERE and key in _PRE_AMPERE_UNSUPPORTED:
        raise ValueError(
            f"kv cache dtype {name!r} needs an fp8 datapath, which {HARDWARE.name} "
            "(pre-Ampere) does not have. Use auto, fp16, or int8 — int8 gives the "
            "same 1 byte/element and carries a per-row scale."
        )
    if IS_PRE_AMPERE and key in ("bf16", "bfloat16"):
        logger.warning(
            "kv cache dtype %s on %s: Turing has no bfloat16 datapath, so XLA will "
            "emulate it through fp32 conversions. Numbers stay correct and it is "
            "slower than fp16 for no benefit.", name, HARDWARE.name,
        )
    return _CACHE_DTYPES[key]


@dataclass
class GenerationStats:
    prompt_tokens: int
    completion_tokens: int
    prefill_ms: float
    decode_ms: float
    finish_reason: str

    @property
    def decode_tok_per_s(self) -> float:
        if self.completion_tokens <= 0 or self.decode_ms <= 0:
            return 0.0
        return self.completion_tokens * 1000.0 / self.decode_ms


def config_from_hf(hf_config: dict[str, Any]) -> Gemma4EConfig:
    """Build a Gemma4EConfig from a Hugging Face config.json.

    Falls back to the E2B defaults for any field the checkpoint omits, so a
    checkpoint that only differs in a couple of dimensions still loads.
    """
    text_cfg = hf_config.get("text_config", hf_config)
    defaults = Gemma4EConfig()
    # Newer exports nest per-attention-type RoPE under rope_parameters.
    rope_params = text_cfg.get("rope_parameters") or {}

    def pick(*names, default=None):
        for n in names:
            if n in text_cfg and text_cfg[n] is not None:
                return text_cfg[n]
        return default

    cfg = Gemma4EConfig(
        vocab_size=pick("vocab_size", default=defaults.vocab_size),
        hidden_size=pick("hidden_size", default=defaults.hidden_size),
        intermediate_size=pick("intermediate_size", default=defaults.intermediate_size),
        num_hidden_layers=pick("num_hidden_layers", default=defaults.num_hidden_layers),
        num_attention_heads=pick("num_attention_heads", default=defaults.num_attention_heads),
        num_key_value_heads=pick("num_key_value_heads", default=defaults.num_key_value_heads),
        head_dim=pick("head_dim", default=defaults.head_dim),
        num_global_key_value_heads=pick(
            "num_global_key_value_heads", "num_key_value_heads",
            default=defaults.num_global_key_value_heads),
        global_head_dim=pick("global_head_dim", default=defaults.global_head_dim),
        rms_norm_eps=pick("rms_norm_eps", default=defaults.rms_norm_eps),
        rope_theta=(rope_params.get("sliding_attention", {}).get("rope_theta")
                    or pick("rope_local_base_freq", "rope_theta", default=defaults.rope_theta)),
        global_rope_theta=(rope_params.get("full_attention", {}).get("rope_theta")
                           or pick("rope_theta", default=defaults.global_rope_theta)),
        logit_softcapping=pick("final_logit_softcapping", default=defaults.logit_softcapping) or 0.0,
        num_kv_shared_layers=pick("num_kv_shared_layers", default=defaults.num_kv_shared_layers),
        hidden_size_per_layer_input=pick(
            "hidden_size_per_layer_input", default=defaults.hidden_size_per_layer_input),
        vocab_size_per_layer_input=pick(
            "vocab_size_per_layer_input", default=defaults.vocab_size_per_layer_input),
        layer_types=pick("layer_types", default=None),
        sliding_window=pick("sliding_window", default=None),
        attn_logit_softcapping=pick("attn_logit_softcapping", default=0.0) or 0.0,
        attention_k_eq_v=bool(pick("attention_k_eq_v", default=False)),
    )
    return cfg


class JaxGemmaEngine:
    """Pure-JAX Gemma 4 E2B inference engine with a static KV cache."""

    def __init__(
        self,
        model_id: str,
        kv_cache_dtype: str = "auto",
        add_bos: bool = True,
        quant_mode: str = "auto",
        max_model_len: int = 4096,
        w4a16_impl: str = "reference",
        w4a16_layout: str = "plane",
        int8_lm_head: bool = False,
        int8_ple: bool = False,
        ple_bits: int = 0,
        dequant_at_load: bool = False,
        window_kv: bool | None = None,
        donate_cache: bool = True,
        prefill_chunk_size: int | None = None,
    ):
        self.model_id = model_id
        self.add_bos = add_bos
        self.bos_token_id: int | None = None
        self.kv_cache_dtype_name = kv_cache_dtype
        self.cache_dtype = resolve_cache_dtype(kv_cache_dtype)
        self.quant_mode = resolve_quant_mode(quant_mode, model_id)
        self.max_model_len = max_model_len
        # Perf knobs. All are numerics-preserving except int8_lm_head, which
        # trades ~0.8% logit error for halving the largest HBM read in a decode
        # step. Benchmark on a TPU before settling on a combination.
        self.w4a16_impl = w4a16_impl
        self.w4a16_layout = w4a16_layout
        self.int8_lm_head = int8_lm_head
        # W4A16 is a memory-for-compute trade: 4x smaller weights, paid for with a
        # dequantize every forward. Measured on v6e-1, dequantizing ONCE at load and
        # running the dense path is 1.60x faster at B=1 and 1.14x at B=32 — worth it
        # whenever the dense model fits (E2B is 3.7 GiB). Keep it off for models that
        # only fit quantized (31B, 26B MoE), where the trade runs the other way.
        # The PLE table is 4.70 GB of E2B's 6.56 GB resident — 72% of the
        # weights, and the shipped QAT checkpoint leaves it in BF16. It is a
        # gather, never a matmul, so dequantizing the fetched rows is nearly free
        # and the saving is pure HBM headroom for KV.
        #   ple_bits=8 -> 2.35 GB (-2.35),  ple_bits=4 -> 1.19 GB (-3.51)
        # `int8_ple=True` is the older spelling of ple_bits=8.
        self.ple_bits = 8 if (int8_ple and not ple_bits) else int(ple_bits)
        if self.ple_bits not in (0, 4, 8):
            raise ValueError(f"ple_bits must be 0, 4 or 8; got {self.ple_bits}")
        self.int8_ple = self.ple_bits == 8
        self.dequant_at_load = dequant_at_load
        # Windowing the sliding layers' KV saves memory once context exceeds the
        # window (2.67x at 8K) but measured ~3% slower on the decode step at short
        # context, where it saves nothing. None = decide per request.
        if window_kv is None and max_model_len and getattr(self, "config", None) is None:
            window_kv = None          # resolved at load(), once the config is known
        self.window_kv = window_kv
        # Buffer donation lets XLA write the updated KV cache into the input
        # buffer instead of allocating a new one. Without it, `dynamic_update_slice`
        # writes ONE token by producing a whole new cache array, so every decode
        # step reads the cache, writes a full copy, then reads it again to attend.
        #
        # Measured on v6e-1 at ctx 8192, B=32, 15 samples per point (IQR < 1.3%):
        #   bf16 cache  21.26 -> 13.14 ms  = 1.62x
        #   int8 cache  13.54 -> 11.09 ms  = 1.22x
        # Consistent across every PLE setting, and output is token-identical, so
        # this is a scheduling change and nothing else. Default ON.
        #
        # The cost: donated buffers are INVALIDATED by the call that consumes them.
        # Anything holding a reference to a cache across steps must re-read it from
        # the step's return value, never reuse the object it passed in. Set False if
        # a caller needs the old cache to stay valid.
        self.donate_cache = donate_cache
        # Prefill, not decode, sets the batch ceiling. Measured on v6e-1 by
        # compile-time memory_analysis: prefill temporaries are LINEAR in the total
        # prompt tokens in the pass, a flat ~2.13 MB per token from 1K to 8K tokens
        # (not quadratic — XLA tiles the softmax, so B x H x S x S scores are never
        # materialized). With ~25 GB left after weights that is a budget of roughly
        # 11,900 prompt tokens per prefill pass, whatever the B/S split.
        #
        # So the rule is B * chunk_size <= ~11,900, and chunking exists to keep that
        # product under the budget for prompts too long to admit in one pass.
        # Decode is governed by a separate and much larger budget: a flat 524,288
        # resident KV tokens at bf16, 1,048,576 at int8.
        # None = one-shot prefill, unchanged behaviour.
        # Requires window_kv=False, since a chunk writes contiguous slots at an
        # arbitrary offset that a shorter ring buffer would wrap.
        self.prefill_chunk_size = prefill_chunk_size
        if prefill_chunk_size is not None and window_kv:
            raise ValueError("prefill_chunk_size requires window_kv=False")

        self.config: Gemma4EConfig | None = None
        self.model: Gemma4EModelJAX | None = None
        self.params: dict[str, Any] | None = None
        self.device = None
        self.weight_bytes = 0
        self._decode_step = None
        self._jit_prefill = None

    # ------------------------------------------------------------------ load

    def load(self, local_dir: str | None = None) -> None:
        """Download (or read) the checkpoint and build JAX parameters on device."""
        from safetensors.flax import load_file

        path = local_dir or self._download()
        with open(os.path.join(path, "config.json")) as fh:
            hf_config = json.load(fh)
        self.config = config_from_hf(hf_config)

        raw: dict[str, Any] = {}
        shards = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
        if not shards:
            raise FileNotFoundError(f"No .safetensors found in {path}")
        # Drop the non-text towers as each shard lands rather than after the whole
        # checkpoint is resident. These checkpoints are multimodal — E2B carries
        # 0.95 GB of vision/audio the text decoder never reads, and the 31B carries
        # proportionally more — and `raw` is held for the entire conversion because
        # the parameter tree aliases its arrays. Filtering here is the difference
        # between peak RSS tracking the text weights and tracking the whole file.
        skipped_bytes = 0
        for shard in shards:
            part = load_file(os.path.join(path, shard))
            for key, arr in part.items():
                if _is_non_text_tensor(key):
                    skipped_bytes += arr.size * arr.dtype.itemsize
                    continue
                raw[key] = arr
            part.clear()
            del part
        if skipped_bytes:
            logger.info("skipped %.2f GB of non-text tower weights",
                        skipped_bytes / 1e9)

        self.params = convert_safetensors_to_jax_params(
            raw,
            num_layers=self.config.num_hidden_layers,
            first_kv_shared_idx=self.config.first_kv_shared_layer_idx,
            attention_k_eq_v=self.config.attention_k_eq_v,
        )

        if self.int8_lm_head:
            self.params = quantize_lm_head(self.params)
        if self.ple_bits:
            # Group by the per-layer slice width: each layer's 256-element slice
            # gets its own scale. A single row-wide scale is workable at 8 bits
            # and far too coarse at 4 (16 levels across all 8960 elements).
            self.params = quantize_ple_table(
                self.params, bits=self.ple_bits,
                group_size=self.config.hidden_size_per_layer_input)
        if self.dequant_at_load:
            self.params = dequantize_params_to_dense(self.params)
            self.quant_mode = "fp16"      # weights are dense now

        set_w4a16_impl(self.w4a16_impl, self.w4a16_layout)

        self.device = jax.devices()[0]
        self.params = jax.device_put(self.params, self.device)
        jax.block_until_ready(self.params)
        self.weight_bytes = sum(
            int(x.size) * int(x.dtype.itemsize)
            for x in jax.tree_util.tree_leaves(self.params)
        )

        self.model = Gemma4EModelJAX(self.config)
        # donate_argnums=(1, 2): caches and the validity mask are threaded through
        # and reassigned every step, so donating lets XLA update them in place
        # instead of copying the whole cache per token. Off by default because a
        # caller that reuses a donated buffer gets a runtime error.
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

    def _download(self) -> str:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            self.model_id,
            allow_patterns=["*.safetensors", "*.json"],
            token=os.environ.get("HF_TOKEN"),
        )

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.params is not None

    # -------------------------------------------------------------- generate

    def generate_stream(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int = 40,
        eos_token_ids: Sequence[int] | None = None,
        seed: int = 0,
    ) -> Iterator[Any]:
        """Yield generated token ids one at a time, then a final GenerationStats.

        Single-sequence (B=1) streaming: each step attends to the full history
        through the static KV cache.
        """
        if not self.is_ready:
            raise RuntimeError("Engine not loaded")

        # Gemma expects a leading <bos>. This checkpoint's tokenizer does NOT add
        # one automatically, and without it the model echoes the prompt back
        # instead of answering. Prepend here so every caller gets it right.
        prompt_ids = list(prompt_ids)
        if self.add_bos and self.bos_token_id is not None and (
            not prompt_ids or prompt_ids[0] != self.bos_token_id
        ):
            prompt_ids = [self.bos_token_id, *prompt_ids]

        eos = set(int(e) for e in (eos_token_ids or []))
        prompt_len = len(prompt_ids)
        budget = self.max_model_len - prompt_len
        if budget <= 0:
            raise ValueError(
                f"Prompt of {prompt_len} tokens leaves no room within "
                f"max_model_len={self.max_model_len}"
            )
        max_new_tokens = max(1, min(max_new_tokens, budget))

        ids = jnp.asarray(prompt_ids, dtype=jnp.int32)[None, :]
        padded, valid_mask = pad_to_bucket(ids)
        bucket_s = int(padded.shape[1])

        t0 = time.perf_counter()
        chunk = self.prefill_chunk_size
        if chunk is not None and bucket_s % chunk == 0:
            last_logits, caches, valid = jax.block_until_ready(
                chunked_prefill_with_kv_cache(
                    self.model, padded, valid_mask, self.params, max_new_tokens,
                    chunk_size=chunk, quant_mode=self.quant_mode,
                    cache_dtype=self.cache_dtype,
                )
            )
        else:
            # Falls back for prompts the chunk size does not divide — the bucket
            # padding is a power of two, so this only happens for chunk sizes that
            # are not.
            last_logits, caches, valid = jax.block_until_ready(
                self._jit_prefill(
                    model=self.model, prompt_ids=padded, prompt_valid=valid_mask,
                    params=self.params, max_new_tokens=max_new_tokens,
                    quant_mode=self.quant_mode, cache_dtype=self.cache_dtype,
                    window_kv=self.window_kv,
                )
            )
        prefill_ms = (time.perf_counter() - t0) * 1000.0

        key = jax.random.PRNGKey(seed)
        prompt_lens = jnp.asarray([prompt_len], dtype=jnp.int32)

        emitted = 0
        finish_reason = "length"
        decode_start = time.perf_counter()
        for step_idx in range(max_new_tokens):
            key, sample_key = jax.random.split(key)
            tok = self._sample(last_logits, sample_key, temperature, top_k)
            tok_id = int(tok[0, 0])

            if tok_id in eos:
                finish_reason = "stop"
                break

            yield tok_id
            emitted += 1

            if step_idx == max_new_tokens - 1:
                break

            caches, valid, last_logits = self._decode_step(
                self.params, caches, valid, tok,
                prompt_lens + step_idx, jnp.int32(bucket_s + step_idx),
            )
        decode_ms = (time.perf_counter() - decode_start) * 1000.0

        yield GenerationStats(
            prompt_tokens=prompt_len,
            completion_tokens=emitted,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _sample(logits: jax.Array, key: jax.Array, temperature: float, top_k: int) -> jax.Array:
        from ports.gemma4.jax_e_model import onchip_sample_jax

        return onchip_sample_jax(logits, key, temperature=temperature, top_k=top_k)

    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int = 40,
        eos_token_ids: Sequence[int] | None = None,
        seed: int = 0,
    ):
        """Non-streaming convenience wrapper. Returns (token_ids, GenerationStats)."""
        tokens: list[int] = []
        stats: GenerationStats | None = None
        for item in self.generate_stream(
            prompt_ids, max_new_tokens, temperature, top_k, eos_token_ids, seed
        ):
            if isinstance(item, GenerationStats):
                stats = item
            else:
                tokens.append(item)
        return tokens, stats

    # ----------------------------------------------------------------- info

    def memory_stats(self) -> dict[str, Any]:
        stats = self.device.memory_stats() if hasattr(self.device, "memory_stats") else {}
        return {
            "weight_bytes": self.weight_bytes,
            "hbm_bytes_in_use": (stats or {}).get("bytes_in_use", 0),
            "hbm_bytes_limit": (stats or {}).get("bytes_limit", 0),
        }
