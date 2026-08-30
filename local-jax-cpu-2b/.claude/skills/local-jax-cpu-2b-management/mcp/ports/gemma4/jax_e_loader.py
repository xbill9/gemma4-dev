"""JAX PyTree Parameter Loader for Gemma 4 E2B QAT Checkpoints.

Loads Hugging Face safetensors checkpoints (e.g. google/gemma-4-E2B-it-qat-w4a16-ct or
google/gemma-4-E2B-it-qat-q4_0-unquantized) directly into JAX PyTree dictionaries.

Pure Python & JAX — ZERO PyTorch, transformers, or confidential repository dependencies.
"""

from typing import Dict, Any
import jax.numpy as jnp



# Gemma 4 E2B ships as a multimodal checkpoint: the text decoder lives under
# `model.language_model.` alongside `model.audio_tower.` / vision towers. Older
# text-only exports use a bare `model.` prefix. Detect rather than assume — the
# wrong prefix silently yields a parameter tree full of None.
_TEXT_PREFIXES = ("model.language_model.", "language_model.", "model.")


def detect_text_prefix(raw_weights: Dict[str, Any]) -> str:
    """Returns the key prefix the text decoder lives under.

    Raises ValueError listing the candidates if none matches, rather than
    letting every subsequent lookup miss and produce an all-None param tree.
    """
    for prefix in _TEXT_PREFIXES:
        if f"{prefix}embed_tokens.weight" in raw_weights:
            return prefix
    sample = sorted(k for k in raw_weights if "embed_tokens" in k)[:5]
    raise ValueError(
        "Could not locate the text decoder in this checkpoint. Tried prefixes "
        f"{_TEXT_PREFIXES}; keys containing 'embed_tokens': {sample or 'none'}"
    )


def convert_safetensors_to_jax_params(
    raw_weights: Dict[str, Any],
    num_layers: int = 35,
    first_kv_shared_idx: int = 15,
    prefix: str = None,
    attention_k_eq_v: bool = False,
) -> Dict[str, Any]:
    """Converts a dict of safetensors numpy/jax arrays into Gemma4EModelJAX parameter PyTree.

    Handles key remapping for:
      - Primary and Per-Layer Embeddings (PLE)
      - Non-shared layers (0..14) with q/k/v/o attention projections
      - Shared layers (15..34) with q/o attention projections (no k/v)
      - MatFormer Double-Wide MLP weights
      - QAT packed W4A16 weights and scales (_packed, _scale suffix mapping)
      - attention_k_eq_v: layers that ship k_proj but no v_proj, because V is K

    attention_k_eq_v: set from the checkpoint config. True on gemma-4-31B, where
      the full-attention layers omit v_proj entirely; False on E2B.
    """
    if prefix is None:
        prefix = detect_text_prefix(raw_weights)

    jax_params: Dict[str, Any] = {}
    missing: list = []

    def get_arr(key: str, default_dtype=jnp.bfloat16):
        # bfloat16, NOT COMPUTE_DTYPE, and that is a known cost rather than an
        # oversight. On a pre-Ampere GPU the compute dtype is float16, so storing
        # bf16 makes XLA insert a convert in front of every use -- MEASURED
        # 2026-08-24 on a T4G as a 1.50 GiB `wrapped_convert` transient on the LM
        # head, and previously as 55% of decode time. Switching the default here
        # is correct and does NOT work yet: the cast has to happen somewhere, and
        # both places tried are worse than the convert. See
        # docs/bf16-weights-on-turing.md before changing it.
        if key in raw_weights:
            arr = raw_weights[key]
            if not isinstance(arr, jnp.ndarray):
                arr = jnp.array(arr)
            if jnp.issubdtype(arr.dtype, jnp.integer):
                return arr
            return arr.astype(default_dtype)
        return None

    def get_linear(key: str):
        """Convert HF ``[out, in]`` dense weights to JAX ``[in, out]``."""
        arr = get_arr(key)
        return None if arr is None else arr.T

    def get_quantized(prefix: str, destination: Dict[str, Any], name: str):
        packed_key = f"{prefix}.{name}.weight_packed"
        scale_key = f"{prefix}.{name}.weight_scale"
        if packed_key not in raw_weights:
            return False
        if scale_key not in raw_weights:
            raise ValueError(f"Missing W4A16 scale tensor: {scale_key}")
        packed = get_arr(packed_key)
        scale = get_arr(scale_key)
        if packed.dtype != jnp.int32:
            raise TypeError(
                f"{packed_key} must be compressed-tensors int32, got "
                f"{packed.dtype}"
            )
        if packed.ndim != 2 or scale.ndim != 2:
            raise ValueError(
                f"{packed_key}/{scale_key} must be rank 2, got "
                f"{packed.shape}/{scale.shape}"
            )
        expected_scale = (packed.shape[0], packed.shape[1] // 4)
        if scale.shape != expected_scale:
            raise ValueError(
                f"{scale_key} has shape {scale.shape}; expected "
                f"{expected_scale} for group_size=32"
            )
        destination[f"{name}_packed"] = packed
        destination[f"{name}_scale"] = scale.astype(jnp.bfloat16)
        return True

    # Global Embeddings & Norms
    jax_params["embed_tokens"] = get_arr(f"{prefix}embed_tokens.weight")
    jax_params["final_norm"] = get_arr(f"{prefix}norm.weight")

    if f"{prefix}embed_tokens_per_layer.weight" in raw_weights:
        jax_params["embed_tokens_per_layer"] = get_arr(f"{prefix}embed_tokens_per_layer.weight")
        if not get_quantized(f"{prefix.rstrip('.')}", jax_params, "per_layer_model_projection"):
            jax_params["per_layer_model_projection"] = get_linear(
                f"{prefix}per_layer_model_projection.weight")
        jax_params["per_layer_projection_norm"] = get_arr(f"{prefix}per_layer_projection_norm.weight")

    # Layer Weights Loop
    for i in range(num_layers):
        lp = f"{prefix}layers.{i}"
        is_shared = i >= first_kv_shared_idx
        layer_p: Dict[str, Any] = {}

        # Layernorms
        layer_p["input_layernorm"] = get_arr(f"{lp}.input_layernorm.weight")
        layer_p["post_attention_layernorm"] = get_arr(f"{lp}.post_attention_layernorm.weight")

        # PLE per-layer weights. Shipped QAT checkpoints store these W4A16-packed
        # (weight_packed/weight_scale), the same scheme vLLM's loader does not
        # implement for per_layer_model_projection — see tpu-inference #3225.
        for ple_name in ("per_layer_input_gate", "per_layer_projection"):
            if not get_quantized(lp, layer_p, ple_name):
                dense = f"{lp}.{ple_name}.weight"
                if dense in raw_weights:
                    layer_p[ple_name] = get_linear(dense)
        layer_p["post_per_layer_input_norm"] = get_arr(f"{lp}.post_per_layer_input_norm.weight")

        # Gemma sandwich norms around the feed-forward block.
        layer_p["pre_feedforward_layernorm"] = get_arr(f"{lp}.pre_feedforward_layernorm.weight")
        layer_p["post_feedforward_layernorm"] = get_arr(f"{lp}.post_feedforward_layernorm.weight")
        # Single learned scalar per layer. Applied by Gemma4EModelJAX to the whole
        # residual stream at the end of each decoder layer; see quirk 3.
        if f"{lp}.layer_scalar" in raw_weights:
            layer_p["layer_scalar"] = get_arr(f"{lp}.layer_scalar")

        # Attention weights
        attn_p: Dict[str, Any] = {}
        for proj in ["q_proj", "o_proj"]:
            proj_prefix = f"{lp}.self_attn"
            if not get_quantized(proj_prefix, attn_p, proj):
                dense_key = f"{proj_prefix}.{proj}.weight"
                if dense_key in raw_weights:
                    attn_p[proj] = get_linear(dense_key)

        if f"{lp}.self_attn.q_norm.weight" in raw_weights:
            attn_p["q_norm"] = get_arr(f"{lp}.self_attn.q_norm.weight")

        if not is_shared:
            for proj in ["k_proj", "v_proj"]:
                proj_prefix = f"{lp}.self_attn"
                if not get_quantized(proj_prefix, attn_p, proj):
                    dense_key = f"{proj_prefix}.{proj}.weight"
                    if dense_key in raw_weights:
                        attn_p[proj] = get_linear(dense_key)

            # attention_k_eq_v: the checkpoint ships no v_proj on these layers
            # because V *is* K — one projection feeds both. Verified on
            # gemma-4-31B, where every full-attention layer (i % 6 == 5) has
            # k_proj and k_norm but no v_proj at all, while sliding layers carry
            # both. E2B sets the flag False and ships v_proj everywhere.
            #
            # Aliasing rather than copying: these are the same arrays, so the
            # duplicate reference costs nothing. Storing K and V separately in
            # the cache is then redundant but correct; collapsing that is a
            # memory optimization for later, worth ~4.5% of the 31B's KV.
            if attention_k_eq_v and attn_p.get("v_proj") is None \
                    and "v_proj_packed" not in attn_p:
                if "k_proj_packed" in attn_p:
                    attn_p["v_proj_packed"] = attn_p["k_proj_packed"]
                    attn_p["v_proj_scale"] = attn_p["k_proj_scale"]
                elif attn_p.get("k_proj") is not None:
                    attn_p["v_proj"] = attn_p["k_proj"]

            if f"{lp}.self_attn.k_norm.weight" in raw_weights:
                attn_p["k_norm"] = get_arr(f"{lp}.self_attn.k_norm.weight")
            if f"{lp}.self_attn.v_norm.weight" in raw_weights:
                attn_p["v_norm"] = get_arr(f"{lp}.self_attn.v_norm.weight")

        layer_p["attn"] = attn_p

        # MLP weights
        mlp_p: Dict[str, Any] = {}
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            proj_prefix = f"{lp}.mlp"
            if not get_quantized(proj_prefix, mlp_p, proj):
                dense_key = f"{proj_prefix}.{proj}.weight"
                if dense_key in raw_weights:
                    mlp_p[proj] = get_linear(dense_key)

        layer_p["mlp"] = mlp_p
        if not mlp_p:
            missing.append(f"layer {i}: no MLP weights")
        if not attn_p:
            missing.append(f"layer {i}: no attention weights")
        jax_params[f"layer_{i}"] = layer_p

    # Fail loudly. A prefix or naming mismatch otherwise yields a tree of Nones
    # that loads "successfully" in seconds and holds zero bytes on device.
    for required in ("embed_tokens", "final_norm"):
        if jax_params.get(required) is None:
            missing.append(f"global tensor '{required}'")
    if "embed_tokens_per_layer" in jax_params and not (
        jax_params.get("per_layer_model_projection") is not None
        or "per_layer_model_projection_packed" in jax_params
    ):
        missing.append("per_layer_model_projection (dense or W4A16-packed)")
    if missing:
        raise ValueError(
            f"Checkpoint is missing required tensors under prefix '{prefix}': "
            + "; ".join(missing[:8])
            + (f" (+{len(missing) - 8} more)" if len(missing) > 8 else "")
        )
    return jax_params
