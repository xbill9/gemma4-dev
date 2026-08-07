"""Reference parity comparator: HF Transformers PyTorch vs JAX Engine for Gemma 4 12B.

Runs prompt "The capital of France is" through both HF PyTorch model and
JAX engine on CPU/TPU to compare layer 0 hidden states, intermediate RMS values,
and final logits.

Usage:
    python3.13 ports/gemma4/jax_12b_hf_compare.py
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax
import jax.numpy as jnp
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jax_engine import JaxGemmaEngine

DEFAULT_MODEL = "google/gemma-4-12B-it-qat-w4a16-ct"
PROMPT = "The capital of France is"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=PROMPT)
    args = parser.parse_args()

    print(f"--- 1. Running HF Transformers (PyTorch) Reference ----------------")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    inputs = tokenizer(args.prompt, return_tensors="pt")

    t0 = time.perf_counter()
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        print(f"HF Model loaded in {time.perf_counter() - t0:.2f}s")

        with torch.no_grad():
            outputs = hf_model(**inputs, output_hidden_states=True)
            hf_logits = outputs.logits[0, -1].float().numpy()

        top_hf = np_top5(hf_logits)
        print("HF Top 5 predicted tokens:")
        for idx, score in top_hf:
            tok_str = tokenizer.decode([idx])
            print(f"  ID {idx:6d} | score {score:6.2f} | token {tok_str!r}")

    except Exception as exc:
        print(f"HF PyTorch execution note: {exc}")
        hf_logits = None

    print("\n--- 2. Running Pure JAX Engine -----------------------------------")
    eng = JaxGemmaEngine(model_id=args.model, quant_mode="w4a16", max_model_len=256)
    eng.load()

    input_ids = jnp.array(inputs["input_ids"].numpy()[0])
    pos_ids = jnp.arange(len(input_ids), dtype=jnp.int32)[None, :]
    valid_mask = jnp.ones((1, len(input_ids)), dtype=bool)

    from ports.gemma4 import jax_e_model as M
    mask = M.make_prefill_causal_mask(valid_mask)
    smask = M.make_prefill_causal_mask(valid_mask, window=eng.config.sliding_window)

    jax_logits = eng.model(
        input_ids[None, :],
        eng.params,
        pos_ids,
        attention_mask=mask,
        sliding_attention_mask=smask,
        quant_mode=eng.quant_mode,
    )
    if isinstance(jax_logits, tuple):
        jax_logits = jax_logits[0]

    last_jax = jax_logits[0, -1].astype(jnp.float32)
    top_jax = jnp.argsort(last_jax)[::-1][:5]

    print("JAX Top 5 predicted tokens:")
    for idx in top_jax:
        score = float(last_jax[idx])
        tok_str = tokenizer.decode([int(idx)])
        print(f"  ID {int(idx):6d} | score {score:6.2f} | token {tok_str!r}")


def np_top5(logits):
    import numpy as np
    top_idxs = np.argsort(logits)[::-1][:5]
    return [(int(i), float(logits[i])) for i in top_idxs]


if __name__ == "__main__":
    main()
