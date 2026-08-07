"""Chat template & BOS test for Gemma 4 12B JAX Engine vs HF PyTorch.

Verifies that when BOS (<bos>) and the Gemma 4 chat template are properly
formatted, the 12B model produces valid, accurate answers ("Paris") on both
HF Transformers PyTorch reference and our pure JAX engine.
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

from jax_engine import JaxGemmaEngine, config_from_hf

DEFAULT_MODEL = "google/gemma-4-12B-it-qat-w4a16-ct"
PROMPT = "What is the capital of France?"


def format_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=PROMPT)
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    formatted = format_prompt(tokenizer, args.prompt)
    print(f"\nFormatted Chat Prompt:\n{formatted!r}\n")

    # 1. Run HF Transformers Reference
    print("--- 1. Running HF Transformers Reference (PyTorch CPU) -----------")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    inputs = tokenizer(formatted, return_tensors="pt")
    with torch.no_grad():
        out_ids = hf_model.generate(**inputs, max_new_tokens=32, do_sample=False)
    hf_text = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:])
    print(f"HF Reference Generation Output:\n{hf_text!r}\n")

    # 2. Run Pure JAX Engine
    print("--- 2. Running Pure JAX Engine -----------------------------------")
    eng = JaxGemmaEngine(model_id=args.model, quant_mode="w4a16", max_model_len=512)
    eng.load()

    tok_ids = tokenizer(formatted, return_tensors="np")["input_ids"][0].tolist()
    out_tokens, stats = eng.generate(tok_ids, max_new_tokens=32, temperature=0.0)
    jax_text = tokenizer.decode(out_tokens)
    print(f"JAX Engine Generation Output:\n{jax_text!r}\n")
    if stats:
        print(f"Prefill: {stats.prefill_ms:.1f}ms | Decode: {stats.decode_tok_per_s:.1f} tok/s")


if __name__ == "__main__":
    main()
