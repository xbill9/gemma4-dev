#!/usr/bin/env python3
"""First light for local-pytorch-cpu-2b: load, generate, report.

Reads its configuration from tpu.env so the run and the rig cannot drift.

WHY IT REPORTS RSS AT TWO POINTS. Safetensors mmaps the checkpoint, so RSS right
after load reads ~1.2 GB for a 10.25 GB file and would suggest the model is tiny.
The number that matters is the peak AFTER generation, once the weights have
actually been walked.
"""

import json
import os
import resource
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

RIG = Path(__file__).resolve().parent
load_dotenv(RIG / "tpu.env")

MODEL = os.environ.get("MODEL_NAME", "google/gemma-4-E2B-it")
DTYPE = getattr(torch, os.environ.get("DTYPE", "bfloat16"))
THREADS = int(os.environ.get("TORCH_NUM_THREADS", "6"))
NEW_TOKENS = int(os.environ.get("FIRST_LIGHT_TOKENS", "128"))
PROMPT = "Name three TPU generations."


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9


def mem_avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) * 1024 / 1e9
    return 0.0


def main():
    torch.set_num_threads(THREADS)
    stamp = lambda: time.strftime("%H:%M:%S")  # noqa: E731
    print(f"[{stamp()}] loading {MODEL} {DTYPE} on cpu, threads={torch.get_num_threads()}", flush=True)

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="cpu")
    model.eval()
    load_s = time.perf_counter() - t0
    params = sum(p.numel() for p in model.parameters())
    rss_after_load = rss_gb()
    print(f"[{stamp()}] LOADED in {load_s:.1f}s | params {params/1e9:.3f}B | "
          f"RSS {rss_after_load:.2f} GB (mmap — NOT the footprint) | "
          f"MemAvailable {mem_avail_gb():.2f} GB", flush=True)

    # transformers 5.x returns a BatchEncoding here, not a bare tensor.
    enc = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                  add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    n_in = enc["input_ids"].shape[-1]
    print(f"[{stamp()}] prompt {n_in} tok; generating {NEW_TOKENS}...", flush=True)

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=NEW_TOKENS, do_sample=False)
    gen_s = time.perf_counter() - t0
    new = out.shape[-1] - n_in
    text = tok.decode(out[0][n_in:], skip_special_tokens=True)
    peak = rss_gb()

    print(f"[{stamp()}] GENERATED {new} tok in {gen_s:.1f}s = {new/gen_s:.3f} tok/s", flush=True)
    print(f"  peak RSS {peak:.2f} GB | MemAvailable {mem_avail_gb():.2f} GB", flush=True)
    print(f"  text_head: {text[:160]!r}", flush=True)

    (RIG / "run").mkdir(exist_ok=True)
    json.dump({
        "model": MODEL, "dtype": str(DTYPE), "threads": THREADS,
        "params_b": round(params / 1e9, 4),
        "load_s": round(load_s, 2),
        "rss_after_load_gb": round(rss_after_load, 3),
        "peak_rss_gb": round(peak, 3),
        "prompt_tokens": n_in, "new_tokens": new,
        "generate_s": round(gen_s, 3),
        "tok_per_s": round(new / gen_s, 4),
        "text_head": text[:200],
    }, open(RIG / "run" / "first_light.json", "w"), indent=1)
    print(f"  wrote {RIG / 'run' / 'first_light.json'}", flush=True)


if __name__ == "__main__":
    main()
