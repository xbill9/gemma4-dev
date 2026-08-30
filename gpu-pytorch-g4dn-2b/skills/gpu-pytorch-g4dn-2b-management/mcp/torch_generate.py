#!/usr/bin/env python3
"""One-shot generate against the local checkpoint, for smoke-testing the box.

    python3 torch_generate.py --prompt "Explain Turing tensor cores in one sentence."
    python3 torch_generate.py --max-new-tokens 64 --stats

Rewritten for CUDA rather than adapted from the TPU sibling: that version imports
`tpu_compiler` and hardcodes `device="tpu"` and bfloat16, none of which exist or
are correct here. What carries over is the shape of the thing, not the code.

This is deliberately NOT the serving path -- `torch_openai_server.py` is. It
exists because the JAX sibling's hardest bug was only separable by driving the
engine directly, outside HTTP: 20 tokens in 0.06 s in-process against 60-126 s
through the server localised the whole cost to process configuration, which no
amount of profiling the model would have found.
"""

import argparse
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def resolve_compute_dtype(device):
    """The dtype the DEVICE has, not the one you hoped for.

    Turing (SM 7.5) has no bf16 datapath. Asking for bfloat16 does not raise --
    CUDA emulates it through fp32 and you simply lose most of decode. Same guard
    as `torch_openai_server.py`; kept in both because either can be run alone.
    """
    import torch

    if device.type != "cuda":
        return torch.float32
    major, minor = torch.cuda.get_device_capability(device)
    dtype = torch.float16 if (major, minor) < (8, 0) else torch.bfloat16
    logging.info(
        "device policy: %s  sm_%d%d  pre_ampere=%s  compute_dtype=%s",
        torch.cuda.get_device_name(device), major, minor,
        (major, minor) < (8, 0), str(dtype).replace("torch.", ""),
    )
    return dtype


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E2B-it")
    p.add_argument("--prompt", default="Explain Turing tensor cores in one sentence.")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--stats", action="store_true", help="print decode tok/s")
    args = p.parse_args()

    import torch
    import transformers

    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device. On this rig that usually means the AMI carries no "
            "NVIDIA driver, or this is not the interpreter holding the DLAMI's "
            "torch. Check `verify_gpu_arch` before anything else."
        )

    device = torch.device("cuda")
    dtype = resolve_compute_dtype(device)

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    t0 = time.perf_counter()
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        .to(device)
        .eval()
    )
    logging.info("loaded in %.1f s", time.perf_counter() - t0)

    ids = tok(args.prompt, return_tensors="pt").to(device)

    # Warm up at the shape you measure. The JAX sibling recorded 18.77 s cold
    # against 4.35 s warm for the same request; torch has its own first-call
    # costs (autotune, allocator growth) and the same rule applies.
    with torch.inference_mode():
        model.generate(**ids, max_new_tokens=4, do_sample=False)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        out = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    print("\n--- output " + "-" * 49)
    print(text)
    if args.stats:
        n = out.shape[1] - ids["input_ids"].shape[1]
        print("-" * 60)
        print(f"{n} tokens in {elapsed:.2f} s  ->  {n / elapsed:.2f} tok/s (warm)")
        print(f"peak device memory: {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
