#!/usr/bin/env python3.12
"""Compiled static-shape greedy decode benchmark for TorchTPU.

HF generate() feeds CPU scalars into the graph (unsupported by TorchTPU's
compiled mode), so this measures torch.compile(backend="tpu") the way the
backend is designed: one static [1, SEQ] buffer, every tensor on device="tpu",
no KV cache (full re-forward per step), no graph breaks, single compile.
"""

import argparse
import base64
import json
import os
import time
import urllib.request

METADATA = "http://metadata.google.internal/computeMetadata/v1"
WARMUP_STEPS = 4
BENCH_STEPS = 64
PROMPT = "Explain in two sentences why TPUs are fast."

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="google/gemma-4-E2B-it")
parser.add_argument("--seq", type=int, default=256, help="static buffer length (512 enables the fused SDPA kernel)")
parser.add_argument("--batch", type=int, default=1, help="lockstep batch size (static multi-user)")
ARGS = parser.parse_args()
SEQ = ARGS.seq
MODEL_ID = ARGS.model
BATCH = ARGS.batch


def _metadata(path):
    req = urllib.request.Request(f"{METADATA}/{path}", headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(req, timeout=10) as res:
        return res.read().decode()


def load_hf_token():
    if os.environ.get("HF_TOKEN"):
        return
    try:
        project = _metadata("project/project-id")
        tok = json.loads(_metadata("instance/service-accounts/default/token"))["access_token"]
        url = f"https://secretmanager.googleapis.com/v1/projects/{project}/secrets/hf-token/versions/latest:access"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
        with urllib.request.urlopen(req, timeout=10) as res:
            os.environ["HF_TOKEN"] = base64.b64decode(json.load(res)["payload"]["data"]).decode()
        print("HF token loaded from Secret Manager.")
    except Exception:
        pass


def main():
    load_hf_token()
    import torch
    import transformers

    device = torch.device("tpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    t0 = time.monotonic()
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    print(f"Loaded in {time.monotonic() - t0:.1f}s")

    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )["input_ids"]
    n_prompt = ids.shape[1]
    assert n_prompt + WARMUP_STEPS + BENCH_STEPS < SEQ

    pad_id = tokenizer.pad_token_id or 0
    tokens = torch.full((BATCH, SEQ), pad_id, dtype=torch.long)
    tokens[:, :n_prompt] = ids[0]
    tokens = tokens.to(device)
    mask = torch.zeros((BATCH, SEQ), dtype=torch.long)
    mask[:, :n_prompt] = 1
    mask = mask.to(device)
    one = torch.ones((BATCH, 1), dtype=torch.long).to(device)

    def step(tokens, mask, last_idx):
        logits = model(input_ids=tokens, attention_mask=mask, use_cache=False).logits
        return logits.index_select(1, last_idx).argmax(-1)  # [1, 1]

    step_c = torch.compile(step, backend="tpu")

    pos = torch.tensor([n_prompt], dtype=torch.long).to(device)  # next write position

    def decode_one():
        nonlocal pos, tokens, mask
        nxt = step_c(tokens, mask, pos - 1)
        tokens = tokens.index_copy(1, pos, nxt)
        mask = mask.index_copy(1, pos, one)
        pos = pos + 1

    print(f"Warmup ({WARMUP_STEPS} steps, includes compile)...")
    t0 = time.monotonic()
    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            decode_one()
        tokens[0, :1].cpu()  # materialize
    print(f"Warmup done in {time.monotonic() - t0:.1f}s")

    print(f"Benchmarking {BENCH_STEPS} decode steps...")
    t0 = time.monotonic()
    with torch.no_grad():
        for _ in range(BENCH_STEPS):
            decode_one()
        tokens[0, :1].cpu()  # materialize before stopping the clock
    elapsed = time.monotonic() - t0

    n_total = n_prompt + WARMUP_STEPS + BENCH_STEPS
    text = tokenizer.decode(tokens[0, n_prompt:n_total].cpu(), skip_special_tokens=True)
    print("\n--- output " + "-" * 49)
    print(text.strip())
    print("-" * 60)
    agg = BATCH * BENCH_STEPS / elapsed
    print(
        f"compiled static-shape decode (SEQ={SEQ}, batch={BATCH}, no KV cache): "
        f"{BENCH_STEPS} steps in {elapsed:.2f}s = {BENCH_STEPS / elapsed:.1f} tok/s/stream, "
        f"{agg:.1f} tok/s aggregate ({1000 * elapsed / BENCH_STEPS:.1f} ms/step)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
