#!/usr/bin/env python3.12
"""Run a stock Hugging Face causal LM on TPU via TorchTPU.

Copy to a workload="pytorch" TPU VM and run with the dedicated interpreter:

    python3.12 torchtpu_generate.py --prompt "Why are TPUs fast?"
    python3.12 torchtpu_generate.py --compile   # torch.compile(backend="tpu")

Needs `transformers` next to torch_tpu — create the VM with
TORCH_TPU_PIP_SPEC="torch_tpu transformers" or install it on the VM with the
same authenticated `pip install` the startup script logs.

Gated models (Gemma) need a Hugging Face token: set HF_TOKEN, or leave it unset
and the script fetches the `hf-token` Secret Manager secret via the VM's
metadata service account (same pattern as the vLLM startup script).

TorchTPU rules honored here: plain torch (no `import torch_tpu`, no torch_xla),
device="tpu", bfloat16, inputs padded to a multiple of 128, no prints or
`.item()` inside the compiled forward.
"""

import argparse
import base64
import json
import time
import urllib.request

METADATA = "http://metadata.google.internal/computeMetadata/v1"
HF_SECRET_ID = "hf-token"


def _metadata(path: str) -> str:
    req = urllib.request.Request(f"{METADATA}/{path}", headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(req, timeout=10) as res:
        return res.read().decode()


def hf_token_from_secret_manager() -> str | None:
    """Best-effort fetch of the hf-token secret using the VM's service account."""
    try:
        project = _metadata("project/project-id")
        access_token = json.loads(_metadata("instance/service-accounts/default/token"))["access_token"]
        url = (
            f"https://secretmanager.googleapis.com/v1/projects/{project}"
            f"/secrets/{HF_SECRET_ID}/versions/latest:access"
        )
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {access_token}"})
        with urllib.request.urlopen(req, timeout=10) as res:
            payload = json.load(res)["payload"]["data"]
        return base64.b64decode(payload).decode()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--prompt", default="Explain in two sentences why TPUs are fast.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--compile", action="store_true", help='torch.compile(backend="tpu") the forward')
    args = parser.parse_args()

    import os

    if not os.environ.get("HF_TOKEN"):
        token = hf_token_from_secret_manager()
        if token:
            os.environ["HF_TOKEN"] = token
            print("HF token loaded from Secret Manager.")

    import torch
    import transformers

    device = torch.device("tpu")
    print(f"Loading {args.model} in bfloat16...")
    t0 = time.monotonic()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    print(f"Loaded {model.num_parameters():,} params in {time.monotonic() - t0:.1f}s")

    if args.compile:
        model.forward = torch.compile(model.forward, backend="tpu")

    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding="longest",
        pad_to_multiple_of=128,  # tile the MXU cleanly
    ).to(device)
    prompt_tokens = inputs["input_ids"].shape[-1]

    if args.compile:
        print("Warmup generate (first compile is slow)...")
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=8, do_sample=False)

    print(f"Generating up to {args.max_new_tokens} tokens...")
    t0 = time.monotonic()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    elapsed = time.monotonic() - t0

    new_tokens = out.shape[-1] - prompt_tokens
    text = tokenizer.decode(out[0][prompt_tokens:].cpu(), skip_special_tokens=True)
    print("\n--- output " + "-" * 49)
    print(text.strip())
    print("-" * 60)
    print(
        f"prompt={prompt_tokens} tok, generated={new_tokens} tok in {elapsed:.1f}s "
        f"({new_tokens / elapsed:.1f} tok/s, compile={'on' if args.compile else 'off'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
