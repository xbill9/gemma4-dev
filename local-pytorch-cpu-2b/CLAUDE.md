# CLAUDE.md — local-pytorch-cpu-2b

Guidance for working inside this rig. Read before changing anything.

## What this rig is

**`transformers` + `torch` on the host CPU**, loading `google/gemma-4-E2B-it` in
bf16 from the HF cache. No engine of our own, no server framework: the model is
held in-process and `generate()` is called on it.

**STATUS 2026-09-04: IT SERVES.** First light the same day it was retargeted —
`benchmarks/runs/2026-09-04-first-light-pytorch-cpu/`.

## Read this first: what this directory was until 2026-09-04

**A verbatim copy of `gpu-jax-g4dn-2b`** — 58 tracked files describing an AWS
G4dn instance with an NVIDIA T4, shipping `jax_engine.py`, `ports/gemma4/`,
`boto3`, `init.sh` and a skill directory named `gpu-jax-g4dn-2b-management`.
**Three of the four name slots were claims nothing in the directory supported**,
and it contained no PyTorch at all. The skill name is the destructive kind:
`make skill-install` `rm -rf`s its destination.

`tests/test_server.py::TestNoCloudControlPlane` asserts that vocabulary is gone.
It splits the check by what each token means, and that split is load-bearing:

- **Cloud words** (`boto3`, `ec2`, `ami`, `spot`, …) are checked in identifiers
  **and string literals** — a region or an instance id hides in a string as
  easily as in a name.
- **Engine words** (`jax`, `jnp`, `flax`, `vllm`) are checked in **imports and
  attributes only, never strings**. This rig's prose legitimately names
  `local-jax-cpu-2b` as the sibling it is a control for, and a string scan turns
  that into a false `jax` hit — which it did on the first run. The invariant is
  "does not import or call another engine", which is a code question.

## MEASURED 2026-09-04 — first light

| | |
| :--- | ---: |
| load | **3.1 s** (warm page cache) |
| params | 5.104 B |
| generate 128 tok | 26.4 s = **4.853 tok/s** |
| repeat | 27.9 s = 4.587 tok/s (5.8% spread) |
| **peak RSS** | **5.71 GB** — identical across both runs |

Single end-to-end figure covering prefill and decode together; `generate()`
exposes no split. **Not comparable to a sibling's decode-only rate.**

## THE FINDING: mmap gives PyTorch the lazy PLE for free

**Peak RSS is 5.71 GB for a 10.25 GB checkpoint — 56% of the file.** That is not
a rounding artifact:

```
safetensors file                10.25 GB
minus embed_tokens_per_layer    -4.70 GB   (@MODELS.md, measured per-tensor)
= everything else                5.55 GB
MEASURED peak RSS                5.71 GB   delta +0.16 GB
```

**Essentially the entire model except the PLE gets touched, and essentially none
of the PLE does.** Safetensors mmaps the file; the PLE is an indexed gather, so
only the rows for tokens actually seen are ever paged in. transformers writes no
special code for this — the OS page cache does it.

**This is the third independent confirmation of the same story**, and the three
together are why this checkpoint behaves the way it does everywhere:

| stack | what happens to the 4.7 GB PLE | result |
| :--- | :--- | :--- |
| llama.cpp | `TENSOR_READ_LAZY` + mmap, never on device | 1612 MiB on a 4 GiB GPU |
| **transformers** | **mmap + gather, only touched rows resident** | **5.71 GB of 10.25** |
| vLLM | copied into a `VocabParallelEmbedding` at `__init__` | **OOM, asks for 4.38 GiB** |

The engine that fails is the one that materialises it eagerly. See
`@QUANTIZATION.md`, "vLLM CANNOT offload Gemma 4's PLE".

**Consequence for this rig: size from the FILE, never from RSS.** RSS right after
load reads 1.20 GB and would say a host that cannot hold the model is fine.
`check_host_capacity` sizes against `MODEL_SAFETENSORS_BYTES` for that reason,
and a test pins it.

## Why bf16, and why the GPU is ignored

**bf16 is a memory decision, not a speed one.** fp32 would be 20.5 GB and does
not fit this host at all. torch accepts bf16 on x86 regardless of AVX512-BF16 —
that flag governs whether there is a fast datapath, not whether the dtype is
legal. This CPU is AVX2-only, so bf16 is emulated and slow, and 4.85 tok/s is
what that costs.

**The box has a GPU and this rig deliberately does not use it.** The GTX 1650 Ti
has 4096 MiB against 10.25 GB of weights, and transformers has no way to keep the
PLE off the device the way llama.cpp does — `device_map="auto"` would spill most
of the model across PCIe and measure the link rather than either processor. A GPU
PyTorch rig here needs its own directory with slot 3 = `1650ti`. **Do not add a
device flag to this one**; a test asserts `DEVICE == "cpu"` and not `"auto"`.

Threads are pinned to **6**, the physical core count. The other six are SMT
siblings sharing execution units and a memory-bound decode gains nothing.

## Two API details that cost a run each

- **`apply_chat_template` returns a `BatchEncoding` in transformers 5.x**, not a
  bare tensor. `ids.shape` raises `KeyError: 'shape'` then `AttributeError`, which
  reads as a broken tokenizer. Use `return_dict=True` and pass `**enc` to
  `generate` so the attention mask goes with it.
- **`device_map` requires `accelerate`**, which is not a transformers dependency.
  Without it `from_pretrained` raises `ValueError` at load. It is in
  `requirements.txt`.

## Gemma 4 reasons — but not on this path

The llama.cpp and Ollama siblings both measured every sweep cell as pure thinking
at 128 tokens. **Here the same prompt at 128 tokens returned a real answer**
("Here are three generations of Google's Tensor Processing Units..."), with no
thinking block. That is the chat template doing it: this rig applies the
checkpoint's own `chat_template.jinja`, while Ollama substitutes its Go renderer
and llama.cpp was driven through a server that enables reasoning. **`MAX_NEW_TOKENS`
is still 1024** — do not assume the direct-answer behaviour holds for every prompt.

## Conventions

- Tests are `unittest`, never pytest: `python3 -m unittest discover -s tests -v`.
- Subprocess calls go through `run_command(cmd: list[str])` with
  `asyncio.create_subprocess_exec`. **Never `shell=True`.**
- MCP tools are `async def` returning markdown with `✅`/`❌`/`📡` prefixes.
- `Optional[str]`, not `X | None`.
- System `python3`, **never a virtualenv**.
- `tpu.env` is the source of truth and is committed.
- **Read `MemAvailable`, never `MemFree`.**
- **torch is unpinned in `requirements.txt` on purpose.** This host's torch is
  whatever the other local work last needed — 2.13.0+cu130 as of 2026-09-04,
  pinned there by vLLM 0.28.0, which requires `torch==2.13.0` exactly. Pinning
  here would silently fight that; the benchmark reports record what each run
  actually used.
- No `.claude-plugin/`, no `.codex/`, no `skills/`, matching the other `local` rigs.

## Canonical root references

`@MODELS.md` (checkpoint properties, the per-tensor PLE breakdown),
`@HARDWARE.md`, `@QUANTIZATION.md` (what each stack supports, and the vLLM PLE
failure), `@NAMING.md`, `@RIG-ANALYSIS.md`.
