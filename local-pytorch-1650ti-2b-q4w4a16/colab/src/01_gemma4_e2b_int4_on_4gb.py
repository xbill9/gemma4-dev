# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Gemma 4 E2B in plain PyTorch on a 4 GB GPU
#
# **Author:** William McLean, Google Developer Expert
#
# **Based on:** the rig [`local-pytorch-1650ti-2b-q4w4a16`](https://github.com/xbill9/gemma4-dev/tree/main/local-pytorch-1650ti-2b-q4w4a16)
# in [`xbill9/gemma4-dev`](https://github.com/xbill9/gemma4-dev), measured on a laptop GTX 1650 Ti (4 GB), and
# Google's quantization-aware-trained (QAT) Gemma 4 E2B, repacked as
# [`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text)
# and [`-text-emb4`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4).
#
# ### What this notebook is for
#
# Running Gemma 4 E2B with plain PyTorch and Hugging Face Transformers inside the memory of a 4 GB GPU. A Colab T4
# has about 15 GB, so the notebook caps PyTorch at the 3,732 MiB a GTX 1650 Ti gives it, and every memory result
# below plays out as it does on that card.
#
# ### What each section shows
#
# * **Section 2, the default path.** Loading the model the usual way inside the 4 GB budget: in 16-bit, and as the
#   4-bit QAT checkpoint through Transformers' own quantization support.
# * **Section 3, the int4 engine.** Every weight and vocabulary table kept in 4-bit on the GPU, a 4-bit
#   matrix-vector kernel, a decode-attention kernel, and the decode step compiled and replayed as one CUDA graph.
#   Speed and memory for a few chat prompts.
# * **Section 4, long context.** A 32,768-token prompt in the same budget: prompt-read speed, decode speed with the
#   context held, and whether the model finds one record planted a quarter of the way in.
# * **Section 5, scorecard.** Your numbers beside the GTX 1650 Ti's.
#
# ### The rules used throughout
#
# 1. **Every result is measured here.** The 1650 Ti numbers appear beside yours as context, never in place of them.
# 2. **One memory budget.** PyTorch is capped at 3,732 MiB in every section.
# 3. **Greedy decoding.** Every reply is deterministic, so two runs of the same cell write the same text.
#
# ### Before you start
#
# Choose **Runtime → Change runtime type → T4 GPU**. No Hugging Face token is needed: every checkpoint used here is
# public. The downloads are about 8 GB.

# %% [markdown]
# ## 1. Setup
#
# Cell 1.1 checks the GPU and installs the library versions the 1650 Ti measurements used.

# %%
# Cell 1.1: GPU and libraries
import subprocess
import sys

print(
    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "transformers==5.16.1",
        "compressed-tensors==0.17.0",
        "accelerate==1.14.0",
    ],
    check=True,
)
import importlib.metadata as md

for package in ("torch", "triton", "transformers", "compressed-tensors", "accelerate"):
    print(f"{package:20s} {md.version(package)}")

# %% [markdown]
# Cell 1.2 fetches the engine, four Python files, from the rig at a fixed commit, and downloads the two checkpoints.

# %%
# Cell 1.2: engine code and checkpoints
import pathlib
import urllib.request

ENGINE_REF = "a836a6aa13d6102edd75025cb1c148be058c58d6"
RIG = f"https://raw.githubusercontent.com/xbill9/gemma4-dev/{ENGINE_REF}/local-pytorch-1650ti-2b-q4w4a16"
for name in ("e2b_engine.py", "w4gemv.py", "w4attn.py", "gemma4_tools.py"):
    pathlib.Path(name).write_bytes(urllib.request.urlopen(f"{RIG}/{name}").read())
    print("fetched", name)

from huggingface_hub import snapshot_download

TEXT = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
EMB4 = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4"
for repo, patterns in ((TEXT, None), (EMB4, ["model-rest-00[01].safetensors", "*.json"])):
    d = pathlib.Path(snapshot_download(repo, allow_patterns=patterns))
    print(f"{repo}: {sum(f.stat().st_size for f in d.rglob('*.safetensors')) / 2**30:.2f} GiB")

# %% [markdown]
# Cell 1.3 sets the budget. `torch.cuda.set_per_process_memory_fraction` makes PyTorch's allocator refuse to hold
# more than a fraction of the card; the fraction here is the 3,732 MiB a GTX 1650 Ti shows PyTorch. Every later
# section, and every helper process, applies the same cap.

# %%
# Cell 1.3: the 4 GB budget
import torch

BUDGET_MIB = 3732
TOTAL_MIB = torch.cuda.get_device_properties(0).total_memory // 2**20
FRACTION = BUDGET_MIB / TOTAL_MIB
print(f"{torch.cuda.get_device_name(0)}: {TOTAL_MIB} MiB; PyTorch capped at {BUDGET_MIB} MiB ({FRACTION:.3f})")

# %% [markdown]
# ## 2. The default path
#
# Two ways a reader would normally load Gemma 4 E2B with Transformers, each in a fresh process under the cap.
#
# **What the 1650 Ti showed.** Unpacked to 16-bit, the decoder layers alone are 3.58 GiB and all the weights
# 8.73 GiB, more than the card. Loaded as the 4-bit checkpoint, the weights fit (1.73 GiB, with the 4.38 GiB
# per-layer table left in CPU memory), but Transformers 5.16 has no 4-bit compute path: the first forward pass
# unpacks every layer to 16-bit and runs out of memory.

# %%
# Cell 2.1: the helper script each attempt runs in
TASKS = pathlib.Path("default_path.py")
TASKS.write_text(r"""
import json, sys, time, torch, transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
torch.cuda.set_per_process_memory_fraction(float(sys.argv[2]))
TEXT, TASK = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text", sys.argv[1]
KEY = {r"^model\.language_model\.": "model."}
out = {"task": TASK}
try:
    if TASK == "fp16":
        # The 4-bit weights unpacked to 16-bit at load: the size of the ordinary 16-bit model.
        m = AutoModelForCausalLM.from_pretrained(TEXT, dtype=torch.float16, key_mapping=KEY, device_map="cuda",
            quantization_config=transformers.CompressedTensorsConfig(run_compressed=False))
        out["loaded_mib"] = torch.cuda.memory_allocated() // 2**20
    else:
        # The 4-bit checkpoint as Transformers loads it; the 4.38 GiB per-layer table kept in CPU memory.
        with torch.device("meta"):
            meta = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(TEXT))
        dm = {f"model.{n}": 0 for n, _ in meta.model.named_children()}
        dm["model.embed_tokens_per_layer"], dm["lm_head"] = "cpu", 0
        m = AutoModelForCausalLM.from_pretrained(TEXT, dtype=torch.float16, key_mapping=KEY, device_map=dm,
            quantization_config=transformers.CompressedTensorsConfig(run_compressed=True))
        out["loaded_mib"] = torch.cuda.memory_allocated() // 2**20
        tok = AutoTokenizer.from_pretrained(TEXT)
        ids = tok.apply_chat_template([{"role": "user", "content": "Hi"}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)["input_ids"].cuda()
        with torch.inference_mode():
            m(input_ids=ids)
        out["forward"] = "ok"
    out["result"] = "fits"
except torch.OutOfMemoryError as e:
    out["result"] = "out of memory"
    out["detail"] = str(e).split(". ")[0][:160]
out["peak_mib"] = torch.cuda.max_memory_allocated() // 2**20
print(json.dumps(out))
""")
print("wrote", TASKS)

# %%
# Cell 2.2: run both
import json


def run_task(task):
    p = subprocess.run([sys.executable, str(TASKS), task, str(FRACTION)], capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if l.startswith("{")]
    return json.loads(line[-1]) if line else {"task": task, "result": "crashed", "detail": p.stderr[-300:]}


default_path = {t: run_task(t) for t in ("fp16", "int4_transformers")}
for r in default_path.values():
    print(json.dumps(r))

# %% [markdown]
# **What you should see.** Both attempts end in `out of memory` inside the 3,732 MiB budget: the 16-bit weights at
# load, and the 4-bit checkpoint at its first forward pass, when Transformers unpacks it.

# %% [markdown]
# ## 3. The int4 engine
#
# `e2b_engine.Engine` keeps the model 4-bit on the GPU end to end:
#
# * the decoder's linear layers stay packed, and a Triton kernel multiplies straight from the packed weights;
# * `embed_tokens`, the per-layer table and `lm_head` come from `-text-emb4`, also 4-bit, on the same QAT grid;
# * a Triton decode-attention kernel reads the cache once for all eight query heads;
# * the decode step is compiled with `torch.compile` and replayed as one CUDA graph.
#
# Cell 3.1 decides how prompts are multiplied on this GPU. On a GTX 1650 Ti, which has no tensor cores, 16-bit
# matrix multiplies ran at about a quarter of the 32-bit rate, so the engine reads prompts in 32-bit there. A T4 has
# tensor cores, so the cell times both and keeps the faster.

# %%
# Cell 3.1: 16-bit or 32-bit multiplies for prompt reads, timed on this GPU
import time

torch.cuda.set_per_process_memory_fraction(FRACTION)


def tflops(dtype, m=256, k=1536, n=24576, reps=10):
    x, w = torch.randn(m, k, device="cuda", dtype=dtype), torch.randn(n, k, device="cuda", dtype=dtype)
    torch.nn.functional.linear(x, w)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        torch.nn.functional.linear(x, w)
    torch.cuda.synchronize()
    return 2 * m * k * n * reps / (time.perf_counter() - t) / 1e12


rates = {dt: tflops(dt) for dt in (torch.float16, torch.float32)}
PREFILL_DTYPE = max(rates, key=rates.get)
torch.cuda.empty_cache()
print({str(k): round(v, 2) for k, v in rates.items()}, "->", PREFILL_DTYPE)

# %%
# Cell 3.2: load the engine (about two minutes: downloads are cached, the decode step compiles)
import logging

from e2b_engine import Engine

# Older torch builds print a long, harmless warning while tracing the attention kernel; keep the output readable.
logging.getLogger("torch._dynamo").addFilter(
    lambda record: not record.getMessage().startswith("Encountered an exception in identify_")
)

t0 = time.perf_counter()
eng = Engine(max_ctx=32768, prefill_dtype=PREFILL_DTYPE)
print(
    f"loaded in {time.perf_counter() - t0:.0f} s; weights {eng.weights_mib} MiB, K/V cache {eng.kv_mib} MiB, "
    f"allocated {torch.cuda.memory_allocated() // 2**20} MiB of {BUDGET_MIB}"
)

# %%
# Cell 3.3: chat
PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Write a Python function that reverses a singly linked list.",
    "A train leaves at 9:40 and arrives at 13:15. How long is the trip in minutes? Show the steps.",
]
torch.cuda.reset_peak_memory_stats()
chat = []
for p in PROMPTS:
    r = eng.generate(eng.render([{"role": "user", "content": p}]), max_new=128)
    chat.append(
        {
            "prompt": p,
            "tokens": len(r["tokens"]),
            "decode_tok_per_s": round((len(r["tokens"]) - 1) / r["decode_s"], 1),
            "text": eng.tok.decode(r["tokens"], skip_special_tokens=True),
        }
    )
    print(f"--- {p}\n{chat[-1]['text']}\n[{chat[-1]['tokens']} tokens, {chat[-1]['decode_tok_per_s']} tok/s]\n")
print(f"peak {torch.cuda.max_memory_allocated() // 2**20} MiB of {BUDGET_MIB}")

# %% [markdown]
# **What you should see.** Three sensible replies, decode in the tens of tokens per second, and a peak well inside
# the budget. On the GTX 1650 Ti the same engine decoded at 87.8 tok/s and llama.cpp at 82.1.

# %% [markdown]
# ## 4. Long context
#
# Gemma 4 E2B keeps a full-length cache in only 3 of its layers; 12 more cache a 512-token window, and the last 20
# reuse those. That is 2 KiB per token per full layer, so 32,768 tokens cost 192 MiB. The engine above was built
# with room for 32,768.
#
# Cell 4.1 fills the context with numbered records, each with a random six-digit code, and asks for the code of the
# record a quarter of the way in.

# %%
# Cell 4.1: a 32k-token recall prompt
import random


def recall_prompt(depth, seed=1234):
    rng = random.Random(seed)
    q = "\n\nQuestion: what is the code for record {r:06d}? Answer with the code only."
    lines, codes, used, i = [], {}, 80, 0
    while True:
        i += 1
        code = rng.randrange(100000, 999999)
        line = f"Record {i:06d}: the code for record {i:06d} is {code}.\n"
        used += len(eng.tok(line, add_special_tokens=False)["input_ids"])
        if used > depth:
            break
        lines.append(line)
        codes[i] = code
    target = len(lines) // 4
    ids = eng.render([{"role": "user", "content": "".join(lines) + q.format(r=target)}])
    return ids, codes[target], len(lines)


ids, code, n_records = recall_prompt(32768 - 200)
torch.cuda.reset_peak_memory_stats()
r = eng.generate(ids, max_new=64, stop_ids=[])  # decode all 64 tokens, for a steady speed
answer = eng.tok.decode(r["tokens"], skip_special_tokens=True).split("\n")[0].strip()
long_ctx = {
    "prompt_tokens": ids.shape[1],
    "records": n_records,
    "prompt_read_tok_per_s": round(ids.shape[1] / r["prompt_s"], 1),
    "decode_tok_per_s": round((len(r["tokens"]) - 1) / r["decode_s"], 1),
    "expected": code,
    "answer": answer[:40],
    "peak_mib": torch.cuda.max_memory_allocated() // 2**20,
}
print(json.dumps(long_ctx))

# %% [markdown]
# **What you should see.** The prompt fits and decodes inside the budget. Whether the answer matches is the model's
# own long-context recall. On the GTX 1650 Ti this exact prompt (seed 1234, 32,491 tokens, 1,082 records) found the
# code. Of two other 32k prompts in the rig's long-context run, both engines missed one and this engine found the
# other: finding one of a thousand near-identical records is at the edge of what E2B does.

# %% [markdown]
# ## 5. Scorecard

# The GTX 1650 Ti column is this notebook run on that card (driver 615.71.10, torch 2.13.0, triton 3.7.1,
# transformers 5.16.1).

# %%
# Cell 5.1: your numbers beside the GTX 1650 Ti's
rows = [
    ("16-bit weights in a 3,732 MiB budget", default_path["fp16"]["result"], "out of memory"),
    ("Transformers' own 4-bit path, first forward", default_path["int4_transformers"]["result"], "out of memory"),
    ("int4 engine, weights + 32k cache (MiB)", f"{eng.weights_mib} + {eng.kv_mib}", "2499 + 198"),
    (
        "decode, short chat (tok/s)",
        f"{min(c['decode_tok_per_s'] for c in chat)}-{max(c['decode_tok_per_s'] for c in chat)}",
        "86.1-87.5",
    ),
    ("prompt read, 32k tokens (tok/s)", long_ctx["prompt_read_tok_per_s"], "147.2 (32-bit multiplies)"),
    ("decode with 32k held (tok/s)", long_ctx["decode_tok_per_s"], "50.8"),
    ("peak memory, 32k (MiB)", long_ctx["peak_mib"], "2908"),
    ("recall, 32k", "found" if long_ctx["answer"] == str(long_ctx["expected"]) else "missed", "found"),
]
print(f"{'':48s} {'here':>22s} {'GTX 1650 Ti':>28s}")
for name, here, ref in rows:
    print(f"{name:48s} {str(here):>22s} {str(ref):>28s}")

# %% [markdown]
# **So, which one?** On a 4 GB GPU, Gemma 4 E2B runs from plain PyTorch only as 4-bit weights end to end, with the
# 4-bit arithmetic done by kernels that read the packed weights. The QAT checkpoint makes the 4-bit weights the
# model Google trained, so nothing is rounded twice.
#
# ## 6. Your notes
#
# Record your GPU, the library versions from cell 1.1 and the scorecard here.
