# How to analyse a rig

**Method, not facts.** The four canonical files hold facts by axis — `MODELS.md` the checkpoints,
`HARDWARE.md` the chips, `QUANTIZATION.md` the stack, `NAMING.md` the spelling. This file is the
order you consult them in, and the questions that decide whether a rig can work *before* anyone
provisions capacity.

Every claim below is measured somewhere in this monorepo and cites where. Nothing here is a
prediction unless it says so.

---

## The decision order

**The order matters more than any individual item**, because each step can kill the plan and they
get progressively more expensive to discover. Steps 1–5 are arithmetic and cost nothing; step 9
costs a provisioning cycle.

| # | Question | Kills the plan if | Source |
| ---: | :--- | :--- | :--- |
| 1 | What does the chip compute **natively**? | you were counting on a format it lacks | `HARDWARE.md` |
| 2 | Does the **compute dtype** match the chip? | no — and it will not error, it will emulate | below |
| 3 | What is the **largest contiguous block**, not free bytes? | fragmentation eats the headroom | below |
| 4 | What does the model **weigh**, as a range? | the range does not fit | `MODELS.md` |
| 5 | What are the **transients**? | they exceed budget while weights fit | below |
| 6 | Is there a **fused kernel** for that format on that chip? | no — quantization buys memory only | `QUANTIZATION.md` |
| 7 | Is the **software path reachable**? | the route is dead in this stack | `QUANTIZATION.md` |
| 8 | Does anything **shard**? | you assumed multi-chip and the engine is single-device | below |
| 9 | Can you **provision it long enough to measure**? | capacity or quota says no | below |

---

## 1. Native numeric format support

`HARDWARE.md` holds the table. The short version, because it decides everything downstream:

| Target | bf16 | fp16 | int8 | fp8 |
| :--- | :---: | :---: | :---: | :---: |
| v5e, v6e | yes | — | **yes, 2x bf16** | **no** |
| v7 / Ironwood | yes | — | yes | **yes** |
| L4 (SM 8.9) | yes | yes | yes | yes |
| **T4G (SM 7.5)** | **no** | **yes** | yes (unused) | **no** |

**Read this as "what has a compute path", not "what can be stored".** A format with no MXU/tensor-core
path still buys footprint and bandwidth — that is the entire case for 4-bit on a 12B — but it will
never buy FLOPS.

## 2. Compute dtype match — the dominant driver, and not a quantization decision

**This is the largest single lever in the repo and it is routinely mistaken for a quantization
question.** It is not: it is whether the weights are *stored* in the dtype the device *computes* in.

MEASURED on a T4G (`gpu-jax-g5g-2b/benchmarks/runs/2026-08-27-baseline-xprof-g5g/`) — weights stored
bf16, device computes float16:

```
dtype conversion   39.60 ms/step   54.0%     computes nothing
fp32 GEMV          24.08 ms/step   32.8%     what the converts leave behind
                                   ------
                                    86.8%    of decode
```

**86.8% of decode goes to dtype work.** Larger than any quantization decision would buy.

**But do not assume the storage dtype is the cause — on this rig it was not.** Converting the whole
tree to float16 was tested on 2026-08-28: the kernels came back identical to the microsecond, for
**+0.0%**. At `B=1` decode is a matrix-*vector* product and cuBLAS's GEMV has **no half-precision
path** (`is_op_tensor_core_eligible = False`), so the weights are promoted to fp32 regardless. Those
converts were promotions *to* fp32, not a bf16→float16 fixup.

**The driver is still real and still first**, because bf16 on a pre-Ampere GPU does not *fail* — it
emulates through fp32, so the numbers come out right and every matmul quietly pays. The lesson is that
**a large dtype cost can have a cause other than the dtype**: read the kernel signature before choosing
a remedy, or you will spend three attempts on the wrong one.

**Resolve it from the live device, not from config.** `ports/gemma4/jax_e_model.py` reads the compute
capability and picks; that is the pattern to copy. A `DTYPE=` in an env file is an override, not the
decision.

## 3. Memory budget — the largest contiguous block

**Never plan against free bytes.** Allocator fragmentation measured **0.661** at peak on a T4G with
2.9 GiB free — and two of three quantization bugs on that rig failed to allocate with GBs nominally
available.

Two further corrections that are easy to miss:

- **Nominal ≠ usable.** T4G is "16 GB" and measures **15360 MiB**; the serving budget after the
  runtime reservation is **14.07 GB**. `HARDWARE.md` has the equivalent for each TPU.
- **A bigger instance may buy host RAM and no device memory.** `g5g.16xlarge` and `g5g.metal` carry
  two T4Gs and **nothing shards across them** — see driver 8.

## 4. Weight footprint — a range, never a point

`MODELS.md` has the table. Use it as **bounds in opposite directions**, because both columns are
wrong and they are wrong the other way from each other:

- **bf16 OVER-predicts by 6–9%** (E2B: 8.97 GiB on v5e-1, 9.257 GB on T4G, against 10.2 GB tabled).
- **int4 UNDER-predicts by ~19%** (E2B at `ple_bits=4`: **3.054 GB measured against 2.4 GiB = 2.58 GB
  tabled**), because `embed_tokens` stays at the storage dtype and the per-group scales are extra.
  **The int8/int4 columns quarter the GiB column, not the GB one** — mixing the two turns a 19% error
  into an apparent 27% one.

**Treat bf16 as a ceiling and int4 as a floor.** A plan that survives both is safe; one that needs the
int4 figure to be exact is not.

**And the `E` prefix is load-bearing** — E4B is 4.5B effective / 8.0B total, not 4B. Reading it as
"4B" understates weights by ~2x, which `MODELS.md` notes is exactly the difference between fitting a
16 GB part and not.

## 5. Transients — what actually binds as models grow

**The weight table invites you to plan against residency alone. That is the mistake.** MEASURED on a
T4G with a 14.07 GB budget:

| Model | Weights | Serves? | What actually failed |
| :--- | ---: | :--- | :--- |
| E2B `ple4` | 3.05 GB | yes | — |
| E4B | fits | **no** | OOM **5.25 GiB during load** |
| **12B** | **8.15 GB — fits easily** | **no** | OOM **12.61 GiB per request** |

Neither failure was residency. Three shapes of transient to budget separately:

- **Load-time.** Quantization itself is memory-hungry while it runs — `quantize_ple_table` upcasts to
  float32 and needs **>15 GiB host RSS** on E2B, and the destination is allocated before the source is
  released unless explicitly told otherwise. **Peak-during-load sets the floor, not steady state.**
- **Per-request, flat.** A term that does not scale with context — dtype conversions of whole weight
  tensors show up here.
- **Per-request, linear.** ~**0.9 MiB/token** above ~4K on E2B. A context limit derived from KV
  arithmetic alone will be wrong: that rig advertised `MAX_MODEL_LEN=8192` while **5,120 tokens
  OOMed**, and was lowered to 4096.

## 6. Fused kernel availability — the exception that does not travel

Quantization buys residency and bandwidth; it buys FLOPS only where a **fused** kernel exists.

- **W4A16 Pallas is fused on TPU** (16 MB VMEM per core) and **refused at startup on Turing** — it
  needs 550 KiB–1.1 MiB of shared memory per block against a **64 KiB** ceiling. Same checkpoint, same
  code, opposite economics.
- **Without a fused kernel it is dequantize-then-matmul.** Measured twice on one rig, same shape both
  times: `ple_bits=4` hit its memory prediction to **0.1%** and moved throughput **0.0%**;
  `int8_lm_head` is byte-exact on memory and worth **+2.3%**, because it dequantizes the full table to
  fp16 every step and never touches the chip's int8 tensor cores.

**Corollary: pick the coarsest quantization that fits.** Finer is not faster — it is more unpack work
in the hot path for the same matmul.

## 7. Software path reachability

`QUANTIZATION.md` is authoritative. The trap is that a route can be *documented*, *accepted at the
CLI*, and *dead*. Two live examples:

- Gemma 4 is **JAX-path only**, so most vLLM quantization routes are unreachable regardless of what
  the flags allow; qwix is the way in.
- A remedy can be **structurally gated**. `PREFILL_CHUNK_SIZE` exists precisely to bound prefill
  temporaries and raises `prefill_chunk_size requires window_kv=False` — while `window_kv`
  auto-resolves to True at any `max_model_len > sliding_window`. The one mitigation for the ceiling in
  driver 5 is behind a flag that is untested.

## 8. Sharding — check, do not assume

A rig with two accelerators is not a rig with twice the memory unless something shards.
`gpu-jax-g5g-2b` is single-device (`jax.devices()[0]`), emits no tensor-parallel flag, and the second
T4G on `16xlarge`/`metal` **idles**. Its `_tensor_parallel_size` reports a GPU count that nothing acts
on.

**Ask: does the engine shard, and does the model's KV geometry permit it?** `MODELS.md` records that a
single KV head does not shard — a constraint of the checkpoint, not the runtime.

## 9. Provisioning — can you measure it at all

An analysis that cannot be verified is a prediction. Two properties to establish early:

- **Capacity and quota**, per zone/AZ, for the exact shape. G5g spot was exhausted in three of four
  us-east-1 AZs on 2026-08-27; only one had any. **Spot price is not a proxy for availability** — the
  one AZ with capacity was the most expensive.
- **Expected lifetime, as a range.** Reclamation on the same instance type in the same region ranged
  from **21 minutes** to **19.2 hours**. Neither is typical. **Checkpoint continuously** rather than
  sizing work to an assumed lifetime.

---

## The verification rule

**A flag being accepted is not evidence it did anything.** The canonical case: an fp8 KV flag was
accepted at the CLI, echoed in `non-default args`, praised in an engine log line, reported in
`/metrics` as `cache_dtype="fp8_e4m3"`, and allocated a genuinely `float8_e4m3fn` tensor — **five
independent signals** — while delivering a **1.000x** capacity ratio.

So, in order of strength:

1. **Cross-check against an absolute physical bound**, never against another config. A decode ceiling
   is **measured bandwidth ÷ bytes STREAMED per token**. Get both terms right — this rule was
   misapplied on 2026-08-29 and produced a confident, wrong conclusion that stood for a day:

   - **Streamed bytes, not resident bytes.** On the `E` sizes most of the checkpoint is a
     **per-layer-embedding table that decode gathers rather than multiplies**, so it is resident and
     never streamed. E2B is 10.209 GB resident but **4.514 GB streamed** — a 2.3x error if you use
     the wrong one. **Reconcile your parameter count against a measured resident figure before
     trusting it**; E2B's `use_double_wide_mlp` doubles the MLP on 20 of 35 layers and is easy to
     miss, and only the reconciliation catches it (9.212 GB computed vs 9.257 GB measured). `MODELS.md` §"Resident is not streamed" has the split per size.
   - **Measured bandwidth, not theoretical peak.** `HARDWARE.md` records both and says which to quote
     (T4G: 277 GB/s measured, 320.1 peak).
   - **If a measurement beats the bound, suspect the bound first.** The original wording said the
     measurement is wrong. That is the less likely case and it is the expensive one to assume: a
     sibling rig's legitimate 31.8 tok/s was briefly read as physically impossible because the bound
     had been built from resident bytes over peak bandwidth. **A violated bound is a hypothesis to
     test, not a verdict** — recompute it before disbelieving the number.
2. **Read the allocation log**, not the flag. If the line never prints, that is also an answer.
3. **Profile the kernels.** `is_kernel_using_tensor_core` settles what a kernel name only implies.
4. **Assert the build id** the server reports equals the payload you shipped. A stale deploy reports
   success.

**Most tests in this repo are parity assertions between two of our own code paths**, so an assumption
both paths share is invisible to all of them. Only a physical bound catches those.

---

## The worksheet

Fill this in before provisioning. Every line is arithmetic or a lookup except the last two.

```
CHIP
  compute dtype the device selects      ____   (from the device, not config)
  native formats with a COMPUTE path    ____   (HARDWARE.md)
  usable memory / serving budget        ____
  largest contiguous block at peak      ____   (assume ~1/3 of free until measured)

MODEL
  weights, bf16 ceiling                 ____   (MODELS.md, over-predicts 6-9%)
  weights, int4 floor                   ____   (MODELS.md x1.19, under-predicts)
  KV per token x max_model_len          ____   (MODELS.md)

TRANSIENTS
  load-time peak (incl. quantization)   ____   host AND device
  per-request flat term                 ____
  per-request linear term x context     ____

FIT
  budget - weights - KV - max transient ____   <- must be positive AND contiguous

FORMAT
  coarsest quantization that fits       ____
  fused kernel for it on this chip?     ____   if no: memory win only, expect 0% speed
  compute win available?                ____   int8 on v5e/v6e, fp8 on v7, else none

VERIFY
  physical bound to cross-check against ____
  how a silent failure would present    ____   <- if you cannot answer this, stop
```

**The last line is the one that matters.** Every expensive incident in this repo — the fp8 KV flag, the
stale deploy, the cache directory that was configured and configured nothing, the bf16 mismatch that
emulated instead of failing — presented as success. If you cannot say in advance what a silent failure
would look like, you will not notice one.

---

## Worked example: `gpu-jax-g5g-2b`

```
CHIP    float16 (device-selected, SM 7.5) | fp16+int8 compute | 14.07 GB | ~1/3 of free
MODEL   9.257 GB bf16 -> 6.155 GB at ple4+int8head | KV 18 KiB/token
TRANS   load >15 GiB host RSS | flat ~1.5 GiB | linear ~0.9 MiB/token above 4K
FIT     positive at E2B; NEGATIVE at E4B (load) and 12B (per-request) despite weights fitting
FORMAT  ple4+int8head; NO fused kernel on Turing; no compute win available
VERIFY  320 GB/s / 6.155 GB = 52 tok/s ceiling; measured 13.6 = 26% of it
        -> the gap is not physics, it is the dtype tax in driver 2
```

That last line is the whole method working: a physical bound turned "13 tok/s" from a number into a
**74% shortfall with an address**.
