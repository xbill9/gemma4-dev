# Reference: TPU inference measurement series (Zimbres)

Four CC-BY-4.0 preprints by **Rubens de Almeida Zimbres** (Intellimetri) measuring LLM
inference on Cloud TPU with vLLM/JAX. They are the closest published work to what this
repo measures, and they both corroborate and challenge our results.

| # | DOI | Title | Hardware |
| --- | --- | --- | --- |
| 1–2 | [10.5281/zenodo.21212010](https://doi.org/10.5281/zenodo.21212010) | From 1,540 to 19,511 Tokens per Second on a Single TPU v5e Chip | v5e ×1, Gemma 2B |
| 3 | [10.5281/zenodo.21227936](https://doi.org/10.5281/zenodo.21227936) | Token Velocity on a Single TPU v5e Chip | v5e ×1, Gemma 2B |
| 4 | [10.5281/zenodo.21404155](https://doi.org/10.5281/zenodo.21404155) | The Decode Block Size Heuristic in TPU Ragged Paged Attention | v6e-4 (TP=4), Gemma 2 27B / Gemma 4 31B |
| 5 | [10.5281/zenodo.21462837](https://doi.org/10.5281/zenodo.21462837) | Batch Scaling and Goodput of a Tuned Attention Kernel on TPU v6e | v6e-4 (TP=4), Gemma 4 31B |

## The framework worth adopting: arithmetic intensity

Critical intensity on v5e is ≈ **240 FLOPs/byte** (peak arithmetic ÷ memory bandwidth).
Below it, an operation is memory bound.

- **Decode matmul intensity ≈ B** (the batch size). Batching amortizes the weight bytes.
- **Decode attention intensity ≈ 8 FLOPs/byte, independent of batch** — a group of query
  heads shares each KV head, but every request must read *its own* cache, so the bytes
  never amortize. Measured cost grew 7.97× across an 8× batch increase (linear).
- Consequence: as batch grows, every other operation's share shrinks and attention's does
  not. Attention became ~41% of the step at batch 512.

## Measured device-time breakdown (Gemma 2B, v5e, optimized, 31.2 ms step)

| component | time | share |
| --- | ---: | ---: |
| attention (18 layers × 714 µs) | 12.9 ms | 41% |
| fused gate+up projection | 6.8 ms | 22% |
| down projection | 4.4 ms | 14% |
| **logits pipeline** (matmul + convert + copy + argmax) | 5.6 ms | 18% |

## What paid off, and what didn't

**Paid:** token-selection fix **2.3×** (the sampling machinery was ~55% of device time —
top-p transforms run even when mathematically a no-op; `temperature=0.0` bypasses them);
batch scaling **4.3×**; attention block-size retuning **+24%**. Total **12.7×**.

**Did not:** speculative decoding **−6×** ("it consumes spare compute that a saturated chip
does not have"); weight and cache quantization unavailable on that stack; a precision change
that "looked certain on paper" **−14%**, reverted.

**Decode KV block size** (paper 4): the RPA v3 heuristic sizes blocks at ~16 MB assuming long
contiguous reads amortize fixed costs. With short sequences the kernel loads, masks, and
discards the padding — ~93% of every block was padding in the v5e case. Overriding to ≤128
gave **+68.7%** on Gemma 4 31B and **+27.7%** on Gemma 2 27B.

## How this bears on our measurements

1. **Our "decode is not bandwidth-bound" conclusion needs revisiting.** We compared a measured
   20.6 ms/token step against a ~3.6 ms "bandwidth floor" and concluded reads were not the
   constraint. That floor counted **weights and the LM head only** — it omitted KV-cache
   traffic (intensity ≈ 8, never amortized) and the logits convert/copy, which this series
   measures at 18% of the step. The framework says decode *is* memory bound; our floor was
   simply incomplete. See `benchmarks/runs/2026-07-28-jax-e2b-v6e1/REPORT.md`.

2. **Our B=1 penalty contradicts the published model.** Weight-reading cost there stayed *flat*
   across an 8× batch increase, and our own B=2/4/8 are flat (7.3–7.9 ms) — but B=1 sits at
   20.5 ms instead of ~7.4. Since intensity ≈ B puts both B=1 and B=2 far below critical, they
   should cost about the same. That points at an implementation artifact in our engine (likely
   an XLA layout/kernel choice for the batch-1 matrix-vector case), not a hardware property.
   This is the strongest argument yet for profiling it rather than theorising.

3. **We waste bandwidth on masked padding, exactly as paper 4 describes.**
   `make_cached_decode_step` attends over the *entire* preallocated KV buffer
   (`bucket_S + max_new_tokens`) every step and masks the invalid slots. That is the same
   load-mask-discard pattern the RPA heuristic is penalised for. Bounding attention to the
   valid prefix is the analogous fix here.

4. **Our fused W4A16 failure was predictable from their rule.** They conclude the stack is best
   modified *above* the compiler (configuration) or *below* it (whole kernels), "rather than
   through point edits to the compiled middle." Our fused dequant-matmul was precisely such a
   point edit, and it regressed (0.59× at B=1, 0.21× at B=2) and OOM'd.

5. **int8 LM head has a ceiling.** The logits pipeline is 18% of the step and only part of it is
   the matmul we shrank, which brackets our measured 4–5% as roughly the available win.

6. **Sampling is already on the fast path.** Their 2.3× came from bypassing top-p transforms;
   `onchip_sample_tpu_v6e_jax` already short-circuits to `argmax` at `temperature <= 0`, and our
   benchmarks run greedy. No gain available here — but worth keeping if sampling is ever enabled
   by default.

## Goodput: the metric this repo should be reporting (paper 5)

Paper 5 is the closest published work to our thesis — its keywords include *multi agent
systems*, on the argument that "multi agent systems consume tokens machine to machine at
rates no human reader imposes, and their unit economics are set almost entirely by the
serving throughput and per step latency measured here."

Its central result is a **goodput cliff**:

- Throughput saturates at **256 concurrent sequences** (7,375 tok/s on a v6e-4 slice).
- Past the knee, tokens/sec stays roughly flat while **time per output token doubles with
  every doubling of batch**.
- Under a **50 ms/token** service objective, goodput therefore **falls to zero exactly where
  raw throughput is still near its maximum**. The throughput and goodput optima coincide at
  the knee.
- "A planner reading only the throughput column would size the fleet at batch 512 or 1,024
  and deliver every user a stream at 70 to 135 ms per token. Sizing at batch 256 costs at
  most 2 percent of peak tokens per second and keeps every stream at reading speed."

The stated conclusion is the one our article should adopt verbatim in spirit: **"the correct
fleet question is not how many tokens per second a slice produces but how many concurrent
streams it supports within the objective"** — there, 256 streams at ~31 tok/s each.

It also finally locates the **regime boundary** the earlier papers could only predict: at
1,500-token histories the shipped heuristic is **28% faster** than the small block, reversing
the sign of the block-size result. Optimizations are claims about an operating regime.

**What this implies for our numbers.** Our measured per-user cost at ctx 512 is 7.4–7.9 ms/token
at B=2–8, and 20.6 ms at B=1 — all comfortably inside a 50 ms/token objective. So our sweep
never approached the goodput cliff; it stopped at B=8, far below the knee. Capacity for this
model on one chip should be expressed as *concurrent streams within a latency objective*, and
we have not yet measured where that runs out. (Batch-scaling sweep to B=256 run on
2026-07-28; see the run report.)

## Decode velocity depends on INPUT length, not output length (paper 3)

The nine-bucket velocity table for Gemma 2B on one v5e chip:

- **Prefill ≈ 38,000 input tok/s, nearly constant across input lengths** (it is compute bound).
- **Decode falls from ~8,300 output tok/s at short inputs to ~2,900 near the context limit** — a
  2.9x degradation "governed almost entirely by input length and almost not at all by output
  length."

For an agent workload this is the single most transferable number in the series. Agent context
grows monotonically with every tool result appended, so decode velocity decays as the
conversation lengthens *regardless of how much the agent emits*. It is not a prefill problem
you can amortize; it is the decode rate itself sagging as history accumulates.

The kernel tuning inverts across the same axis: **+84% on short input buckets, −24% on long
ones**, with the sign flipping between 1,024 and 7,168 input tokens. Hence the paper's
conclusion that decode velocity is a property of the hardware, model **and kernel configuration**
triple — "a kernel change deployed after profiling shifts the table in both directions at once."

## A profiling trap to avoid when we chase the B=1 anomaly

Paper 1/2 discloses a mistake worth internalising before we profile anything:

> Aggregating trace events **by operation name** suggested the decode feed-forward ops ran with a
> 1,024-token shape, implying half of each step was padding at batch 512. Resolving every event to
> the tensor shape in its own HLO text disproved it.

Compiled programs **reuse operation names across bucket sizes**, so warmup, chunked-prefill and
decode populations collide under one name. Aggregation must key on **shape**, and occurrence
counts must be checked against the expected number of layers/steps. Our unexplained 1.40x
single-stream penalty is exactly the kind of thing a name-keyed profile would mis-attribute.

## Where the series disagrees with this repo's framing

Paper 1/2 argues the opposite of our article's opening premise, and the disagreement is worth
keeping in view rather than resolving by assertion:

> agents exchange messages among themselves, multiplying token volume by an order of magnitude or
> more per user task, with no human waiting on any individual token. There, **per-token latency
> loses most of its meaning and aggregate throughput becomes the binding constraint**.

Our article says the reverse — "aggregate throughput is irrelevant here; per-stream latency is
everything." Both hold, for different agent topologies:

- **Serial/interactive agents** (a turn blocks on the previous one, a human waits at the end):
  latency compounds across round trips and binds.
- **Machine-to-machine fleets** (agents messaging in parallel, no human per token): throughput
  binds, and the paper maximizes exactly that.

Paper 5's **goodput** — throughput subject to a per-token latency objective — is the metric that
subsumes both, which is why it is the right axis to report. The paper also concedes the
qualification directly: "a production service with irregular arrivals, longer prompts, and
latency targets will operate below the maximum."

## Two claims of ours the series independently supports

1. **Prefix caching is the top agent gap.** "Prefix caching ... is at its most effective in agent
   fleets, because agents share long system prompts and accumulated context, so the
   prompt-processing cost of inter-agent messages collapses." That is our ranked #1 missing
   feature, argued from the other direction.
2. **Small models are the right shape for agent fleets.** "the model scale studied here, two and a
   half billion parameters, is the natural size for role-specialized agents, where a fleet of small
   specialists ... replaces a single large generalist at a fraction of the serving cost."

Also worth stealing: **determinism makes agent pipelines idempotent** — "the same input state
produces the same action, which turns debugging from replaying probabilities into replaying
facts." Our engine is greedy by default and bit-exact under the parity tests, so this is free.

Efficiency figures for context: energy fell ~128 → 10 mJ/token and cost rose from ~4.6M to
58.5M tokens per dollar across the 12.7x — every efficiency metric moved by the same factor
because the chip and its power envelope were constant.

## The operating recipe (paper 5 §8) — worth following verbatim

1. **Measure the sequence lengths your traffic actually serves**; they decide the kernel
   configuration. Histories up to a few hundred tokens favour a small decode block (32–128 all
   captured the effect); beyond ~1,000 tokens the shipped heuristic wins; **mixed traffic argues
   for separate pools or length-aware selection**.
2. **Find the saturation knee by sweeping batch** with everything else fixed. The signature to
   look for is the throughput multiplier per doubling: **1.8 → 1.6 → 1.2 → flat**.
3. **Set the operating batch at the knee, not past it.**
4. **Express capacity in streams within a latency objective**, not peak tokens/sec.
5. **State the objective explicitly.** 50 ms/token "is a defensible default for human facing
   streams and the wrong number for offline pipelines."

Cost of doing this: "about two hours of a four chip slice ... under 26 dollars to locate an
operating point worth thousands per month." Our equivalent sweep cost ~$1.38 on one chip.

**Applying step 2 to our own data** (ctx 512, real config): multipliers per doubling are
2.80 → 1.99 → 2.00 → 1.92 → **1.82** at B=32. We are still on the linear stretch and never
reached the 1.2×/flat signature — HBM ran out first. Our knee is unmeasured, and it is above 32.

## Determinism is not a compromise for specialized agents (paper 1/2 §11)

The strongest argument in the series for what this repo already does by default:

> Language models are trained under a cross-entropy objective with teacher forcing... Deterministic
> selection, taking the highest-probability token, is the purest consumption of exactly that
> objective. Sampling parameters exist to deviate from the trained distribution on purpose...
> **fine-tuning is what makes deterministic decoding lossless.**

A model fine-tuned to a narrow task develops a sharply peaked output distribution, so the gap
between sampled and greedy decoding "shrinks toward zero exactly as specialization increases."
A generalist chat model gives up real variety at temperature zero; **a model emitting tool calls,
JSON, routing decisions or extractions gives up almost nothing** — and gains determinism, which
has independent production value: "runs become reproducible, failures become debuggable, outputs
become cacheable, and behavior becomes testable."

Note also *why* their 2.3× was available: not sampling as a concept, but "a binary search over a
256,000-entry vocabulary executing 32 iterations per step for a top-p value that made the search a
no-op." Requesting the permissive value of a sampling parameter still pays its full cost.

## Pallas is cheaper to compile than XLA, not more expensive

Same attention computation, five trials, fresh processes, compile cache disabled: **XLA median
297.5 ms, Pallas via Mosaic ~50 ms** — roughly 6× faster to compile, with equivalent results and
equivalent execution time once compiled. Writing a kernel is not a compile-time penalty, which
removes one common objection to the Pallas route.

Related operational costs: engine startup (compiling the full bucket menu) ran 2–5 minutes
depending on cache state, each new maximum batch size adds compilation work, and the compile cache
"does not persist across rebuilt environments in the hosted setting."

## The methodological argument (paper 1/2 §10)

Worth reading in full before optimizing anything. Their two winning interventions look obvious in
hindsight — use direct token selection, raise the batch — and the paper anticipates the objection:

> That reading mistakes the fix for the diagnosis... nobody's checklist contains the item that
> actually held 55 percent of this chip.

The counterfactual is the point. A team following the mainstream sequence without profiling would
have "deployed speculative decoding, measured here at six times slower in this regime; pursued
quantization, measured at zero functioning routes on this stack, **one failing silently**; or
hand-optimized a genuine numeric inefficiency, measured at 14 percent slower when removed at the
wrong layer."

That is a precise description of this session: we pursued fused int4 quantization (regressed and
OOM'd, and our kernel *did* fail silently — a bare `except` hid wrong results wherever Pallas
compiled) and an int8 LM head (1.00–1.05×). "The simple strategies did not win by default. They won
an elimination against the sophisticated ones."

## Caveats when transferring these numbers

Different hardware (v5e ×1, and v6e-4 sharded) and a different serving stack (vLLM/tpu-inference,
not our raw JAX engine). The *mechanisms* — intensity, padding waste, regime dependence —
transfer; the absolute figures do not. Paper 3's headline is precisely that decode velocity is a
property of the hardware, model **and kernel configuration** triple, not a constant of a
hardware/model pair.

PDFs are not vendored here (CC-BY-4.0, but large); fetch via the DOIs above.
