# 2026-08-31 — three runtimes, one harness, one statistic

**The first controlled comparison of vLLM, JAX and PyTorch on this silicon.** Every prior
cross-rig number in this family was three harnesses computing three different statistics on
three different days and two instance sizes. This run fixes that.

| | value |
| --- | --- |
| Hardware | `g5g.2xlarge` spot, **`us-east-1a`**, all three, 2026-08-31 |
| Instances | vLLM `i-04a83e4a41be52ffa` · JAX `i-0ffdb819663bacc77` · PyTorch `i-02e79988a6cbeecbf` |
| Checkpoint | `google/gemma-4-E2B-it`, float16, all three |
| Harness | `gpu-pytorch-g5g-2b/sweep.py`, identical invocation, 3 repeats, warm-at-shape |
| Statistic | client-side inter-token latency off the SSE stream (`--decode-source`) |

Artifacts: `sweep.json`, `sweep.log`, `metrics.prom` in each rig's
`benchmarks/runs/2026-08-31-crossrig-*-g5g/`.

## Result

| runtime | decode tok/s (stream) | % of 61.4 ceiling | vs PyTorch | cells |
| --- | ---: | ---: | ---: | ---: |
| **vLLM** v0.27.2rc0 | **32.53** | 53.0% | **3.18x** | 12/12 |
| JAX (ple4+int8head) | 12.69 | 20.7% | 1.24x | 10/12 |
| **PyTorch** + transformers | **10.24** | 16.7% | 1.00x | 10/12 |

**The ranking is unchanged and the magnitude is now trustworthy.** The previously quoted
~31.8 tok/s for vLLM — derived from a TPOT figure taken 17 days earlier on a `g5g.4xlarge`
with a different harness — lands within **2.3%** of the controlled 32.53. The old number was
right despite six uncontrolled variables; it just could not be shown to be right.

## What only a controlled run could show: vLLM's lead shrinks with context

| input tok | out | vLLM | JAX | PyTorch | vLLM/PT | JAX/PT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 92 | 32 | 36.15 | 12.63 | 10.32 | **3.50** | 1.22 |
| 92 | 128 | 33.90 | 12.80 | 9.44 | 3.59 | 1.36 |
| 633 | 32 | 35.22 | 12.73 | 10.53 | 3.35 | 1.21 |
| 633 | 128 | 32.15 | 12.73 | 10.07 | 3.19 | 1.26 |
| 1259 | 32 | 33.01 | 12.71 | 10.56 | 3.13 | 1.20 |
| 1259 | 128 | 32.60 | 12.72 | 10.13 | 3.22 | 1.25 |
| 2501 | 32 | 32.74 | 12.67 | 11.05 | 2.96 | 1.15 |
| 2501 | 128 | 32.45 | 12.66 | 10.17 | 3.19 | 1.24 |
| 3746 | 32 | 29.82 | 12.60 | 10.57 | **2.82** | 1.19 |
| 3746 | 128 | 29.78 | 12.60 | 10.10 | 2.95 | 1.25 |
| 4630 | 32 | 28.22 | — | — | — | — |
| 4630 | 128 | 27.94 | — | — | — | — |

**vLLM's decode falls 22.7% across the context range (36.15 → 27.94 over 50x), while both of
ours are flat.** PyTorch spans 9.44–11.05 with no trend; JAX falls 3.4%. So the headline
multiple is not a constant — it is **3.5x at short context and 2.8x at 3,746 tokens**, and
quoting a single figure without naming the context is how these comparisons go wrong.

The likely reading, consistent with this rig's profile: our two runtimes are so launch-bound
(~5,650 kernel launches/step at 1–3 us on a chip with 5–10 us launch overhead) that attention
cost is invisible behind the overhead — attention measured **1.0% of decode** here. vLLM has
removed enough of that overhead for attention's O(S) KV read to become visible. **Being
context-sensitive is what running closer to the hardware looks like.**

**Only vLLM reaches 4,630 input tokens.** Both of ours refuse it — `--seq`/`max_model_len` is
4096 against vLLM's 16384 — with clean errors naming both numbers. Those two cells are
`infeasible`, not failures.

## The calibration, which is why the numbers can be compared at all

`sweep.py --decode-source both` runs each repeat twice, once non-streaming (reading the
server's own `usage.decode_tokens_per_second`) and once streaming (timing inter-token gaps
client-side). The ratio is the offset between the two statistics:

| rig | usage | stream | stream/usage |
| --- | ---: | ---: | ---: |
| JAX | 12.962 | 12.687 | **0.9799** |
| PyTorch | 10.814 | 10.243 | **0.9543** |

**The two rigs have different offsets — 2.0% against 4.6% — so one rig's ratio cannot be used
to convert another's number.** That is the whole justification for measuring it rather than
assuming a constant, and it is why the cross-rig table above is built from `stream` throughout
rather than from each rig's native gauge.

vLLM emits no such gauge, so `auto` correctly selected `stream` for it. That is the entire
reason this harness could not measure vLLM before today: `sweep.py` read a `usage` field that
only our two servers invent.

## Two fixes this run depended on

**1. The JAX gauge was rounded to 1 decimal** — `:.1f` at `jax_openai_server.py:569` and
`round(..., 1)` at `:693`. At ~13 tok/s that is 0.78% resolution, which is why every prior JAX
sweep shows byte-identical repeats (12.8, 12.8, 12.8) and why that rig could not resolve the
~2% effects being argued about on it. Now 4 decimals; this run reads 12.962, not 13.0.

**2. `sweep.py` gained `--decode-source {auto,usage,stream,both}`.** `usage` preserves every
pre-2026-08-31 number; `stream` matches `vllm bench serve`'s TPOT definition exactly
(`(latency − ttft) / (output_len − 1)`) so it is comparable to that tool's published figures.

## Operational notes

- **g5g spot was exhausted region-wide again**, for ~9 minutes across all four AZs, exactly as
  in the morning run. Both launch loops landed in `us-east-1a`, which is why all three
  instances share an AZ — luck, not design, but it improves the control.
- **The vLLM prebuilt AMI took 8m17s to load weights**, not the ~180 s its rig's `CLAUDE.md`
  predicts: `weight_utils` reports a 9.54 GiB checkpoint against **11.19 GiB available RAM**,
  so the load thrashes page cache. That doc says `g5g.2xlarge` "needs no swapfile and buys
  that time back"; on this run it did not. Engine init itself was 207 s, as documented.
- **KV cache came out at 329,579 tokens, matching the 2026-08-14 run exactly** — independent
  confirmation the engine config is unchanged between the two.
- The vLLM rig's `create_g5g_instance` **cannot launch its own prebuilt AMI**: `_resolve_ami`
  always resolves the base DLAMI from SSM and there is no `ami_id` override, so the tool would
  start the multi-hour from-source rebuild its own `CLAUDE.md` says not to do. This run
  launched `ami-0b44b90b3d02430ee` directly with boto3, carrying the rig's own tags.
- All three instances terminated as soon as their artifacts were captured.

## Boot and revision time — 3 repeats per rig, measured 2026-08-31

**The axis that reverses the performance ranking.** Same `g5g.2xlarge`, `us-east-1a`, nine cold
boots and nine warm reloads. Harness and raw data ship beside this file as `boot_campaign.py` and
`boot_results.json`.

Start line is the moment `run_instances` returns an id — capacity wait is excluded deliberately,
it measures AWS and not the rig. **Two stop lines, not one:** health 200, and a first chat
completion. They are not the same instant on every runtime, and the gap is a finding.

| | cold boot | spread | warm reload | cold/warm | first completion (cold / warm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **PyTorch** | **195.2 s** | 11.8% | **24.5 s** | 8.0x | 1.0 s / 0.7 s |
| **JAX** | 242.2 s | 11.5% | 74.1 s | 3.3x | **22.9 s** / 9.2 s |
| **vLLM** | **1417.8 s** | 12.6% | 264.3 s | 5.4x | 0.5 s / 0.2 s |

**vLLM takes 23m 38s to serve — 7.3x PyTorch — from a PREBUILT AMI that downloads nothing.**
The image carries the ~67-minute SM 7.5 build and an offline model cache; PyTorch installs from
wheels *and* pulls 9.5 GB over the network and is still 7x faster. Weight loading alone is
**546 s median**, 2.8x PyTorch's entire cold boot, because `weight_utils` reports a 9.54 GiB
checkpoint against 11.19 GiB of available RAM and thrashes page cache. The PyTorch rig hit the
same wall on 2026-08-29 and fixed it with `device_map={"": 0}` — shard-by-shard to the GPU, host
peak 10.52 GB, zero swap. vLLM has no equivalent on a 16 GiB host.

That rig's `CLAUDE.md` predicts ~4 minutes and says `g5g.2xlarge` "needs no swapfile and buys that
time back." **Measured across three boots it is 23-25 minutes.** The doc is wrong by ~6x.

### Health 200 is not "ready to serve", and only JAX is badly caught by it

**JAX's first completion costs 22.9 s** against PyTorch's 1.0 s and vLLM's 0.5 s — it returns
health 200 and *then* compiles XLA per shape bucket on the first real request. Quoting health
alone understates JAX's true time-to-serving by 22 seconds. **It does not fully go away warm
either: 9.2 s.** So JAX's real revision cycle is 74.1 + 9.2 = **83 s**, against PyTorch's
24.5 + 0.7 = **25 s**.

vLLM is the opposite: slowest to boot by far, then the fastest first token of the three (0.5 s
cold, 0.2 s warm), because CUDA graph capture and AOT compile are already paid before the port
binds.

### Revision cost is where the ranking inverts hardest

| | to change serving code |
| --- | --- |
| PyTorch | ship 3 files (16 KiB) over SSM, restart — **25 s** |
| JAX | same mechanism — **83 s** |
| vLLM | **no deploy path exists**: from-source rebuild, ~67 min, plus reapplying the out-of-tree Turing patch |

The 264.3 s warm figure for vLLM is a `systemctl restart`, **not a code revision** — it is the
closest analogue available and it flatters vLLM, because a real change to that rig means rebuilding
the image. So vLLM wins decode 3.2x and TTFT ~30x, and loses the iteration loop by 3-100x
depending on what you are changing.

### Boot variance is ~12% and belongs to the platform

11.5% / 11.8% / 12.6% across three different runtimes is too consistent to be a runtime property.
**That is ~6x this rig's decode noise floor (~2%), so a single boot measurement is worth very
little** — which is the whole reason this section has repeats.

Two second-order effects worth keeping. **PyTorch's warm spread is 51.1%** (20.8-33.4 s) because
the first redeploy after a cold boot pays a page-cache cost the later ones do not — median-of-3
understates the first revision you would actually make. **JAX's warm spread is 4.2%**: its reload
is dominated by deterministic compile work, so it is tighter but 3x slower.

### Method notes

- **A first version of this campaign was discarded and re-run.** Its poll intervals (10 s / 20 s /
  5 s) were coarse enough that two genuinely similar boots landed on the same tick and reported
  **byte-identical** times — 214.4 s twice. The data was real (log wall-clock confirms 215 s and
  216 s) but publishing it would have presented the measurement floor as reproducibility. Health
  polling is now 0.5 s. The coarse results are kept as `boot_results_v1_coarse.json`.
- **Markets are mixed and recorded per rep** — spot where available, on-demand after 3 dry rounds.
  Same hardware either way, so it cannot move a boot number, and the data shows no effect: JAX
  rep 3 ran on-demand at 242.21 s, between its two spot reps at 237.66 and 265.52.
- **g5g spot was exhausted region-wide four separate times**, 5-11 minutes each. On-demand is
  **$0.556/hr** against measured spot of $0.3813-$0.4416 — a 26-46% premium — so the fallback cost
  ~$0.24 across the whole campaign. The entire day, ~15 instances and ~3.5-4 instance-hours, came
  to under $2. **Cheap hardware is what made discarding a bad campaign a 20-minute decision
  rather than an argument.**
