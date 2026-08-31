# 2026-08-31 — vLLM leg of the three-runtime controlled comparison

**This rig's contribution to a cross-rig run.** Full analysis in
`gpu-pytorch-g5g-2b/benchmarks/runs/2026-08-31-crossrig-torch-g5g/REPORT.md`.

`g5g.2xlarge` spot, `us-east-1a`, `i-04a83e4a41be52ffa`, launched from the prebuilt
**`ami-0b44b90b3d02430ee`**. vLLM v0.27.2rc0, float16, `max_model_len=16384`,
`gpu_memory_utilization=0.90`, `max_num_seqs=8`, TP=1. Driven by
**`gpu-pytorch-g5g-2b/sweep.py`** at concurrency 1.

**This is the first sweep artifact this rig has ever had.** The 2026-08-14 run left only a
`REPORT.md`; there was no `sweep.json` and no validated report anywhere in the rig, which is
why its numbers could not be recomputed and had to be read out of prose.

## Result

**Decode 32.528 tok/s median (client-side stream), 12/12 cells, 0 degenerate.**

| input tok | out | decode tok/s | TPOT ms | e2e tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 92 | 32 | 36.15 | 27.66 | 33.56 |
| 92 | 74 | 33.90 | 29.50 | 32.74 |
| 633 | 32 | 35.22 | 28.40 | 32.01 |
| 633 | 83 | 32.15 | 31.10 | 31.11 |
| 1259 | 32 | 33.01 | 30.30 | 30.21 |
| 1259 | 83 | 32.60 | 30.67 | 31.52 |
| 2501 | 32 | 32.74 | 30.54 | 29.45 |
| 2501 | 85 | 32.45 | 30.81 | 31.13 |
| 3746 | 32 | 29.82 | 33.53 | 26.23 |
| 3746 | 85 | 29.78 | 33.58 | 28.33 |
| 4630 | 32 | 28.22 | 35.44 | 24.33 |
| 4630 | 85 | 27.94 | 35.79 | 26.36 |

**The 2026-08-14 figure is confirmed.** That run reported TPOT 31.44 ms at c=1 (~31.8 tok/s)
on a `g5g.4xlarge` with `vllm bench serve`. This run, on a different instance size 17 days
later with a different harness, gives **32.53** — within **2.3%**. `sweep.py`'s `stream` path
uses `vllm bench serve`'s exact TPOT definition, which is what makes the two comparable.

**KV cache: 329,579 tokens — identical to the 2026-08-14 run**, independently confirming the
engine config has not drifted.

## Decode is context-sensitive here, unlike the JAX and PyTorch rigs

**36.15 -> 27.94 tok/s over a 50x context range, a 22.7% decline**, monotonic. The two
sibling runtimes on the same chip are flat over the same range (PyTorch no trend, JAX −3.4%).
Those two are launch-bound hard enough to hide attention entirely — attention measures 1.0% of
decode on the PyTorch rig — so **being context-sensitive is evidence vLLM has removed enough
per-step overhead for the KV read to matter.**

Consequence: **the advantage over the sibling rigs is 3.5x at 92 tokens and 2.8x at 3,746.**
Quote it with a context or not at all.

**Only this rig reaches 4,630 input tokens.** Both siblings cap at 4096.

## Two things about the AMI that its own docs get wrong

- **Weight loading took 498 s**, not the ~180 s `CLAUDE.md` predicts. `weight_utils` reports a
  9.54 GiB checkpoint against **11.19 GiB available RAM**, so the load thrashes page cache.
  The claim that `g5g.2xlarge` "needs no swapfile and buys that time back" did not hold here;
  total boot-to-serving was **~21 minutes**. Engine init proper was 207 s, as documented.
- **`create_g5g_instance` cannot launch the prebuilt AMI.** `_resolve_ami` always resolves the
  base DLAMI from SSM and the tool takes no `ami_id`, so the documented "use it, do not
  rebuild" path is unreachable and the tool would instead start the ~67-minute from-source
  build. This run launched the AMI directly with boto3, carrying the rig's own
  `ManagedBy=gpu-vllm-g5g-2b` tag so the rig's list/terminate tools still see it. **Adding an
  `ami_id` override is the fix.**

## Boot and revision time (3 repeats, 2026-08-31)

Cold boot **1417.8 s = 23m 38s** [1346.8–1525.2, 12.6%] · warm `systemctl restart` **264.3 s** [12.0%].

**This rig takes 7.3x longer to serve than the PyTorch sibling, from a prebuilt AMI that downloads
nothing** while that sibling installs from wheels and pulls 9.5 GB. Weight loading alone is
**546 s median** — 2.8x the sibling's entire cold boot — because `weight_utils` reports a 9.54 GiB
checkpoint against 11.19 GiB available RAM and thrashes page cache.

**`CLAUDE.md`'s "~4 minutes" and "`g5g.2xlarge` needs no swapfile and buys that time back" are
wrong by ~6x**, measured across three independent boots. That claim should be corrected.

The compensation: **fastest first token of the three** — 0.5 s cold, 0.2 s warm, against JAX's
22.9 s — because CUDA graph capture and AOT compile are paid before the port binds.

**Revision cost has no number because there is no deploy path.** The 264.3 s above is a
`systemctl restart`, not a code change; a real change means rebuilding the image (~67 min) and
reapplying the out-of-tree Turing patch. Full table and method in
`gpu-pytorch-g5g-2b/benchmarks/runs/2026-08-31-crossrig-torch-g5g/REPORT.md`.
