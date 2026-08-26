# 2026-08-26 — configuration sweep: every quantization lever is broken

**`g5g.2xlarge` spot (`i-00fc456e54538727a`), `us-east-1a`, build id `6852f5680f43`.
Terminated after the run.**

The 2026-08-25 sweep varied the request shape. This one varies the **serving config** —
`ple_bits` and `int8_lm_head`, the rig's only quantization levers — which nothing here had
ever measured. It also finally locates the dense prefill ceiling.

**Result: only the baseline loads. All three levers fail, for three different reasons, and
every failing allocation matches its tensor exactly.**

## Config matrix

| Config | Result | Fails at | Allocation |
| --- | --- | --- | ---: |
| `ple_bits=0` (baseline) | **ok** — 3/3 cells, 9.257 GB weights, ready in 199 s | — | — |
| `ple_bits=8`, no swap | never ready | **host RAM** | 14.3 GB RSS |
| `ple_bits=4`, no swap | never ready | **host RAM** | 14.1 GB RSS |
| `ple_bits=8`, **with swap** | never ready | **device** | 2.19 GiB |
| `int8_lm_head` | never ready | **device** | 1.50 GiB |
| `ple4 + int8_lm_head` | not run — both components already fail | — | — |

### Bug 1 — `g5g.2xlarge` gets no swapfile, and PLE needs one

`ple_bits` was killed by the Linux OOM killer, five times, under `Restart=on-failure`:

```
Out of memory: Killed process 37418 (python3.14)
  anon-rss:14312752kB  shmem-rss:1128448kB   ≈ 15.4 GB
Mem: 15 GiB total      Swap: 0
```

`quantize_ple_table` upcasts the 4.70 GB PLE table to float32 in host chunks while the full
parameter tree is still resident, and peak RSS exceeds the box.

The cause is our own gate:

```python
_SWAP_BELOW_HOST_RAM_GB = 16
def _needs_swap(t): return 0 < _host_memory_gb(t) < 16
```

`g5g.2xlarge` has **exactly 16** GiB, so `< 16` is false and no swapfile is provisioned. The
rig provisions swap for `g5g.xlarge` and skips the one size where this feature needs it.
Adding 16 GiB by hand — the same `fallocate`/`mkswap`/`swapon` block `_user_data` already
renders — stopped the OOM kills dead (kill count frozen at 6, all pre-swap).

**Fix:** make the threshold inclusive, or raise it. The remedy already exists; it is gated off.

### Bug 2 — `quantize_lm_head` upcasts the whole table on the device

```
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1.50GiB
  [executable_name='jit_convert_element_type']
```

`262144 × 1536 × 4 B = 1.50 GiB` — exact. `quantize_lm_head` does
`emb.astype(jnp.float32)` over the entire embedding table, on device, while the full 9.26 GB
tree is resident.

The correct pattern is **1,200 lines away in the same file**: `quantize_ple_table` quantizes
on the host, in row chunks, with a long comment explaining exactly why the naive version
OOMs. `quantize_lm_head` never received that treatment.

This matters more than the others: `int8_lm_head` is the lever most likely to *help*. The
LM-head conversion `wrapped_convert_61` was **14.3% of decode on its own** in the 2026-08-25
profile. It cannot be enabled on this rig at all today.

### Bug 3 — the quantized PLE table is placed while the original is still resident

With swap, `ple_bits=8` clears the host OOM and fails on the device instead:

```
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 2.19GiB
```

`262144 × 8960 × 1 B = 2.19 GiB` — exact, the int8 output table. At that moment the
**4.38 GiB bf16 original is still on the device**: `quantize_ple_table` carefully does its
arithmetic on the host, then `jax.device_put`s the result before the source is released.

Peak is roughly 11.5 GB against a 14.07 GB budget — which would fit, except the free space is
**66% fragmented** (measured 2026-08-25, `fragmentation 0.661`). A 2.19 GiB *contiguous*
block is not available.

**Bugs 2 and 3 are one pattern:** allocate the destination before freeing the source, on a
device whose free memory is fragmented. Same root as the `device_put` fragmentation already
written up in `docs/bf16-weights-on-turing.md`.

## The dense prefill ceiling is between 4,105 and 5,120 tokens

`MAX_MODEL_LEN=8192`, and the rig cannot prefill anywhere near it.

| prompt tokens | status | failed allocation |
| ---: | --- | ---: |
| 4,105 | ok (2026-08-25) | — |
| 5,120 | **infeasible** | 2.59 GiB |
| 6,144 | **infeasible** | 3.48 GiB |
| 7,168 | **infeasible** | 4.42 GiB |
| 7,800 | **infeasible** | 5.11 GiB |

The transient grows with context at roughly **0.9 MiB/token** in this range. That is worth
recording carefully, because `docs/bf16-weights-on-turing.md` measured the prefill transient
as **FLAT** in the bucket (1.504 GiB at 512 and at 1,536, 1.742 GiB at 4,096). Both are true:
there is a large flat term that dominates below ~4K, and a linear term that takes over above
it. The flat measurement was not wrong, it was taken entirely inside the flat region.

**`MAX_MODEL_LEN=8192` is therefore not an honest number** — it is a little over half
reachable.

### The documented remedy is structurally unreachable

`PREFILL_CHUNK_SIZE` exists precisely to bound prefill temporaries. It cannot be enabled in
the shipped configuration:

```python
if prefill_chunk_size is not None and window_kv:
    raise ValueError("prefill_chunk_size requires window_kv=False")
```

`window_kv` auto-resolves to **True** whenever `max_model_len > sliding_window` — 8192 > 512,
verified in this run's log (`window_kv=auto resolved to True`). So setting
`PREFILL_CHUNK_SIZE` raises at startup, and the only way to reach chunked prefill is
`window_kv=off`, which `CLAUDE.md` records as untested.

The one mitigation for the ceiling is gated behind an untested flag.

## What to fix, in order

1. **Make `_needs_swap` inclusive** (`<= 16`, or raise to 24). One line, unblocks `ple_bits`
   past its host-side OOM. Cheapest fix here by a wide margin.
2. **Rewrite `quantize_lm_head` to match `quantize_ple_table`** — host-side, chunked, and
   release the source before placing the result. This unblocks the only lever with a
   plausible *throughput* win.
3. **Release the source before `device_put` in `quantize_ple_table`.** Delete the bf16 table
   from the tree and drop the local reference before placing the int8 copy, so peak is
   `max(source, dest)` rather than `source + dest`.
4. **Set `MAX_MODEL_LEN` to something reachable** (4096) until prefill temporaries are
   bounded, or fix the `window_kv` / `prefill_chunk_size` interaction so chunking is usable.
   Advertising 8192 when 5,120 fails is a trap for anything sizing against it.
5. **Quote largest-contiguous-block, not free bytes**, in any capacity claim on this rig.
   Bugs 2 and 3 both fail with GBs nominally free.

## Artifacts

| File | What |
| --- | --- |
| `config_sweep.py` | the driver — rewrites `ExecStart` per config, reloads, sweeps |
| `config_sweep.json` | per-config results, checkpointed after each |
| `driver.log` | full driver console |
| `sweep_ple0.json`, `requests_ple0.jsonl`, `console_ple0.log` | the one config that loaded |
| `ceiling.json`, `ceiling.jsonl`, `ceiling.log` | the 5,120–7,800 token ceiling probe |
| `journal_ple8_hostoom.txt` | journal capture of the host-OOM crash loop |

Baseline throughput reproduced yesterday's numbers on fresh hardware; no new xprof trace was
taken, because no non-baseline config loads and it would only have reproduced the profile
already committed in `857835a`.
