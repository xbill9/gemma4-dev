# 2026-08-31 — 32 GiB host: the RAM hypothesis, falsified

**One `g5g.4xlarge` cold boot to test why weight loading takes ~546 s.** Same AMI
(`ami-0b44b90b3d02430ee`), same AZ, same harness as the 9-boot campaign.
`i-0785db75689508c2a`, spot, `us-east-1a`.

## Result: more RAM does not fix it

| | 16 GiB (`g5g.2xlarge`, n=3) | 32 GiB (`g5g.4xlarge`, n=1) | delta |
| --- | ---: | ---: | ---: |
| available RAM | 11.19 GiB | **26.49 GiB** | 2.4x |
| **weight loading** | 546.45 s | **467.97 s** | **−14.4%** |
| engine init | 207.19 s | 198.74 s | −4.1% |
| launch → health 200 | 1417.8 s | **1351.0 s** | −4.7% |
| first completion | 0.5 s | 0.46 s | — |
| KV cache | 329,579 tokens | 329,579 tokens | identical |

**−4.7% on total boot sits inside the ±12.6% run-to-run band the campaign measured, so the
honest reading is: no detectable improvement.** Weight loading moved 14%, which is real but
nowhere near the collapse a page-cache explanation predicts — 9.54 GiB fits 26.49 GiB with
room to spare, and the load still took nearly eight minutes.

n=1 is adequate here only because the *predicted* effect was large (hundreds of seconds). It
would not be adequate to characterise the 14% that actually appeared.

## Two claims falsified, one after the other

1. **"g5g.2xlarge needs no swapfile and buys that time back"** (`CLAUDE.md`, pre-2026-08-31) —
   falsified by the 3-boot campaign: 23m 38s, not ~4 minutes.
2. **"a bigger host will not fix it"** — my own replacement text, retracted the same day as
   unsupported, since no large host had been measured.
3. **"a larger host would very plausibly fix it"** — the retraction's *own* replacement, and
   **this run falsifies that too.** It does not.

Three wrong statements about the same 546 seconds, each written more carefully than the last.
The lesson is not to reason harder about the cause; it is that **the box costs $0.56/hour and
the answer took 25 minutes.**

## The actual lead

```
9.54 GiB / 467.97 s = 20.9 MB/s off LOCAL DISK
```

That is far below gp3's baseline, so this is neither RAM nor disk bandwidth — it is the
loader. vLLM says so itself, in a line present in every one of these boot logs:

```
Auto-prefetch is disabled because the filesystem (EXT4) is not a recognized network FS
(NFS/Lustre). If you want to force prefetching, start vLLM with
--safetensors-load-strategy=prefetch
```

**`--safetensors-load-strategy=prefetch` is the next experiment**, and it is a one-line change
to `/opt/serve.sh` rather than a bigger instance. Untested as of this run.

For contrast, the PyTorch sibling loads the same checkpoint with `device_map={"": 0}` — shard
by shard straight to the GPU, host peak 10.52 GB, and a full serving endpoint in 195 s.
