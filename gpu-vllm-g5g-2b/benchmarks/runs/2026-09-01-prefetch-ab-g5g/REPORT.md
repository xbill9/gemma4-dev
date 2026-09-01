# 2026-09-01 — `--safetensors-load-strategy=prefetch`: no effect. But the cold/warm gap is 6x.

**Within-box A/B on `i-02109c6c38368dfec`** (`g5g.2xlarge`, on-demand, `us-east-1a`,
`ami-0b44b90b3d02430ee`). One instance, restarts only, so there is no host variance in the
comparison. Cold boot 1228.8 s, then three timed restarts.

## The flag does nothing

| restart | config | weight load | launch→health | first completion |
| --- | --- | ---: | ---: | ---: |
| A | as shipped | 76.13 s | 274.64 s | 0.65 s |
| B | `--safetensors-load-strategy=prefetch` | **75.12 s** | 272.48 s | 0.28 s |
| C | prefetch, repeat | 32.07 s | 222.65 s | 0.17 s |

**A → B is −1.3% on weight load. That is nothing.** The lead from 2026-08-31 — vLLM's own log
suggesting the flag because auto-prefetch is disabled on EXT4 — **is dead.**

**C is not evidence that it works.** 76 → 75 → 32 across three successive restarts is a
monotonic warming trend on an unchanged config between B and C; attributing the 32 s to the flag
would be reading a trend as a treatment effect.

**Caveat on scope, and it matters:** the flag was tested on the WARM path, where loading is
already ~6x faster than the problem being chased. This rules out a warm-path effect. It does
**not** strictly rule out a cold-boot effect, which would need the flag baked in before first
boot.

## The finding that actually matters: cold is 6x warm on identical hardware

| | weight load | n |
| --- | ---: | --- |
| **cold boot** (fresh instance) | **468–561 s** | 4 |
| **warm restart** (same box) | **32–76 s** | 3 |

Same instance, same volume, same EXT4, same vLLM, same checkpoint. **The only thing that
changed is that the blocks had been read once.**

That is consistent with **EBS lazily hydrating the volume from the AMI snapshot** — first touch
of each block fetches from S3, subsequent reads come off the volume. It fits every measurement
taken so far:

- more RAM did not help (2026-08-31, 32 GiB host: 468 s) — it is not page cache;
- the loader's read strategy did not help (this run) — the bytes are not on the volume yet;
- restarts get progressively faster (76 → 75 → 32) — progressive hydration;
- 9.54 GiB / 468 s = 20.9 MB/s, far below gp3 steady state but entirely ordinary for
  first-touch snapshot reads.

**This is a hypothesis, not a result.** It is the fourth explanation offered for these seconds
and the previous three were wrong, so it is written down as the next thing to TEST, not as the
answer.

## The test that would settle it

On a fresh instance, before starting vLLM, force the whole volume to hydrate and time the boot:

```bash
dd if=/dev/nvme0n1 of=/dev/null bs=1M status=progress   # or fio, or EBS fast snapshot restore
systemctl start vllm
```

If weight loading drops from ~500 s to ~76 s, the hypothesis holds and the fix is a
volume pre-warm or **EBS Fast Snapshot Restore on the AMI** — neither of which is a vLLM change.
If it does not, the hypothesis joins the other three.

## Operational note

The first attempt at this run was on **spot and was reclaimed 10 minutes into the boot**
(`Server.SpotInstanceTermination`). Re-run on on-demand for ~$0.42, because the experiment needs
~40 minutes of instance life and a reclamation mid-A/B would corrupt the comparison rather than
merely waste it. Spot first, on-demand when the workload outlives it.
