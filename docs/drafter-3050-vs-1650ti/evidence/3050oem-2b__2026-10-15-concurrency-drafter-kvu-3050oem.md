# 2026-10-15 — the repeated-paragraph concurrency sweep with one shared KV cache, RTX 3050 OEM

**Question:** `2026-10-15-concurrency-mixed-3050oem` found llama-server's default per-slot KV
cache halving decode speed at random on short mixed prompts, and a shared cache (`-kvu`) giving
1.2–1.9x more output there. Does `2026-10-14-concurrency-drafter-3050oem` (job A) change when
rerun with `-kvu`?

**Answer: its drafter ratios stand, and on these longer requests the shared cache is slower.**
Drafter/none matches job A within 0.05 at every level up to 32 except E4B 128 in / 512 out at 8
requests (1.60x against 1.77x) and 32 (1.08x against 1.15x). Output with `-kvu` falls behind
per-slot from 16 requests up: at 64 it is 0.55x (E2B, 512 in / 128 out) to 0.79x (E4B, 128 in /
512 out), and time to first token about doubles. It is steadier: spreads at 4 requests and up are
at most 3.8% on the 512-in shape, against 23–36% per-slot.

| drafter / no drafter, `-kvu` | c=1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2B, 512 in / 128 out | 1.56 | 1.17 | 1.28 | 1.28 | 1.05 | 0.99 | 0.94 |
| E2B, 128 in / 512 out | 1.79 | 1.28 | 1.43 | 1.54 | 1.18 | 1.06 | 0.97 |
| E4B, 512 in / 128 out | 1.86 | 1.38 | 1.32 | 1.38 | 1.14 | 1.04 | 0.97 |
| E4B, 128 in / 512 out | 2.05 | 1.44 | 1.41 | 1.60 | 1.26 | 1.08 | 0.95 |

| shared ÷ per-slot output, no drafter | c=1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2B, 512 / 128 | 1.00 | 0.96 | 0.95 | 0.94 | 0.86 | 0.73 | 0.55 |
| E2B, 128 / 512 | 1.00 | 0.97 | 0.97 | 0.94 | 0.90 | 0.82 | 0.70 |
| E4B, 512 / 128 | 1.00 | 0.98 | 0.97 | 0.95 | 0.91 | 0.81 | 0.64 |
| E4B, 128 / 512 | 1.00 | 0.99 | 0.97 | 1.00 | 0.94 | 0.89 | 0.79 |

- **E4B with the drafter fits 64 slots at `-c 65536` with `-kvu`** (6,936 MiB); per-slot it did
  not start. Its ratio at 64 is new: 0.97x and 0.95x.
- **Which cache wins depends on how many tokens all requests hold together.** Here each request
  holds up to 640 tokens (up to 41k cells across 64), and shared is slower; in the mixed-prompt
  run each held under 300 and shared was faster. That fits each request's attention covering the
  whole shared cache, so its cost grows with the total held (inferred; not traced in the source).
- Prompt cache 0 tokens in every cell (`metrics-*.prom`).

**What this means:** neither cache setting is right for every load on llama.cpp `77c4540ce`:
per-slot stalls at random, shared costs more as total context grows. The rigs serve at
`--parallel 1`, where the two are identical. Job A's no-drafter E2B numbers, and the 2026-10-13
sweep they reproduce, stand as the per-slot result.

## Method

`run.sh`, job 1 of `../2026-10-15-queue-3050oem`, 16:10–17:33. Job A's script unchanged except the
server binary (`../2026-10-15-queue-3050oem/kvu-server.sh` adds `-kvu`): `sweep.py`, c = 1–64,
5 repeats, unique prompts with the prompt cache defeated, 512 in / 128 out and 128 in / 512 out,
thinking on (as `sweep.py` sends), one server per (model, arm) at `-c 65536 --parallel 64`, the
rigs' flags, drafted arms at 2 tokens. `summarize.py` → `summary.txt`. Job A ran 2026-10-14 on
the same binary; one request at a time reproduces it within 0.6%.
