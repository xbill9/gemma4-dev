# 2026-10-14 — the 1650 Ti on llama.cpp `77c4540ce`: drafter speed and answer quality

**Answer: the drafter costs no answers on this card, and with 8 requests at once it slows the work
down.** GSM8K and BFCL with the exact Q4_0 drafter are level with no drafter on both models (every
95% range includes zero), but at `--parallel 8` the drafted arms finish 0.88–0.96x as fast. One
request at a time, the drafter speeds E2B up 1.16x and E4B 1.50x. On this build E4B's best draft
length is 1 token (1.50x against 1.44x at 2), and at 1 token its greedy text matches no drafter.

## Setup

The laptop `yogagpu` (GTX 1650 Ti Max-Q, 4096 MiB, driver 615.71.09). llama.cpp `77c4540ce`, the
RTX 3050 runs' build, built here into its own worktree `~/llama.cpp-77c4540ce` with the rig's
`make build` flags (`GGML_CUDA=ON`, `CUDA_ARCHITECTURES=75`, `GGML_NATIVE=ON`, Release, `-j6`,
665 s). The rigs keep serving on `f95b0d9`. Drafters: the exact Q4_0 files, hashes checked against
the 3050 run's `sha256.txt` (`sha256.log`). `run.sh` ran everything; `finish.sh` reran the one
piece a memory-pressure stop on the host killed (below).

## Speed, one request at a time (`spec_ab.py`, `summarize.py`)

`2026-10-14-drafter-1650ti`'s method with `LLAMA_SERVER_BIN` pointed at `77c4540ce`: `-c 8192
--parallel 1`, six prompts, greedy ×3 plus one sampled pass, 256 tokens, prompt cache off.

| greedy, geometric mean of 6 prompts | E2B tok/s | x | E4B tok/s | x |
| --- | ---: | ---: | ---: | ---: |
| no drafter | 77.0 | 1.000 | 37.8 | 1.000 |
| drafter, 1 token | 89.6 | 1.164 | **56.6** | **1.497** |
| drafter, 2 tokens | 84.3 | 1.095 | 54.4 | 1.438 |
| no drafter, repeated last | 77.2 | 1.003 | 37.8 | 1.000 |

On `f95b0d9` (`2026-10-14-drafter-1650ti`): E2B 1.180 / 1.109x, E4B 1.432 / 1.430x.

- **Greedy text, E2B**: both drafted arms differ from no drafter on all 18 replies, as on `f95b0d9`.
  On the 3050, E2B at 1 token matched no drafter, so this difference follows the card.
- **Greedy text, E4B**: both no-drafter arms match on all 18 replies, and the 1-token arm matches
  them on all 18; the 2-token arm differs on 15. On `f95b0d9` E4B's greedy text did not repeat
  even without a drafter (`local-llamacpp-1650ti-4b-q4_0/benchmarks/runs/2026-10-14-repeatability-1650ti`).

## Answer quality, 8 requests at once (`accuracy/`)

GSM8K (1,319, 2,048-token limit) and BFCL simple (400) through `jev-tpu-v5e1/gen_eval.py`, 8 at a
time, with the 3050 accuracy run's server flags (`-c 32768 --parallel 8 -ctk f16 -ctv f16 -fa 1`)
and this rig's own (`per_layer_token_embd` on the host). Pairing and 95% ranges are
`gen_compare.compare` (10,000 resamples, seed 0).

| | no drafter | drafter | difference (95% range) | wall time, x vs no drafter |
| --- | ---: | ---: | --- | ---: |
| E2B GSM8K, 1 token | 0.8992 | 0.8946 | −0.0045 (−0.0152 to +0.0061) | 3,096 → 3,479 s, 0.89x |
| E2B BFCL, 1 token | 0.9225 | 0.9250 | +0.0025 (0.0000 to +0.0075) | 284 → 297 s, 0.95x |
| E4B GSM8K, 2 tokens | 0.9318 | 0.9295 | −0.0023 (−0.0091 to +0.0038) | 4,890 → 5,554 s, 0.88x |
| E4B GSM8K, 1 token | 0.9318 | 0.9310 | −0.0008 (−0.0076 to +0.0061) | 4,890 → 5,293 s, 0.92x |
| E4B BFCL, 2 tokens | 0.9100 | 0.9125 | +0.0025 (0.0000 to +0.0075) | 555 → 597 s, 0.93x |
| E4B BFCL, 1 token | 0.9100 | 0.9150 | +0.0050 (−0.0050 to +0.0150) | 555 → 577 s, 0.96x |

- **Scores match the 3050's** no-drafter run of the same day (E2B 0.8984 / 0.9250, E4B 0.9340 /
  0.9100).
- **The drafter slows batched work here; on the 3050 it sped the same work up** (GSM8K 1.51x E2B,
  1.69x E4B at `--parallel 8`). The 1650 Ti rigs serve one request at a time (`PARALLEL_SLOTS=1`),
  where the drafter wins.
- **VRAM**: E2B 1,710 → 1,778 MiB with the drafter, E4B 3,534 → 3,604 MiB, of 4,096.
- The E4B 1-token arm ran the next morning from `../2026-10-14-drafter-gaps-1650ti/run.sh`, same
  flags (`driver.log` there).
- **The interruption**: at about 01:45 the host's Claude Code session stopped `run.sh` for low
  memory partway into BFCL for E4B with 2 tokens. `finish.sh` reran that piece from the start at
  09:54 (`gen_eval.py` rewrites its output; the server log is `server-e4b-q4x-n2-bfcl.log`). Every
  other piece ran once, uninterrupted.
