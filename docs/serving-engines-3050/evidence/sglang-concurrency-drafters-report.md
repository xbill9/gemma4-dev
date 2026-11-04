# Drafters under load, SGLang, RTX 3050 OEM — 2026-11-04

The 1–64-request sweep of `../2026-11-04-concurrency-3050oem` repeated with Google's E2B QAT drafter as W4A16
at 1–4 drafted tokens, one server per arm (`run.sh`, `w4a16-nN/`), the arguments of vLLM's drafter sweep
(`local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-concurrency-drafters-3050oem`): two shapes, unique
prompts, thinking on, median of 5 repeats. **The 4-token arm holds at most 16 running requests** (64 do not
load beside the drafter on 8 GB, `../2026-11-04-engines-3050oem`); beyond 16 its requests queue. The arms ran
14:38–15:51, before an Android build started on the host. `compare.py` prints `COMPARE.txt`.

#### Output tok/s, ratio to SGLang without a drafter (4 drafted tokens, the best length at every level)

| requests | 512/128 none | 4 drafted | 128/512 none | 4 drafted |
|---|---|---|---|---|
| 1 | 103.4 | 157.6 (1.52x) | 119.6 | 227.7 (1.90x) |
| 2 | 178.0 | 214.2 (1.20x) | 231.1 | 363.8 (1.57x) |
| 4 | 279.7 | 242.9 (0.87x) | 434.7 | 427.0 (0.98x) |
| 8 | 391.2 | 258.7 (0.66x) | 779.7 | 491.4 (0.63x) |
| 16 | 487.1 | 269.2 (0.55x) | 1,269.1 | 516.5 (0.41x) |
| 64 | 544.0 | 269.1 (0.49x) | 1,711.3 | 517.3 (0.30x) |

**On SGLang the drafter pays only at 1–2 requests at once.** With a drafter, output stops growing from 8 requests on, at
every draft length (at most 271 tok/s on 512/128 and 518 on 128/512, both at 4 tokens), while without one it
keeps growing to 544 and 1,711. Shorter drafts lose more at every level (1 token: 0.20–0.95x).

**vLLM's drafter holds up under load and SGLang's does not.** vLLM's best draft length falls from 4 tokens to 1
as requests rise and is level with no drafter from 32 requests on (`COMPARE.txt`); SGLang's best stays at 4 and
falls to 0.30–0.49x. Taking each engine's best setting per level (drafter or none), SGLang serves 0.57–0.67x
vLLM at 1–2 requests, 0.44–0.76x at 4–8, and 0.86–1.03x at 16–64, where both run without a drafter.

Tokens per step (median of SGLang's `accept len` lines) on these prompts: 1.88 / 2.60 / 3.17 / 3.61 at 1–4
drafted tokens, higher than on the six chat prompts of `../2026-11-04-engines-3050oem` (1.62–2.15). Why output
stops growing with the drafter was not measured.

Peak memory 6,760–7,716 MiB.

#### Files

`w4a16-nN/{in512-out128,in128-out512}/` (`concurrency.json` and per-request records), `w4a16-nN/server.log`,
`gpu.csv`, `vram.txt`, `command.txt`, `run.log`.
