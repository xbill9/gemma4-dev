# 2026-11-04 — SGLang, one request, with and without Google's drafter, on the 3050

SGLang 0.5.23 serving `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` @ `db715f83` on the RTX 3050 OEM, with
the rig's two patches, against vLLM 0.33.0 on the same checkpoint the same day. Runner and prompts:
`local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-engines-3050oem/engines.py` (six prompts,
greedy, 256 tokens, three repeats; four requests at once; memory after warm-up). Drafter: Google's E2B QAT
assistant as W4A16 (`gemma-4-E2B-it-qat-q4_0-w4a16-ct-assistant`, vLLM's best), through
`--speculative-algorithm NEXTN`, which SGLang runs as `FROZEN_KV_MTP`. `SUMMARY.txt` is
`summarize.py` over `results.jsonl`.

#### Result

| arm | tok/s, one request | x no drafter | 4 at once | memory | tokens per step | accepted per drafted token |
|---|---|---|---|---|---|---|
| SGLang | 119.9 | 1.00 | 446.7 | 6,352 MiB | 1 | - |
| SGLang, 1 drafted | 118.9 | 0.99 | 275.1 | 6,768 MiB | 1.62 | 0.62 |
| SGLang, 2 drafted | 141.7 | 1.18 | 319.9 | 7,024 MiB | 1.90 | 0.45 |
| SGLang, 3 drafted | 155.5 | 1.30 | 333.2 | 7,264 MiB | 2.05 | 0.35 |
| SGLang, 4 drafted | **158.9** | **1.33** | 288.0 | 6,198 MiB | 2.15 | 0.29 |
| vLLM | 121.0 | 1.00 | 451.0 | 6,697 MiB | | |
| vLLM, 4 drafted | 229.8 | 1.90 | 481.8 | 7,345 MiB | | |

**Without a drafter SGLang is level with vLLM** (0.99x for one request and for four). **With the drafter vLLM
pays more**: 1.90x against SGLang's best 1.33x, so vLLM with its drafter serves one request 1.45x faster than
SGLang with its own. Acceptance is the same on both engines (vLLM 0.602 / 0.482 / 0.378 / 0.310 at 1–4 drafted
tokens, 2026-11-03), so the difference is in what a drafted step costs SGLang. At four requests at once every
drafted arm is slower than none (0.62–0.75x).

Tokens per step are the median of SGLang's `accept len` log lines (107–146 per arm); accepted per drafted token
is (tokens per step − 1) / drafted tokens.

**4 drafted tokens need fewer running requests.** At `MEM_FRACTION_STATIC=0.80` and 64 running requests the
sliding-window cache leaves no room for the full one, and at 0.85 graph capture runs out of memory
(`server-sglang-d4-mem085.log`). The 4-token arm ran with `MAX_RUNNING_REQUESTS=16` and decode graphs up to 16;
this test sends at most 4 at once.

Greedy text: the drafted arms match no drafter on 1–2 of 6 prompts, and vLLM matches SGLang on 1 of 6.
Every arm repeats its own text on all six prompts.

#### Files

`results.jsonl`, `SUMMARY.txt`, `server-<arm>.log`, `run.out`. Earlier arms in `run.out` failed before the
patches; the server logs of those starts were overwritten by the later ones.
