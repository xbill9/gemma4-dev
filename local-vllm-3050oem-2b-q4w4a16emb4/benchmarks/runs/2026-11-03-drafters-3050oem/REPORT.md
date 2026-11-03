# Drafters on vLLM and llama.cpp, one request at a time, RTX 3050 OEM — 2026-11-03

Gemma 4 E2B on the QAT Q4_0 grid on both engines, with Google's E2B QAT assistant drafter in every
precision each engine can load, at 1–4 drafted tokens. One script (`drafters.py`) starts each server,
sends the same requests and times both engines the same way; `summarize.py` prints `SUMMARY.txt`.
`chain.sh` ran every arm in one sitting (17:18–18:01), vLLM first, then llama.cpp.

#### Setup

- **vLLM 0.33.0**: this rig's `serve.py` and `tpu.env` (`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` @ `db715f83`),
  drafter through `--speculative-config` (vLLM resolves it as `Gemma4MTPModel`, method `mtp`).
- **llama.cpp `7182ef71e`**: the exact Q4_0 E2B GGUF of `local-llamacpp-3050oem-2b-q4_0` with that rig's serving
  flags at `-c 8192 --parallel 1`, drafter through `-md … --spec-type draft-mtp -ngld all --spec-draft-n-max N`.

| drafter | engine | file | size |
|---|---|---|---|
| bf16, Google's weights | vLLM | `google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant` snapshot `ebc7e1a2` | 158 MB |
| W4A16 (int4 linears, bf16 embedding table) | vLLM | built by `../../../repack/repack_drafter.py`; SHA-256 in `sha256.txt` | 143 MB |
| bf16 | llama.cpp | `gguf_exact.py --bf16` (`local-llamacpp-3050oem-2b-q4_0/.../2026-11-03-drafters-3050oem`) | 170 MB |
| Q8_0 | llama.cpp | llama.cpp's converter at `d1044bc82` | 98 MB |
| exact Q4_0 | llama.cpp | the rig's `SPEC_DRAFT_MODEL` | 59 MB |

The W4A16 drafter converts all 22 linear layers. Read back as bf16, 1,053,736 of its 10,092,544 values
(10.4%) differ from Google's bf16 file, by at most 0.98% (`repack.log`, `repack_report.json`). The exact
Q4_0 GGUF reproduces the source exactly.

**Requests:** six prompts (code, chat, story, explain, facts, summary), each greedy three times and once
sampled (temperature 0.8, seed 1234), 256 tokens, thinking off; prompt cache off on llama.cpp.
- **Speed:** decode tok/s is measured on the stream, from the first to the last token chunk. The figure
  per arm is the geometric mean over the six prompts of each prompt's median.
- **Acceptance:** read from each engine's counters (llama.cpp's reply `timings`, vLLM's spec-decode
  metrics).
- **Drift:** each engine repeats its no-drafter arm last. Both came back within 0.4% of the first.
- **Temperature:** the card was cooled to 50 °C or below before each arm.

#### Result

Greedy decode tok/s and ratio to the same engine with no drafter:

| drafted tokens | vLLM bf16 | vLLM W4A16 | llama.cpp exact Q4_0 | llama.cpp Q8_0 | llama.cpp bf16 | accepted |
|---|---|---|---|---|---|---|
| none | 125.5 | | 110.4 | | | |
| 1 | 188.7 (1.50x) | 191.1 (1.52x) | 149.6 (1.36x) | 149.0 (1.35x) | 138.6 (1.26x) | 0.59–0.60 |
| 2 | 225.2 (1.79x) | 227.3 (1.81x) | **151.3 (1.37x)** | 150.6 (1.36x) | 133.4 (1.21x) | 0.45–0.49 |
| 3 | 237.0 (1.89x) | 239.8 (1.91x) | 145.1 (1.31x) | 143.9 (1.30x) | 123.0 (1.11x) | 0.37–0.38 |
| 4 | 241.1 (1.92x) | **244.0 (1.94x)** | 135.4 (1.23x) | 135.0 (1.22x) | 113.1 (1.02x) | 0.31 |

**The drafter pays more on vLLM.** At its best setting vLLM generates 244.0 tok/s, 1.94x its own speed without
a drafter. llama.cpp's best is 151.3 tok/s, 1.37x. With drafters, vLLM serves one request 1.61x faster than
llama.cpp; without them the gap is 1.14x (125.5 against 110.4). The engines accept the same share of drafted
tokens at each length (vLLM 0.602 / 0.482 / 0.378 / 0.310 W4A16, llama.cpp 0.588 / 0.455 / 0.376 / 0.311 exact
Q4_0), so the difference is in what a drafted step costs. On vLLM the gain keeps growing to 4 tokens. On
llama.cpp it peaks at 2 and falls after, as in its October runs on this card.

**By prompt**, vLLM W4A16 at 4 tokens ranges from 1.42x (story) to 3.32x (code); llama.cpp exact Q4_0 at
2 tokens from 1.13x (story) to 1.82x (code). The sampled pass gives the same order: vLLM 1.84x, llama.cpp
1.33x at those settings.

**Drafter precision:**
- **vLLM:** W4A16 is level with or slightly ahead of bf16 at every length (1.009–1.013x) at the same acceptance.
- **llama.cpp:** exact Q4_0 and Q8_0 are level, and bf16 is behind (0.93x of Q4_0 at 1 token, 0.88–0.84x at 2–4) at the
  same acceptance, as in the 2026-10-14 run.

A 4-bit drafter costs nothing in acceptance on either engine.

#### Memory

- **llama.cpp:** 2,779 MiB with no drafter; the drafter adds 82 MiB (Q4_0), 120 MiB (Q8_0) or 188 MiB (bf16).
- **vLLM:** takes 0.85 of the card in every arm, so the drafter comes out of the context cache. The cache
  holds 351,060 tokens with no drafter, 180,254 at 1 drafted token and 129,049–129,703 at 4. At 4 drafted tokens that is
  enough for 7 requests of the full 16,384 tokens at once, or 64 requests of about 2,000 each.

#### Greedy text

- **llama.cpp:** each server repeats its greedy text exactly. At 1 drafted token the text matches no drafter
  on all six prompts; at 2–4 it differs on all six.
- **vLLM:** each server also repeats its greedy text exactly within one run, but **two vLLM servers started
  with no drafter differ from each other on 3 of 6 prompts** (chat, explain, story). Every drafted vLLM arm
  matches the first no-drafter server on code, facts and summary. So on vLLM, a drafter's effect on greedy
  text can't be told apart from restarting the server.
- **Answers:** on llama.cpp the drafter changes no answer score (GSM8K 0.8969 / 0.9014, BFCL 0.925 / 0.9225
  with / without, `local-llamacpp-3050oem-2b-q4_0/.../2026-10-14-accuracy-fullctx-3050oem`). vLLM with a
  drafter has not been scored.

#### Files

- `results.jsonl`: one row per request.
- `vllm.out`, `llama.out`: per-request lines.
- `server-ENGINE-ARM.log`: one log per server.
- `SUMMARY.txt`: per-prompt ratios.
