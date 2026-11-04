# Drafters under load, vLLM and llama.cpp, RTX 3050 OEM — 2026-11-03

The 1–64-request sweep of `../2026-11-03-concurrency-3050oem` repeated with drafters on both engines. It uses
the same `sweep.py` arguments, two shapes (512 in / 128 out, 128 in / 512 out), unique prompts, thinking on,
and the median of 5 repeats per level.
- **vLLM 0.33.0:** the W4A16 drafter at 1–4 drafted tokens (`run.sh`, one server per arm, `w4a16-nN/`).
- **llama.cpp `7182ef71e`:** the exact Q4_0 drafter at 1 and 2 drafted tokens, with the same 64-slot server
  as its morning run (`local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-03-concurrency-drafters-3050oem`).
  At one request, 3 and 4 tokens lose to 2 there (`../2026-11-03-drafters-3050oem`).

`chain.sh` ran both engines in one sitting (18:02–18:57). The no-drafter numbers are from the same morning.
`compare.py` prints `COMPARE.txt`.

#### Output tok/s, ratio to the same engine without a drafter

512 in / 128 out:

| requests | vLLM none | vLLM best drafter | llama.cpp none | llama.cpp 2 tokens | vLLM / llama.cpp, best of each |
|---|---|---|---|---|---|
| 1 | 105.4 | 234.8 (2.23x, 4 tokens) | 95.3 | 151.7 (1.59x) | 1.55 |
| 2 | 183.3 | 340.6 (1.86x, 4) | 160.9 | 191.6 (1.19x) | 1.78 |
| 4 | 294.0 | 449.6 (1.53x, 3) | 201.2 | 257.0 (1.28x) | 1.75 |
| 8 | 422.2 | 512.5 (1.21x, 3) | 228.1 | 291.2 (1.28x) | 1.76 |
| 16 | 531.0 | 566.0 (1.07x, 1) | 321.5 | 341.6 (1.06x) | 1.66 |
| 32 | 587.3 | 590.0 (1.00x, 1) | 354.7 | 351.5 (0.99x) | 1.66 |
| 64 | 603.1 | 601.0 (1.00x, 1) | 367.6 | 350.0 (0.95x) | 1.64 |

128 in / 512 out:

| requests | vLLM none | vLLM best drafter | llama.cpp none | llama.cpp 2 tokens | vLLM / llama.cpp, best of each |
|---|---|---|---|---|---|
| 1 | 120.0 | 377.2 (3.14x, 4 tokens) | 107.1 | 189.8 (1.77x) | 1.99 |
| 2 | 231.5 | 643.2 (2.78x, 4) | 193.2 | 255.1 (1.32x) | 2.52 |
| 4 | 435.0 | 993.6 (2.28x, 3) | 262.8 | 376.2 (1.43x) | 2.64 |
| 8 | 777.2 | 1,264.7 (1.63x, 3) | 312.6 | 475.7 (1.52x) | 2.66 |
| 16 | 1,235.9 | 1,462.8 (1.18x, 1) | 508.9 | 614.3 (1.21x) | 2.38 |
| 32 | 1,544.3 | 1,576.4 (1.02x, 1) | 592.7 | 658.8 (1.11x) | 2.39 |
| 64 | 1,651.2 | 1,657.7 (1.00x, 1) | 633.3 | 630.1 (0.99x) | 2.62 |

On llama.cpp, 2 drafted tokens beat 1 at every level of both shapes. The full table for every arm is in `COMPARE.txt`.

**With drafters on both engines, vLLM leads llama.cpp at every level**: 1.55–1.78x on 512 in / 128 out
and 1.99–2.66x on 128 in / 512 out. Without drafters the lead is 1.11–1.85x and 1.12–2.61x. The
drafter widens vLLM's lead at 1–4 requests. From 16 requests up it adds little on either engine.

**The best draft length falls as load rises on vLLM**: 4 tokens at 1–2 requests, 3 at 4–8, 1 from 16 up.
At 32–64 requests every length is level with no drafter or behind it: 4 tokens gives 0.87–0.93x, 1 token
1.00–1.02x. **On llama.cpp**, 2 drafted tokens pay up to 16 requests (1.06–1.77x), are level at 32 on
512 / 128 (0.99x) and still ahead there on 128 / 512 (1.11x), and are level or behind at 64 (0.95–0.99x).
The same held on this card in October.

#### These prompts draft easily

The sweep prompt is repeated filler text with a request to summarize it, read with thinking on. vLLM
accepted 0.87 of drafted tokens at 1 token and 0.63–0.72 at 4 here. On the six chat prompts of
`../2026-11-03-drafters-3050oem` it accepted 0.60 and 0.31. So the one-request gains here (2.23x and 3.14x)
are larger than the 1.94x measured on chat prompts. Read the ratios as how the gain changes with load,
and the chat-prompt run as what one user sees. llama.cpp's `/metrics` has no drafted-token counters, so its
acceptance here is not recorded.

#### Memory and variation

- **vLLM:** the drafter comes out of the context cache, since vLLM keeps a fixed share of the card. The
  cache holds 294,299 / 258,017 / 250,172 / 243,094 tokens at 1–4 drafted tokens, against 64 requests of
  about 700 here.
- **llama.cpp:** 2,854 and 2,922 MiB at 1 and 2 tokens, against 2,612 without.
- **Variation:** on 128 / 512, repeats agree within 3.3% at every level. On 512 / 128 llama.cpp varies
  by up to 32%, as it did without the drafter (21–26%). vLLM varies by up to 6% there, except at 2 requests:
  one repeat in five per arm ran at 64–118 tok/s while vLLM compiled an attention kernel for the new batch
  size (`kernel_unified_attention`, logged in `w4a16-nN/vllm-server.log`). The medians are within 1–3% of
  each arm's best repeat.
