# 2026-10-23 — answer quality through the OpenAI server, against llama.cpp on the same card

GSM8K (1,319, 2,048-token limit) and BFCL simple (400) through `jev-tpu-v5e1/gen_eval.py`, greedy, one request at a
time, against `openai_server.py --max-ctx 8192` on the GTX 1650 Ti. Reference: llama.cpp `77c4540ce` on the same card
with the exact Q4_0 GGUF, no drafter, 8 at a time (`local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-14-drafter-accuracy-1650ti`,
arm `e2b-none`). Same harness, sets, token limits and chat template (byte-identical; `/apply-template` renders the
same prompt text). Paired with `jev-tpu-v5e1/gen_compare.py` (95% range from 10,000 resamples of questions).

#### Results

| | llama.cpp | PyTorch | Difference (95% range) | Only llama.cpp right | Only PyTorch right |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K | 0.8992 | 0.8992 | +0.000 (−0.011 to +0.011) | 26 | 26 |
| BFCL simple | 0.9225 | 0.9200 | −0.003 (−0.007 to +0.000) | 1 | 0 |

Files: `gsm8k-pytorch-fixed.jsonl`, `bfcl_simple-pytorch-fixed.jsonl`, `driver.log`. GSM8K took 6,775 s one at a
time (llama.cpp 3,096 s at 8 at a time); 4 replies hit the 2,048-token limit on each engine.

#### The prompt-attention fix

The first pass (`gsm8k-pytorch.jsonl`, `bfcl_simple-pytorch.jsonl`) scored GSM8K 0.881 and BFCL 0.915. On sliding
layers the engine passed PyTorch's memory-efficient SDPA an explicit window mask with K/V broadcast across the eight
heads (stride 0). That kernel, torch 2.13 on this card, gets the last query row wrong whenever a chunk holds 32k+1
queries (`sdpa_repro.txt`: relative error 1.0–1.7 on that row; contiguous K/V or one head give 0.000). The last row is
the last prompt token, whose state picks the first reply token and whose K/V every later token reads.

- `prefill_attn_sweep.txt`: sliding layers wrong at n = 33, 65, …, 225 at four chunk offsets; full layers, which use
  the built-in causal mask, clean at every n.
- `first_token_vs_llamacpp.jsonl`, `first_token_stock_cpu.jsonl`: on three 161-token BFCL prompts llama.cpp and stock
  transformers (bf16, CPU) give `<|tool_call>` log-probability −0.11 to −0.16; the engine gave −1.39 to −2.79. Plain
  prompts of 10–279 tokens agreed within about 0.1 (`first_token_bisect.jsonl`).
- `emb4_rows_check.txt`: the int4 vocabulary rows match Google's QAT tables at 0.0015–0.0019 relative error for special
  and ordinary tokens alike. `layer_maxabs.txt`: no fp16 overflow (largest activation 109).
- `layer0_submodules.txt`: q, k and v at the last position match stock; the attention output does not.

The fix copies K/V for sliding layers (at most 768 keys); every chunk length 1–256 at four offsets then matches an
fp32 reference, worst relative error 2.5e-4 (`prefill_attn_sweep_fixed.txt`). `fix_scope.txt`: the fix left every reply
on the 1,281 GSM8K and 388 BFCL prompts it cannot reach byte-identical, and the 38 GSM8K prompts it reaches went from 12
right to 36 (llama.cpp 36). `affected_prompts.txt` counts them against the first pass.

#### The server

`curl_examples.sh` / `curl_examples.out` (fixed engine, server log `server-curl.log`): a plain chat (34 tokens, 85.23 tok/s decode) and a tool call returned as
OpenAI `tool_calls` (`get_weather`, `{"city": "Paris"}`). The 6 BFCL replies with no tool call on the fixed engine are
the model asking a question back; `replay_bfcl.jsonl` replays the first pass's 8, the other 2 being the attention bug.

#### Not measured

Thinking on; more than one request at a time (the server serves one).
