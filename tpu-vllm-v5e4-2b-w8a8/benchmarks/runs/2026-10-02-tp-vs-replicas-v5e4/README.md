# 2026-10-02-tp-vs-replicas-v5e4

Which way of using the four chips of a v5e-4 serves `xbill9/gemma-4-E2B-it-qat-w8a8-int8` fastest. Every layout ran on
one VM, one boot: flex-start `v5litepod-4` in `us-west4-a`, created through this rig's own `create_tpu_queued_resource`,
on the patched image the boot script builds (`vllm-tpu-w8a8:patched`). Each later container reuses the boot container's
image, environment (`MIN_TOKEN_BUCKET=64`, `GMM_V2_TILE_VMEM_FRACTION=0.85`) and `vllm serve` arguments
(`--max-model-len 16384 --max_num_batched_tokens 512 --gpu-memory-utilization 0.80`, the gemma4 tool and reasoning
parsers), changing only what its row names.

#### Result

- 🟢 **Four one-chip engines are the fastest layout**, and `--data-parallel-size 4` delivers them behind one port: 2.52x
  TP=4's output at 64 concurrent requests and 3.36x at 256, 1.77x on long prompts with a third of the time to first token.
- 🟢 **The rig should default to `--data-parallel-size 4 --tensor-parallel-size 1`.** The image supports it on TPU:
  tpu_inference resolves it to `TPU_MULTIPROCESS_DP=1`, starts four engine processes each pinned to one chip
  (`TPU_VISIBLE_CHIPS=0..3`, `TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1`), and runs four API servers on port 8000.
- ⚠️ At 1 request DP=4 gives 0.93x of one chip and TP=4 gives 1.08x; one engine handles each request, so a lone
  request gets one chip's speed in any data-parallel layout.
- ⚠️ Each DP engine holds 333,344 KV tokens, the one-chip pool. TP=4 holds 631,968 in one pool, which only matters
  for a single request longer than 333k tokens; `--max-model-len` is 16,384.
- 🟢 **TP=1, TP=2 and TP=4 give the same output within 11%.** A one-chip server here matches the v5e-1 sibling's
  same-method boot (174.7 / 581.1 / 1,386.0 at 1 / 4 / 16), so TP=4 is about one chip's worth of serving spread over four.
- ❌ **Splitting does not shorten the step because the attention decode kernel does not shrink.** Per chip it costs
  803 ms of a 1,461 ms burst at TP=1, 839 at TP=2 and 831 at TP=4. E2B has one KV head, so every chip holds and reads
  the whole KV cache. Matmul time does split (443 → 230 → 130 ms) and collectives add back 206 ms at TP=2 and 243 at
  TP=4, so the two cancel.
- 🟢 **`--max-model-len` sets the attention kernel's cost, and it is the largest lever measured.** At 2,048 the same
  burst's attention kernel takes 117 ms against 803, and one chip serves 2,744 tok/s at 16 requests (1.98x) and 6,705
  at 64 (3.12x). The kernel variant compiled for 16,384 reads `bkv_16384` blocks on every layer, including the
  sliding-window layers that attend over 512 tokens (`RPAd-...-bkv_16384_16384-sw_512`, 666 of the 803 ms).
  DP=4 at `--max-model-len 8192` reaches 7,788 tok/s at 64 requests and 14,189 at 256, 3.59x and 5.85x TP=4.
- ⚠️ `--max-num-seqs 16` costs throughput on its own (0.92x at 16 requests, capped at 1,308 tok/s at 64). The
  2.0x gap to the `jev-tpu-v5e1` sweep's numbers comes from that sweep's `--max-model-len 2048`.

#### Method

Load client: `scripts/multi_client.py`, a copy of `../../../../jev-tpu-v5e1/tpu/w4a16_client.py` that takes a
comma-separated `JEV_LOAD_URLS` and sends request i of each pass to URL `i mod n`; with one URL it behaves as the
original. `load` = N parallel chats of exactly 256 output tokens (`ignore_eos`, greedy), one untimed warm-up pass, three
timed passes, median. `loadlong` = 16 streamed requests of about 2,048 prompt tokens with a unique prefix per request
(`JEV_LOAD_INPUT_TOKENS=2048`). Run on the VM against localhost. The four-replica row splits each pass evenly over ports
8000-8003; the DP rows send everything to port 8000 and vLLM's DP coordinator balances it.

Chip pinning for the one-chip and two-chip servers: `TPU_VISIBLE_CHIPS`, `TPU_CHIPS_PER_PROCESS_BOUNDS`,
`TPU_PROCESS_BOUNDS=1,1,1`, `TPU_PROCESS_PORT` (unique per process), `TPU_PROCESS_ADDRESSES=localhost:<port>`,
`CLOUD_TPU_TASK_ID=0` — the same variables tpu_inference's own multi-process DP sets. Chips 0 and 1 form a pair with
`TPU_CHIPS_PER_PROCESS_BOUNDS=2,1,1`; `1,2,1` fails at libtpu init with `Mesh build failed, duplicate coordinate
assignment` (`vllm-tp2-bounds121-fail.log`). `scripts/launch.py` builds each `docker run` from `docker inspect` of the
boot container.

Profiles: vLLM's `--profiler-config '{"profiler":"torch",...}'` plus `/start_profile` and `/stop_profile` around one burst
of 16 requests x 128 output tokens after a warm-up burst (`scripts/prof.py`), summarized with xprof's `hlo_stats`
inside the container (`scripts/anahlo.py`, `scripts/rpanames.py`): self time per HLO category, summed over chips and
divided by the chip count. "Collectives" is all-reduce + all-gather + collective-permute + all-to-all.

Compile cache: from the TP=4 relaunch on, every container ran with `-e VLLM_XLA_CACHE_PATH=/xla-cache -v
/opt/xla-cache:/xla-cache`, one directory shared by every engine, with no write errors. The engine-start table says which
starts found their own programs in it.

#### Verification

- `/v1/models` lists the checkpoint (`verify-models.json`).
- Greedy chat, "What is the capital of Australia? Answer in one sentence.": `The capital of Australia is Canberra.` from
  the TP=4 boot (`verify-chat-tp4.json`), from each of the four replicas (`verify-chat-replicas.jsonl`) and from the
  DP=4 server (`verify-chat-dp4.json`).
- Each layout's KV pool and weights per chip are in its log: 1.75 GiB per chip at TP=4, 3.46 at TP=2, 6.88 at TP=1.

#### Layouts, output tok/s by concurrent requests

| Layout | Chips per engine | Engines | KV tokens per engine | 1 | 4 | 16 | 64 | 256 | Long 16 x 2,048: output tok/s | total tok/s | TTFT median s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TP=4, one server (rig boot) | 4 | 1 | 631,968 | 189.0 | 643.7 | 1,464.6 | 2,167.5 | 2,426.2 | 1,044.8 | 9,600.6 | 0.579 |
| TP=2, one server, chips 0-1 | 2 | 1 | 532,416 | 175.6 | 580.7 | 1,373.4 | 2,111.0 | — | 953.1 | 8,758.0 | 0.688 |
| TP=1, one server, chip 0 | 1 | 1 | 333,344 | 174.7 | 580.4 | 1,386.7 | 2,148.9 | — | 1,006.7 | 9,250.7 | 0.57 |
| 4 x TP=1, ports 8000-8003 | 1 | 4 | 333,344 | 174.7 | 696.4 | 2,319.2 | 5,537.0 | 8,513.4 | 2,004.2 | 18,416.3 | 0.181 |
| `--data-parallel-size 4`, one port | 1 | 4 | 333,344 | 162.1 | 654.3 | 2,033.2 | 5,471.8 | 8,158.5 | 1,852.8 | 17,025.6 | 0.181 |
| `--data-parallel-size 4`, second start | 1 | 4 | 333,344 | 162.2 | 584.8 | 2,102.0 | 5,270.3 | 8,115.8 | 1,852.2 | 17,019.7 | 0.182 |
| `--data-parallel-size 4 --max-model-len 8192` | 1 | 4 | 333,312 | 194.3 | 773.7 | 2,872.4 | 7,788.2 | 14,188.7 | 2,433.6 | 22,362.3 | 0.168 |

At 16 requests: DP=4 / TP=4 = 1.39x; DP=4 at 8,192 / TP=4 = 1.96x.  
At 64 requests: DP=4 / TP=4 = 2.52x; DP=4 at 8,192 / TP=4 = 3.59x.  
At 256 requests: DP=4 / TP=4 = 3.36x; DP=4 at 8,192 / TP=4 = 5.85x.  

#### Ratio to TP=1 on one chip, same VM, same arguments

| Layout | 1 | 4 | 16 | 64 | Long 16 |
|---|---:|---:|---:|---:|---:|
| TP=4, one server (rig boot) | 1.08x | 1.11x | 1.06x | 1.01x | 1.04x |
| TP=2, one server, chips 0-1 | 1.01x | 1.00x | 0.99x | 0.98x | 0.95x |
| TP=1, one server, chip 0 | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |
| 4 x TP=1, ports 8000-8003 | 1.00x | 1.20x | 1.67x | 2.58x | 1.99x |
| `--data-parallel-size 4`, one port | 0.93x | 1.13x | 1.47x | 2.55x | 1.84x |

TP=4 relaunched (same arguments plus profiler and compile cache), 16 requests: 1,464.4; at boot 1,464.6.

#### One chip, TP=1: `--max-model-len` and `--max-num-seqs`

| `--max-model-len` | `--max-num-seqs` | KV tokens | 1 | 4 | 16 | 64 | Long 16 x 2,048: output tok/s | 16 vs 16,384/256 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16,384 | 256 (default) | 333,344 | 174.7 | 580.4 | 1,386.7 | 2,148.9 | 1,006.7 | 1.00x |
| 8,192 | 256 (default) | 333,312 | 207.8 | 741.4 | 2,056.3 | 3,751.9 | 1,506.4 | 1.48x |
| 4,096 | 256 (default) | 333,312 | 214.8 | 798.4 | 2,478.4 | 5,384.5 | 1,629.9 | 1.79x |
| 2,048 | 256 (default) | 333,312 | 217.4 | 823.0 | 2,743.5 | 6,705.3 | infeasible | 1.98x |
| 16,384 | 16 | 333,376 | 188.6 | 593.3 | 1,280.4 | 1,308.1 | 967.4 | 0.92x |
| 2,048 | 16 | 333,312 | 218.7 | 831.3 | 2,757.2 | 2,898.5 | infeasible | 1.99x |

#### Profiles: one burst of 16 requests x 128 output tokens, per chip

| Server | Chips | Burst wall s | Device busy ms | Attention decode kernel ms | Matmul fusions ms | Collectives ms | Collectives share |
|---|---:|---:|---:|---:|---:|---:|---:|
| tp1 | 1 | 1.602 | 1,460.9 | 803.0 | 443.1 | 0.0 | 0.0% |
| tp2 | 2 | 1.624 | 1,468.6 | 838.9 | 230.4 | 205.8 | 14.0% |
| tp4 | 4 | 1.539 | 1,364.7 | 831.1 | 130.2 | 243.3 | 17.8% |
| tp1-maxlen2048 | 1 | 1.11 | 727.8 | 116.6 | 450.9 | 0.0 | 0.0% |

| Server | Attention decode kernel variant | ms per chip | calls per chip |
|---|---|---:|---:|
| tp1 | `RPAd-p_32-bq_1_1-bkv_16384_16384-sw_512` | 666.1 | 3,836 |
| tp1 | `RPAd-p_32-bq_1_1-bkv_8192_8192` | 137.0 | 959 |
| tp4 | `RPAd-p_32-bq_1_1-bkv_16384_16384-sw_512` | 690.4 | 3,808 |
| tp4 | `RPAd-p_32-bq_1_1-bkv_8192_8192` | 140.8 | 952 |
| len2k | `RPAd-p_128-bq_1_1-bkv_2048_2048-sw_512` | 84.5 | 3,864 |
| len2k | `RPAd-p_128-bq_1_1-bkv_2048_2048` | 32.1 | 966 |

#### Engine start: `init engine` seconds from vLLM's log

| Container | Compile cache | init engine s | of which compilation s |
|---|---|---:|---:|
| TP=4 boot | off | 297.86 | 258.13 |
| TP=2 | off | 314.03 | 261.17 |
| TP=1 | off | 302.05 | 224.82 |
| 4 x TP=1, replica 0 | off | 305.55 | 227.13 |
| DP=4 (each engine) | off | 307.55 / 308.19 / 308.10 / 308.42 | 229.70 / 229.85 / 229.89 / 230.10 |
| TP=4 relaunch | on, empty | 297.89 | 258.30 |
| TP=1, max-model-len 2048 | on, other shapes only | 170.37 | 115.58 |
| TP=1, max-num-seqs 16 | on, other shapes only | 398.79 | 366.90 |
| TP=1, 2048 and 16 | on, other shapes only | 145.57 | 116.82 |
| TP=1, max-model-len 4096 | on, other shapes only | 125.45 | 113.28 |
| TP=1, max-model-len 8192 | on, other shapes only | 128.83 | 116.67 |
| DP=4, first start with cache (each engine) | on, other shapes only | 229.30 / 229.49 / 229.97 / 229.53 | 217.19 / 216.89 / 217.33 / 217.37 |
| DP=4, second start (each engine) | on, holds this layout | 137.68 / 138.59 / 139.80 / 140.05 | 125.51 / 126.40 / 127.23 / 127.81 |
| DP=4, max-model-len 8192 (each engine) | on, holds TP=1 at 8,192 | 53.34 / 53.43 / 54.08 / 54.57 | 41.11 / 41.22 / 41.53 / 41.91 |

Queued Resource created 2026-10-02T21:57:16.933861380Z, delete issued 2026-10-02T23:53:11+00:00: 1.93 h, 4.64 USD at 2.40 USD/h.

#### What the root files should say (not edited here)

- `QUANTIZATION.md` or `HARDWARE.md`: on this image the RPA v3 decode kernel's block is sized from `--max-model-len`,
  and sliding-window layers run the full-length block; for E2B on v5e, attention cost per step scales with
  `--max-model-len`, not with the context in use.
- `MODELS.md`, E2B: one KV head means tensor parallelism splits matmuls and leaves the attention kernel whole on every
  chip; on a multi-chip host serve E2B data-parallel.

#### Files

| File | What |
|---|---|
| `startup.log`, `queued-resource.json` | boot script output and the Queued Resource at collection time |
| `verify-*.json*` | verification responses |
| `<layout>.load.c<N>.json`, `<layout>.loadlong.c16.json` | load client output; layouts `tp4`, `tp2`, `tp1`, `rep4`, `dp4`, `dp4-rerun`, `dp4-len8k`, `tp4-relaunch`, and the one-chip settings `len8k`, `len4k`, `len2k`, `seqs16`, `jev` (2,048 and 16) |
| `vllm-*.log` | `docker logs` of each container, HTTP access lines removed, addresses redacted |
| `hlo-stats-*.json`, `attention-kernels.json` | profile summaries from xprof `hlo_stats` |
| `trace-steps-tp1.json`, `trace-steps-tp2.json` | per-step module times from the same bursts' trace viewer export (137 model steps; 9.53 ms TP=1, 10.19 ms TP=2) |
| `profile-bursts.jsonl` | wall time of each profiled burst |
| `scripts/` | load client copy, launcher, profiler and summary scripts, `mktables.py` that prints every table above |
