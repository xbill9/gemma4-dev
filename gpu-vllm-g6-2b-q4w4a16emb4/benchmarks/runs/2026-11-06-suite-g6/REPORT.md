# 2026-11-06 — concurrency sweeps and GSM8K/BFCL for the six engines on one NVIDIA L4 (EC2 G6)

Step 6 of `../../../flight/FLIGHT.md`: the 3050's concurrency sweeps (1–64 requests) and GSM8K/BFCL accuracy, run
for every engine of the G6 flight on the same instance as `../2026-11-05-engines-g6` (`i-040ed957b67c2d514`,
g6.4xlarge, one L4, driver 595.112.07, the same engine builds, weights and drafters). One engine served at a time;
all 19 jobs ran in one sitting from 04:05 UTC (LiteRT Q4_0 GSM8K still running at this commit), driven by `suite.py`.

## Method

- **Sweeps** (`sweep.py`, the 3050 rig's, unchanged): 1, 2, 4, 8, 16, 32, 64 requests at once, two shapes
  (512 tokens in / 128 out and 128 in / 512 out), 5 repeats per level, a unique prompt per request, streaming.
  Output tok/s is the median repeat. Servers as on the 3050: vLLM and SGLang at 0.85 of the card, 64 sequences,
  16,384 tokens; llama.cpp `-c 65536 --parallel 64`; drafters W4A16 at 1 and 4 tokens (vLLM), 4 (SGLang),
  exact Q4_0 at 2 (llama.cpp). `mistralrs` keeps the 3050 limits (4 sequences, 8,192-token cache);
  `mistralrs-card` is the most the L4 loads (32 sequences, 16,384 tokens, `../2026-11-05-engines-g6`).
  PyTorch and LiteRT-LM serve one request at a time, so they have no sweep.
- **Accuracy** (`jev-tpu-v5e1/gen_eval.py`, as on the 3050): GSM8K (1,319 questions, up to 2,048 tokens) and BFCL
  simple (400), greedy, thinking off, **one request at a time**. llama.cpp `-c 131072 --parallel 1`, the 3050
  setting. LiteRT-LM through `proxy.py` (the 3050's: it moves `max_tokens` to `max_completion_tokens`).
- 3050 references: each engine's own 3050 sweep and accuracy run at the same settings (listed in `analyze.py`).
  llama.cpp's 3050 accuracy is on `77c4540ce`; everything else on both cards is the same build.

## Sweeps (`SWEEP.txt`)

```
in512-out128: output tok/s on the L4 (5 repeats, unique prompts)
arm                   1       2       4       8      16      32      64   peak@c  x c=1  vs vllm@64
vllm              132.8   249.5   447.7   738.3  1080.7  1393.8  1593.7   1593.7@64 12.00   1.000
vllm-d1           223.4   406.7   690.5  1042.2  1388.9  1605.6  1710.0   1710.0@64  7.66   1.073
vllm-d4           345.2   597.1   899.5  1217.1  1337.6  1499.8  1629.3   1629.3@64  4.72   1.022
sglang            129.6   242.8   433.9   708.3  1025.8  1208.6  1373.0   1373.0@64 10.60   0.862
sglang-d4         213.7   385.9   576.5   710.5   850.5   969.0  1004.5   1004.5@64  4.70   0.630
llama             125.4   219.5   326.3   428.1   523.3   589.0   630.8    630.8@64  5.03   0.396
llama-d2          224.4   327.4   426.9   494.4   571.9   591.6   622.0    622.0@64  2.77   0.390
mistralrs         133.1   243.6   433.7   412.8   401.3   394.5   389.5    433.7@4   3.26   0.244
mistralrs-card    133.1   243.9   438.6   591.2  1120.8  1030.8  1132.0   1132.0@64  8.51   0.710
drafter against no drafter, same engine and shape:
  vllm-d1      /vllm     c1=1.682  c2=1.630  c4=1.542  c8=1.412  c16=1.285  c32=1.152  c64=1.073
  vllm-d4      /vllm     c1=2.600  c2=2.393  c4=2.009  c8=1.648  c16=1.238  c32=1.076  c64=1.022
  sglang-d4    /sglang   c1=1.649  c2=1.590  c4=1.329  c8=1.003  c16=0.829  c32=0.802  c64=0.732
  llama-d2     /llama    c1=1.790  c2=1.492  c4=1.308  c8=1.155  c16=1.093  c32=1.005  c64=0.986

in128-out512: output tok/s on the L4 (5 repeats, unique prompts)
arm                   1       2       4       8      16      32      64   peak@c  x c=1  vs vllm@64
vllm              138.8   272.9   526.1   991.0  1744.8  2800.2  3715.7   3715.7@64 26.78   1.000
vllm-d1           244.7   478.0   910.6  1654.5  2733.2  3664.7  4364.5   4364.5@64 17.83   1.175
vllm-d4           452.9   855.7  1482.6  2398.8  2823.1  3479.3  4516.3   4516.3@64  9.97   1.215
sglang            134.6   265.1   514.0   967.4  1673.3  2257.1  2967.1   2967.1@64 22.04   0.799
sglang-d4         283.4   552.2   874.6  1223.0  1511.0  1829.1  2032.6   2032.6@64  7.17   0.547
llama             133.8   245.7   408.6   597.5   794.3   977.6  1056.3   1056.3@64  7.89   0.284
llama-d2          271.5   429.6   596.2   791.3  1020.8  1131.9  1170.3   1170.3@64  4.31   0.315
mistralrs         132.9   252.2   461.1   464.7   464.5   463.6   462.1    464.7@8   3.50   0.124
mistralrs-card    133.9   254.3   468.0   626.5  1339.0  1843.0  1825.2   1843.0@32 13.77   0.491
drafter against no drafter, same engine and shape:
  vllm-d1      /vllm     c1=1.764  c2=1.752  c4=1.731  c8=1.670  c16=1.567  c32=1.309  c64=1.175
  vllm-d4      /vllm     c1=3.264  c2=3.136  c4=2.818  c8=2.421  c16=1.618  c32=1.243  c64=1.215
  sglang-d4    /sglang   c1=2.105  c2=2.083  c4=1.701  c8=1.264  c16=0.903  c32=0.810  c64=0.685
  llama-d2     /llama    c1=2.029  c2=1.748  c4=1.459  c8=1.324  c16=1.285  c32=1.158  c64=1.108
```

#### Findings

- **vLLM is the fastest engine at every level from 2 requests up**: 1,593.7 tok/s at 64 requests on 512 / 128 and
  3,715.7 on 128 / 512, 12.0x and 26.8x one request, with no level failing. SGLang matches it for one request
  (129.6 against 132.8 tok/s) and reaches 0.86x / 0.80x of it at 64. llama.cpp reaches 0.40x / 0.28x
  (630.8 / 1,056.3 tok/s).
- **vLLM's drafter pays at every level on the L4**: with 4 drafted tokens 2.60x / 3.26x no drafter for one
  request (345.2 / 452.9 tok/s), falling to 1.02x / 1.22x at 64; with 1 token 1.07x / 1.18x at 64. 4 tokens is
  ahead of 1 token up to 8 requests on 512 / 128 and up to 16 on 128 / 512. **The highest output measured is vLLM
  with 4 drafted tokens at 64 requests on 128 / 512: 4,516.3 tok/s.**
- **SGLang's drafter loses from 16 requests** (0.83x / 0.90x at 16, 0.73x / 0.69x at 64), as on the 3050, but by
  less: its output keeps growing to 1,004.5 / 2,032.6 tok/s at 64, where on the 3050 it stopped at about 270 / 520.
- **llama.cpp's drafter (2 tokens) pays more than on the 3050 and loses only once**: 1.79x / 2.03x for one request
  (3050: 1.59x / 1.77x), still 1.11x at 64 on 128 / 512 (3050: 1.00x) and 0.99x at 64 on 512 / 128 (3050: 0.95x).
- **mistral.rs at the 3050 limits stops at 4 requests** (433.7 / 464.7 tok/s), because it serves 4 at once.
  At the L4's limits (32 sequences) it reaches 1,132.0 / 1,843.0 tok/s, 0.71x / 0.49x vLLM at 64, ahead of
  llama.cpp at every level (1.79x / 1.73x at 64).

## Against the 3050 (`COMPARE-3050.txt`)

```
in512-out128: L4 output tok/s / 3050 output tok/s at each level
arm                   1       2       4       8      16      32      64   L4 peak  3050 peak  ratio
vllm              1.260   1.361   1.523   1.749   2.035   2.373   2.643    1593.7      603.1  2.643
vllm-d1           1.346   1.513   1.742   2.053   2.454   2.721   2.845    1710.0      601.0  2.845
vllm-d4           1.471   1.753   2.165   2.528   2.674   2.830   2.952    1629.3      551.9  2.952
sglang            1.253   1.364   1.551   1.811   2.106   2.267   2.524    1373.0      544.0  2.524
sglang-d4         1.356   1.802   2.373   2.746   3.159   3.576   3.733    1004.5      271.0  3.707
llama             1.315   1.365   1.622   1.877   1.627   1.661   1.716     630.8      367.6  1.716
llama-d2          1.480   1.709   1.661   1.698   1.674   1.683   1.777     622.0      351.5  1.770
mistralrs         1.234   1.218   1.562   1.528   1.516   1.513   1.507     433.7      277.7  1.562

in128-out512: L4 output tok/s / 3050 output tok/s at each level
arm                   1       2       4       8      16      32      64   L4 peak  3050 peak  ratio
vllm              1.156   1.178   1.209   1.275   1.412   1.813   2.250    3715.7     1651.2  2.250
vllm-d1           1.157   1.197   1.270   1.431   1.869   2.325   2.633    4364.5     1657.7  2.633
vllm-d4           1.201   1.330   1.731   2.160   2.326   2.593   2.952    4516.3     1529.9  2.952
sglang            1.125   1.147   1.182   1.241   1.319   1.422   1.734    2967.1     1711.3  1.734
sglang-d4         1.245   1.518   2.048   2.489   2.925   3.534   3.930    2032.6      517.6  3.927
llama             1.249   1.272   1.555   1.912   1.561   1.649   1.668    1056.3      633.3  1.668
llama-d2          1.430   1.684   1.585   1.663   1.662   1.718   1.857    1170.3      658.8  1.776
mistralrs         1.189   1.139   1.490   1.505   1.507   1.506   1.500     464.7      309.4  1.502
```

For one request the L4 is 1.13–1.48x the 3050. **The gap grows with load for vLLM and SGLang**: at 64 requests
vLLM gives 2.64x / 2.25x the 3050's output, and SGLang with its drafter 3.73x / 3.93x. llama.cpp stays at
1.56–1.91x from 4 requests up, and mistral.rs at the 3050's limits at 1.49–1.56x. Two hosts on different days, so a ratio here compares two machines, card and host together.

## Accuracy (`ACCURACY.txt`)

```
gsm8k: one request at a time, thinking off
arm                n right     acc trunc   wall s x llama  3050 acc  3050 s L4/3050 time  L4-3050 paired [95%]       differ   L4 arm - L4 llama [95%]
llama           1319  1185  0.8984     5   3970.9   1.000    0.9014    4993        0.795  -0.0030 [-0.0136, +0.0068]    50
llama-d2        1319  1184  0.8976     2   2006.5   0.505    0.8969    2831        0.709  +0.0008 [-0.0091, +0.0106]    43   -0.0008 [-0.0099, +0.0091]
vllm            1319  1183  0.8969     5   3918.4   0.987    0.9014    4360        0.899  -0.0045 [-0.0129, +0.0038]    32   -0.0015 [-0.0121, +0.0099]
vllm-d4         1319  1186  0.8992     4   1270.2   0.320    0.8992    1478        0.859  +0.0000 [-0.0091, +0.0091]    36   +0.0008 [-0.0106, +0.0121]
sglang          1319  1181  0.8954     5   4035.2   1.016    0.8992    4491        0.898  -0.0038 [-0.0121, +0.0038]    29   -0.0030 [-0.0136, +0.0076]
sglang-d4       1319  1186  0.8992     2   1955.5   0.492    0.8992    2601        0.752  +0.0000 [-0.0083, +0.0083]    28   +0.0008 [-0.0091, +0.0114]
mistralrs       1319  1189  0.9014     2   3918.4   0.987    0.9030    4664        0.840  -0.0015 [-0.0121, +0.0099]    54   +0.0030 [-0.0076, +0.0144]
pytorch         1319  1182  0.8961     5   3990.8   1.005    0.8992    4377        0.912  -0.0030 [-0.0068, +0.0008]     6   -0.0023 [-0.0129, +0.0083]
litert-int248   1319  1175  0.8908     0   5017.7   1.264    0.8795    7266        0.691  +0.0114 [-0.0023, +0.0258]    93   -0.0076 [-0.0220, +0.0068]
litert-q4      (not run)

bfcl_simple: one request at a time, thinking off
arm                n right     acc trunc   wall s x llama  3050 acc  3050 s L4/3050 time  L4-3050 paired [95%]       differ   L4 arm - L4 llama [95%]
llama            400   370  0.9250     0    111.2   1.000    0.9225     152        0.730  +0.0025 [-0.0050, +0.0125]     3
llama-d2         400   368  0.9200     0     62.7   0.564    0.9250      98        0.641  -0.0050 [-0.0125, +0.0000]     2   -0.0050 [-0.0125, +0.0000]
vllm             400   369  0.9225     0    102.6   0.923    0.9250     140        0.733  -0.0025 [-0.0075, +0.0000]     1   -0.0025 [-0.0075, +0.0000]
vllm-d4          400   369  0.9225     0     35.1   0.316    0.9225      58        0.606  +0.0000 [+0.0000, +0.0000]     0   -0.0025 [-0.0075, +0.0000]
sglang           400   366  0.9150     0    109.7   0.987    0.9150     133        0.827  +0.0000 [-0.0075, +0.0075]     2   -0.0100 [-0.0200, -0.0025]
sglang-d4        400   366  0.9150     0     56.9   0.512    0.9175      75        0.759  -0.0025 [-0.0075, +0.0000]     1   -0.0100 [-0.0200, -0.0025]
mistralrs        400   368  0.9200     0     92.8   0.835    0.9150     119        0.779  +0.0050 [+0.0000, +0.0125]     2   -0.0050 [-0.0175, +0.0075]
pytorch          400   369  0.9225     0    124.5   1.120    0.9225     156        0.800  +0.0000 [+0.0000, +0.0000]     0   -0.0025 [-0.0125, +0.0050]
litert-int248    400   330  0.8250     0    311.2   2.799    0.8250     478        0.652  +0.0000 [-0.0125, +0.0125]     6   -0.1000 [-0.1350, -0.0675]
litert-q4      (not run)
```

`differ` counts questions right on one card and wrong on the other; the paired range is a 95% bootstrap over
questions. `x llama` and `L4/3050 time` are wall-clock ratios.

#### Findings

- **The L4 gives the same answers as the 3050**: every engine's paired range against its own 3050 run includes
  zero on both tasks. GSM8K 0.8954–0.9014 for the five CUDA engines, with and without drafters; LiteRT int248
  0.8908. BFCL 0.915–0.925; LiteRT int248 0.825, the same 330 of 400 as on the 3050, and LiteRT Q4_0 0.860
  against 0.8625.
- **Between engines on the L4, GSM8K is level for every engine against llama.cpp** (paired ranges include zero).
  On BFCL, SGLang (0.915, with or without its drafter) is below llama.cpp (0.925, range −0.0200 to −0.0025), as it
  was 4 behind vLLM on the 3050; LiteRT int248 is 0.100 below.
- **Drafters change no score and cut the time**: GSM8K takes 1,270 s with vLLM's drafter, 0.32x llama.cpp without
  one (3,971 s); llama.cpp's drafter 0.51x, SGLang's 0.49x. Without drafters the CUDA engines take
  3,918–4,035 s, and LiteRT int248 5,018 s (1.26x).
- One request at a time, GSM8K runs 0.69–0.91x the 3050's time. The 2026-10-26 L4 llama.cpp run
  (`gpu-llamacpp-g6-2b-q4_0/benchmarks/runs/2026-10-26-drafter-g6`, 1,013.6 s) sent 8 requests at once, so its
  time does not compare with these.

## Files

- `sweep-<arm>/` — `in512-out128/` and `in128-out512/` (`concurrency.json`, `metrics.prom`), the sweep logs,
  `command.txt` (the server's command line), `server.log`, `gpu.csv` (nvidia-smi once a second).
- `acc-<arm>/` — `gsm8k.jsonl`, `bfcl_simple.jsonl` (every question, reply and verdict), their logs, `command.txt`,
  `server.log`, `gpu.csv`; LiteRT arms also `config.json`.
- `driver.log` — every job's start, readiness, result and end.
- `suite.py`, `proxy.py`, `sweep.py` — the driver, the LiteRT proxy and the sweep as run (copies of `../../../flight/`
  and `../../../sweep.py`); `analyze.py` writes `SWEEP.txt`, `COMPARE-3050.txt` and `ACCURACY.txt`.
