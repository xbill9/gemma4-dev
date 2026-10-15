# Preregistration: 4-bit QAT repacks against bf16 on one MI300X

Written 2026-10-02, before any droplet was created for this study and before any 4-bit checkpoint
was served on gfx942 in this monorepo. Everything below the line "Results" is filled in after the runs;
nothing above it is edited once the first stage starts, except to append dated amendments at the
end, each saying what changed and why.

## The question

On a TPU v5e the QAT Q4_0 repacks earned their place by making large models fit. On an MI300X
(191.7 GiB) bf16 31B fits with room to spare, so the repack has only one thing left to offer:
**is it faster?** And does it keep the accuracy it kept on v5e?

This card has already turned one kernel-level win into an end-to-end loss: fp8 ran the E2B decode
GEMMs 1.50x faster than bf16 and served at 0.53x–0.80x bf16 (`../gpu-vllm-mi300x-2b`, 2026-09-18).
That is the prior this study is up against.

## Arms

Four rigs, one checkpoint each. Within a pair, the `tpu.env` files differ only in the checkpoint lines.

| Arm | Rig | Checkpoint | Weights on disk |
| --- | --- | --- | --- |
| E2B bf16 | `../gpu-vllm-mi300x-2b` | `google/gemma-4-E2B-it`, served text-only (see below) | 10.28 GB |
| E2B repack | `../gpu-vllm-mi300x-2b-q4w4a16` | `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text` | 6.56 GB |
| 31B bf16 | `../gpu-vllm-mi300x-31b` | `google/gemma-4-31B-it`, served text-only | 62.55 GB |
| 31B repack | this rig | `xbill9/gemma-4-31B-it-qat-q4_0-w4a16-ct-text` | 19.3 GB |

Both repacks are symmetric int4 in groups of 32, compressed-tensors `pack-quantized`, `lm_head`
left bf16 and tied, `Gemma4ForCausalLM` with no vision or audio tower. The E2B bf16 rig's committed
`tpu.env` admits 4 images; for this study it is run with
`LIMIT_MM_PER_PROMPT='{"image": 0, "audio": 0}'` in the environment (a real environment variable
wins over `tpu.env`), so that every arm serves text only. Both bf16 arms still hold their vision
tower's weights in memory. That residual difference is known and is not corrected for.

**Shared settings, every arm:** one droplet, one session, `--max-model-len 32768`,
`--gpu-memory-utilization 0.90`, prefix caching left on (vLLM's default), the gemma4 reasoning and
tool-call parsers, `--async-scheduling`, no `--quantization` flag. **One image for every arm:**
at the start of the session, `check_image` runs on `vllm/vllm-openai-rocm:nightly-rocm100`, and the
digest it resolves to is written into each arm's `VLLM_IMAGE` as `@sha256:…` for the whole session.
A number from one image is never differenced against a number from another. The bf16 E2B rig's
September sweeps are a consistency check only, never a comparison arm.

## Stage 0 — can this image serve a 4-bit checkpoint at all (go / no-go)

`check_image` on the pinned digest must report all three:

1. `Gemma4ModelArchConfigConvertor` registered;
2. `gfx942` in torch's arch list;
3. `TritonW4A16LinearKernel` in vLLM's ROCm mixed-precision kernel list.

If (3) fails, the study stops there. The finding is that the shipped ROCm image has no 4-bit path
for this checkpoint on this card, and it is reported as that. **No patching, no source builds, and
no swapping in an older image inside this study.** Any of those is a new study with its own
preregistration.

## Stage 1 — E2B pair. Stage 2 — 31B pair

Run order: E2B bf16, E2B repack, 31B repack, 31B bf16. The 31B bf16 weights (62.55 GB) are fetched
with `download_weights` while the E2B arms run. Each arm goes through steps 1 to 4, then
`stop_vllm`, before the next arm starts.

### 1. Boot checks — before any number is read

From the boot log (`server_logs`, `grep="Using"` and the memory lines) and `serving_status`, record:

- **which linear kernel each quantized layer got.** A repack arm whose log does not name
  `TritonW4A16LinearKernel` has not tested the 4-bit path. It is recorded as a failed arm, not as a
  slow one;
- weights resident (GiB), KV pool (tokens and GiB), and the time from container start to ready;
- `verify_capabilities`: text, thinking and tool calling must all pass. Vision is not probed.
  The E2B bf16 rig's `server.py` predates the text-only change and still probes vision. With
  images set to 0 that probe fails by design, and the failure is ignored for that arm only.

**How a silent failure would present.** The repack loads, but the weights are dequantized to bf16
at load, or a fallback kernel is chosen. Both show up as a KV pool no larger than its bf16 twin's
and a log that does not name the Triton kernel. Expected pool difference, from arithmetic: the
repack's pool should be larger than its twin's by roughly the difference in resident weights, about
3.7 GB at E2B and about 43 GB at 31B.

### 2. Throughput

`benchmarking_suite.py --concurrency 1 8 64 --input-len 128 1024 8192 --output-len 512 --repeat 3`,
with `--seed-base` distinct per arm, so no arm reads another arm's prefix-cache entries. Nine
cells, three repeats each. The report records `weights_dtype` and `quantization` from `tpu.env`.

### 3. Quality

`../jev-tpu-v5e1/gen_eval.py` runs against each arm over an SSH tunnel
(`ssh -L 8000:127.0.0.1:8000`; the serving port is not opened to the internet):

- GSM8K test, all 1,319 problems, greedy, `JEV_GEN_MAX_TOKENS=2048`. On v5e, 768 tokens truncated
  QAT-derived answers more often than bf16 ones;
- BFCL v3 simple, all 400 records, `tool_choice` auto.

The records are `../jev-tpu-v5e1/data/gsm8k.jsonl` and `bfcl_simple.jsonl`, pinned by
`gen_manifest.json`. Pairs are compared with `../jev-tpu-v5e1/gen_compare.py`: record for record,
with a 95% range from 10,000 bootstrap resamples (seed 0). The 3,880-record classification suite is
not run. Its harness reads logprobs through a vendored server, and porting that to this card is
not part of this study.

### 4. Filing

Each arm's run goes in its own rig, at `benchmarks/runs/2026-10-XX-<what>-mi300x/`, with the
report in `benchmarks/reports/`. Comparisons sit beside this file, in `results/`. After that,
`make benchmarks` runs at the monorepo root.

## Primary outcomes

1. **31B, one request, input 128:** time per output token for the repack against bf16. This is the
   cell where the repack has the most to gain.
2. **31B and E2B at 1 / 8 / 64 requests:** the ratio of output tok/s, repack over bf16, at every
   input length.
3. **GSM8K and BFCL:** the repack minus bf16, paired, for each size.

Everything else is secondary: TTFT, total tok/s, KV pool, boot time.

## Predictions, with the bounds they come from

Copy bandwidth is 3.76 TB/s, as measured on this card (`../HARDWARE.md`). The weight-stream
figures below are arithmetic; none of them has been measured.

- **31B bf16, one request:** one decode step streams about 60 GB, so TPOT is at least about 16 ms
  (at most about 62 tok/s per stream). The repack streams about 19 GB, so its TPOT is at least
  about 5 ms. **Prediction: the repack's TPOT at one request is 1.5x–3x lower than bf16's.**
  Below 1.5x would mean the Triton kernel, not memory, sets the pace. Above 3.2x would beat the
  ratio of bytes and means something is wrong with the measurement.
- **31B at 64 requests:** decode is no longer bound by weight bandwidth. **Prediction: ratio
  between 0.7x and 1.2x.** The dequantization cost is paid on every step, and the Triton kernel
  competes with hipBLASLt's bf16 GEMMs.
- **E2B at every concurrency: ratio between 0.8x and 1.2x.** bf16 E2B already ran at 2.9 ms per
  token at one request (2026-09-18, a different nightly), against a weight-stream floor of about
  1.2 ms (4.51 GB streamed per token, `../MODELS.md`). It is bound by overhead, not bandwidth, so
  cutting its bytes buys little.
- **Quality:** each repack within 2 points of its bf16 twin on GSM8K and on BFCL. On v5e the E2B
  repack scored −0.9 (GSM8K) and −0.7 (BFCL) against bf16. There is no 31B bf16 reference on v5e.
- **Cross-platform consistency check:** the E2B repack scored 0.901 GSM8K and 0.920 BFCL on v5e,
  with the same records and the same `gen_eval.py`. On this card it should land within about
  1.5 points of both. A larger gap means the kernel's numbers need checking before any speed
  result is reported. One possible cause: the Triton kernel reads the symmetric zero point wrong
  (uint4b8, bias 8).

## What would change the conclusion

- **The 31B repack is not faster at one request.** Then 4-bit does not pay on this card through
  vLLM's shipped ROCm path. It becomes the second format, after fp8, whose expected win disappears
  between the kernel and the server.
- **A repack loses more than 2 points on either task, with a range that excludes zero.** Then the
  accuracy claim from v5e does not carry over to this card, whatever the speed.
- **The repack wins at one request and loses at 64.** Then the recommendation depends on the
  workload, and it says so: 4-bit for one user or a few agents, bf16 for a batch server.

## Budget and stop rules

- $1.99/hr, billed from create to destroy, whether or not the droplet is powered on. Estimated
  4–6 droplet-hours: an image pull of about 35 GB, about 90 GB of weights, and four arms of nine
  cells each plus gen_eval. **Hard cap: 8 droplet-hours, about $16.** At the cap the arm in
  progress finishes and the rest are deferred. The droplet is not left up overnight.
- The droplet is created and destroyed by a person, in the console. The rigs have no tool for
  either, on purpose.
- A repack arm that fails to boot gets **one** retry, with the boot log read first. A second
  failure is recorded with its log and the study moves on to the next arm.

## Results

*(empty — filled in after the runs)*
