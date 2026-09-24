# Pre-registration: Jev-style label reads of Gemma 4 on one TPU v6e chip

Committed before any model call in this directory. Sibling of `../jev`, which measured the same reads on one NVIDIA L4; the prompts, data, label tokens, probability code and scoring are copied from there unchanged (`vendor/structured_server.py` is byte-identical, sha256 `7cd9aa00…`).

## Question

How does a plain Gemma 4 label read hold up across model sizes, served by vLLM on one TPU v6e chip, on the same 1,200 labelled examples and the same 3,880-record public suite where Jev 1.13.0 has published results? And can a quantized 26B-A4B or 31B be served this way on one v6e chip at all?

## Hardware and serving

- One TPU v6e chip (`ct6e-standard-1t`, Compute Engine, flex-start, project `aisprint-491218`), zone chosen by availability; the zone, image digest and vLLM version are recorded per run.
- `vllm/vllm-tpu:nightly`, pulled once per run and pinned by digest for every arm in that run.
- Serving flags for every arm: `--tensor-parallel-size 1 --max-model-len 2048 --max-num-seqs 16 --max-logprobs 32 --generation-config vllm --limit-mm-per-prompt '{"image":0,"audio":0}' --enable-prefix-caching`. Anything added for a specific checkpoint is recorded as a deviation.
- The client runs on the same VM against `localhost`.

## Arms

Registered (bf16, the reference checkpoints):

| Arm | Checkpoint |
|---|---|
| E2B | `google/gemma-4-E2B-it` |
| E4B | `google/gemma-4-E4B-it` |
| 12B | `google/gemma-4-12B-it` |

Exploratory (26B-A4B and 31B do not fit one v6e chip at bf16; none of these has been served by vLLM on one v6e chip in this monorepo). Tried in this order, each given one boot attempt of at most 25 minutes; the first that serves for a size is run in full, the rest are recorded with their failure:

| Size | Checkpoint | Format | Size on disk |
|---|---|---|---|
| 26B-A4B | `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` | compressed-tensors fp8 | 26.67 GiB |
| 26B-A4B | `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` | AWQ 4-bit (the checkpoint of the L4 run) | 16.01 GiB |
| 31B | `google/gemma-4-31B-it-qat-w4a16-ct` | compressed-tensors w4a16 | 21.67 GiB |
| 31B | `cyankiwi/gemma-4-31B-it-AWQ-4bit` | AWQ 4-bit | 19.47 GiB |

The 31B fp8 builds (30.98 GiB) exceed the chip's usable HBM and are not tried. A checkpoint "serves" when `/v1/models` answers and a five-example smoke read returns a probability for every label.

## Read and data

- The plain read of `../jev`: the served model's own chat template, then the proxy's answer lead, one completion token, label log-probabilities via `logprob_token_ids`; the proxy's single-token labels and rescaling.
- Four tasks, 300 examples each (sst2, AG News, DAIR Emotion, tweet_eval irony, seed 20260923), plus the three choice tasks with options reversed.
- Bespoke Labs' 13-subset public suite, 3,880 records, built with Nimble's converters at `0e67403` (the build used for `../jev`), copied to the VM, and checked there against all 13 `dataset_sha256` values; a mismatched subset is excluded.
- Latency: the first 100 examples of each task, concurrency 1, client on the VM.

## Outcomes

- Per task: accuracy with 95% range, ECE (15 bins), Brier, calibration after one temperature fitted on 0 to 150 labels (single split, pre-registered) and on 50 labels (mean of 20 splits), option-order changes.
- Suite: Bespoke Labs' definitions (accuracy, 10-bin ECE, multiclass Brier) per subset and pooled by question type, beside Jev 1.13.0's published figures; ECE after 50 labels (20 splits).
- Median and 90th-percentile time per decision.
- A cross-platform check: E2B and E4B were also read on the L4 in `../jev` with the same checkpoints and prompts. The share of examples with the same predicted label on both platforms is reported. It checks the plumbing and is not a hardware comparison.

## Commitments

- Every result is published, including arms that fail to serve and any size that does worse.
- Cost cap: 6 hours of flex-start v6e-1 time. The VM deletes itself when the run ends and carries `--max-run-duration` as a backstop.
- Deviations are recorded here with a date before the affected results are scored.
