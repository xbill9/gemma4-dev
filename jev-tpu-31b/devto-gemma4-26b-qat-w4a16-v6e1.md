---
title: "Google's QAT Gemma 4 26B-A4B on One TPU v6e: 15.6x the KV Cache and 1.9x the Throughput of FP8"
published: false
description: "Google ships its quantization-aware-trained Gemma 4 26B-A4B as GGUF and as a 48 GiB bf16 export, and as compressed-tensors W4A16 for every size but this one. A lossless repack to W4A16, a W4A16 mixture-of-experts method for vLLM's JAX path on TPU, and one v6e chip: 17.43 GiB of HBM, 53,888 KV tokens and 1,283 output tokens per second, against RedHat's FP8 build at 27.99 GiB, 3,456 tokens and 668. The same checkpoint loads unpatched on vLLM 0.30.0 on an NVIDIA L4."
tags: gemma, googlecloud, machinelearning, llm
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/jev-tpu-31b/devto-26b-qat-cover.ff0758f6.jpg
---

This article provides a step by step guide to serving Google's quantization-aware-trained (QAT) Gemma 4 26B-A4B on one Google Cloud TPU v6e chip with vLLM, and compares it with the FP8 build that is the only 26B serving on one chip today. Every per-record output, log and script is committed.

The QAT 26B serves on one v6e chip at 17.43 GiB of HBM, with 53,888 tokens of KV cache and 1,283 output tokens per second. RedHat's FP8 build uses 27.99 GiB, holds 3,456 tokens and serves 668. On a 3,880-record classification suite the two land within a point of each other. The route is a lossless repack of Google's "unquantized" QAT export and one new method in vLLM's TPU backend; the same repacked checkpoint loads unpatched on vLLM 0.30.0 on an NVIDIA L4.

https://github.com/xbill9/gemma4-dev/tree/main/jev-tpu-31b

---

#### Why Measure This?

Gemma 4 26B-A4B is a mixture-of-experts model: 25.8B parameters, 128 experts per layer, 8 active per token. At bf16 it needs 48.07 GiB, and one v6e chip has 28.74 GiB of usable HBM.

Google trained 4-bit versions of every Gemma 4 size with quantization-aware training, and its model card lists three formats. For 26B-A4B two of them apply:

- **GGUF (Q4_0)**, for llama.cpp. vLLM has no GGUF loader.
- **"Unquantized" QAT checkpoints**, bf16 at 48.07 GiB.
- **Compressed Tensors (w4a16)**, "for native, optimized inference with vLLM", listed for "Gemma 4 E2B, E4B, 12B, and 31B".

So for vLLM on one chip the only 26B choice is a third-party build. RedHat's FP8 checkpoint fits with 0.75 GiB to spare, enough KV cache for 3,456 tokens, which is one and a half 2,048-token requests.

This article builds the missing W4A16 checkpoint from Google's own QAT weights and serves it.

---

#### At This Point You Should Have…

- A Google Cloud project with v6e quota in a region that offers `ct6e-standard-1t`, and the `gcloud` CLI logged in
- A Cloud Storage bucket for checkpoints and results
- About 70 GB of local disk and Python 3 with `numpy`, for the repack
- Two clones: `git clone https://github.com/xbill9/gemma4-dev` and `git clone -b gemma4-w4a16-moe https://github.com/xbill9/tpu-inference`

---

#### Step 1 — Look Inside the "Unquantized" Export

`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized` stores bf16 tensors, but the values came out of QAT for Q4_0: every group of 32 weights along the input dimension already sits on a 16-level grid, `step × level` with level from −8 to 7. Group size 32 is measured: sampled groups of 64 fail the same test.

That makes a 4-bit checkpoint a change of container. The one trap is the step. The textbook Q4_0 rule, `step = max|w| / 8`, assumes the largest weight in a group sits at level ±8. When a group's peak sits at level 5, that rule derives 5/8 of the true step and re-rounds every weight onto a grid that does not contain it: about 5% median error per group, and every shape check still passes.

So the repack recovers the step instead: for m from 1 to 8 it tries `max|w| / m` and keeps the first that reproduces the whole group, then refines the step by least squares over the 32 values.

---

#### Step 2 — Repack to Compressed-Tensors W4A16

```bash
huggingface-cli download google/gemma-4-26B-A4B-it-qat-q4_0-unquantized \
  --local-dir ~/models/gemma-4-26B-A4B-it-qat-q4_0-unquantized
python3 repack_q4_0.py repack ~/models/gemma-4-26B-A4B-it-qat-q4_0-unquantized \
  ~/models/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct --workers 6
```

```plaintext
model-layer-000.safetensors: 1186 tensors, 0 kept bf16
model-layer-001.safetensors: 1186 tensors, 0 kept bf16
...
total 15.29 GiB in 31 shards; 0 tensors kept bf16 for off-grid groups; 222 ignored modules
```

It writes compressed-tensors `pack-quantized` symmetric int4 with group size 32, the format of Google's other QAT releases. Attention, the dense MLP and all 3,840 experts are quantized; the router, embeddings, norms and vision tower are copied unchanged. The experts ship fused in the source, `experts.gate_up_proj` as `[128, 1408, 2816]`, and are written one module per expert, `experts.{i}.{gate,up,down}_proj`, which is the layout vLLM already reads for int4 mixture-of-experts checkpoints. A tensor with any group off the grid would stay bf16; none did. It ran in 388 seconds on a 16-core machine with 15 GB of RAM, streaming one layer at a time.

---

#### Step 3 — Verify Every Group

```bash
python3 repack_q4_0.py verify ~/models/gemma-4-26B-A4B-it-qat-q4_0-unquantized \
  ~/models/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct
```

`verify` rereads both checkpoints from disk and counts, for every group, whether the stored level is the source value's rank on the stored grid:

| Tensors | Groups of 32 | Levels off the grid | Values bit-identical |
| :--- | ---: | ---: | ---: |
| Experts, gate and up | 475,791,360 | 0 | 92.6% |
| Experts, down | 237,895,680 | 0 | 92.6% |
| Dense MLP | 16,727,040 | 0 | 92.0–92.5% |
| Attention | 34,693,120 | 0 | 89.7–90.4% |

All 748 copied tensors are byte-identical. The 7–10% of values that differ in their last bits do so through the scale: Q4_0 carries a 16-bit float step, and the scale is stored at bf16 because that is the precision vLLM's TPU W4A16 layers load it at. The worst relative difference is 1.1e-2. The levels, which carry the QAT training, all match.

---

#### Step 4 — Add W4A16 Experts to vLLM's TPU Backend

Gemma 4 runs only on the JAX path of `tpu-inference`, vLLM's TPU backend. Two pieces serve this checkpoint there:

- **W4A16 for linear layers**: [tpu-inference #3653](https://github.com/vllm-project/tpu-inference/pull/3653), approved, which serves Google's other QAT sizes.
- **W4A16 for the experts**: a `WNA16FusedMoEMethod` on the `gemma4-w4a16-moe` branch, stacked on #3653.

The experts method loads each expert's packed int4 weights and 32-wide scales, fuses gate and up, and hands them to the GMM kernel unchanged. Three properties of the existing mixture-of-experts path decide how:

- The default processing re-quantizes expert weights with one scale per output channel. For QAT weights that replaces the trained grid, so the method lays out the weights itself.
- The fused mixture-of-experts kernel needs quantization blocks that are multiples of 256, so group 32 goes to the GMM backend, whose `gmm_v2` kernel takes a scale per 32-wide group.
- `gmm_v2` keeps activations in bf16 when the group is narrower than the matrix unit, 256 columns on v6e, and dequantizes each weight tile in fast on-chip memory before the multiply. A group of 32 stays weight-only 4-bit.

Its unit tests include a forward pass through a real expert layer at the 26B's shape, 2816 hidden and 704 intermediate, against a NumPy reference.

---

#### Step 5 — Launch One v6e Chip

The VM applies the patches to the pinned vLLM TPU image, runs the unit tests on the chip, copies the checkpoint from Cloud Storage, serves it, runs the read, and deletes itself.

```bash
gcloud compute instances create jev-tpu-31b-moe2 --zone europe-west4-a \
  --machine-type ct6e-standard-1t \
  --image-family ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e --image-project ubuntu-os-accelerator-images \
  --boot-disk-size 200GB --scopes cloud-platform --maintenance-policy TERMINATE \
  --provisioning-model FLEX_START --reservation-affinity none --request-valid-for-duration 2h \
  --max-run-duration 4h --instance-termination-action DELETE \
  --metadata jev-code=<code-tarball>,jev-run=2026-09-26-moe2,jev-patches="kvshare.diff wna16.diff moe.diff",jev-gcs-models=gs://<bucket>/jev-tpu-31b/models/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct \
  --metadata-from-file startup-script=tpu/startup_quant.sh,jev-arms=<arms-file>
```

The server runs with the flags of every other run in the repository, so results pair record for record:

```bash
vllm serve /work/models/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct --tensor-parallel-size 1 \
  --max-model-len 2048 --max-num-seqs 16 --max-logprobs 32 --generation-config vllm \
  --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}' --enable-prefix-caching
```

---

#### Step 6 — Serve and Read

```plaintext
[jev-quant 2026-09-26T13:39:47Z] pytest: 36 passed, 17 warnings in 40.64s
[jev-quant 2026-09-26T13:53:13Z] READY /work/models/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct after 796s
[jev-quant 2026-09-26T13:53:31Z] 26b-q4w4 smoke: ok 20 records, labels returned [6, 6, 5, 5, 6, 2, 2, 2, 2, 2, 1, 2, 3, 3, 4, 2, 2, 2, 2, 2]
[jev-quant 2026-09-26T13:53:55Z] 26b-q4w4 four tasks: 24s for 1200 decisions at concurrency 8
[jev-quant 2026-09-26T13:55:36Z] 26b-q4w4 suite: 58s for 3880 records
[jev-quant 2026-09-26T13:55:56Z] 26b-q4w4 load: 1283.3 tok/s, range 1260.1 to 1288.4
```

The read is the one used throughout this repository: four 300-example classification tasks, the same three-choice tasks with the options reversed, and Bespoke Labs' 3,880-record public suite, each answer taken as the highest-probability label. Throughput is 16 concurrent requests of exactly 256 output tokens, median of three passes. The first boot compiled for 334 seconds; the saved compile cache removes that on later boots.

---

#### What Fits on One v6e Chip?

```plaintext
Memory statistics | total_hbm_limit_gb=31.24GiB | total_hbm_limit_cap_gb=28.74GiB | total_hbm_used_gb=17.43GiB | total_hbm_avail_gb=11.32GiB
```

| 26B-A4B build | HBM used | Free | KV cache |
| :--- | ---: | ---: | ---: |
| RedHat FP8 | 27.99 GiB | 0.75 GiB | 3,456 tokens |
| QAT W4A16 | 17.43 GiB | 11.32 GiB | 53,888 tokens |

The 17.43 GiB breaks down, by arithmetic from the checkpoint's tensor headers, into 14.10 GiB of experts, 1.38 GiB of embeddings, 1.07 GiB of vision encoder and 0.86 GiB of attention and dense MLP. The vision encoder is resident because the 26B loads through the multimodal model class even with images switched off. The KV cache costs 120 KiB per token at the fp8 that vLLM picks on v6e (arithmetic from the allocation shape, 8 heads × K and V × 256 × 30 layers), so 53,888 tokens is 26 concurrent 2,048-token requests.

---

#### Does the Accuracy Hold?

Both builds are quantized, and no bf16 26B fits one chip, so the comparison is W4A16 against FP8, paired on the same records:

| Read | FP8 | QAT W4A16 | Difference (95% range) |
| :--- | ---: | ---: | :--- |
| 3,880-record suite | 76.0% | 75.3% | −0.7 points (−1.5 to +0.1) |
| SST-2 | 94.7% | 95.0% | +0.3 (−0.7 to +1.7) |
| AG News | 86.0% | 86.0% | 0.0 (−1.7 to +1.7) |
| DAIR Emotion | 58.3% | 60.0% | +1.7 (−0.3 to +4.0) |
| tweet_eval irony | 91.0% | 90.7% | −0.3 (−2.3 to +2.0) |

Every range spans zero. On the suite 133 records go from right to wrong and 106 from wrong to right.

---

#### The Same Checkpoint on an NVIDIA GPU

The repacked checkpoint loads on stock vLLM with no patches. On one NVIDIA L4 (`g2-standard-8`) with `vllm/vllm-openai` at vLLM 0.30.0 and the same flags:

```plaintext
[jev-gpu 2026-09-26T14:11:08Z] READY after 255s
[jev-gpu 2026-09-26T14:11:08Z] memory: Model loading took 14.8 GiB memory and 55.495697 seconds GPU KV cache size: 17,990 tokens, Maximum concurrency for 2,048 tokens per request: 8.78x
[jev-gpu 2026-09-26T14:16:23Z] load: 393.7 tok/s, range 389.5 to 396.3
```

vLLM picks its existing Marlin int4 kernels for both the experts and the linear layers. Paired with the TPU run, the four tasks agree within one example each, and the suite reads 0.7 points higher on the L4 (76.0% against 75.3%, 95% range +0.3 to +1.1).

KVRESULT_PENDING

---

#### 🔎 Tip: 12B Needs One Flag

Gemma 4 12B ships as `Gemma4UnifiedForConditionalGeneration`, which the TPU backend does not register, so vLLM falls back to its PyTorch path. Its decoder is the one `Gemma4ForCausalLM` already serves:

```bash
vllm serve google/gemma-4-12B-it --hf_overrides '{"architectures": ["Gemma4ForCausalLM"]}'
```

That loads bf16 12B on the JAX path at 22.18 GiB against 24.56 GiB on the PyTorch path, and the same flag serves the QAT W4A16 12B at 9.46 GiB with #3653. With the override, vLLM treats the model as text-only and needs no `--limit-mm-per-prompt`.

---

#### Compare and Contrast

| One v6e chip | RedHat FP8 | QAT W4A16 |
| :--- | :---: | :---: |
| Source weights | bf16 model, quantized by RedHat | Google's QAT weights |
| HBM used | 27.99 GiB | 🥇 17.43 GiB |
| KV cache | 3,456 tokens | 🥇 53,888 tokens |
| Output tokens/s | 668 | 🥇 1,283 |
| Suite accuracy | 🥇 76.0% | 75.3% |
| Flex-start cost per million output tokens | $0.56 | 🥇 $0.29 |
| Runs on stock vLLM TPU today | 🥇 yes | needs #3653 and the experts method |

---

#### So, Which One?

For serving on one v6e chip, the QAT W4A16 build: it holds 15.6 times the KV cache, serves 1.92 times the tokens per second, and reads the suite 0.7 points below FP8, a difference whose 95% range reaches zero. FP8 remains the build that runs on a stock vLLM TPU image today. On NVIDIA GPUs the same QAT checkpoint needs nothing beyond stock vLLM.

---

#### What Does It Cost?

At europe-west4's v6e rates of $1.35 per chip-hour on flex-start and $2.97 on demand, the measured throughputs work out to, by arithmetic:

| Build | Flex-start | On demand |
| :--- | ---: | ---: |
| QAT W4A16, 1,283 tok/s | $0.29 per million output tokens | $0.64 |
| FP8, 668 tok/s | $0.56 | $1.23 |

These assume the chip runs at the measured load around the clock, 16 concurrent requests of 256 tokens.

---

#### Teardown

Each VM deletes itself when its run ends, and `--max-run-duration` deletes it regardless. Confirm nothing is left:

```bash
gcloud compute instances list --format='value(name,zone.basename(),status)'
```

The checkpoint stays in Cloud Storage for the next boot; delete it with `gcloud storage rm -r gs://<bucket>/jev-tpu-31b/models/` when done.

---

#### Summary

The goal of this article was to serve Google's QAT Gemma 4 26B-A4B on one TPU v6e chip with vLLM. The key to the solution was a lossless repack of Google's bf16 QAT export into compressed-tensors W4A16, and a W4A16 experts method for vLLM's TPU backend that keeps the checkpoint's 32-wide scales. The results were:

- 🟢 The QAT 26B serves on one v6e chip at 17.43 GiB, against 27.99 GiB for RedHat's FP8 build
- 🟢 53,888 tokens of KV cache against 3,456, and 1,283 output tokens per second against 668
- 🟢 Every one of 765 million weight groups repacks onto its 4-bit grid, and all 748 unquantized tensors copy byte for byte
- ⚠️ Suite accuracy is 0.7 points below FP8, with a 95% range of −1.5 to +0.1
- 🟢 The repacked checkpoint loads unpatched on vLLM 0.30.0 on an NVIDIA L4
- ⚠️ On TPU it needs #3653 and the experts method, neither merged yet
- 🟢 Gemma 4 12B serves on the TPU backend's JAX path with one `--hf_overrides` flag

Scope: one TPU v6e chip (`ct6e-standard-1t`, flex-start) in europe-west4-a, vLLM `0.29.1rc1.dev468+g0b7f11a1e` at `vllm/vllm-tpu@sha256:19a1a052…` with #3299, #3653 and the experts method applied, one run per build, `--max-model-len 2048`, vLLM's default fp8 KV cache on v6e. The FP8 read comes from a run two days earlier on the same image without the patches, and its throughput from a second VM the same day; the four-task reads repeat record for record across VMs here, the suite within 0.1 points, and throughput moved about 2% between VMs. The GPU run used one NVIDIA L4 in us-central1-a with vLLM 0.30.0 and a bf16 KV cache. Costs are arithmetic from list prices and the measured throughput. The repacked checkpoint is unofficial and derived from Google's release under Apache 2.0. Parts of the analysis and writing were done with AI assistance (Claude); every figure comes from the committed output files.

The strategy for serving Google's QAT Gemma 4 26B on one TPU v6e chip was validated with an incremental step by step approach.

---

#### References

- Code, repack, per-record results and logs: https://github.com/xbill9/gemma4-dev/tree/main/jev-tpu-31b
- W4A16 experts method (branch): https://github.com/xbill9/tpu-inference/tree/gemma4-w4a16-moe
- tpu-inference #3653, W4A16 linear method: https://github.com/vllm-project/tpu-inference/pull/3653
- Google's QAT source checkpoint: https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized
- RedHat FP8 build: https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic
- Google's QAT announcement: https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/
- Bespoke Labs public suite: https://github.com/bespokelabsai/nimble/blob/0e67403/docs/PUBLIC_BENCHMARKS.md
- vLLM TPU documentation: https://docs.vllm.ai/projects/tpu/en/latest/
- Cloud TPU v6e: https://cloud.google.com/tpu/docs/v6e
