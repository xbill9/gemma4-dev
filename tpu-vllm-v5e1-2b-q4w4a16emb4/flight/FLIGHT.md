# The v5e-1 ladder

The G6 ladder (`gpu-vllm-g6-2b-q4w4a16emb4/benchmarks/runs/2026-11-0{5,6}-*-g6`) on one Cloud TPU v5e chip:
every engine that serves Gemma 4 E2B on a v5e-1, the same requests and statistics. The CUDA engines of the G6
ladder (llama.cpp, SGLang, mistral.rs, LiteRT-LM) have no TPU build; the TPU set is vLLM (`tpu_inference`), the
pure-JAX engine, its Rust front end, and TorchTPU in Python and in Rust. Started 2026-11-06.

| Arm | Engine | Rig | Weights |
| --- | --- | --- | --- |
| `vllm` | vLLM, this rig's patched image | `tpu-vllm-v5e1-2b-q4w4a16emb4` | `xbill9/...-w4a16-ct-text-emb4` (the G6 ladder's) |
| `vllm-text`, `vllm-text-dN` | the same | the same | `xbill9/...-w4a16-ct-text`; `-dN` adds `google/gemma-4-E2B-it-assistant`, N drafted tokens |
| `jax` | pure-JAX engine | `~/jax-dev/tpu-jax-v5e1-2b-q4w4a16` | `xbill9/...-w4a16-ct-text`, `--quant-mode w4a16` |
| `jax-bf16` | pure-JAX engine | `~/jax-dev/tpu-jax-v5e1-2b` | `google/gemma-4-E2B-it`, dense |
| `jaxrust` | Rust model over PJRT | `~/jax-dev/tpu-jaxrust-v5e1-2b` | `google/gemma-4-E2B-it` (bf16 only) |
| `torchtpu`, `torchtpu-bf16` | TorchTPU 0.2.0, Python | `~/torchtpu-dev/tpu-pytorch-v5e1-2b-w4a16` | `google/gemma-4-E2B-it-qat-w4a16-ct`; `-bf16` `google/gemma-4-E2B-it` |
| `torchtpu-rust` | TorchTPU 0.2.0, Rust (PyO3) | `~/torchtpu-dev/tpu-pytorchrust-v5e1-2b-q4w4a16emb4` | `-text` linears with the `-emb4` int4 tables |

**Dropped from the 2026-11-06 run (user, 2026-11-07): both TorchTPU arms.** `torchtpu` (w4a16) does not
serve: `torch.compile` fails tracing compressed-tensors' int4 unpack (`index_put_(): cannot record EventSnapshot
during an FX trace`). `torchtpu-bf16` served, 5/5 on the first five GSM8K questions in 1,000 s, three of them at
the 2,048-token cap. `torchtpu-rust` answered GSM8K at about 29 s a question. Their `one` results (six prompts)
were measured and are in the run's `results.jsonl`; their GSM8K runs were stopped and deleted. Also not run:
`vllm-text-d1` (time).

Why the drafter arms use `-text`: vLLM's MTP drafter copies the target's bf16 `embed_tokens` and cannot pair with
the int4-table build; Google's QAT drafter was slower than none on this chip at every load (`QUANTIZATION.md`,
2026-09-29), so the drafter is the bf16-release `-it-assistant`.

## Where it runs

Three flex-start v5e-1s in `us-west4-a`, 24-hour run limit, one engine on a chip at a time:

- `tpu-vllm-v5e1-2b-q4w4a16emb4` (this rig's queued resource, its own startup script: pulls and patches the pinned
  image, serves vLLM): every vLLM and JAX arm. `setup-jax.sh` adds Python 3.13, `jax[tpu]==0.12.0.1` (the
  2026-11-03 v5e-1 runs' version), Rust nightly, the `jaxrust_rs` module and every checkpoint in `/opt/hf`.
- `ladder-v5e1-torchtpu`: the `torchtpu` arm. `ladder-v5e1-torchtpurs`: the `torchtpu-rust` arm. Both run the
  TorchTPU Rust rig's rendered startup script (Python 3.14, torch 2.14.1, torch_tpu 0.2.0, the crate built on the
  VM, the `torchtpu-rs` unit). TorchTPU decodes at roughly 12 to 50 tok/s on this chip, so its GSM8K runs take
  hours; separate chips keep them off the vLLM and JAX timeline.

Every arm on a chip of the same type and zone; a ratio between two engines on different chips is a ratio between
two v5e-1s.

## Jobs (`ladder.py`)

- `one:<arm>` for every arm: the G6 `engines.py` requests (six prompts, 256 tokens, greedy, thinking off,
  e2e and streamed, 3 repeats; then 4 at once).
- `sweep:<arm>` for the vLLM arms: `sweep.py` (the G6 copy), 512/128 and 128/512, 1-64 requests, 5 repeats.
  The JAX and TorchTPU servers take one request at a time; their 4-at-once row shows it.
- `acc:<arm>` for every arm: GSM8K (1,319, 2,048 tokens), one request at a time; BFCL simple (400) for the vLLM
  arms only. The JAX and TorchTPU servers render the chat template without the request's `tools` and return no
  `tool_calls`, so BFCL there would score a missing server feature, not the model.

Not measured: TPU memory in use (no `nvidia-smi` equivalent wired in).

## The 2026-11-06 run, and how to pick it up

Output directory on every VM: `/opt/ladder-out/2026-11-06-ladder-v5e1` (`driver.log`, `results.jsonl` for the
`one` jobs, `one-*/`, `sweep-*/`, `acc-*/`, `versions.txt`, `setup-jax*.log`, `finish.log`). Settings that differ
from the defaults above, found by the five-question smoke tests (`/opt/ladder-out/smoke*`):

- vLLM drafter arms serve at `--gpu-memory-utilization 0.70`: `-text` plus the drafter ran out of HBM during
  warmup at 0.80 (`RESOURCE_EXHAUSTED ... Attempting to allocate 64.00M ... 25.12M free`).
- vLLM's first start on each arm takes 31 to 34 minutes (ready_s 2,055 and 1,860), so `ladder.py` runs all of an
  arm's jobs (`one+sweep+acc:<arm>`) against one server start.
- `setup-jax.sh` exports `HOME`: under `systemd-run` it was unset and the first run stopped at the Rust step; it
  was rerun on B and C after the TorchTPU `one` jobs had finished (`setup-jax-2.log`).

| VM (queued resource) | Jobs | Unit |
| --- | --- | --- |
| `tpu-vllm-v5e1-2b-q4w4a16emb4` (A) | `one+sweep+acc` for `vllm`, `vllm-text`, `vllm-text-d4` | `ladder-full` |
| `ladder-v5e1-torchtpu` (B) | `one+acc:jax-bf16` | `ladder-jax` |
| `ladder-v5e1-torchtpurs` (C) | `one+acc:jaxrust`, then `one+acc:jax` (moved off A: A's `one-jax/`, `acc-jax/` hold only `MOVED` and `DONE`) | `ladder-jax` |

**Each VM finishes by itself.** `finish.sh` runs as the `ladder-finish` unit: it waits for the ladder unit to
end, uploads the run directory as `gs://aisprint-491218-ladder-v5e1/2026-11-06-ladder-v5e1/<queued resource>/ladder-run.tgz`
(with `ladder-run.sha256` and `finish.log`), downloads it back and compares SHA-256, then deletes its own queued
resource. It does **not** delete when the upload fails or `driver.log` has a `FAIL` / `ok=False` line outside
the TorchTPU arms; that VM then stays up until the 24-hour flex-start limit (about 19:35 UTC on 2026-11-07). The
VMs run as the default compute service account, which holds `roles/editor` and `roles/tpu.admin`.

To pick up:

```
gcloud storage ls -r gs://aisprint-491218-ladder-v5e1/2026-11-06-ladder-v5e1/
gcloud storage cat gs://aisprint-491218-ladder-v5e1/2026-11-06-ladder-v5e1/*/finish.log
gcloud compute tpus queued-resources list --zone us-west4-a --project aisprint-491218
```

A VM still listed with a `NOT deleting` finish.log needs its failed job rerun (`ladder.py` skips jobs with a
`DONE` marker, so rerunning the same job list redoes only the failed ones), then a new `ladder-finish` unit. Then:
unpack the three archives into `benchmarks/runs/2026-11-06-ladder-v5e1/` (one subdirectory per VM, or merged:
the arm directories do not overlap except A's `MOVED` placeholders), copy `ladder.py`, `setup-jax.sh`,
`finish.sh` beside them, write REPORT.md with the tables computed by a script, commit, push, and delete the
bucket. No comparison paper.

State at 2026-11-07 01:19 UTC (from each `driver.log`): A has finished `vllm` (BFCL 368/400 = 0.920, GSM8K
1188/1319 = 0.9007, 3,630 s) and `vllm-text` (BFCL 368/400 = 0.920, GSM8K 1186/1319 = 0.8992, 3,948 s), and is
starting `vllm-text-d4`. B is on `acc:jax-bf16` (GSM8K since 00:41), C on `acc:jaxrust` (since 00:41), every
`one` job done. Expected ends, from smoke-test rates and so approximate: A about 04:00, B about 04:30, C
about 08:30 UTC.

**Result (2026-11-07).** B and C finished every job and deleted themselves (01:48 and 03:11 UTC). On A
`vllm-text-d4` ran out of TPU memory in warmup at 0.70 and again on a rerun at 0.60 (`ladder.py` now passes 0.60
for d4; `finish.sh` takes a start time so an earlier failure does not block the delete); A was deleted by hand at
15:01 UTC. Archives verified by SHA-256 and unpacked into `benchmarks/runs/2026-11-06-ladder-v5e1/`, one
directory per VM, with `REPORT.md` and `SUMMARY.txt` from `analyze.py`; the bucket is deleted.
