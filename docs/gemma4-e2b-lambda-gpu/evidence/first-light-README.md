# First light: Gemma 4 E2B as a Lambda GPU function — 2026-11-18

Lambda Managed Instances GPU support, announced 2026-11-18, on its first day. One g6.2xlarge (NVIDIA L4)
in us-east-1, created by Lambda behind capacity provider `gemma4-e2b-g6`, serving
`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` @ `db715f83` from AWS's Lambda vLLM image (vLLM
`0.33.0+cu130`, from the server's banner in `startup.log`). Settings are the rig's `tpu.env`;
`function-config.json` is the published version as Lambda reports it (account id redacted).

## Deploy

| Step | Time | Source |
| --- | --- | --- |
| Image push to ECR, first time (base 9.13 GB compressed + 2.86 GB checkpoint), from a home connection | 398.4 s | `lmi.py image` |
| `publish-version` → version Active | 484.0 s | `lmi.py function` |
| of which: publish (22:52:59 UTC) → vLLM banner (22:58:11) | 312 s | `startup.log` |
| Lambda Init (`platform.initReport`, `lambda-managed-instances`, success) | 156,397 ms | `startup.log` |
| of which: model load (2.85 GiB) | 5.64 s | `startup.log` |
| of which: `torch.compile` | 43.54 s | `startup.log` |
| of which: engine init (profile, KV cache, warmup) | 64.94 s | `startup.log` |

The time before the banner covers instance launch and image fetch, which Lambda does not log separately.
vLLM reports `Using MarlinLinearKernel`, 14.28 GiB of KV cache and `GPU KV cache size: 1,593,517 tokens`
at `--gpu-memory-utilization 0.85`, `--max-model-len 16384`.

Two things Lambda refused or reported differently, now handled in `lmi.py`:
`CreateFunction` rejects an OCI image index ("The image manifest, config or layer media type … is not
supported"), which Docker 29 pushes by default with a provenance attestation; the image is built with
`buildx --provenance=false --sbom=false` and `oci-mediatypes=false`. And an unpublished LMI function
reports `ActiveNonInvocable`; only a published version turns `Active`.

## Requests

`bench.py`: synchronous boto3 `Invoke` from a desktop in the US to us-east-1, so every time includes the
round trip and the Lambda invoke path. Each request has a unique nonce, greedy, thinking off,
`max_tokens` 256 with `ignore_eos`, so every reply is 256 tokens. All requests went to one execution
environment (one CloudWatch log stream). Two passes, `bench-run1.json` and `bench-run2.json`:

| Requests at once | Output tok/s, pass 1 | Output tok/s, pass 2 | Slowest request, pass 2 |
| ---: | ---: | ---: | ---: |
| 1 | 130.5, 133.2 | 128.9 | 1.984 s |
| 4 | 306.7 | 479.3 | 2.122 s |
| 16 | 1,640.9 | 1,709.0 | 2.356 s |
| 48 | 3,482.7 | 3,441.8 | 3.472 s |

Output tok/s is 256 × requests over the wall time of the batch. Pass 1's 4-request batch (3.21–3.33 s per
request) is the one outlier, with no cause in the logs: vLLM's only Triton JIT warning
(`kernel_unified_attention`) came at the first `lmi.py ask`, before either pass. 48 is the most one environment takes
(`PerExecutionEnvironmentMaxConcurrency`); every request at every level succeeded.

Two invocations through `lmi.py ask` (aws CLI, start-up included): a 41-token answer in 2.41 s, and with
`chat_template_kwargs.enable_thinking` a 283-token reply with its thinking returned in `reasoning` in 2.92 s.

**Not comparable to `gpu-vllm-g6-2b-q4w4a16emb4`'s sweeps as they stand**: those were 512 in / 128 out and
128 in / 512 out with streaming, timed on the instance; these are short prompts, 256 out, timed from a
desktop through Lambda, with no streaming (the image's handler returns whole completions).
