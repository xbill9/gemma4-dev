# 2026-09-30 — E2B emb4 on a raw G4dn, against the same build on SageMaker

First serve of `gpu-vllm-g4dn-2b-w4a16`: `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` on one
on-demand `g4dn.xlarge` (1× NVIDIA T4) in us-east-2, `vllm/vllm-openai:v0.30.0` with the Turing
shared-memory clamp built into a derived tag on the instance, fp16, `max_model_len` 8192, GPU memory
utilization 0.90, vLLM's default `max_num_seqs` (`settings.txt`). Measured with sagemaker-gemma's
`compare.py` — the script that measured the SageMaker endpoint `gemma-4-e2b-emb4-t4`
(`ml.g4dn.xlarge`, AWS vLLM container 0.30.0 with the same clamp, 2026-09-30) — run **on the
instance against localhost**, over Systems Manager. The security group has no inbound rules.

## Results (`measure-gemma-4-e2b-emb4-g4dn.json`)

`compare.py combine`, SageMaker first:

```
                            gemma-4-e2b-emb4-t4  gemma-4-e2b-emb4-g4dn  ratio
weights_gib                           2.86                2.86  1.0
kv_cache_tokens                     660033              660108  1.0
load_seconds                     30.890975           21.123713
decode_tokens_per_second             108.5               111.2  1.02
load_c1_tokens_per_second             91.1              113.85  1.25
load_c4_tokens_per_second           308.25               394.9  1.28
load_c16_tokens_per_second          787.05              1005.1  1.28
quality_correct                         36                  36
identical answers: 40/40
```

Per-call fixed cost: 0.561 s on SageMaker (aws CLI from a workstation), 0.006 s here (client on
the instance). As on the G6 twin, that cost is what separates the 1- to 16-parallel rates; the
model server is the same.

## Timeline (`timeline.txt`)

Launched 16:09:10Z, image pull done +170 s, patch applied +191 s, derived image built +194 s,
clamp verified in the image +209 s, vLLM healthy 16:18:54Z (9.7 min), measured by 16:21:21Z,
terminated 16:28:11Z.
