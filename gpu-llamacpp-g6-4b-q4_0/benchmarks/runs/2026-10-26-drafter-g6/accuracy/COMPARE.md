# GSM8K and BFCL simple with and without the drafter, NVIDIA L4

| task | n | no drafter | drafter, 2 tokens | difference | 95% range | none only | drafter only | same text | wall none, s | wall drafter, s | speed-up |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| gsm8k | 1319 | 0.9310 | 0.9287 | **-0.0023** | -0.0099 to +0.0053 | 14 | 11 | 409 | 1416 | 893 | 1.58x |
| bfcl_simple | 400 | 0.9100 | 0.9150 | **+0.0050** | +0.0000 to +0.0125 | 0 | 2 | 392 | 86 | 69 | 1.25x |

Drafter counters over both tasks (after − before): `spec_decode_num_draft_tokens_total` 347382, `spec_decode_num_drafts_total` 173691
