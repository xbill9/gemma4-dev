# GSM8K and BFCL simple with and without the drafter, NVIDIA L4

| task | n | no drafter | drafter, 2 tokens | difference | 95% range | none only | drafter only | same text | wall none, s | wall drafter, s | speed-up |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| gsm8k | 1319 | 0.8992 | 0.8984 | **-0.0008** | -0.0099 to +0.0083 | 18 | 17 | 240 | 1014 | 713 | 1.42x |
| bfcl_simple | 400 | 0.9300 | 0.9250 | **-0.0050** | -0.0125 to +0.0000 | 2 | 0 | 396 | 50 | 44 | 1.15x |

Drafter counters over both tasks (after − before): `spec_decode_num_draft_tokens_total` 411008, `spec_decode_num_drafts_total` 205505
