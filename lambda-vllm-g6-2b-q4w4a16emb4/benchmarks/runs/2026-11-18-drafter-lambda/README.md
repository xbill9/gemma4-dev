# Drafter on the Lambda GPU function — 2026-11-18/19

Same function, image base, checkpoint and `bench.py` as `../2026-11-18-first-light-lambda`, plus Google's E2B
QAT assistant drafter repacked as W4A16 (`repack/repack_drafter.py`; `model.safetensors` SHA-256
`66e683ad…860b`, identical to the drafter on the gpu host), baked in at `/opt/drafter`, 4 drafted tokens:
`--speculative-config '{"model": "/opt/drafter", "num_speculative_tokens": 4}'`. vLLM's engine line in the
serving environment: `speculative_config=SpeculativeConfig(method='mtp', model='/opt/drafter', num_spec_tokens=4)`;
weights 2.98 GiB with the drafter against 2.85 without, KV cache 1,486,429 tokens against 1,593,517.

**Every request below went to that environment**: all `platform.start` events after the run began are in
one log stream (`…14390ffe`). `nodrafter-repeat/` is a run made while the republish was failing: all 75
of its requests went to an old environment without the drafter, so it is a second no-drafter run, kept as such.

## Output tok/s, 256 tokens per request

| Requests at once | No drafter (4 passes) | Drafter (2 passes) | Drafter / no drafter, medians |
| ---: | --- | --- | ---: |
| 1 | 126.8–133.4 | 224.3, 212.8 (69.3 first request) | 1.63 |
| 4 | 306.7, 479.3, 497.3, 493.1 | 386.3, 723.2 | 1.47 |
| 16 | 1,640.9–1,738.9 | 1,515.3, 1,510.5 | 0.88 |
| 48 | 3,441.8–3,614.4 | 2,249.1, 2,266.8 | 0.63 |

Ratios computed in code from the `bench-run*.json` files (median of each column; with an even count the
upper middle value). The first drafter request (3.69 s) and the first 4-request batch (2.63 s each) were
the first on a new environment; the second pass at the same levels took 1.20 s and 1.37 s.

**The drafter pays for 1–4 requests at once and loses from 16**: one reply of 256 tokens came back in
1.14–1.20 s against 1.92–2.02 s without it. At 48 requests it gives 0.63x.

vLLM's acceptance over the run (`SpecDecoding metrics` lines in the environment's log): mean acceptance
length 1.92–2.06 tokens per step, draft acceptance 22.9–26.5%, per position about 0.41–0.52, 0.25, 0.18,
0.07–0.10. vLLM also warned that 4 drafted tokens reuse the one MTP layer and "may result in lower acceptance
rate", and capped `max_num_scheduled_tokens` at 2,048.

The G6 flight measured the same drafter at 4 tokens paying at every level up to 64 requests
(`gpu-vllm-g6-2b-q4w4a16emb4/benchmarks/runs/2026-11-06-suite-g6`), on different prompts (512 in / 128 out and
128 / 512, streamed, timed on the instance); this run is short prompts, 256 out with `ignore_eos`, timed from
a desktop through Lambda. Why the drafter loses here from 16 requests is not established.

## Republishing on a capped capacity provider

A republish starts the new environment before stopping the old one. The old version had scaled to two
g6.2xlarge (the capacity provider's `MaxVCpuCount` of 16, also the account's G quota), so the first republish
ended `Failed` ("An internal error occurred") and the second "MinimumExecutionEnvironments can't be
provisioned because CapacityProvider has reached maxVCpuCount limit". After the failure Lambda rolled
`$LATEST.PUBLISHED` back to the previous image and environment. `republish.py` scales to 0, publishes and
scales back to `MIN_ENVS`/`MAX_ENVS` (1/1): published code in place 18.2 s after publish, then Active
552 s after scaling up, Lambda Init 205,420 ms.
