# jev-tpu

Jev-style label reads of Gemma 4 served by vLLM on one TPU v6e chip: E2B, E4B and 12B at bf16, and 26B-A4B through an fp8 checkpoint, on the four labelled tasks and Bespoke Labs' 3,880-record public suite where Jev 1.13.0 has published results. Sibling of `../jev`, which ran the same read on one NVIDIA L4; the read, data and scoring code are copied from there.

- `PREREGISTRATION.md` — design, committed before any model call, with dated deviations
- `RESULTS.md` — what served, accuracy and calibration beside Jev, checks, cost
- `tpu/` — `startup.sh` (unattended run), `boot.sh`, `serve.sh`, `run.sh`
- `sweep_score.py`, `nimble_suite/suite_sweep.py` — the figures in `RESULTS.md`
- `results/` — every per-record output, logs and raw request probes

On TPU the read asks for the top 32 log-probabilities (`JEV_TOPK=32`), because `tpu_inference` has no per-request `logprob_token_ids`.
