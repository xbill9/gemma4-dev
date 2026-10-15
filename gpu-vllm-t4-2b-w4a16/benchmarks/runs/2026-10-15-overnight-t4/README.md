# 2026-10-15 — queued T4 drafter gap jobs (not yet run)

Scripts for the gaps `2026-10-15-drafter-sweep-t4` left open. `queue.sh` never ran; a trimmed
plan did (`trimmed.sh`, `trimmed2.sh`, results in `REPORT.md`). Jobs 3 and 4 below have not run.

| job | script | question |
| --- | --- | --- |
| 1 | `conc.py`, `summarize_conc.py` | Does the drafter still pay at 1, 2, 4 and 8 requests in flight? The rigs serve `--max-num-seqs 8` with 3 drafted tokens; every drafter run so far was one request at a time, and on TPU v5e the drafter turned into a loss at 16. |
| 2 | `quality.py`, `build_suite.sh` | Does serving with the drafter change GSM8K and BFCL scores? With a drafter, 3 of 6 E2B prompts give different greedy text. Also the first quality scores for the T4 builds, and the Nimble suite once it is rebuilt on this host. |
| 3 | `spec_ab.py` (from the sweep) | Drafters against the bf16 E2B target (`gpu-vllm-t4-2b`), which has never served with one. |
| 4 | `longctx.py` | Does the gain hold with 4k, 8k and 15k tokens of context? |

`t4serve.py` starts and stops one vLLM server with a rig's `tpu.env` flags. `queue.sh` runs the
four jobs in order; as written it first waits for the sweep's `run.sh` and `run_q4ct.sh` to finish,
so that wait has to be removed or satisfied before it is started again. Start it detached
(`setsid nohup ./queue.sh &`) so it outlives the shell that launched it.
