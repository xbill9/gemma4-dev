# gpu-pytorch-g6-2b

Serve **`google/gemma-4-E2B-it`** with **stock PyTorch + HF transformers** on **AWS EC2 G6** —
an **x86_64** host paired with an **NVIDIA L4** (Ada, SM 8.9, 23034 MiB).

> **Status: this rig has served.** `benchmarks/runs/2026-08-29-first-serve-g6/` —
> **20.93 tok/s** median decode on a `g6.2xlarge` spot instance, 8/8 sweep cells, 0
> degenerate responses. 75 tests pass offline. Retargeted from
> [`gpu-pytorch-g5g-2b`](../gpu-pytorch-g5g-2b/) on 2026-08-29.

## Why this exists

This is the **ordinary-runtime control** for this silicon. No custom model port, no XLA, no
compiled static shapes, no from-source build — `AutoModelForCausalLM` and an HF KV cache. It
is the number the JAX and vLLM rigs on identical hardware should be asked to beat, and by far
the cheapest of the three to stand up.

It also completes a 2x2. `gpu-pytorch-g5g-2b` is the same runtime on a T4G; `gpu-jax-g6-2b`
is the same chip under JAX:

| | T4G (Turing, SM 7.5) | **L4 (Ada, SM 8.9)** |
| --- | --- | --- |
| **PyTorch** | `gpu-pytorch-g5g-2b` — 10.88 tok/s | **this rig — 20.93 tok/s** |
| **JAX** | `gpu-jax-g5g-2b` — 12.80 tok/s | `gpu-jax-g6-2b` — 48.40 tok/s |

## The headline, and the caveat that goes with it

**Against the T4G, this is a clean 1.92x** — identical runtime, identical dense checkpoint,
byte-identical `tpu_jax_weight_bytes` (10,208,595,008). Only the GPU differs.

**Against the JAX rig on the same chip it is not clean at all.** That rig serves **6.155 GB**
(`ple4 + int8_lm_head`) against this rig's **10.209 GB**, and decode is bandwidth-bound on
exactly those bytes. The raw 2.31x gap is mostly quantization; normalising it away leaves
JAX ahead by **~1.66x**, and *that* is the runtime difference.

## Where the time goes

```
10.209 GB / 300.05 GB/s = 34.02 ms/step  ->  29.4 tok/s   (bandwidth floor)
measured                  47.78 ms/step  ->  20.93 tok/s  (71% of it)
```

**~13.75 ms of every decode step is not weight streaming.** That is eager mode: HF
transformers launches hundreds of small kernels per step from Python, with no fusion and no
graph capture. The JAX rig on the same chip sits at **99.3%** of its roofline.

The two untried levers are **batching** (`MAX_NUM_SEQS` is 1, so there is nothing to amortise
the launch overhead against) and **`torch.compile` / CUDA graphs**. Neither is measured.

## Quick start

```bash
pip install -r requirements.txt          # control plane only; no torch, no GPU needed
python3 -m unittest discover -s tests -v # 75 tests, fully offline
make lint
./project-setup.sh                       # writes .mcp.json
```

Then, through the MCP server (`mcp__gpu-pytorch-g6-2b__…`):

```
create_g6_instance → get_install_progress → verify_gpu_arch → deploy_torch_server
                   → get_torch_logs → verify_model_health
```

**Run `make skill` before `deploy_torch_server`.** The deploy resolves its payload next to
`server.py`, so from the MCP snapshot it ships the previous `make skill` output — silently.
`verify_model_health` compares the served build id against the local digest and reports
`STALE DEPLOY`.

## Benchmarking

```bash
python3 sweep.py --base http://<ip>:8000/v1 --out benchmarks/runs/<date>-<what>-g6/
python3 make_report.py --run benchmarks/runs/<date>-<what>-g6/ --instance i-... --az ...
```

`sweep.py` writes per cell so a spot reclamation cannot cost the whole run;
`make_report.py` is separate so a report can be shaped after the instance is gone.

**Quote `tpu_jax_decode_tokens_per_second`, not an end-to-end rate.** End-to-end carries
prefill and the HTTP round trip, so it falls with context (19.93 → 16.33) while decode does
not (21.37 → 20.75). The `tpu_jax_` prefix is an identifier, not a description — it is kept
so the sibling rigs' reports still compare by name.

## Things that will bite you

- **`g6.16xlarge` is single-GPU** despite the suffix; the multi-GPU sizes are 12/24/48xlarge.
  There is no `g6.metal`. Nothing shards regardless — a bigger instance buys host RAM, not
  device memory.
- **G6 has twice the host RAM of G5g at every suffix**, so never transfer a G5g verdict onto
  the same G6 size name.
- **The arch is not in an x86_64 DLAMI's name.** Only ARM64 images announce theirs, so an
  `arm64`→`x86_64` substitution in the name filter matches nothing.
- **`torch.cuda.get_arch_list()` has no `sm_89`** on this AMI, and that is fine — cubins run
  on any device of the same major with a minor ≥ their own. An exact-match check fails a
  healthy L4; it did, once, and aborted the first install.
- **G6 spot is priced at ~96–100% of on-demand** in us-east-1. Little saving, real
  interruption risk, and price is not a proxy for capacity.

See `CLAUDE.md` for the full set, and `docs/INHERITED.md` for what came from the fork.
