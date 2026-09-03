# gpu-llamacpp-g5g-2b-q4_0

An MCP devops agent for **EC2 G5g** (AWS Graviton2 host + NVIDIA **T4G**, Turing SM 7.5),
serving **`google/gemma-4-E2B-it-qat-q4_0-gguf`** through **`llama-server`**.

> **This rig has served nothing.** Scaffolded 2026-09-02. `benchmarks/` is empty on purpose,
> and every number in this repo's prose is either arithmetic from the GGUF's tensor table or a
> measurement from a sibling rig — each says which. See `CLAUDE.md`.

## What makes it different

**It is the only rig in the monorepo serving 4-bit weights on a GPU.** Every other route was
checked on 2026-09-02 and closed (root `QUANTIZATION.md` has the evidence):

- **vLLM 0.26.0** has no `gguf` module at all — CUDA build or TPU.
- **JAX** has no GGUF reader anywhere in its ecosystem.
- **transformers 5.12.1** reads the file, then dequantizes to fp32 (a 9.4 GB host transient on
  one tensor) and **silently drops 35 `layer_scalar` tensors**.

It also **compiles**, because llama.cpp publishes no prebuilt Linux aarch64 CUDA binary. That
is the main asymmetry against its planned sibling `gpu-ollama-g5g-2b-q4_0`, which ships one
with native sm_75 SASS already in it.

## Quick start

```bash
pip install -r requirements.txt        # the MCP server's own deps, on your machine
make test                              # 50 offline tests: no AWS, no network, no GPU
make lint
make skill                             # regenerate the skill + plugin snapshots
```

Then, through the MCP server:

```
create_g5g_instance  ->  get_install_progress  ->  verify_gpu_arch
                     ->  verify_model_health   ->  get_metrics
```

**There is no deploy step.** Cloud-init builds llama.cpp, verifies the built binary sees a CUDA
device, enables the unit, and llama-server fetches its own checkpoint.

## Three failures that are silent by construction

Each yields a server that starts, binds, and answers **correctly**:

| Failure | What catches it |
| --- | --- |
| CPU-only build | `-DGGML_CUDA=ON` at configure; `verify_gpu_arch` greps the built binary's `--list-devices`; `verify_model_health` flags decode < 3 tok/s |
| Partial GPU offload | `--n-gpu-layers 999` |
| No `nvcc` on the AMI | the bootstrap exits 1 rather than letting cmake configure without CUDA |

If a number looks low, run `verify_gpu_arch` **before** investigating anything else.

## Reading a number off it

Quote the decode rate `get_metrics` derives from `llamacpp:tokens_predicted_total` /
`llamacpp:tokens_predicted_seconds_total`. Never an end-to-end rate — the PyTorch sibling
measured the two disagreeing by up to 36% on the same rows. Prefill is reported separately and
must never be folded in.

```bash
python3 sweep.py --base http://<ip>:8000/v1 --out benchmarks/runs/<date>-<what>-g5g
```

## Configuration

`tpu.env` is the source of truth, and **every key in it is read by `server.py`** — a test
asserts this, because the PyTorch fork carried five inert keys for a month. The directory name
is documentation, not config: never copy a slot value into a CLI flag.

## Layout

```
server.py        the MCP server: EC2 lifecycle, bootstrap rendering, SRE tools
tpu.env          authoritative configuration
sweep.py         benchmark harness (shared shape with every sibling)
make_report.py   sweep output -> serving-report.schema.json
tests/           offline; test_server.py asserts on the RENDERED BOOTSTRAP
```

There is no serving payload. The server is upstream `llama-server`, built from a pinned ref.
