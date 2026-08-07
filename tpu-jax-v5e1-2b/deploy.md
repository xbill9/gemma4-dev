# 🚀 Deploying Gemma 4 E2B QAT on Cloud TPU v5e-1 (flex-start) using JAX

This document outlines the deployment architecture, precision specifications, memory
budget, API endpoints, auto-start `systemd` configuration, and reusable GCE image
staging for running **Gemma 4 E2B QAT** on a **single-chip Cloud TPU v5e** provisioned
as a **flex-start GCE instance** via **JAX**.

> **Provenance of the numbers below.** Every measured figure in this repo — throughput,
> HBM occupancy, KV sizes — was taken on a **v6e-1**. The v5e-1 sections here are
> *derived* from those measurements plus the v5e chip's published 16 GB HBM, and are
> labelled as such. Nothing on this page is a v5e-1 measurement yet. Re-measure before
> quoting a v5e-1 tok/s anywhere.

---

## 📌 Overview & Key Highlights

- **Hardware Target**: Cloud TPU v5e single chip — GCE machine type `ct5lp-hightpu-1t`,
  **16 GB HBM** (half a v6e chip).
- **Provisioning**: `--provisioning-model=FLEX_START` GCE instance (not a queued
  resource). Capacity for the single-chip v5e shape has only been granted in
  **`us-west4-a`**, which is why that is the repo's default zone.
- **Backend Stack**: Pure JAX `0.11.0`, `libtpu-0.0.44`, `flax`,
  `compressed-tensors 0.17.1`, `google-deepmind/gemma 4.1.0`.
- **Recommended Checkpoint**: `google/gemma-4-E2B-it-qat-w4a16-ct` — see
  [Which checkpoint fits 16 GB](#-which-checkpoint-fits-16-gb) below. This differs from
  the v6e-1 recommendation, where the faster `-q4_0-unquantized` variant fits easily.
- **API Server**: Pure JAX FastAPI + Uvicorn server exposing OpenAI-compatible endpoints
  + Prometheus `/metrics`.
- **Auto-Start & Staging**: `systemd` daemon enabled + a GCE image for fast redeploys.
- **Model Reference Guide**: See [models.md](models.md) for the complete reference of all
  model checkpoints, flags, and quantization options.

---

## 🏗️ Provisioning the flex-start v5e-1

### Via the MCP agent (preferred)

The `tpu-jax-v5e1-2b` server defaults to this shape (`ACCELERATOR_TYPE=v5e-1`,
`TENSOR_PARALLEL_SIZE=1`, `GOOGLE_CLOUD_ZONE=us-west4-a`):

```
create_tpu_vm_instance(workload="jax")           # bare jax[tpu], no docker, no HF token
wait_for_jax_ready(instance_name="jax-tpu-vm")   # polls the serial console
verify_jax_tpu(instance_name="jax-tpu-vm")       # asserts jax.devices() sees the TPU
```

`find_tpu_vm(workload="jax")` sweeps zones instead of pinning one; its zone list comes
from the quota metric of the accelerator's own family, so a v5e sweep queries
`TPUV5sLitepodPerProjectPerZoneForTPUAPI`. Pass `zones=["us-west4-a"]` to skip the
lookup entirely.

### Equivalent raw `gcloud`

```bash
gcloud compute instances create jax-tpu-vm \
    --project=aisprint-491218 \
    --zone=us-west4-a \
    --machine-type=ct5lp-hightpu-1t \
    --provisioning-model=FLEX_START \
    --request-valid-for-duration=2h \
    --max-run-duration=4h \
    --instance-termination-action=DELETE \
    --image-project=ubuntu-os-accelerator-images \
    --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=200GB \
    --scopes=cloud-platform \
    --metadata-from-file=startup-script=startup_script_jax_template.sh
```

The legacy queued-resources path also reaches v5e; it spells the accelerator
`v5litepod-1` and needs the `v2-alpha-tpuv5-lite` runtime image (a v6e image will not
boot a v5e node). `create_tpu_queued_resource` derives that image from
`ACCELERATOR_TYPE` automatically; `TPU_RUNTIME_VERSION` overrides it.

**Billing:** flex-start bills until the instance is deleted, and self-deletes at
`--max-run-duration`. Tear down with `destroy_tpu_vm_instance` when finished.

---

## 🎛️ Model & CLI Server Flags

For a detailed walkthrough of all model flags and checkpoint options, refer to
[models.md](models.md).

The JAX TPU server [`jax_openai_server.py`](jax_openai_server.py) and CLI runner
[`jax_gemma4_e2b.py`](jax_gemma4_e2b.py) support the following flags:

### 1. Model & Precision Flags

| Flag | Type | Default Value | Valid Options / Examples | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`--model`** | `str` | `google/gemma-4-E2B-it-qat-w4a16-ct` | `google/gemma-4-E2B-it-qat-w4a16-ct`<br>`google/gemma-4-E2B-it-qat-q4_0-unquantized`<br>`google/gemma-4-E2B-it` | Specifies the Hugging Face model checkpoint repository or path to load onto the TPU. |
| **`--kv-cache-dtype`** | `str` | `fp8` | `fp8` (`fp8_e4m3fn`), `bfloat16`, `int8` | Sets the precision for Key-Value attention cache memory in TPU HBM. `fp8` reduces KV memory by 50%. |
| **`--torch-dtype`** / **`--dtype`** | `str` | `bfloat16` | `bfloat16`, `float32` | Data type for intermediate activations and MXU matrix multiplications. `bfloat16` is native on TPU. |
| **`--max-new-tokens`** | `int` | `128` | Positive integers (e.g. `128`, `512`, `2048`) | Maximum number of new tokens generated per completion request. |
| **`--temperature`** | `float` | `0.7` | `0.0` (greedy) to `1.0` | Controls sampling randomness. `0.0` uses deterministic greedy decoding. |

### 2. Network & Server Flags

| Flag | Type | Default Value | Example Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`--host`** | `str` | `0.0.0.0` | `0.0.0.0`, `127.0.0.1` | Network interface address for the FastAPI Uvicorn web server. |
| **`--port`** | `int` | `8000` | `8000`, `8080` | HTTP port on which the OpenAI REST API & `/metrics` endpoints listen. |
| **`--prompt`** | `str` | *"Explain why TPUs excel..."* | `"Any text prompt"` | Input prompt for single-shot CLI testing via `jax_gemma4_e2b.py`. |

---

## 🎯 Model Precision & Quantization Architecture

| Component | Precision / Scheme | Description |
| :--- | :--- | :--- |
| **Model Weights (`W4`)** | **INT4 (4-bit integer)** | QAT compressed-tensors with `group_size=32` (~3.52 GB compressed disk size) |
| **Activations (`A16`)** | **BF16 (`bfloat16`)** | 16-bit floating point for full precision matrix multiplies on TPU MXU |
| **KV Cache (`KV8`)** | **FP8 (`fp8_e4m3fn`)** | 8-bit floating point Key-Value cache (50% memory saving) |

---

## 🧠 Which checkpoint fits 16 GB

*Derived, not measured on v5e-1.* The component sizes come from
`jax.devices()[0].memory_stats()` on a v6e-1; only the HBM ceiling changes.

| Component | Size | `-w4a16-ct` (int4) | `-q4_0-unquantized` (bf16) |
| :--- | ---: | ---: | ---: |
| Model weights | 3.52 GB int4 / 10.21 GB bf16 | 3.52 GB | 10.21 GB |
| MXU activations & XLA buffers | 1.50 GB | 1.50 GB | 1.50 GB |
| libtpu & XLA runtime reserves | 2.00 GB | 2.00 GB | 2.00 GB |
| **Fixed subtotal** | | **7.02 GB** | **13.71 GB** |
| **Left for KV cache** (of ~15.6 GiB usable) | | **~9.7 GB** | **~3.0 GB** |

A 128K-token fp8 KV cache measured 2.40 GB on v6e-1. So:

- **`-w4a16-ct` is the safe default on v5e-1**: ~9.7 GB of KV headroom, roughly 4x the
  128K-context footprint, leaving room for batching.
- **`-q4_0-unquantized` is the faster checkpoint (10.1 vs 8.1 tok/s on v6e-1) but is
  tight here** — ~3.0 GB of headroom is barely above one 128K context and leaves nothing
  for batch. Expect to cap `--max-model-len` and batch size, and expect OOM if you do
  not. This is the opposite of the v6e-1 guidance, where 32 GB made it the free choice.
- Anything larger than E2B does not belong on a v5e-1 in bf16.

---

## 🌐 Server Endpoints & Usage (`jax_openai_server.py`)

The server script [`jax_openai_server.py`](jax_openai_server.py) runs on port `8000`.

### Exposed REST Endpoints

- **`GET /health`**: Health & precision status check.
- **`GET /metrics`**: Prometheus formatted metrics (request count, token count, tok/s, HBM bytes used).
- **`GET /v1/models`**: List active served models.
- **`POST /v1/chat/completions`**: OpenAI Chat Completions API.
- **`POST /v1/completions`**: OpenAI Text Completions API.

Get the live address with `get_tpu_vm_endpoint` rather than hardcoding one — a
flex-start instance gets a fresh external IP each time it is created.

### Sample cURL Request

```bash
curl http://<instance-external-ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-E2B-it-qat-w4a16-ct",
    "messages": [
      {"role": "user", "content": "Explain why TPUs excel at JAX workloads."}
    ],
    "max_tokens": 64
  }'
```

### Sample Prometheus Metrics Output (`/metrics`)

```prometheus
# HELP tpu_jax_requests_total Total HTTP requests processed by JAX TPU server
# TYPE tpu_jax_requests_total counter
tpu_jax_requests_total{model="google/gemma-4-E2B-it-qat-w4a16-ct",status="success"} 1

# HELP tpu_jax_tokens_per_second Current generation throughput in tokens per second
# TYPE tpu_jax_tokens_per_second gauge
tpu_jax_tokens_per_second{model="google/gemma-4-E2B-it-qat-w4a16-ct"} 8.1

# HELP tpu_jax_hbm_used_bytes High Bandwidth Memory used in bytes
# TYPE tpu_jax_hbm_used_bytes gauge
tpu_jax_hbm_used_bytes{device="TPU_0(process=0,(0,0,0,0))"} 2765312
```

The `8.1` above is the v6e-1 figure carried over as an example of the metric's shape —
it is not a v5e-1 result.

---

## ⚙️ Systemd Auto-Start Configuration

The server runs under `systemd` to ensure automatic startup whenever the VM boots up.

Service file at `/etc/systemd/system/jax-openai.service`:

```ini
[Unit]
Description=JAX Gemma 4 OpenAI API Server
After=network.target

[Service]
Type=simple
User=xbill
WorkingDirectory=/home/xbill
ExecStart=/home/xbill/venv_jax312/bin/python3 /home/xbill/jax_openai_server.py --model google/gemma-4-E2B-it-qat-w4a16-ct --kv-cache-dtype fp8 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Managing the Service
```bash
sudo systemctl status jax-openai
sudo systemctl restart jax-openai
sudo systemctl stop jax-openai
```

---

## 📦 Instant Boot via Reusable GCE Image

Staging a GCE image with the environment and cached model weights cuts a redeploy to
well under a minute — worth doing on flex-start, where the instance self-deletes at
`--max-run-duration` and gets rebuilt often.

- **GCP Project**: `aisprint-491218`
- **Existing image**: `gemma4-jax-v6e1-image` (built on a v6e-1). The JAX stack in it is
  hardware-independent, but build a `gemma4-jax-v5e1-image` from a v5e-1 boot disk
  before relying on it — the image name encodes where it was verified, and the v5e-1
  image has not been staged yet.

### Booting a new v5e-1 from a staged image

```bash
gcloud compute instances create jax-tpu-vm \
    --project=aisprint-491218 \
    --zone=us-west4-a \
    --machine-type=ct5lp-hightpu-1t \
    --provisioning-model=FLEX_START \
    --max-run-duration=4h \
    --instance-termination-action=DELETE \
    --maintenance-policy=TERMINATE \
    --scopes=cloud-platform \
    --image=gemma4-jax-v5e1-image
```
