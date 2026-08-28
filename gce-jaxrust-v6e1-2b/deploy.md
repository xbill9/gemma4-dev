# 🚀 Deploying Gemma 4 E2B QAT on Cloud TPU v6e-1 (flex-start) using Python JAX

This document outlines the deployment architecture, precision specifications, memory
budget, API endpoints, auto-start `systemd` configuration, and reusable GCE image
staging for running **Gemma 4 E2B QAT** on a **single-chip Cloud TPU v6e (Trillium)**
provisioned as a **flex-start Compute Engine instance** via **JAX**.

> **This page describes the Python JAX engine, which is no longer this rig's default.**
> It was inherited from `tpu-jax-v6e1-2b` at the 2026-08-28 fork and is kept because
> `workload="jax"` still provisions that engine as the parity oracle. The default path
> here is Rust — see `rust/README.md` and the "Rust/XLA on TPU" section of the skill.
>
> **Provenance of the numbers below.** Every measured figure on this page was taken on a
> **v6e-1**, on the **Python JAX** engine, through the **Cloud TPU API** — before the
> migration to Compute Engine. The control plane changes provisioning and not serving
> (the KV cache allocation came out at the same integer on both), so they remain valid
> *as JAX results*. **None of them is evidence about the Rust engine**, which has not run
> on a chip. In particular the checkpoint-choice argument below is a JAX-path argument:
> rlx reads safetensors and GGUF both, so the constraint on the Rust path is different
> and has to be re-derived rather than carried across.

---

## 📌 Overview & Key Highlights

- **Hardware Target**: Cloud TPU v6e (Trillium) single chip — Compute Engine machine
  type `ct6e-standard-1t`, **32 GB HBM**.
- **Provisioning**: `--provisioning-model=FLEX_START` Compute Engine instance. This rig
  has **no queued-resource path at all** — see the skill for why, and for the four
  provisioning models. `europe-west4-a` granted a single v6e chip first try with no
  queueing, which is why it is the default zone; it is a starting point rather than a
  guarantee, since capacity is zonal and moves within minutes.
- **Backend Stack**: Pure JAX `0.11.0`, `libtpu-0.0.44`, `flax`,
  `compressed-tensors 0.17.1`, `google-deepmind/gemma 4.1.0`.
- **Recommended Checkpoint**: `google/gemma-4-E2B-it-qat-q4_0-unquantized` — the
  faster of the two (10.1 vs 8.1 tok/s) and, with 32 GB, the one that fits with room to
  spare. See [Which checkpoint fits 32 GB](#-which-checkpoint-fits-32-gb) below. Note
  `jax_openai_server.py` still *defaults* to `-w4a16-ct`; pass `--model` to override.
- **API Server**: Pure JAX FastAPI + Uvicorn server exposing OpenAI-compatible endpoints
  + Prometheus `/metrics`.
- **Auto-Start & Staging**: `systemd` daemon enabled + a GCE image for fast redeploys.
- **Model Reference Guide**: See [models.md](models.md) for the complete reference of all
  model checkpoints, flags, and quantization options.

---

## 🏗️ Provisioning the flex-start v6e-1

### Via the MCP agent (preferred)

The `gce-jaxrust-v6e1-2b` server defaults to this shape (`ACCELERATOR_TYPE=v6e-1`,
`TENSOR_PARALLEL_SIZE=1`, `GOOGLE_CLOUD_ZONE=europe-west4-a`,
`PROVISIONING_MODEL=flex-start`), and `workload="jax"` is the default:

```
create_tpu_vm_instance()     # bare jax[tpu], no docker, no HF token
wait_for_jax_ready()         # polls the serial console — RUNNING is NOT ready
verify_jax_tpu()             # asserts jax.devices() sees the TPU
```

`find_tpu_vm()` sweeps zones instead of pinning one. Its candidates are the
intersection of the zones publishing `ct6e-standard-1t` and the regions holding quota
on the pool flex-start actually spends (`PREEMPTIBLE-TPU-V6E-per-project-region`, with
the family quota as fallback). Pass `zones=["europe-west4-a"]` to skip the lookup.

**If a create sits in PENDING, do not wait it out.** PENDING means either no quota or
no capacity and looks identical either way; `probe_zone_capacity()` fires a throwaway
spot create — spot does not queue — and names which, in seconds.

### Equivalent raw `gcloud`

```bash
gcloud compute instances create gce-jaxrust-v6e1-2b \
    --project=aisprint-491218 \
    --zone=europe-west4-a \
    --machine-type=ct6e-standard-1t \
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

`--scopes=cloud-platform`, `--boot-disk-size` and `--maintenance-policy=TERMINATE` are
all required and **all fail late** — long after the create, with nothing in the failure
pointing back at the flag. `get_deployment_command` emits this command with the current
configuration filled in.

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

## 🧠 Which checkpoint fits 32 GB

Component sizes measured with `jax.devices()[0].memory_stats()` on a v6e-1 — the chip
this page describes, so these are measurements rather than derivations.

| Component | Size | `-w4a16-ct` (int4) | `-q4_0-unquantized` (bf16) |
| :--- | ---: | ---: | ---: |
| Model weights | 3.52 GB int4 / 10.21 GB bf16 | 3.52 GB | 10.21 GB |
| MXU activations & XLA buffers | 1.50 GB | 1.50 GB | 1.50 GB |
| libtpu & XLA runtime reserves | 2.00 GB | 2.00 GB | 2.00 GB |
| **Fixed subtotal** | | **7.02 GB** | **13.71 GB** |
| **Left for KV cache** (of ~31.2 GiB usable) | | **~24.2 GB** | **~17.5 GB** |

A 128K-token fp8 KV cache measured 2.40 GB. So:

- **`-q4_0-unquantized` is the right default on v6e-1**: it is the faster checkpoint
  (10.1 vs 8.1 tok/s) and ~17.5 GB of headroom is about 7x a 128K context, so speed
  costs nothing here. On a 16 GB chip this choice inverts — that is the v5e trade-off,
  and it does not apply.
- **`-w4a16-ct` remains the smaller-footprint option**, worth choosing when you want
  maximum batch or context rather than maximum single-stream speed. It is also what
  `jax_openai_server.py` loads by default and what `w4a16_impl` decodes.
- A bf16 12B fits one v6e chip; 26B and 31B do not, quantized or otherwise
  (`_min_chips_for_model` blocks those before the long load).

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

The `8.1` above is the measured `-w4a16-ct` figure on this chip; `-q4_0-unquantized`
reaches 10.1.

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
- **Existing image**: `gemma4-jax-v6e1-image`, built on a v6e-1 — the chip this rig
  targets, so it is verified where it is used.

### Booting a new v6e-1 from a staged image

```bash
gcloud compute instances create gce-jaxrust-v6e1-2b \
    --project=aisprint-491218 \
    --zone=europe-west4-a \
    --machine-type=ct6e-standard-1t \
    --provisioning-model=FLEX_START \
    --max-run-duration=4h \
    --instance-termination-action=DELETE \
    --maintenance-policy=TERMINATE \
    --scopes=cloud-platform \
    --image=gemma4-jax-v6e1-image
```
