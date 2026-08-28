# 🎛️ Gemma 4 Model Flags & Configuration Guide (`models.md`)

This guide provides a comprehensive reference for all model flags, CLI arguments, data types, quantization schemes, and supported checkpoints when deploying **Gemma 4** on **Google Cloud TPUs** using **JAX**.

---

## 📌 Supported Gemma 4 Model Checkpoints

| Model ID | Effective Parameters | Quantization Scheme | Format | Target TPU Hardware | Recommended Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`google/gemma-4-E2B-it-qat-w4a16-ct`** | **2.3B** (5.1B total) | QAT W4A16 | `compressed-tensors` | TPU v6e-1 (32 GB HBM) | High-throughput, lowest disk/load footprint (**Default**) |
| **`google/gemma-4-E2B-it-qat-q4_0-unquantized`** | **2.3B** (5.1B total) | QAT Q4_0 Baseline | Unquantized `bfloat16` | TPU v6e-1 (32 GB HBM) | Maximum QAT generation quality |
| **`google/gemma-4-E2B-it-qat-mobile-transformers`** | **2.3B** (5.1B total) | QAT Mixed (W4/W2, A8, KV8) | `transformers` | Edge / Mobile / TPU | Low-power mobile & edge deployment |
| **`google/gemma-4-E2B-it`** | **2.3B** (5.1B total) | Unquantized | Full `bfloat16` | TPU v6e-1 (32 GB HBM) | Baseline non-QAT model |
| **`google/gemma-4-E4B-it`** | **4.5B** (8.0B total) | Unquantized / QAT | `bfloat16` / `W4A16` | TPU v6e-1 / v6e-4 | Mid-sized multimodal serving |
| **`google/gemma-4-31B-it`** | **31.0B** | Unquantized / QAT | `bfloat16` / `W4A16` | TPU v6e-8 / v5p-8 | Production multi-chip TPU pod serving |

---

## 🎛️ Complete Model & CLI Flags Reference

### 1. Model Selection & Precision Flags

| Flag | Type | Default Value | Valid Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`--model`** | `str` | `google/gemma-4-E2B-it-qat-w4a16-ct` | Any HF Repo ID or local path | Specifies the model checkpoint to load into JAX TPU memory. |
| **`--kv-cache-dtype`** | `str` | `fp8` | `fp8` (`fp8_e4m3fn`), `bfloat16`, `int8` | Data type for Key-Value attention cache in TPU HBM. `fp8` cuts KV memory usage by 50%. |
| **`--torch-dtype`** / **`--dtype`** | `str` | `bfloat16` | `bfloat16`, `float32` | Precision for intermediate activations and MXU matrix multiplications. |
| **`--max-new-tokens`** | `int` | `128` | Positive integers (`1` to `8192`) | Maximum number of new tokens generated per inference request. |
| **`--temperature`** | `float` | `0.7` | `0.0` to `2.0` | Sampling temperature. `0.0` enables deterministic greedy decoding. |
| **`--top-p`** | `float` | `0.95` | `0.0` to `1.0` | Nucleus sampling probability threshold. |
| **`--top-k`** | `int` | `50` | Positive integers (e.g. `40`, `50`) | Top-K sampling token candidate pool size. |

---

### 2. Server & Network Execution Flags

| Flag | Type | Default Value | Example Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`--host`** | `str` | `0.0.0.0` | `0.0.0.0`, `127.0.0.1` | Network interface IP address for the FastAPI / Uvicorn server. |
| **`--port`** | `int` | `8000` | `8000`, `8080` | HTTP port for OpenAI API (`/v1/chat/completions`) & Prometheus (`/metrics`). |
| **`--prompt`** | `str` | *"Explain why TPUs excel..."* | `"Custom user prompt"` | Input prompt string for single-shot testing with [`jax_gemma4_e2b.py`](jax_gemma4_e2b.py). |

---

## 🔬 Quantization Precision Matrix

```
+---------------------------------------------------------------------------------------+
|                                    Gemma 4 QAT Model                                  |
+-------------------+-------------------+-------------------+---------------------------+
| Layer Component   | Storage Precision | Execution Dtype   | Scaling & Metadata        |
+-------------------+-------------------+-------------------+---------------------------+
| Model Weights     | INT4 (4-bit)      | bfloat16          | group_size = 32           |
| Layer Scale/Zero  | bfloat16          | bfloat16          | Per-channel / per-group   |
| Layer Biases      | bfloat16          | bfloat16          | Unquantized float16       |
| Activations       | bfloat16          | bfloat16          | MXU native execution      |
| KV Cache          | FP8 (e4m3fn)      | FP8 / bfloat16    | 50% HBM reduction         |
+-------------------+-------------------+-------------------+---------------------------+
```

---

## 🚀 Execution Examples

### 1. Launch OpenAI-Compatible API Server with W4A16 QAT & FP8 KV Cache
```bash
python3 jax_openai_server.py \
  --model google/gemma-4-E2B-it-qat-w4a16-ct \
  --kv-cache-dtype fp8 \
  --host 0.0.0.0 \
  --port 8000
```

### 2. Run CLI Benchmark Test
```bash
python3 jax_gemma4_e2b.py \
  --model google/gemma-4-E2B-it-qat-w4a16-ct \
  --prompt "Explain why TPUs excel at JAX workloads in two sentences." \
  --max-new-tokens 128
```
