# 📊 Gemma 4 QAT vLLM GPU 2D Grid Concurrency Benchmark Report

Generated at: 2026-06-21 10:09:10
Endpoint: `https://gpu-12b-qat-mtp-wgcq55zbfq-uk.a.run.app`
Model: `/mnt/models/gemma-4-12B-it-qat-w4a16-ct` (NVIDIA L4 GPU Cloud Run)

## 🕒 Average Latency Matrix (seconds)

| Context \ Users | 1 | 8 | 64 | 512 | 2048 |
|---:|---:|---:|---:|---:|---:|
| **8** | 0.37s | 3.90s | 1.31s | 8.01s | 31.12s |
| **128** | 0.40s | 0.76s | 2.95s | 20.34s | 34.15s |
| **2048** | 0.00s | 1.09s | 3.35s | 22.81s | 33.76s |
| **16384** | 0.00s | 55.80s | 4.69s | 32.16s | 44.31s |

## 🚀 Throughput Matrix (Requests per second)

| Context \ Users | 1 | 8 | 64 | 512 | 2048 |
|---:|---:|---:|---:|---:|---:|
| **8** | 2.6 | 2.0 | 28.8 | 32.9 | 29.1 |
| **128** | 2.4 | 8.5 | 12.1 | 12.8 | 8.8 |
| **2048** | 0.0 | 6.0 | 10.6 | 7.9 | 9.8 |
| **16384** | 0.0 | 0.1 | 7.6 | 8.1 | 6.7 |
