## start_model_server

📡 Started llama-server (pid 781760) → http://127.0.0.1:8090

```
/home/xbill/llama.cpp-fc07d78/build-cpu/bin/llama-server -m /home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact/gemma-4-E2B-it-q4_0-exact.gguf --host 127.0.0.1 --port 8090 -ngl 0 -c 8192 -ctk f16 -ctv f16 -fa 1 -t 8 -tb 8 --parallel 1 --metrics
```

Loading is not instant. Poll `model_server_status`; log at `run/llama-server.log`.

## model_server_status

✅ Serving at http://127.0.0.1:8090 (pid 781760). `/health` → 200.

Arm attested **cpu** — device=cpu · pid=781760 · -ngl 0 · /home/xbill/llama.cpp-fc07d78/build-cpu/bin/llama-server

## attest_arm

✅ **Arm attested: cpu** (this rig expects **cpu**)

- **pid:** 781760
- **exe:** `/home/xbill/llama.cpp-fc07d78/build-cpu/bin/llama-server`
- **sha256:** `b50156c3dfba9175…` — the pairing identity. Two arms are comparable only if they were built from one commit; the hash is what actually ran.
- **`-ngl`:** 0
- **GPU libraries mapped:** none
- **CUDA_VISIBLE_DEVICES:** (empty — devices hidden)
- **model:** `/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact/gemma-4-E2B-it-q4_0-exact.gguf`
- **threads:** `-t 8` `-tb 8` · **ctx:** 8192

## cpu_status

📡 **CPU** — `local-llamacpp-i71360p-2b-q4_0`

- **Model:** 13th Gen Intel(R) Core(TM) i7-1360P
- **Logical CPUs:** 16
- **Hybrid:** 8 P-core threads, 8 E-core threads
- **SIMD:** avx avx2 avx_vnni
- **RAM:** 7.91 GiB available of 15.36 GiB
- **Threads configured:** decode `-t 8`, prefill `-tb 8` (spot-checked 2026-09-29, not swept — see tpu.env)
- **CPU affinity:** not set. The kernel places the threads across P- and E-cores.
- **Measurement health:** thermal behaviour of this host is not characterized. Treat an absolute t/s as one reading until repeated with a stated cooldown.

⚠️  No AVX-512. llama.cpp takes its AVX2 kernels here; do not compare against a number from an AVX-512 host on the strength of the same build flags.

## model_info

📡 **Model** — `local-llamacpp-i71360p-2b-q4_0`

- **Name:** `xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`
- **Path:** `/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact/gemma-4-E2B-it-q4_0-exact.gguf`
- **On disk:** 2.64 GB
- **Quantization slot:** `q4_0`, and every quantized tensor is Q4_0 — both embedding tables included (Google's GGUF stores those as Q6_K).
- **Touched every token:** ~1.30 GB. `per_layer_token_embd` (1.32 GB, 50% of the file) is `TENSOR_READ_LAZY` and is served by GET_ROWS out of the mmap, a few rows per token.

Run `inspect_gguf.py` to re-derive the split from the artifact rather than trusting these numbers.

## query_model('What is the capital of Australia? One sentence.', max_tokens=1024)

✅ **Reply**

The capital of Australia is Canberra.

---
_(plus 486 chars of reasoning, suppressed)_
prompt 26 tok · completion 123 tok · 25.8 tok/s

## /proc/<pid>/status after the query

```
VmRSS:	 4011904 kB
RssAnon:	 1433372 kB
RssFile:	 2578532 kB
```

## stop_model_server

✅ Sent SIGTERM to llama-server (pid 781760). Memory is released on exit.
