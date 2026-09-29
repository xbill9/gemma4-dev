## start_model_server

📡 Started llama-server (pid 857775) → http://127.0.0.1:8090

```
/home/xbill/llama.cpp-fc07d78/build-cpu/bin/llama-server -m /home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf --host 127.0.0.1 --port 8090 -ngl 0 -c 8192 -ctk f16 -ctv f16 -fa 1 -t 8 -tb 8 --parallel 1 --metrics
```

Loading is not instant. Poll `model_server_status`; log at `run/llama-server.log`.

## model_server_status

✅ Serving at http://127.0.0.1:8090 (pid 857775). `/health` → 200.

Arm attested **cpu** — device=cpu · pid=857775 · -ngl 0 · /home/xbill/llama.cpp-fc07d78/build-cpu/bin/llama-server

## model_info

📡 **Model** — `local-llamacpp-i71360p-2b-q4_0`

- **Name:** `xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`
- **Path:** `/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf`
- **On disk:** 2.62 GB
- **Quantization slot:** `q4_0`, and every weight matrix is Q4_0 — both embedding tables included (Google's GGUF stores those as Q6_K).
- **Touched every token:** ~1.28 GB. `per_layer_token_embd` (1.32 GB, 51% of the file) is `TENSOR_READ_LAZY` and is served by GET_ROWS out of the mmap, a few rows per token.

Run `inspect_gguf.py` to re-derive the split from the artifact rather than trusting these numbers.

## query_model('What is the capital of Australia? One sentence.')

✅ **Reply**

The capital of Australia is Canberra.

---
_(plus 429 chars of reasoning, suppressed)_
prompt 26 tok · completion 110 tok · 26.1 tok/s

## stop_model_server

✅ Sent SIGTERM to llama-server (pid 857775). Memory is released on exit.
