# local-llamacpp-i71360p-2b-q4_0

Gemma 4 E2B-it served by `llama-server` on the CPU of a local workstation, from the exact
Q4_0 GGUF: Google's QAT weights on their trained 4-bit grid, embeddings included.

| | |
| --- | --- |
| Platform | `local` — no control plane, nothing provisioned, Ctrl-C is a complete teardown |
| Runtime | `llama-server` from llama.cpp `fc07d781e`, CPU-only build (`GGML_CUDA=OFF`) |
| Hardware | `i71360p` — Lenovo Yoga 9 14IRP8, i7-1360P (4 P + 8 E cores, 16 threads), AVX2 + AVX-VNNI, 15 GiB, no discrete GPU |
| Model | [`xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf), 2.64 GB |
| Encoding | `q4_0` — every quantized tensor, both embedding tables included |
| Endpoint | `http://127.0.0.1:8090` |

`tpu.env` holds the authoritative values.

## Results — 2026-09-29

Rebuilt GGUF against Google's `gemma-4-E2B-it-qat-q4_0-gguf`, both measured against a bf16
GGUF of the same QAT weights:

- 🟢 **2.64 GB against 3.35 GB.**
- 🟢 **Mean KL divergence 0.00175 against 0.0543**; same top token 98.06% against 87.18%.
- 🟢 **Generation 7% faster** in both run orders (1.069x, 1.074x). Prefill shows no stable
  difference.
- ⚠️ Absolute tok/s on this laptop follows package temperature: the same file generated at
  21.70 tok/s in a pass that started at 76 °C and 24.45 in one that started at 53 °C. Quote
  ratios from interleaved runs.

Details: [`benchmarks/runs/2026-09-29-exact-gguf-i71360p/REPORT.md`](benchmarks/runs/2026-09-29-exact-gguf-i71360p/REPORT.md).

## Use

```bash
make install     # system python3, no virtualenv
make build       # llama.cpp worktree at LLAMA_CPP_COMMIT, CPU only
make download    # the GGUF into MODEL_PATH's directory
make serve       # foreground; Ctrl-C tears down
make query       # one chat completion
make test        # offline unittest suite
```

The MCP server (`server.py`, registered as `local-llamacpp-i71360p-2b-q4_0`) exposes
`start_model_server`, `stop_model_server`, `model_server_status`, `attest_arm`,
`query_model`, `cpu_status`, `model_info` and `get_help`. There are no provisioning tools.

Gemma 4 thinks before it answers. `llama-server` routes that to `reasoning_content`; send
`"chat_template_kwargs": {"enable_thinking": false}` for a direct reply, or keep
`max_tokens` at 512 or more.
