# local-llamacpp-i71360p-4b-q4_0

Gemma 4 E4B-it served by `llama-server` on the CPU of a local workstation, from the exact
Q4_0 GGUF: Google's QAT weights on their trained 4-bit grid, embeddings included. The E4B
sibling of [`local-llamacpp-i71360p-2b-q4_0`](../local-llamacpp-i71360p-2b-q4_0/): same
host, same binary, same flags, and slot 4 is the only slot that differs.

| | |
| --- | --- |
| Platform | `local` — no control plane, nothing provisioned, Ctrl-C is a complete teardown |
| Runtime | `llama-server` from llama.cpp `fc07d781e`, CPU-only build (`GGML_CUDA=OFF`) |
| Hardware | `i71360p` — Lenovo Yoga 9 14IRP8, i7-1360P (4 P + 8 E cores, 16 threads), AVX2 + AVX-VNNI, 15 GiB, no discrete GPU |
| Model | [`xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf`](https://huggingface.co/xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf), 4.22 GB — 4.5B effective, 8.0B total |
| Encoding | `q4_0` — every weight matrix, both embedding tables included |
| Endpoint | `http://127.0.0.1:8091` |

`tpu.env` holds the authoritative values.

## Results

None on this host yet. The file's quality was measured by
[`local-llamacpp-1650ti-4b-q4_0`](../local-llamacpp-1650ti-4b-q4_0/): mean KL divergence from
bf16 0.00103 against Google's 0.0353.

## Use

```bash
make install     # system python3, no virtualenv
make build       # the shared llama.cpp worktree at LLAMA_CPP_COMMIT, CPU only
make download    # the GGUF into MODEL_PATH's directory, SHA-256 checked
make serve       # foreground; Ctrl-C tears down
make query       # one chat completion
make test        # offline unittest suite
```

The MCP server (`server.py`, registered as `local-llamacpp-i71360p-4b-q4_0`) exposes
`start_model_server`, `stop_model_server`, `model_server_status`, `attest_arm`,
`query_model`, `cpu_status`, `model_info` and `get_help`. There are no provisioning tools.

Gemma 4 thinks before it answers. `llama-server` routes that to `reasoning_content`; send
`"chat_template_kwargs": {"enable_thinking": false}` for a direct reply, or keep
`max_tokens` at 512 or more.
