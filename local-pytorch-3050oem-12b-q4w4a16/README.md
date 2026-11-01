# local-pytorch-3050oem-12b-q4w4a16

Gemma 4 12B QAT int4 in plain PyTorch on an RTX 3050 OEM. Short prompts only; see `CLAUDE.md`.

    python3 openai_server.py --port 8100 --max-ctx 4096
    curl localhost:8100/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hi"}]}'
