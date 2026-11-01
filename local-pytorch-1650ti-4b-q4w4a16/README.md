# local-pytorch-1650ti-4b-q4w4a16

Gemma 4 E4B, Google's QAT weights in int4, in plain PyTorch on a 4 GB GTX 1650 Ti. Local: nothing to provision.

    python3 openai_server.py --max-ctx 40960
    curl localhost:8100/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hi"}]}'

Loads in about 5 minutes on this host (15 GB of RAM) and needs the card to itself. `CLAUDE.md` has the details;
`benchmarks/runs/2026-10-31-accuracy-1650ti/REPORT.md` has the measurements.
