# CLAUDE.md — gpu-vllm-mi300x-embed2

Read `../CLAUDE.md` and `../gpu-vllm-mi300x-2b/CLAUDE.md` first; the MI300X facts there hold here.

- **Slot 4 is `embed2`**, the one model-slot value that is not a Gemma 4 size (`../NAMING.md`).
- **No MCP server, no `tpu.env`.** `embed_run.py` is the whole rig: it takes the droplet IP and runs
  everything over ssh. Create and destroy of the droplet stay human console steps, as in every
  MI300X rig. A stopped droplet still bills.
- **Both servers stay up for the whole run** (generator at `--gpu-memory-utilization 0.70`, embedder
  at 0.15), so an "alone" number means the other server was loaded and idle, the same memory split
  as the side-by-side numbers.
- **The reference prompts are in the text.** EmbeddingGemma 2 expects `task: … | query: ` and
  `title: none | text: ` prefixes; `/v1/embeddings` adds none, so `reference.py` writes them into the
  strings both sides embed.
- Never run ROCm commands locally; the reference runs on the local CPU only.
