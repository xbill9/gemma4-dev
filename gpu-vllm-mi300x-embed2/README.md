# gpu-vllm-mi300x-embed2

`google/embeddinggemma-2` (EmbeddingGemma 2, 740M, built on the Gemma 4 architecture) served by
stock vLLM on one AMD Instinct MI300X droplet from AMD Developer Cloud, alone and beside a Gemma 4
generator (`xbill9/gemma-4-12B-it-qat-q4_0-fp8-text`) on the same card.

| File | What it does |
| --- | --- |
| `reference.py` | CPU reference vectors with sentence-transformers → `reference/reference.json` |
| `embed_run.py` | One droplet run: correctness against the reference, embedding throughput, the generator alone, then each under the other's load |
| `benchmarks/runs/<run-id>/` | `results.json`, `check.json`, per-cell bench logs, both servers' boot logs |

```bash
python3 reference.py                       # local CPU, ~10 minutes
python3 embed_run.py --ip <droplet ip> --run-id 2026-10-09-embed2-mi300x
```

Image: `vllm/vllm-openai-rocm@sha256:3b5af9b0…` (vLLM `0.31.1rc1.dev173`, transformers 5.19.0), the
nightly of 2026-10-09. The data-type sweep's digest (`ec62abec…`) registers `EmbeddingGemma2Model` in
vLLM, but its transformers 5.18.0 cannot parse the `embedding_gemma2` config and the server exits at
startup. Text only: `--limit-mm-per-prompt '{"image": 0, "audio": 0}'`; no ROCm image ships the audio
extras.
