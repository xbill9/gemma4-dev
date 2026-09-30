# gpu-vllm-g4dn-2b-w4a16

vLLM on one EC2 `g4dn.xlarge` (NVIDIA T4, Turing SM 7.5) serving
`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4`, as the raw-EC2 counterpart of the
sagemaker-gemma SageMaker T4 endpoint. Forked from `gpu-vllm-g4dn-2b` on 2026-09-30; slot 5 is
the only slot that differs (`@../NAMING.md`). The parent's `CLAUDE.md` covers the Turing clamp
and the derived-image path, which are unchanged here.

    python3 ec2_measure.py benchmarks/runs/<date>-<label> <endpoint-name> [model-id]

launches on-demand, serves, runs `compare.py` on the instance and terminates.

## Runs

| Run | Result |
| --- | --- |
| `benchmarks/runs/2026-09-30-sagemaker-match-g4dn` | decode 111.2 tok/s vs 108.5 on SageMaker, 40/40 answers identical |
