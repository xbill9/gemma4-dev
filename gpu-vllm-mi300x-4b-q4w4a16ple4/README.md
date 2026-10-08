# gpu-vllm-mi300x-4b-q4w4a16ple4

Results for one arm of the MI300X data-type sweep of Gemma 4 E4B: **`xbill9/gemma-4-E4B-it-qat-q4_0-w4a16-ct-text-ple4`** through vLLM on **one AMD Instinct MI300X** (DigitalOcean GPU droplet via AMD Developer Cloud). An artifact rig: measurements only.

Slot 5, `q4w4a16ple4`: `q4w4a16` with only the per-layer embedding table int4, text only.

The checkpoint was built 2026-10-07 by `gpu-vllm-mi300x-2b/repack/build_missing.py` and served from the droplet's disk; not on Hugging Face yet. 

Every arm of this sweep serves through `gpu-vllm-mi300x-2b-q4w4a16`'s server.py with the checkpoint settings in the environment, driven by `gpu-vllm-mi300x-2b/dtype_sweep.py --size 4b` on one pinned image digest; the ratios against bf16 are in `gpu-vllm-mi300x-4b/benchmarks/runs/<run-id>/dtype-summary.json`.
