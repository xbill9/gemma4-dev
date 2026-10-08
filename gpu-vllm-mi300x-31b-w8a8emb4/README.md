# gpu-vllm-mi300x-31b-w8a8emb4

Results for one arm of the MI300X data-type sweep of Gemma 4 31B: **`xbill9/gemma-4-31B-it-qat-w8a8-int8-emb4`** through vLLM on **one AMD Instinct MI300X** (DigitalOcean GPU droplet via AMD Developer Cloud). An artifact rig: measurements only.

Slot 5, `w8a8emb4`: `w8a8` with the vocabulary tables and an untied `lm_head` int4, text only.

The checkpoint was built 2026-10-07 by `gpu-vllm-mi300x-2b/repack/build_missing.py` and served from the droplet's disk; not on Hugging Face yet. 

Every arm of this sweep serves through `gpu-vllm-mi300x-2b-q4w4a16`'s server.py with the checkpoint settings in the environment, driven by `gpu-vllm-mi300x-2b/dtype_sweep.py --size 31b` on one pinned image digest; the ratios against bf16 are in `gpu-vllm-mi300x-31b/benchmarks/runs/<run-id>/dtype-summary.json`.
