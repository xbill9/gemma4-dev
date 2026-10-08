# gpu-vllm-mi300x-4b

Results for one arm of the MI300X data-type sweep of Gemma 4 E4B: **`google/gemma-4-E4B-it`** through vLLM on **one AMD Instinct MI300X** (DigitalOcean GPU droplet via AMD Developer Cloud). An artifact rig: measurements only.

The checkpoint is pulled from Hugging Face. The reference instruction-tuned release in bf16, served text only.

Every arm of this sweep serves through `gpu-vllm-mi300x-2b-q4w4a16`'s server.py with the checkpoint settings in the environment, driven by `gpu-vllm-mi300x-2b/dtype_sweep.py --size 4b` on one pinned image digest; the ratios against bf16 are in `gpu-vllm-mi300x-4b/benchmarks/runs/<run-id>/dtype-summary.json`.
