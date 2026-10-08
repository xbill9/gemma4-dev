# gpu-vllm-mi300x-26b-q4w4a16

Results for one arm of the MI300X data-type sweep of Gemma 4 26B-A4B: **`xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct-text`** through vLLM on **one AMD Instinct MI300X** (DigitalOcean GPU droplet via AMD Developer Cloud). An artifact rig: measurements only.

Slot 5, `q4w4a16`: the QAT Q4_0 grid kept exactly as int4 W4A16 (compressed-tensors, group 32), text only.

The checkpoint is pulled from Hugging Face. 

Every arm of this sweep serves through `gpu-vllm-mi300x-2b-q4w4a16`'s server.py with the checkpoint settings in the environment, driven by `gpu-vllm-mi300x-2b/dtype_sweep.py --size 26b` on one pinned image digest; the ratios against bf16 are in `gpu-vllm-mi300x-26b/benchmarks/runs/<run-id>/dtype-summary.json`.
