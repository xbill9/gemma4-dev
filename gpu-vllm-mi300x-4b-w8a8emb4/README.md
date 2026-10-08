# gpu-vllm-mi300x-4b-w8a8emb4

Results for one arm of the MI300X data-type sweep of Gemma 4 E4B: **`xbill9/gemma-4-E4B-it-qat-w8a8-int8-emb4`** through vLLM on **one AMD Instinct MI300X** (DigitalOcean GPU droplet via AMD Developer Cloud). An artifact rig: measurements only.

Slot 5, `w8a8emb4`: `w8a8` with the vocabulary tables and an untied `lm_head` int4, text only.

The checkpoint is pulled from Hugging Face. 

Every arm of this sweep serves through `gpu-vllm-mi300x-2b-q4w4a16`'s server.py with the checkpoint settings in the environment, driven by `gpu-vllm-mi300x-2b/dtype_sweep.py --size 4b` on one pinned image digest; the ratios against bf16 are in `gpu-vllm-mi300x-4b/benchmarks/runs/<run-id>/dtype-summary.json`.
