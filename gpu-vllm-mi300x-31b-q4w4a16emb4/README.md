# gpu-vllm-mi300x-31b-q4w4a16emb4

Results for one arm of the MI300X data-type sweep of Gemma 4 31B: **`xbill9/gemma-4-31B-it-qat-q4_0-w4a16-ct-text-emb4`** through vLLM on **one AMD Instinct MI300X** (DigitalOcean GPU droplet via AMD Developer Cloud). An artifact rig: measurements only.

Slot 5, `q4w4a16emb4`: `q4w4a16` with `embed_tokens`, the per-layer embeddings where present and an untied `lm_head` int4 too, text only.

The checkpoint is pulled from Hugging Face. 

Every arm of this sweep serves through `gpu-vllm-mi300x-2b-q4w4a16`'s server.py with the checkpoint settings in the environment, driven by `gpu-vllm-mi300x-2b/dtype_sweep.py --size 31b` on one pinned image digest; the ratios against bf16 are in `gpu-vllm-mi300x-31b/benchmarks/runs/<run-id>/dtype-summary.json`.
