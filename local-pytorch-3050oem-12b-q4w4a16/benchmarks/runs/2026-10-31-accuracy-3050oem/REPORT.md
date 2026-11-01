# Accuracy: 12B int4 PyTorch on the RTX 3050 OEM, 2026-10-31

**No score. BFCL did not complete.** `openai_server.py --max-ctx 4096`, one request at a time.

- Prompts 1 to 46 answered, 46 of 46 right; llama.cpp also has these 46 right.
- Request 47 onward: 354 of 354 returned HTTP 500. `server.log` shows CUDA out of memory, and no later request recovered.
- `gen_eval.py` printed accuracy 0.115 because it counts those failures as wrong. That figure measures the crash, not the
  model. llama.cpp's full score is 0.948 on the same 400.

GSM8K was not run. Cause of the failure at request 47 not investigated.
