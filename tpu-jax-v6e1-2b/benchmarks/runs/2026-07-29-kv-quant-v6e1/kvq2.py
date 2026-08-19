"""Quantized KV cache quality on the real checkpoint.

Same harness as verify_gen.py (which works): tokenize with the chat template,
let the engine prepend BOS, decode greedily. Only the cache dtype varies.
"""
import os, sys, time
sys.path.insert(0, os.path.expanduser("~/gemma"))
from jax_engine import JaxGemmaEngine
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

MID = "google/gemma-4-E2B-it-qat-w4a16-ct"
snap = snapshot_download(MID, allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"],
                         token=os.environ.get("HF_TOKEN"))
tok = AutoTokenizer.from_pretrained(snap)
QS = ["What is 2+2?", "The capital of France is", "Name three colours.",
      "Explain gravity in one sentence."]
eos = [t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>"))
       if isinstance(t, int) and t >= 0]

for dtype in ("bf16", "int8", "fp8_e4m3", "fp8_e5m2"):
    print(f"\n########## KV cache = {dtype} ##########", flush=True)
    try:
        eng = JaxGemmaEngine(MID, kv_cache_dtype=dtype, quant_mode="w4a16",
                             max_model_len=512)
        t0 = time.time(); eng.load(local_dir=snap)
        eng.bos_token_id = tok.bos_token_id
        print(f"loaded {eng.weight_bytes/1e9:.2f} GB in {time.time()-t0:.1f}s", flush=True)
        for q in QS:
            ids = tok(f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n",
                      add_special_tokens=False)["input_ids"]
            out, st = eng.generate(ids, max_new_tokens=24, temperature=0.0,
                                   eos_token_ids=eos)
            print(f"  Q: {q}\n  A: {tok.decode(out, skip_special_tokens=True)!r}"
                  f"   [{st.decode_tok_per_s:.0f} tok/s, {st.finish_reason}]", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  FAILED: {str(e).splitlines()[0][:120]}", flush=True)

print("\nKVQ2_DONE", flush=True)
