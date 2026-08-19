"""Does quantizing the PLE table cost output quality on the real checkpoint?

Capacity said PLE quantization is throughput-neutral. That only matters if the
model still works: int4 round-trips the table at ~7% relative error against
int8's 0.5%, and a synthetic bound says nothing about trained weights.
"""
import os, sys, gc, json, time
sys.path.insert(0, os.path.expanduser("~/gemma"))
from jax_engine import JaxGemmaEngine
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

MID = "google/gemma-4-E2B-it-qat-w4a16-ct"
snap = snapshot_download(MID, allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"])
tok = AutoTokenizer.from_pretrained(snap)
eos = [t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>"))
       if isinstance(t, int) and t >= 0]
QS = ["What is 2+2?",
      "The capital of France is",
      "Name three colours.",
      "Explain gravity in one sentence.",
      "Write a haiku about TPUs.",
      "List the first five prime numbers.",
      "Translate 'good morning' into Spanish."]
OUT = []
for bits in (0, 8, 4):
    print(f"\n=== ple_bits={bits or 'bf16'} ===", flush=True)
    eng = JaxGemmaEngine(MID, kv_cache_dtype="int8", quant_mode="w4a16",
                         max_model_len=512, ple_bits=bits)
    eng.load(local_dir=snap)
    eng.bos_token_id = tok.bos_token_id
    print(f"  weights {eng.weight_bytes/1e9:.2f} GB", flush=True)
    for q in QS:
        ids = tok(f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n",
                  add_special_tokens=False)["input_ids"]
        o, st = eng.generate(ids, max_new_tokens=28, temperature=0.0, eos_token_ids=eos)
        txt = tok.decode(o, skip_special_tokens=True)
        print(f"  {q!r}\n     -> {txt!r}  [{st.decode_tok_per_s:.0f} tok/s]", flush=True)
        OUT.append(dict(ple_bits=bits, q=q, a=txt,
                        weights_gb=round(eng.weight_bytes/1e9, 2)))
    del eng; gc.collect()

with open(os.path.expanduser("~/ple_quality.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print("\nQUALITY_DONE", flush=True)
