"""CPU reference embeddings for google/embeddinggemma-2, to check the MI300X server against.

    python3 reference.py            -> reference/reference.json

Runs sentence-transformers on the local CPU (no GPU on this machine). The model card's
prompts are written into the text itself, so the served model gets byte-identical input
through /v1/embeddings, which applies no prompt of its own.
"""
import json
import platform
import time

import sentence_transformers as st
import torch
import transformers

MODEL = "google/embeddinggemma-2"
REVISION = "914f7f8"
Q = "task: search result | query: "
D = "title: none | text: "
QUERIES = [
    "Which number format runs fastest on an AMD MI300X?",
    "How much memory does Gemma 4 31B need in bf16?",
    "What is quantization-aware training?",
    "How do I serve a model with vLLM on ROCm?",
    "Why is int4 slow on gfx942?",
    "What is a mixture-of-experts model?",
    "How many tokens fit in the KV cache?",
    "Is fp8 e4m3fnuz the same as e4m3fn?",
]
DOCS = [
    "fp8 is faster than bf16 for Gemma 4 12B and 31B on one MI300X, from 1.12x to 1.45x.",
    "Gemma 4 31B loads in 58.99 GiB of HBM at bf16 and 30.63 GiB at fp8.",
    "Quantization-aware training teaches a model's weights to sit on a low-bit grid during training.",
    "vllm serve starts an OpenAI-compatible server; on ROCm use the vllm-openai-rocm image.",
    "TritonW4A16LinearKernel unpacks 4-bit weights to bf16 before each multiply.",
    "A mixture-of-experts layer routes each token through a few of many expert networks.",
    "The KV cache stores attention keys and values for every token in flight.",
    "E4M3FNUZ has an exponent bias one higher than E4M3FN, a largest value of 240 and no negative zero.",
    "Bananas are rich in potassium and grow in tropical climates.",
    "The 1969 Moon landing was broadcast live to an estimated 600 million people.",
    "Sourdough bread rises using wild yeast and lactic acid bacteria.",
    "A haiku has three lines of five, seven and five syllables.",
]
LONG = D + " ".join(DOCS * 20)  # about 5,500 tokens, inside the 8K context, to check long inputs too


def main():
    t = time.time()
    m = st.SentenceTransformer(MODEL, revision=REVISION, device="cpu")
    texts = [Q + q for q in QUERIES] + [D + d for d in DOCS] + [LONG]
    full = m.encode(texts, normalize_embeddings=True, batch_size=4)
    out = {
        "model": MODEL, "revision": REVISION, "device": "cpu", "host": platform.processor() or platform.machine(),
        "torch": torch.__version__, "transformers": transformers.__version__, "sentence_transformers": st.__version__,
        "seconds": round(time.time() - t, 1), "n_queries": len(QUERIES), "n_docs": len(DOCS),
        "long_tokens": len(m.tokenizer(LONG)["input_ids"]),
        "texts": texts, "embeddings": [[round(float(x), 7) for x in v] for v in full],
    }
    for dim in (512, 256, 128):
        cut = full[:, :dim]
        cut = cut / (cut ** 2).sum(1, keepdims=True) ** 0.5
        out[f"embeddings_{dim}"] = [[round(float(x), 7) for x in v] for v in cut]
    sims = full[: len(QUERIES)] @ full[len(QUERIES): len(QUERIES) + len(DOCS)].T
    out["top_doc_per_query"] = [int(i) for i in sims.argmax(1)]
    json.dump(out, open("reference/reference.json", "w"))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith(("embeddings", "texts"))}, indent=1))


if __name__ == "__main__":
    main()
