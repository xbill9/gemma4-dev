import json, sys, time, urllib.request
import numpy as np
from sentence_transformers import SentenceTransformer
Q = "task: search result | query: "
D = "title: none | text: "
long = " ".join(f"Sentence {i}: tensor processing units accelerate matrix multiplication for neural networks." for i in range(120))
texts = [Q + "Which accelerator has no bf16 datapath?",
         D + "The Tesla T4 is a Turing part with fp16 tensor cores and no bfloat16 support.",
         D + "Sourdough bread needs a starter, flour, water and salt.",
         "task: sentence similarity | query: A cat sat on the mat.",
         D + long]
req = urllib.request.Request("http://127.0.0.1:8000/v1/embeddings",
      data=json.dumps({"model": "google/embeddinggemma-2", "input": texts}).encode(),
      headers={"Content-Type": "application/json"})
import os
if os.environ.get("PHASE")=="ref":
    V = np.load(os.environ["OUT"]); r={"usage":{"prompt_tokens":"-"}}; dt=0.0
else:
    t = time.time(); r = json.load(urllib.request.urlopen(req)); dt = time.time() - t
    V = np.array([d["embedding"] for d in r["data"]], dtype=np.float64)
print("vllm: n", len(V), "dim", V.shape[1], "norms", np.round(np.linalg.norm(V, axis=1), 5), "prompt_tokens", r["usage"]["prompt_tokens"], f"{dt:.2f}s")
import os
if os.environ.get("PHASE")=="vllm":
    np.save(os.environ["OUT"], V); sys.exit(0)
st = SentenceTransformer("google/embeddinggemma-2", device="cpu", model_kwargs={"dtype": "float32"},
                         config_kwargs={"vision_config": None, "audio_config": None})
R = st.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float64)
cos = (V * R).sum(1) / np.linalg.norm(V, axis=1) / np.linalg.norm(R, axis=1)
print("1-cos(vllm, sentence-transformers):", ["%.1e" % x for x in 1 - cos]); print("max|vllm-st|:", "%.1e" % np.abs(V - R).max())
print("token counts (ST tokenizer):", [len(st.tokenizer(x)["input_ids"]) for x in texts])
print("query·T4 doc", round(float(V[0] @ V[1]), 4), " query·bread doc", round(float(V[0] @ V[2]), 4))
