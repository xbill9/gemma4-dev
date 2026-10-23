import json, time, torch, transformers
from transformers import AutoModelForCausalLM
MODEL = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
t = time.perf_counter()
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, key_mapping={r"^model\.language_model\.": "model."},
    device_map="cpu", quantization_config=transformers.CompressedTensorsConfig(run_compressed=False)).eval()
print(json.dumps({"load_s": round(time.perf_counter()-t,1)}))
def gib(mod): return round(sum(p.numel()*p.element_size() for p in list(mod.parameters())+list(mod.buffers()))/2**30, 3)
print(json.dumps({n: gib(c) for n, c in m.model.named_children()} | {"total": gib(m)}))
print(json.dumps({"dtypes": sorted({str(p.dtype) for p in m.parameters()}), "layer0": gib(m.model.layers[0]), "layer_sizes_distinct": sorted({gib(l) for l in m.model.layers})}))
