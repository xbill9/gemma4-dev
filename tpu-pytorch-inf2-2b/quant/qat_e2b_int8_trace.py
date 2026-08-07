"""Gemma-4 E2B QAT int8-sym-g32 on a single NeuronCore via torch_neuronx.

Same Option-B recipe as qat_e2b_trace.py (two graphs, KV as graph I/O,
host-side PLE, eager attn + tanh GELU + softcap), but every decoder-layer
linear is swapped for a QuantLinear that stores int8 weights + fp32 group
scales (from quant/quantize_int8.py) and dequantizes to bf16 in-graph, so
the neff carries int8 weights and the dequant runs on-device each call.

The CPU reference runs the SAME swapped modules, so SEQ_MATCH compares
identical int8-dequant math on host vs device.

Env:
  QUANT_CKPT  path to model_int8_g32.safetensors (default /workspace)
  INLINE      1 (default) inline weights into the neff; 0 = weight
              separation (fallback if the compiler constant-folds the
              dequant back to bf16 — check neff sizes / decode rate)
"""
import os
import time

import torch

torch.manual_seed(0)

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-E2B-it-qat-q4_0-unquantized")
QUANT_CKPT = os.environ.get("QUANT_CKPT", "/workspace/model_int8_g32.safetensors")
INLINE = os.environ.get("INLINE", "1") == "1"
GROUP = 32
MAX = 128
BUCKET = 32
NEG = torch.finfo(torch.float32).min
PRE_OUT = os.environ.get("PRE_OUT", "/workspace/qat_e2b_int8_prefill.pt")
DEC_OUT = os.environ.get("DEC_OUT", "/workspace/qat_e2b_int8_decode.pt")

from transformers import AutoTokenizer, DynamicCache, Gemma4ForConditionalGeneration
from safetensors import safe_open

print("loading", MODEL_ID, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
probe = tok("hello world").input_ids
print("tokenizer probe:", probe, flush=True)
assert len(set(probe)) >= 2, "tokenizer maps everything to one id — bad download"

m = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="eager"
).eval()
lang = m.model.language_model
lm_head = m.lm_head
cfg = lang.config
SW = cfg.sliding_window
WDT = m.dtype
softcap = getattr(m.config.text_config, "final_logit_softcapping", None)


class GeluTanh(torch.nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


for mod in lang.modules():
    if hasattr(mod, "act_fn"):
        mod.act_fn = GeluTanh()


class QuantLinear(torch.nn.Module):
    """int8-sym-g32 linear: dequant to bf16 in-graph, then F.linear."""

    def __init__(s, wq, ws, bias):
        super().__init__()
        s.register_buffer("wq", wq)                     # int8 [out, in]
        s.register_buffer("ws", ws)                     # fp32 [out, in/GROUP]
        s.bias_t = bias
        s.out_features, s.in_features = wq.shape

    def forward(s, x):
        o, g = s.ws.shape
        w = s.wq.to(WDT) * s.ws.to(WDT).unsqueeze(-1).expand(o, g, GROUP).reshape(o, g * GROUP)
        return torch.nn.functional.linear(x, w, s.bias_t)


print("loading int8 checkpoint", QUANT_CKPT, flush=True)
qt = {}
with safe_open(QUANT_CKPT, framework="pt") as f:
    for k in f.keys():
        if k.endswith(".weight_i8") or k.endswith(".weight_scale"):
            qt[k] = f.get_tensor(k)

swapped = 0
for name, mod in list(m.named_modules()):
    if not isinstance(mod, torch.nn.Linear):
        continue
    qk = name + ".weight_i8"
    if qk not in qt:
        continue
    wq, ws = qt[qk], qt[name + ".weight_scale"]
    assert tuple(wq.shape) == tuple(mod.weight.shape), (name, wq.shape, mod.weight.shape)
    bias = mod.bias.detach() if mod.bias is not None else None
    parent = m
    *parents, leaf = name.split(".")
    for p in parents:
        parent = getattr(parent, p)
    setattr(parent, leaf, QuantLinear(wq, ws.float(), bias))
    swapped += 1
print(f"swapped {swapped} linears to int8 (of {len(qt)//2} quantized in ckpt)", flush=True)
assert swapped == len(qt) // 2, "quantized tensors in ckpt without a matching Linear"
assert swapped > 0, "no linears swapped — checkpoint/model name mismatch?"

NONSHARED, LINFO = [], {}
for i, lyr in enumerate(lang.layers[: cfg.num_hidden_layers]):
    a = lyr.self_attn
    if not a.is_kv_shared_layer:
        hd = a.head_dim
        NONSHARED.append(i)
        LINFO[i] = (a.k_proj.out_features // hd, hd)
print("non-shared KV layers:", NONSHARED, flush=True)


def softcap_logits(lg):
    return softcap * torch.tanh(lg / softcap) if softcap else lg


class PreWrap(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.lang = lang
        s.head = lm_head

    def forward(s, ie, am, ple):
        cache = DynamicCache()
        out = s.lang(inputs_embeds=ie, per_layer_inputs=ple, attention_mask=am,
                     use_cache=True, past_key_values=cache)
        lg = softcap_logits(s.head(out.last_hidden_state))
        ks = [cache.layers[i].keys for i in NONSHARED]
        vs = [cache.layers[i].values for i in NONSHARED]
        return (lg, ks, vs)


class StaticKV:
    is_compileable = False

    def __init__(s, key_bufs, val_bufs, onehot):
        s.key = {i: key_bufs[j] for j, i in enumerate(NONSHARED)}
        s.val = {i: val_bufs[j] for j, i in enumerate(NONSHARED)}
        s.oh = onehot

    def update(s, k, v, idx, *a, **kw):
        s.key[idx] = s.key[idx] * (1.0 - s.oh) + k * s.oh
        s.val[idx] = s.val[idx] * (1.0 - s.oh) + v * s.oh
        return s.key[idx], s.val[idx]

    def get_seq_length(s, *a, **k):
        return 0

    def export(s):
        return [s.key[i] for i in NONSHARED], [s.val[i] for i in NONSHARED]


class DecWrap(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.lang = lang
        s.head = lm_head

    def forward(s, ie, ple, position_ids, onehot, full_mask, slide_mask, key_bufs, val_bufs):
        cache = StaticKV(key_bufs, val_bufs, onehot)
        masks = {"full_attention": full_mask, "sliding_attention": slide_mask}
        out = s.lang(inputs_embeds=ie, per_layer_inputs=ple, position_ids=position_ids,
                     attention_mask=masks, use_cache=True, past_key_values=cache)
        lg = softcap_logits(s.head(out.last_hidden_state))
        ks, vs = cache.export()
        return (lg, ks, vs)


pre = PreWrap().eval()
dec = DecWrap().eval()


def embed_ids(id_list):
    ids = torch.tensor([id_list])
    with torch.no_grad():
        ie = lang.embed_tokens(ids)
        ple = lang.get_per_layer_inputs(ids, ie)
    return ie, ple


def host_pos_tensors(pos):
    ar = torch.arange(MAX)
    onehot = (ar == pos).view(1, 1, MAX, 1).to(WDT)
    valid = ar <= pos
    full = torch.where(valid, 0.0, NEG).view(1, 1, 1, MAX)
    slide = torch.where(valid & (ar > pos - SW), 0.0, NEG).view(1, 1, 1, MAX)
    return torch.tensor([[pos]], dtype=torch.long), onehot, full, slide


ec = m.generation_config.eos_token_id
EOS = set(ec) if isinstance(ec, (list, tuple)) else {ec}


def run_greedy(pre_fn, dec_fn, prompt, maxnew=30):
    n0 = len(prompt)
    pad = prompt + [0] * (BUCKET - n0)
    ie, ple = embed_ids(pad)
    am = torch.tensor([[1] * n0 + [0] * (BUCKET - n0)])
    with torch.no_grad():
        lg, ks, vs = pre_fn(ie, am, ple)
    first = int(lg[0, n0 - 1].argmax())

    key_bufs = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1], dtype=WDT) for i in NONSHARED]
    val_bufs = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1], dtype=WDT) for i in NONSHARED]
    for j in range(len(NONSHARED)):
        key_bufs[j][:, :, :n0, :] = ks[j][:, :, :n0, :]
        val_bufs[j][:, :, :n0, :] = vs[j][:, :, :n0, :]

    seq, cur = [first], n0
    for _ in range(maxnew):
        if seq[-1] in EOS:
            break
        ie1, ple1 = embed_ids([seq[-1]])
        position_ids, onehot, full_mask, slide_mask = host_pos_tensors(cur)
        with torch.no_grad():
            lg1, key_bufs, val_bufs = dec_fn(ie1, ple1, position_ids, onehot,
                                             full_mask, slide_mask, key_bufs, val_bufs)
        seq.append(int(lg1[0, 0].argmax()))
        cur += 1
    return seq


msgs = [{"role": "user", "content": "What is the capital of France?"}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
prompt = enc["input_ids"][0].tolist()
assert len(prompt) <= BUCKET, "prompt longer than BUCKET"
print("prompt ids:", prompt, flush=True)

import torch_neuronx

ie, ple = embed_ids(prompt + [0] * (BUCKET - len(prompt)))
am = torch.tensor([[1] * len(prompt) + [0] * (BUCKET - len(prompt))])
CARGS = ["--model-type", "transformer", "--auto-cast", "all", "--auto-cast-type", "bf16"]

t = time.time()
pre_neff = torch_neuronx.trace(pre, (ie, am, ple), compiler_args=CARGS,
                               inline_weights_to_neff=INLINE)
print(f"PREFILL_COMPILED {time.time()-t:.0f}s", flush=True)

key_bufs = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1], dtype=WDT) for i in NONSHARED]
val_bufs = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1], dtype=WDT) for i in NONSHARED]
ie1, ple1 = embed_ids([prompt[-1]])
position_ids, onehot, full_mask, slide_mask = host_pos_tensors(len(prompt))
t = time.time()
dec_neff = torch_neuronx.trace(
    dec, (ie1, ple1, position_ids, onehot, full_mask, slide_mask, key_bufs, val_bufs),
    compiler_args=CARGS, inline_weights_to_neff=INLINE)
print(f"DECODE_COMPILED {time.time()-t:.0f}s", flush=True)

torch.jit.save(pre_neff, PRE_OUT)
torch.jit.save(dec_neff, DEC_OUT)
for p in (PRE_OUT, DEC_OUT):
    print(f"neff saved: {p} {os.path.getsize(p)/1e9:.2f} GB", flush=True)

cpu_seq = run_greedy(pre, dec, prompt)
dev_seq = run_greedy(pre_neff, dec_neff, prompt)
print("CPU:   ", repr(tok.decode([s for s in cpu_seq if s not in EOS], skip_special_tokens=True)), flush=True)
print("Device:", repr(tok.decode([s for s in dev_seq if s not in EOS], skip_special_tokens=True)), flush=True)
print("SEQ_MATCH:", cpu_seq == dev_seq, flush=True)
assert cpu_seq == dev_seq, "device output diverged from CPU reference"

# the serving container loads the SAVED artifact — verify the round trip
pre_re = torch.jit.load(PRE_OUT)
dec_re = torch.jit.load(DEC_OUT)
re_seq = run_greedy(pre_re, dec_re, prompt)
print("RELOAD_MATCH:", re_seq == cpu_seq, flush=True)
assert re_seq == cpu_seq, "saved+reloaded neff diverged"
del pre_re, dec_re


def bench(question, maxnew=80):
    e = tok.apply_chat_template([{"role": "user", "content": question}],
                                add_generation_prompt=True, return_tensors="pt", return_dict=True)
    p = e["input_ids"][0].tolist()
    t0 = time.time()
    seq = run_greedy(pre_neff, dec_neff, p, maxnew=maxnew)
    dt = time.time() - t0
    print(tok.decode([s for s in seq if s not in EOS], skip_special_tokens=True), flush=True)
    print(f"BENCH {len(seq)} tokens in {dt:.1f}s = {len(seq)/dt:.1f} tok/s", flush=True)


def bench_decode_fixed(n=100):
    """Fixed-length decode-only benchmark: no EOS, no HTTP, prefill excluded."""
    p = prompt
    n0 = len(p)
    pad = p + [0] * (BUCKET - n0)
    ie_, ple_ = embed_ids(pad)
    am_ = torch.tensor([[1] * n0 + [0] * (BUCKET - n0)])
    with torch.no_grad():
        lg, ks, vs = pre_neff(ie_, am_, ple_)
    nxt = int(lg[0, n0 - 1].argmax())
    kb = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1], dtype=WDT) for i in NONSHARED]
    vb = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1], dtype=WDT) for i in NONSHARED]
    for j in range(len(NONSHARED)):
        kb[j][:, :, :n0, :] = ks[j][:, :, :n0, :]
        vb[j][:, :, :n0, :] = vs[j][:, :, :n0, :]
    cur = n0
    t0 = time.time()
    for _ in range(n):
        ie1_, ple1_ = embed_ids([nxt])
        pid, oh, fm, sm = host_pos_tensors(cur)
        with torch.no_grad():
            lg1, kb, vb = dec_neff(ie1_, ple1_, pid, oh, fm, sm, kb, vb)
        nxt = int(lg1[0, 0].argmax())
        cur += 1
    dt = time.time() - t0
    print(f"DECODE_FIXED {n} steps in {dt:.1f}s = {n/dt:.1f} tok/s", flush=True)


bench("In one sentence, what is AWS Inferentia?")
bench_decode_fixed(96)  # cap: n0 + 96 < MAX=128
print("QAT_E2B_INT8_DONE", flush=True)
