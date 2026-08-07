"""Gemma-4 E2B QAT bf16, lockstep batch-B Option-B trace (default B=4).

Same two-graph recipe as qat_e2b_trace.py, but every graph input carries a
batch dimension of B. Lockstep: all streams share the same position, so the
one-hot/masks stay batch-1 and broadcast. Decode weights are read once per
step regardless of B — on a weight-bandwidth-bound decoder the extra
streams should ride nearly free (TPU-measured law; this run prices it on
Inferentia2).

Validation: all B device streams (same prompt) must match the CPU batch-B
greedy reference token-for-token, and each other.

Env: MODEL_ID, B (default 4), PRE_OUT, DEC_OUT.
"""
import os
import time

import torch

torch.manual_seed(0)

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-E2B-it-qat-q4_0-unquantized")
B = int(os.environ.get("B", "4"))
MAX = 128
BUCKET = 32
NEG = torch.finfo(torch.float32).min
PRE_OUT = os.environ.get("PRE_OUT", f"/workspace/qat_e2b_b{B}_prefill.pt")
DEC_OUT = os.environ.get("DEC_OUT", f"/workspace/qat_e2b_b{B}_decode.pt")

from transformers import AutoTokenizer, DynamicCache, Gemma4ForConditionalGeneration

print("loading", MODEL_ID, "B =", B, flush=True)
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


def embed_ids(batch_ids):
    """batch_ids: list of B token-id lists (equal length)."""
    ids = torch.tensor(batch_ids)
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
    return torch.full((B, 1), pos, dtype=torch.long), onehot, full, slide


ec = m.generation_config.eos_token_id
EOS = set(ec) if isinstance(ec, (list, tuple)) else {ec}


def zero_kv():
    kb = [torch.zeros(B, LINFO[i][0], MAX, LINFO[i][1], dtype=WDT) for i in NONSHARED]
    vb = [torch.zeros(B, LINFO[i][0], MAX, LINFO[i][1], dtype=WDT) for i in NONSHARED]
    return kb, vb


def run_greedy(pre_fn, dec_fn, prompt, maxnew=30):
    """Lockstep greedy, no EOS early-exit (streams can't stop independently)."""
    n0 = len(prompt)
    pad = prompt + [0] * (BUCKET - n0)
    ie, ple = embed_ids([pad] * B)
    am = torch.tensor([[1] * n0 + [0] * (BUCKET - n0)] * B)
    with torch.no_grad():
        lg, ks, vs = pre_fn(ie, am, ple)
    first = lg[:, n0 - 1].argmax(-1)

    key_bufs, val_bufs = zero_kv()
    for j in range(len(NONSHARED)):
        key_bufs[j][:, :, :n0, :] = ks[j][:, :, :n0, :]
        val_bufs[j][:, :, :n0, :] = vs[j][:, :, :n0, :]

    seqs = [[int(first[b])] for b in range(B)]
    cur = n0
    for _ in range(maxnew - 1):
        ie1, ple1 = embed_ids([[seqs[b][-1]] for b in range(B)])
        position_ids, onehot, full_mask, slide_mask = host_pos_tensors(cur)
        with torch.no_grad():
            lg1, key_bufs, val_bufs = dec_fn(ie1, ple1, position_ids, onehot,
                                             full_mask, slide_mask, key_bufs, val_bufs)
        nxt = lg1[:, 0].argmax(-1)
        for b in range(B):
            seqs[b].append(int(nxt[b]))
        cur += 1
    return seqs


msgs = [{"role": "user", "content": "What is the capital of France?"}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
prompt = enc["input_ids"][0].tolist()
assert len(prompt) <= BUCKET, "prompt longer than BUCKET"
print("prompt ids:", prompt, flush=True)

import torch_neuronx

ie, ple = embed_ids([prompt + [0] * (BUCKET - len(prompt))] * B)
am = torch.tensor([[1] * len(prompt) + [0] * (BUCKET - len(prompt))] * B)
CARGS = ["--model-type", "transformer", "--auto-cast", "all", "--auto-cast-type", "bf16"]

t = time.time()
pre_neff = torch_neuronx.trace(pre, (ie, am, ple), compiler_args=CARGS)
print(f"PREFILL_COMPILED {time.time()-t:.0f}s", flush=True)

key_bufs, val_bufs = zero_kv()
ie1, ple1 = embed_ids([[prompt[-1]]] * B)
position_ids, onehot, full_mask, slide_mask = host_pos_tensors(len(prompt))
t = time.time()
dec_neff = torch_neuronx.trace(
    dec, (ie1, ple1, position_ids, onehot, full_mask, slide_mask, key_bufs, val_bufs),
    compiler_args=CARGS)
print(f"DECODE_COMPILED {time.time()-t:.0f}s", flush=True)

torch.jit.save(pre_neff, PRE_OUT)
torch.jit.save(dec_neff, DEC_OUT)
for p in (PRE_OUT, DEC_OUT):
    print(f"neff saved: {p} {os.path.getsize(p)/1e9:.2f} GB", flush=True)

cpu_seqs = run_greedy(pre, dec, prompt)
dev_seqs = run_greedy(pre_neff, dec_neff, prompt)
print("CPU[0]:   ", repr(tok.decode([s for s in cpu_seqs[0] if s not in EOS], skip_special_tokens=True)), flush=True)
print("Device[0]:", repr(tok.decode([s for s in dev_seqs[0] if s not in EOS], skip_special_tokens=True)), flush=True)
streams_equal = all(dev_seqs[b] == dev_seqs[0] for b in range(B))
print("STREAMS_EQUAL:", streams_equal, flush=True)
print("SEQ_MATCH:", cpu_seqs == dev_seqs, flush=True)
assert streams_equal, "device streams diverged from each other on identical prompts"
assert cpu_seqs == dev_seqs, "device output diverged from CPU reference"


def bench_decode_fixed(n=96):
    """Fixed-length lockstep decode: no EOS, no HTTP, prefill excluded."""
    p = prompt
    n0 = len(p)
    ie_, ple_ = embed_ids([p + [0] * (BUCKET - n0)] * B)
    am_ = torch.tensor([[1] * n0 + [0] * (BUCKET - n0)] * B)
    with torch.no_grad():
        lg, ks, vs = pre_neff(ie_, am_, ple_)
    nxt = lg[:, n0 - 1].argmax(-1)
    kb, vb = zero_kv()
    for j in range(len(NONSHARED)):
        kb[j][:, :, :n0, :] = ks[j][:, :, :n0, :]
        vb[j][:, :, :n0, :] = vs[j][:, :, :n0, :]
    cur = n0
    t0 = time.time()
    for _ in range(n):
        ie1_, ple1_ = embed_ids([[int(nxt[b])] for b in range(B)])
        pid, oh, fm, sm = host_pos_tensors(cur)
        with torch.no_grad():
            lg1, kb, vb = dec_neff(ie1_, ple1_, pid, oh, fm, sm, kb, vb)
        nxt = lg1[:, 0].argmax(-1)
        cur += 1
    dt = time.time() - t0
    ms = dt / n * 1000
    print(f"DECODE_FIXED B={B} {n} steps in {dt:.1f}s = {ms:.1f} ms/step | "
          f"{B*n/dt:.1f} tok/s aggregate | {n/dt:.1f} tok/s per stream", flush=True)


bench_decode_fixed(96)
print(f"QAT_E2B_B{B}_DONE", flush=True)
