"""Gemma-4 E2B QAT bf16, batch-B Option-B trace with PER-STREAM positions.

Continuous-batching variant of qat_e2b_b4_trace.py: position_ids [B,1],
one-hot [B,1,MAX,1] and decode masks [B,1,1,MAX] carry a real batch
dimension, so every stream tracks its own position — requests with
different prompt lengths and arrival times can share the batch.

Validation is adversarial for stream isolation: 4 distinct prompts placed
as [A,B,A,C,B,D,A,D] (duplicates in different neighborhoods, heterogeneous
prompt lengths → heterogeneous positions from step 0). Device must match
the CPU batch reference token-for-token AND duplicate slots must emit
identical sequences — a per-stream mask bug breaks the duplicate check
even if single-prompt decoding looks fine.

Env: MODEL_ID, B (default 8), PRE_OUT, DEC_OUT.
"""
import os
import time

import torch

torch.manual_seed(0)

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-E2B-it-qat-q4_0-unquantized")
B = int(os.environ.get("B", "8"))
MAX = 128
BUCKET = 32
NEG = torch.finfo(torch.float32).min
PRE_OUT = os.environ.get("PRE_OUT", f"/workspace/qat_e2b_cb{B}_prefill.pt")
DEC_OUT = os.environ.get("DEC_OUT", f"/workspace/qat_e2b_cb{B}_decode.pt")

from transformers import AutoTokenizer, DynamicCache, Gemma4ForConditionalGeneration

print("loading", MODEL_ID, "B =", B, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
probe = tok("hello world").input_ids
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
    ids = torch.tensor(batch_ids)
    with torch.no_grad():
        ie = lang.embed_tokens(ids)
        ple = lang.get_per_layer_inputs(ids, ie)
    return ie, ple


def host_pos_tensors(pos_list):
    """Per-stream position tensors. pos_list: list of B ints."""
    pos = torch.tensor(pos_list)                                   # [B]
    ar = torch.arange(MAX)
    onehot = (ar[None, :] == pos[:, None]).view(B, 1, MAX, 1).to(WDT)
    valid = ar[None, :] <= pos[:, None]                            # [B, MAX]
    full = torch.where(valid, 0.0, NEG).view(B, 1, 1, MAX)
    slide = torch.where(valid & (ar[None, :] > (pos[:, None] - SW)), 0.0, NEG).view(B, 1, 1, MAX)
    return pos.view(B, 1).long(), onehot, full, slide


ec = m.generation_config.eos_token_id
EOS = set(ec) if isinstance(ec, (list, tuple)) else {ec}


def zero_kv():
    kb = [torch.zeros(B, LINFO[i][0], MAX, LINFO[i][1], dtype=WDT) for i in NONSHARED]
    vb = [torch.zeros(B, LINFO[i][0], MAX, LINFO[i][1], dtype=WDT) for i in NONSHARED]
    return kb, vb


def prep_prompts(questions):
    """questions: list of B strings → (padded ids [B][BUCKET], am rows, n0s)."""
    pads, ams, n0s = [], [], []
    for q in questions:
        enc = tok.apply_chat_template([{"role": "user", "content": q}],
                                      add_generation_prompt=True, return_tensors="pt", return_dict=True)
        p = enc["input_ids"][0].tolist()
        assert len(p) <= BUCKET, "prompt longer than BUCKET"
        pads.append(p + [0] * (BUCKET - len(p)))
        ams.append([1] * len(p) + [0] * (BUCKET - len(p)))
        n0s.append(len(p))
    return pads, ams, n0s


def run_greedy(pre_fn, dec_fn, questions, maxnew=24):
    """Per-stream-position lockstep greedy, fixed maxnew (no EOS exit)."""
    pads, ams, n0s = prep_prompts(questions)
    ie, ple = embed_ids(pads)
    am = torch.tensor(ams)
    with torch.no_grad():
        lg, ks, vs = pre_fn(ie, am, ple)
    nxt = [int(lg[b, n0s[b] - 1].argmax()) for b in range(B)]

    key_bufs, val_bufs = zero_kv()
    for j in range(len(NONSHARED)):
        key_bufs[j][:, :, :BUCKET, :] = ks[j][:, :, :BUCKET, :]
        val_bufs[j][:, :, :BUCKET, :] = vs[j][:, :, :BUCKET, :]

    seqs = [[nxt[b]] for b in range(B)]
    cur = list(n0s)                                # per-stream positions
    for _ in range(maxnew - 1):
        ie1, ple1 = embed_ids([[seqs[b][-1]] for b in range(B)])
        position_ids, onehot, full_mask, slide_mask = host_pos_tensors(cur)
        with torch.no_grad():
            lg1, key_bufs, val_bufs = dec_fn(ie1, ple1, position_ids, onehot,
                                             full_mask, slide_mask, key_bufs, val_bufs)
        for b in range(B):
            seqs[b].append(int(lg1[b, 0].argmax()))
            cur[b] += 1
    return seqs


QS = [
    "What is the capital of France?",
    "Name the largest planet in our solar system.",
    "What is 17 multiplied by 23? Reply with the number only.",
    "In one short sentence, what is AWS Inferentia?",
]
LAYOUT = [0, 1, 0, 2, 1, 3, 0, 3]        # duplicates in different neighborhoods
questions = [QS[LAYOUT[b]] for b in range(B)]
print("slot layout:", LAYOUT, flush=True)

import torch_neuronx

pads, ams, n0s = prep_prompts(questions)
print("prompt lengths per slot:", n0s, flush=True)
ie, ple = embed_ids(pads)
am = torch.tensor(ams)
CARGS = ["--model-type", "transformer", "--auto-cast", "all", "--auto-cast-type", "bf16"]

t = time.time()
pre_neff = torch_neuronx.trace(pre, (ie, am, ple), compiler_args=CARGS)
print(f"PREFILL_COMPILED {time.time()-t:.0f}s", flush=True)

key_bufs, val_bufs = zero_kv()
ie1, ple1 = embed_ids([[pads[b][n0s[b] - 1]] for b in range(B)])
position_ids, onehot, full_mask, slide_mask = host_pos_tensors(n0s)   # heterogeneous example
t = time.time()
dec_neff = torch_neuronx.trace(
    dec, (ie1, ple1, position_ids, onehot, full_mask, slide_mask, key_bufs, val_bufs),
    compiler_args=CARGS)
print(f"DECODE_COMPILED {time.time()-t:.0f}s", flush=True)

torch.jit.save(pre_neff, PRE_OUT)
torch.jit.save(dec_neff, DEC_OUT)
for p in (PRE_OUT, DEC_OUT):
    print(f"neff saved: {p} {os.path.getsize(p)/1e9:.2f} GB", flush=True)

cpu_seqs = run_greedy(pre, dec, questions)
dev_seqs = run_greedy(pre_neff, dec_neff, questions)
for b in range(B):
    txt = tok.decode([s for s in dev_seqs[b] if s not in EOS], skip_special_tokens=True)
    print(f"slot {b} (Q{LAYOUT[b]}):", repr(txt), flush=True)
dup_ok = all(dev_seqs[b] == dev_seqs[LAYOUT.index(LAYOUT[b])] for b in range(B))
print("DUP_ISOLATION:", dup_ok, flush=True)
print("SEQ_MATCH:", cpu_seqs == dev_seqs, flush=True)
assert dup_ok, "duplicate prompts in different slots diverged — stream isolation broken"
assert cpu_seqs == dev_seqs, "device output diverged from CPU reference"

# saved-artifact round trip (this is what the serving container loads)
pre_re = torch.jit.load(PRE_OUT)
dec_re = torch.jit.load(DEC_OUT)
re_seqs = run_greedy(pre_re, dec_re, questions)
print("RELOAD_MATCH:", re_seqs == cpu_seqs, flush=True)
assert re_seqs == cpu_seqs, "saved+reloaded neff diverged"
del pre_re, dec_re


def bench_decode_fixed(n=90):
    pads_, ams_, n0s_ = prep_prompts(questions)
    ie_, ple_ = embed_ids(pads_)
    am_ = torch.tensor(ams_)
    with torch.no_grad():
        lg, ks, vs = pre_neff(ie_, am_, ple_)
    nxt = [int(lg[b, n0s_[b] - 1].argmax()) for b in range(B)]
    kb, vb = zero_kv()
    for j in range(len(NONSHARED)):
        kb[j][:, :, :BUCKET, :] = ks[j][:, :, :BUCKET, :]
        vb[j][:, :, :BUCKET, :] = vs[j][:, :, :BUCKET, :]
    cur = list(n0s_)
    t0 = time.time()
    for _ in range(n):
        ie1_, ple1_ = embed_ids([[nxt[b]] for b in range(B)])
        pid, oh, fm, sm = host_pos_tensors(cur)
        with torch.no_grad():
            lg1, kb, vb = dec_neff(ie1_, ple1_, pid, oh, fm, sm, kb, vb)
        nxt = [int(lg1[b, 0].argmax()) for b in range(B)]
        cur = [c + 1 for c in cur]
    dt = time.time() - t0
    ms = dt / n * 1000
    print(f"DECODE_FIXED CB B={B} {n} steps in {dt:.1f}s = {ms:.1f} ms/step | "
          f"{B*n/dt:.1f} tok/s aggregate | {n/dt:.1f} tok/s per stream", flush=True)


bench_decode_fixed(90)
print(f"QAT_E2B_CB{B}_DONE", flush=True)
