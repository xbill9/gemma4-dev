"""Diagnose CB SEQ_MATCH divergence: per-stream first-divergence index + logit margins.

Loads the already-saved cb8 neffs, reruns CPU vs device greedy, and for each
stream reports where they first diverge and the CPU logit margin at that
position (tiny margin -> bf16 tie-flip, benign; early/large -> real bug).
"""
import os
import torch

torch.manual_seed(0)

MODEL_ID = os.environ.get("MODEL_ID", "/workspace/model")
B = 8
MAX = 128
BUCKET = 32
NEG = torch.finfo(torch.float32).min
PRE_OUT = "/workspace/qat_e2b_cb8_prefill.pt"
DEC_OUT = "/workspace/qat_e2b_cb8_decode.pt"

from transformers import AutoTokenizer, DynamicCache, Gemma4ForConditionalGeneration

tok = AutoTokenizer.from_pretrained(MODEL_ID)
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
        NONSHARED.append(i)
        LINFO[i] = (a.k_proj.out_features // a.head_dim, a.head_dim)


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
import torch_neuronx  # registers torch.classes.neuron.Model for jit.load
pre_neff = torch.jit.load(PRE_OUT)
dec_neff = torch.jit.load(DEC_OUT)


def embed_ids(batch_ids):
    ids = torch.tensor(batch_ids)
    with torch.no_grad():
        ie = lang.embed_tokens(ids)
        ple = lang.get_per_layer_inputs(ids, ie)
    return ie, ple


def host_pos_tensors(pos_list):
    pos = torch.tensor(pos_list)
    ar = torch.arange(MAX)
    onehot = (ar[None, :] == pos[:, None]).view(B, 1, MAX, 1).to(WDT)
    valid = ar[None, :] <= pos[:, None]
    full = torch.where(valid, 0.0, NEG).view(B, 1, 1, MAX)
    slide = torch.where(valid & (ar[None, :] > (pos[:, None] - SW)), 0.0, NEG).view(B, 1, 1, MAX)
    return pos.view(B, 1).long(), onehot, full, slide


QS = [
    "What is the capital of France?",
    "Name the largest planet in our solar system.",
    "What is 17 multiplied by 23? Reply with the number only.",
    "In one short sentence, what is AWS Inferentia?",
]
LAYOUT = [0, 1, 0, 2, 1, 3, 0, 3]
questions = [QS[LAYOUT[b]] for b in range(B)]

pads, ams, n0s = [], [], []
for q in questions:
    enc = tok.apply_chat_template([{"role": "user", "content": q}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    p = enc["input_ids"][0].tolist()
    pads.append(p + [0] * (BUCKET - len(p)))
    ams.append([1] * len(p) + [0] * (BUCKET - len(p)))
    n0s.append(len(p))


def zero_kv():
    kb = [torch.zeros(B, LINFO[i][0], MAX, LINFO[i][1], dtype=WDT) for i in NONSHARED]
    vb = [torch.zeros(B, LINFO[i][0], MAX, LINFO[i][1], dtype=WDT) for i in NONSHARED]
    return kb, vb


def run(pre_fn, dec_fn, maxnew=24):
    """Returns per-stream token lists AND per-step top-2 logit margins."""
    ie, ple = embed_ids(pads)
    am = torch.tensor(ams)
    with torch.no_grad():
        lg, ks, vs = pre_fn(ie, am, ple)
    seqs, margins = [], []
    nxt = []
    for b in range(B):
        row = lg[b, n0s[b] - 1].float()
        top2 = torch.topk(row, 2).values
        margins.append([float(top2[0] - top2[1])])
        nxt.append(int(row.argmax()))
    key_bufs, val_bufs = zero_kv()
    for j in range(len(NONSHARED)):
        key_bufs[j][:, :, :BUCKET, :] = ks[j][:, :, :BUCKET, :]
        val_bufs[j][:, :, :BUCKET, :] = vs[j][:, :, :BUCKET, :]
    seqs = [[nxt[b]] for b in range(B)]
    cur = list(n0s)
    for _ in range(maxnew - 1):
        ie1, ple1 = embed_ids([[seqs[b][-1]] for b in range(B)])
        position_ids, onehot, full_mask, slide_mask = host_pos_tensors(cur)
        with torch.no_grad():
            lg1, key_bufs, val_bufs = dec_fn(ie1, ple1, position_ids, onehot,
                                             full_mask, slide_mask, key_bufs, val_bufs)
        for b in range(B):
            row = lg1[b, 0].float()
            top2 = torch.topk(row, 2).values
            margins[b].append(float(top2[0] - top2[1]))
            seqs[b].append(int(row.argmax()))
            cur[b] += 1
    return seqs, margins


cpu_seqs, cpu_margins = run(pre, dec)
dev_seqs, dev_margins = run(pre_neff, dec_neff)

for b in range(B):
    if cpu_seqs[b] == dev_seqs[b]:
        print(f"slot {b} (Q{LAYOUT[b]}, n0={n0s[b]}): MATCH ({len(cpu_seqs[b])} tokens)", flush=True)
        continue
    div = next(i for i in range(len(cpu_seqs[b])) if cpu_seqs[b][i] != dev_seqs[b][i])
    print(f"slot {b} (Q{LAYOUT[b]}, n0={n0s[b]}): DIVERGES at token {div} "
          f"(cpu_margin={cpu_margins[b][div]:.4f}, dev_margin={dev_margins[b][div]:.4f})", flush=True)
    print(f"  cpu[{div}:]={cpu_seqs[b][div:div+6]} dev[{div}:]={dev_seqs[b][div:div+6]}", flush=True)
    print(f"  cpu text: {tok.decode(cpu_seqs[b], skip_special_tokens=True)!r}", flush=True)
    print(f"  dev text: {tok.decode(dev_seqs[b], skip_special_tokens=True)!r}", flush=True)
print("DIAG_DONE", flush=True)
