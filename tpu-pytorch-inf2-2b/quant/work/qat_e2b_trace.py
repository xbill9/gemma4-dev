"""Gemma-4 E2B QAT (q4_0-unquantized) on a single NeuronCore via torch_neuronx.

Adapted from aws-neuron-samples-pr/gemma4_e2b sample: same Option-B recipe
(two graphs, KV as graph I/O, host-side PLE, eager attn + tanh GELU + softcap),
pointed at the ungated QAT checkpoint. Compiles, validates device==CPU greedy
(SEQ_MATCH), saves the neffs, and benchmarks decode throughput.
"""
import time

import torch

torch.manual_seed(0)

MODEL_ID = "google/gemma-4-E2B-it-qat-q4_0-unquantized"
MAX = 128
BUCKET = 32
NEG = torch.finfo(torch.float32).min
PRE_OUT = "/workspace/qat_e2b_prefill.pt"
DEC_OUT = "/workspace/qat_e2b_decode.pt"

from transformers import AutoTokenizer, DynamicCache, Gemma4ForConditionalGeneration

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
pre_neff = torch_neuronx.trace(pre, (ie, am, ple), compiler_args=CARGS)
print(f"PREFILL_COMPILED {time.time()-t:.0f}s", flush=True)

key_bufs = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1], dtype=WDT) for i in NONSHARED]
val_bufs = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1], dtype=WDT) for i in NONSHARED]
ie1, ple1 = embed_ids([prompt[-1]])
position_ids, onehot, full_mask, slide_mask = host_pos_tensors(len(prompt))
t = time.time()
dec_neff = torch_neuronx.trace(
    dec, (ie1, ple1, position_ids, onehot, full_mask, slide_mask, key_bufs, val_bufs),
    compiler_args=CARGS)
print(f"DECODE_COMPILED {time.time()-t:.0f}s", flush=True)

torch.jit.save(pre_neff, PRE_OUT)
torch.jit.save(dec_neff, DEC_OUT)
print("neffs saved:", PRE_OUT, DEC_OUT, flush=True)

cpu_seq = run_greedy(pre, dec, prompt)
dev_seq = run_greedy(pre_neff, dec_neff, prompt)
print("CPU:   ", repr(tok.decode([s for s in cpu_seq if s not in EOS], skip_special_tokens=True)), flush=True)
print("Device:", repr(tok.decode([s for s in dev_seq if s not in EOS], skip_special_tokens=True)), flush=True)
print("SEQ_MATCH:", cpu_seq == dev_seq, flush=True)
assert cpu_seq == dev_seq, "device output diverged from CPU reference"


def bench(question, maxnew=80):
    e = tok.apply_chat_template([{"role": "user", "content": question}],
                                add_generation_prompt=True, return_tensors="pt", return_dict=True)
    p = e["input_ids"][0].tolist()
    t0 = time.time()
    seq = run_greedy(pre_neff, dec_neff, p, maxnew=maxnew)
    dt = time.time() - t0
    print(tok.decode([s for s in seq if s not in EOS], skip_special_tokens=True), flush=True)
    print(f"BENCH {len(seq)} tokens in {dt:.1f}s = {len(seq)/dt:.1f} tok/s", flush=True)


bench("In one sentence, what is AWS Inferentia?")
print("QAT_E2B_DONE", flush=True)
