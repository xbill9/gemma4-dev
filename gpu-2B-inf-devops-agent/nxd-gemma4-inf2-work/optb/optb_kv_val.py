"""Validate sliding-window masking at total position >512 for the 2048/512 build.
Device static-KV greedy decode vs HF reference generate() on a LONG prompt, so decode
crosses the SW=512 window boundary (never exercised at the old MAX=512, where SW==buffer).
Pass criterion: device generated ids == HF reference ids through positions >512."""
import os
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "1")   # container keeps core 0
import sys, time, torch
torch.manual_seed(0)
MP = "/workspace/real-gemma4-E2B-it"
MAX = int(os.environ.get("KV_MAX", "2048"))
BUCKET = int(os.environ.get("KV_BUCKET", "512"))
PRE = os.environ.get("KV_PRE_OUT", "/workspace/kv_pre_2048.pt")
DEC = os.environ.get("KV_DEC_OUT", "/workspace/kv_dec_2048.pt")
NGEN = int(os.environ.get("NGEN", "120"))
NEG = torch.finfo(torch.float32).min

from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
import torch_neuronx
tok = AutoTokenizer.from_pretrained(MP)
m = Gemma4ForConditionalGeneration.from_pretrained(MP, torch_dtype=torch.float32, attn_implementation="eager"); m.eval()
lang = m.model.language_model; lm_head = m.lm_head; cfg = lang.config; SW = cfg.sliding_window
softcap = getattr(m.config.text_config, "final_logit_softcapping", None)

class GeluTanh(torch.nn.Module):
    def forward(self, x): return 0.5*x*(1.0+torch.tanh(0.7978845608028654*(x+0.044715*x*x*x)))
for mod in lang.modules():
    if hasattr(mod, "act_fn"): mod.act_fn = GeluTanh()

NONSHARED, LINFO = [], {}
for i, lyr in enumerate(lang.layers[:cfg.num_hidden_layers]):
    a = lyr.self_attn
    if not a.is_kv_shared_layer:
        hd = a.head_dim; NONSHARED.append(i); LINFO[i] = (a.k_proj.out_features // hd, hd)

def embed_ids(id_list):
    ids = torch.tensor([id_list])
    with torch.no_grad():
        ie = lang.embed_tokens(ids); ple = lang.get_per_layer_inputs(ids, ie)
    return ie, ple

def host_pos_tensors(pos):
    ar = torch.arange(MAX)
    onehot = (ar == pos).view(1, 1, MAX, 1).to(torch.float32)
    valid = ar <= pos
    full = torch.where(valid, 0.0, NEG).view(1, 1, 1, MAX)
    slide = torch.where(valid & (ar > pos - SW), 0.0, NEG).view(1, 1, 1, MAX)
    return torch.tensor([[pos]], dtype=torch.long), onehot, full, slide

# ---- build a long prompt (~BUCKET-30 tokens) so decode crosses position 512 ----
content = "Summarize the following text in detail.\n\n"
filler = "Neural networks process information through layers of interconnected nodes that transform inputs. "
while True:
    msgs = [{"role": "user", "content": content}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = enc["input_ids"][0].tolist()
    if len(ids) >= BUCKET - 30: break
    content += filler
prompt = ids; n0 = len(prompt)
print("prompt tokens", n0, "| MAX", MAX, "BUCKET", BUCKET, "SW", SW, "NGEN", NGEN,
      "| will reach abs pos", n0 + NGEN - 1, "(window active from pos", SW, ")", flush=True)
assert n0 <= BUCKET

# ---- device: prefill (core-agnostic load) + static-KV decode ----
t = time.time()
pre = torch.jit.load(PRE)
pad = prompt + [0]*(BUCKET-n0)
ie, ple = embed_ids(pad); am = torch.tensor([[1]*n0 + [0]*(BUCKET-n0)])
lg, ks, vs = pre(ie, am, ple)
first = int(lg[0, n0-1].argmax())
key_bufs = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1]) for i in NONSHARED]
val_bufs = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1]) for i in NONSHARED]
for j in range(len(NONSHARED)):
    key_bufs[j][:, :, :n0, :] = ks[j][:, :, :n0, :]
    val_bufs[j][:, :, :n0, :] = vs[j][:, :, :n0, :]
print("prefill done, secs", round(time.time()-t, 1), flush=True)
dec = torch.jit.load(DEC)
dev = [first]; cur = n0
tl = time.time()
for _ in range(NGEN-1):
    ie1, ple1 = embed_ids([dev[-1]])
    pi, oh, fm, sm = host_pos_tensors(cur)
    lg1, key_bufs, val_bufs = dec(ie1, ple1, pi, oh, fm, sm, key_bufs, val_bufs)
    dev.append(int(lg1[0, 0].argmax())); cur += 1
loop_s = time.time() - tl
print("decode", NGEN-1, "steps", round(loop_s, 2), "s =", round((NGEN-1)/loop_s, 1), "tok/s (post neff-load)", flush=True)

# ---- reference: HF generate greedy, EOS disabled + min_new forced so it crosses 512 ----
with torch.no_grad():
    out = m.generate(input_ids=torch.tensor([prompt]), attention_mask=torch.ones(1, n0, dtype=torch.long),
                     max_new_tokens=NGEN, min_new_tokens=NGEN, do_sample=False, num_beams=1)
ref_gen = out[0][n0:].tolist()

# ---- compare ----
L = min(len(dev), len(ref_gen))
firstdiff = -1
for k in range(L):
    if dev[k] != ref_gen[k]: firstdiff = k; break
match = firstdiff == -1 and len(dev) == len(ref_gen)
# count how many compared positions are strictly past the window boundary
past512 = sum(1 for k in range(L) if (n0 + k) >= SW)
print("device gen", len(dev), "ref gen", len(ref_gen), "| compared", L, flush=True)
print("positions_compared_past_window", past512, flush=True)
print("SEQ_MATCH", match, "| first_diff_idx", firstdiff,
      "(abs pos", (n0 + firstdiff) if firstdiff >= 0 else -1, ")", flush=True)
print("DEVICE TXT:", repr(tok.decode(dev, skip_special_tokens=True))[:300], flush=True)
print("REF    TXT:", repr(tok.decode(ref_gen, skip_special_tokens=True))[:300], flush=True)
print("VAL_DONE", flush=True)
