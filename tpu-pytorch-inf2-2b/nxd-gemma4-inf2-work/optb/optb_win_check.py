"""Isolate masking-correctness from bf16 precision: run the EAGER fp32 static-KV decode
(same host sliding-window masks as the neff) and compare to HF generate() past pos 512.
If fp32 static-KV == HF, the window masking is correct and any device divergence is bf16."""
import os
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "1")
import torch
torch.manual_seed(0)
MP = "/workspace/real-gemma4-E2B-it"
MAX = int(os.environ.get("KV_MAX", "2048"))
BUCKET = int(os.environ.get("KV_BUCKET", "512"))
NGEN = int(os.environ.get("NGEN", "45"))
NEG = torch.finfo(torch.float32).min

from transformers import AutoTokenizer, Gemma4ForConditionalGeneration, DynamicCache
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

def softcap_logits(lg): return softcap*torch.tanh(lg/softcap) if softcap else lg
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

class StaticKV:
    is_compileable = False
    def __init__(s, kb, vb, oh):
        s.key = {i: kb[j] for j, i in enumerate(NONSHARED)}
        s.val = {i: vb[j] for j, i in enumerate(NONSHARED)}
        s.oh = oh
    def update(s, k, v, idx, *a, **kw):
        s.key[idx] = s.key[idx]*(1.0-s.oh) + k*s.oh
        s.val[idx] = s.val[idx]*(1.0-s.oh) + v*s.oh
        return s.key[idx], s.val[idx]
    def get_seq_length(s, *a, **k): return 0

def pre_forward(ie, am, ple):
    cache = DynamicCache()
    out = lang(inputs_embeds=ie, per_layer_inputs=ple, attention_mask=am, use_cache=True, past_key_values=cache)
    lg = softcap_logits(lm_head(out.last_hidden_state))
    return lg, [cache.layers[i].keys for i in NONSHARED], [cache.layers[i].values for i in NONSHARED]

def dec_forward(ie, ple, pi, oh, fm, sm, kb, vb):
    cache = StaticKV(kb, vb, oh)
    out = lang(inputs_embeds=ie, per_layer_inputs=ple, position_ids=pi,
               attention_mask={"full_attention": fm, "sliding_attention": sm},
               use_cache=True, past_key_values=cache)
    lg = softcap_logits(lm_head(out.last_hidden_state))
    return lg, [cache.key[i] for i in NONSHARED], [cache.val[i] for i in NONSHARED]

# long prompt
content = "Summarize the following text in detail.\n\n"
filler = "Neural networks process information through layers of interconnected nodes that transform inputs. "
while True:
    enc = tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = enc["input_ids"][0].tolist()
    if len(ids) >= BUCKET - 30: break
    content += filler
prompt = ids; n0 = len(prompt)
print("prompt", n0, "NGEN", NGEN, "reach pos", n0+NGEN-1, "window from", SW, flush=True)

# fp32 eager static-KV greedy
pad = prompt + [0]*(BUCKET-n0)
ie, ple = embed_ids(pad); am = torch.tensor([[1]*n0 + [0]*(BUCKET-n0)])
with torch.no_grad():
    lg, ks, vs = pre_forward(ie, am, ple)
first = int(lg[0, n0-1].argmax())
kb = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1]) for i in NONSHARED]
vb = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1]) for i in NONSHARED]
for j in range(len(NONSHARED)):
    kb[j][:, :, :n0, :] = ks[j][:, :, :n0, :]; vb[j][:, :, :n0, :] = vs[j][:, :, :n0, :]
cpu = [first]; cur = n0
for _ in range(NGEN-1):
    ie1, ple1 = embed_ids([cpu[-1]])
    pi, oh, fm, sm = host_pos_tensors(cur)
    with torch.no_grad():
        lg1, kb, vb = dec_forward(ie1, ple1, pi, oh, fm, sm, kb, vb)
    cpu.append(int(lg1[0, 0].argmax())); cur += 1

# HF reference
with torch.no_grad():
    out = m.generate(input_ids=torch.tensor([prompt]), attention_mask=torch.ones(1, n0, dtype=torch.long),
                     max_new_tokens=NGEN, min_new_tokens=NGEN, do_sample=False, num_beams=1)
ref = out[0][n0:].tolist()

L = min(len(cpu), len(ref)); fd = -1
for k in range(L):
    if cpu[k] != ref[k]: fd = k; break
print("fp32_staticKV vs HF: MATCH", (fd == -1 and len(cpu) == len(ref)),
      "| first_diff_idx", fd, "(abs pos", (n0+fd) if fd >= 0 else -1, ")", flush=True)
print("past_window_compared", sum(1 for k in range(L) if (n0+k) >= SW), flush=True)
print("CHECK_DONE", flush=True)
