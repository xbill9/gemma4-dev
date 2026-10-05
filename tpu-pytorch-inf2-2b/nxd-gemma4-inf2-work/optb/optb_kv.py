"""Proper two-graph KV-cache decode for Gemma-4-E2B via torch_neuronx (Option B).
mode 'cpu'  : validate the prefill+static-buffer-decode logic in eager, no trace.
mode 'trace': compile prefill + decode graphs, run device greedy, compare to CPU.
"""
import os, sys, time, torch
torch.manual_seed(0)
MODE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
MP = "/workspace/real-gemma4-E2B-it"
MAX = int(os.environ.get("KV_MAX", "128"))       # max total sequence (buffer length)
BUCKET = int(os.environ.get("KV_BUCKET", "32"))  # prefill bucket (right-padded prompt length)
PRE_OUT = os.environ.get("KV_PRE_OUT", "/workspace/kv_pre_neff.pt")
DEC_OUT = os.environ.get("KV_DEC_OUT", "/workspace/kv_dec_neff.pt")
NEG = torch.finfo(torch.float32).min

from transformers import AutoTokenizer, Gemma4ForConditionalGeneration, DynamicCache
tok = AutoTokenizer.from_pretrained(MP)
m = Gemma4ForConditionalGeneration.from_pretrained(MP, torch_dtype=torch.float32, attn_implementation="eager"); m.eval()
lang = m.model.language_model; lm_head = m.lm_head
softcap = getattr(m.config.text_config, "final_logit_softcapping", None)
cfg = lang.config

class GeluTanh(torch.nn.Module):
    def forward(self, x): return 0.5*x*(1.0+torch.tanh(0.7978845608028654*(x+0.044715*x*x*x)))
for mod in lang.modules():
    if hasattr(mod, "act_fn"): mod.act_fn = GeluTanh()

# ---- discover non-shared layers + their (n_kv, head_dim) ----
NONSHARED = []
LINFO = {}
for i, lyr in enumerate(lang.layers[:cfg.num_hidden_layers]):
    a = lyr.self_attn
    if not a.is_kv_shared_layer:
        hd = a.head_dim
        nkv = a.k_proj.out_features // hd
        NONSHARED.append(i); LINFO[i] = (nkv, hd)
print("non-shared layers", NONSHARED, flush=True)
print("layer info (nkv,hd)", {i: LINFO[i] for i in NONSHARED}, flush=True)
SW = cfg.sliding_window

def softcap_logits(lg):
    return softcap*torch.tanh(lg/softcap) if softcap else lg

# ---------- PREFILL (Option B forward + return non-shared K/V) ----------
class PreWrap(torch.nn.Module):
    def __init__(s): super().__init__(); s.lang = lang; s.head = lm_head
    def forward(s, ie, am, ple):
        cache = DynamicCache()
        out = s.lang(inputs_embeds=ie, per_layer_inputs=ple, attention_mask=am,
                     use_cache=True, past_key_values=cache)
        lg = softcap_logits(s.head(out.last_hidden_state))
        ks = [cache.layers[i].keys for i in NONSHARED]
        vs = [cache.layers[i].values for i in NONSHARED]
        return (lg, ks, vs)
pre = PreWrap().eval()

# ---------- static KV cache with one-hot masked write ----------
class StaticKV:
    is_compileable = False
    def __init__(s, key_bufs, val_bufs, onehot):
        s.key = {i: key_bufs[j] for j, i in enumerate(NONSHARED)}
        s.val = {i: val_bufs[j] for j, i in enumerate(NONSHARED)}
        s.oh = onehot                              # [1,1,MAX,1] float, from host
    def update(s, k, v, idx, *a, **kw):            # k,v: [1,nkv,1,hd]
        s.key[idx] = s.key[idx]*(1.0-s.oh) + k*s.oh
        s.val[idx] = s.val[idx]*(1.0-s.oh) + v*s.oh
        return s.key[idx], s.val[idx]
    def get_seq_length(s, *a, **k): return 0        # unused (position_ids passed)
    def export(s):
        return [s.key[i] for i in NONSHARED], [s.val[i] for i in NONSHARED]

def host_pos_tensors(pos):
    """All position-dependent inputs, computed on host so the graph has no int-index ops."""
    ar = torch.arange(MAX)
    onehot = (ar == pos).view(1, 1, MAX, 1).to(torch.float32)
    valid = ar <= pos
    full = torch.where(valid, 0.0, NEG).view(1, 1, 1, MAX)
    slide = torch.where(valid & (ar > pos - SW), 0.0, NEG).view(1, 1, 1, MAX)
    position_ids = torch.tensor([[pos]], dtype=torch.long)
    return position_ids, onehot, full, slide

# ---------- DECODE (single token, static buffers as I/O) ----------
class DecWrap(torch.nn.Module):
    def __init__(s): super().__init__(); s.lang = lang; s.head = lm_head
    def forward(s, ie, ple, position_ids, onehot, full_mask, slide_mask, key_bufs, val_bufs):
        cache = StaticKV(key_bufs, val_bufs, onehot)
        masks = {"full_attention": full_mask, "sliding_attention": slide_mask}
        out = s.lang(inputs_embeds=ie, per_layer_inputs=ple,
                     position_ids=position_ids, attention_mask=masks,
                     use_cache=True, past_key_values=cache)
        lg = softcap_logits(s.head(out.last_hidden_state))
        ks, vs = cache.export()
        return (lg, ks, vs)
dec = DecWrap().eval()

# ---------- host helpers ----------
ec = m.generation_config.eos_token_id
EOS = set(ec) if isinstance(ec, (list, tuple)) else {ec}

def embed_ids(id_list):
    ids = torch.tensor([id_list])
    with torch.no_grad():
        ie = lang.embed_tokens(ids)
        ple = lang.get_per_layer_inputs(ids, ie)
    return ie, ple

msgs = [{"role": "user", "content": "What is the capital of France?"}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
prompt = enc["input_ids"][0].tolist(); n0 = len(prompt)
print("prompt tokens", n0, "| MAX", MAX, "| BUCKET", BUCKET, flush=True)
assert n0 <= BUCKET

def run_greedy(pre_fn, dec_fn, maxnew=30):
    # --- prefill (padded to BUCKET) ---
    pad = prompt + [0]*(BUCKET-n0)
    ie, ple = embed_ids(pad)
    am = torch.tensor([[1]*n0 + [0]*(BUCKET-n0)])
    with torch.no_grad():
        lg, ks, vs = pre_fn(ie, am, ple)
    first = int(lg[0, n0-1].argmax())
    # --- init MAX buffers, copy prompt K/V ---
    key_bufs = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1]) for i in NONSHARED]
    val_bufs = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1]) for i in NONSHARED]
    for j in range(len(NONSHARED)):
        key_bufs[j][:, :, :n0, :] = ks[j][:, :, :n0, :]
        val_bufs[j][:, :, :n0, :] = vs[j][:, :, :n0, :]
    # --- decode loop ---
    seq = [first]; cur = n0
    for _ in range(maxnew):
        if seq[-1] in EOS: break
        ie1, ple1 = embed_ids([seq[-1]])
        position_ids, onehot, full_mask, slide_mask = host_pos_tensors(cur)
        with torch.no_grad():
            lg1, key_bufs, val_bufs = dec_fn(ie1, ple1, position_ids, onehot,
                                             full_mask, slide_mask, key_bufs, val_bufs)
        nxt = int(lg1[0, 0].argmax())
        seq.append(nxt); cur += 1
    return seq

if MODE == "cpu":
    t = time.time()
    seq = run_greedy(pre, dec)
    txt = tok.decode([s for s in seq if s not in EOS], skip_special_tokens=True)
    print("CPU KV-DECODE GEN:", repr(txt), flush=True)
    print("ids", seq, "| secs", round(time.time()-t, 1), flush=True)
    print("CPU_DONE", flush=True)
    sys.exit(0)

# ---------- MODE trace ----------
print("=== tracing prefill + decode ===", flush=True)
import torch_neuronx
os.environ["NEURON_RT_VISIBLE_CORES"] = "0,1"
# example inputs
pad = prompt + [0]*(BUCKET-n0)
ie, ple = embed_ids(pad); am = torch.tensor([[1]*n0 + [0]*(BUCKET-n0)])
t = time.time()
pre_neff = torch_neuronx.trace(pre, (ie, am, ple), compiler_workdir="/workspace/kv_pre_wd",
    compiler_args=["--model-type", "transformer", "--auto-cast", "all", "--auto-cast-type", "bf16"])
torch.jit.save(pre_neff, PRE_OUT)
print("PREFILL_TRACE_DONE secs", round(time.time()-t, 1), flush=True)

key_bufs = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1]) for i in NONSHARED]
val_bufs = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1]) for i in NONSHARED]
ie1, ple1 = embed_ids([prompt[-1]])
position_ids, onehot, full_mask, slide_mask = host_pos_tensors(n0)
t = time.time()
dec_neff = torch_neuronx.trace(dec, (ie1, ple1, position_ids, onehot, full_mask, slide_mask, key_bufs, val_bufs),
    compiler_workdir="/workspace/kv_dec_wd",
    compiler_args=["--model-type", "transformer", "--auto-cast", "all", "--auto-cast-type", "bf16"])
torch.jit.save(dec_neff, DEC_OUT)
print("DECODE_TRACE_DONE secs", round(time.time()-t, 1), flush=True)

print("=== CPU greedy ===", flush=True)
cpu_seq = run_greedy(pre, dec)
print("=== DEVICE greedy ===", flush=True)
t = time.time(); dev_seq = run_greedy(pre_neff, dec_neff)
print("dev secs", round(time.time()-t, 1), flush=True)
print("CPU GEN:", repr(tok.decode([s for s in cpu_seq if s not in EOS], skip_special_tokens=True)), flush=True)
print("DEV GEN:", repr(tok.decode([s for s in dev_seq if s not in EOS], skip_special_tokens=True)), flush=True)
print("SEQ_MATCH", cpu_seq == dev_seq, flush=True)
print("CPU_IDS", cpu_seq, flush=True)
print("DEV_IDS", dev_seq, flush=True)
print("ALL_DONE", flush=True)
