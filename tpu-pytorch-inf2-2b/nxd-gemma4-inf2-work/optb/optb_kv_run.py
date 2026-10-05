"""Run the saved two-graph KV-cache decode on device, one neff resident at a time
(prefill runs once, is freed, then decode loops) so a single 16GB NeuronCore suffices.
Reuses /workspace/kv_pre_neff.pt + kv_dec_neff.pt (no recompile)."""
import os
os.environ["NEURON_RT_VISIBLE_CORES"] = "0,1"   # prefill->core0, decode->core1 (each ~4.5GB fits a 16GB core)
import sys, time, gc, torch
torch.manual_seed(0)
MP = "/workspace/real-gemma4-E2B-it"; MAX = 128; BUCKET = 32
NEG = torch.finfo(torch.float32).min
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France?"

from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
import torch_neuronx
tok = AutoTokenizer.from_pretrained(MP)
m = Gemma4ForConditionalGeneration.from_pretrained(MP, torch_dtype=torch.float32, attn_implementation="eager"); m.eval()
lang = m.model.language_model; cfg = lang.config; SW = cfg.sliding_window

NONSHARED, LINFO = [], {}
for i, lyr in enumerate(lang.layers[:cfg.num_hidden_layers]):
    a = lyr.self_attn
    if not a.is_kv_shared_layer:
        hd = a.head_dim; NONSHARED.append(i); LINFO[i] = (a.k_proj.out_features // hd, hd)

ec = m.generation_config.eos_token_id
EOS = set(ec) if isinstance(ec, (list, tuple)) else {ec}

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

msgs = [{"role": "user", "content": PROMPT}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
prompt = enc["input_ids"][0].tolist(); n0 = len(prompt)
print("PROMPT", repr(PROMPT), "| prompt tokens", n0, flush=True)
assert n0 <= BUCKET

# ---- prefill (load, run once, free) ----
t = time.time()
pre = torch.jit.load("/workspace/kv_pre_neff.pt")
pad = prompt + [0]*(BUCKET-n0)
ie, ple = embed_ids(pad); am = torch.tensor([[1]*n0 + [0]*(BUCKET-n0)])
lg, ks, vs = pre(ie, am, ple)
first = int(lg[0, n0-1].argmax())
key_bufs = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1]) for i in NONSHARED]
val_bufs = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1]) for i in NONSHARED]
for j in range(len(NONSHARED)):
    key_bufs[j][:, :, :n0, :] = ks[j][:, :, :n0, :]
    val_bufs[j][:, :, :n0, :] = vs[j][:, :, :n0, :]
print("prefill done, first tok", first, repr(tok.decode([first])), "| secs", round(time.time()-t, 1), flush=True)

# ---- decode (loads onto core 1; prefill stays resident on core 0) ----
tl = time.time()
dec = torch.jit.load("/workspace/kv_dec_neff.pt")
print("decode neff load secs", round(time.time()-tl, 1), flush=True)
seq = [first]; cur = n0
tloop = time.time(); steps = 0
for _ in range(MAX - n0):
    if seq[-1] in EOS: break
    ie1, ple1 = embed_ids([seq[-1]])
    position_ids, onehot, full_mask, slide_mask = host_pos_tensors(cur)
    lg1, key_bufs, val_bufs = dec(ie1, ple1, position_ids, onehot, full_mask, slide_mask, key_bufs, val_bufs)
    seq.append(int(lg1[0, 0].argmax())); cur += 1; steps += 1
loop_s = time.time() - tloop
txt = tok.decode([s for s in seq if s not in EOS], skip_special_tokens=True)
print("DEVICE GEN:", repr(txt), flush=True)
print("ids", seq, flush=True)
print("decode loop", round(loop_s, 2), "s /", steps, "steps =", round(loop_s/max(steps,1)*1000), "ms/tok =",
      round(steps/loop_s, 1), "tok/s (after one-time neff load)", flush=True)
print("ALL_DONE", flush=True)
