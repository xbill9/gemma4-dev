"""Gemma 4 E2B QAT W4A16 with long context on a 4 GiB Turing card.

    python3 e2b_long.py OUT.jsonl [DEPTHS]     # DEPTHS: comma-separated context lengths, default 0,4096,...,131000

Every weight and vocabulary table int4 on the GPU (decoder linears from -text, tables from -text-emb4), merged
gate/up and q/k/v projections, int4 GEMV kernel v5. The KV cache lives here, not in transformers: the 3 full-attention
layers that own K/V get MAX_CTX slots (slot = position), the 12 sliding layers a 512-slot ring that also records each
slot's position. Prompts are read in CHUNK-token pieces; decode is one compiled step replayed as a CUDA graph.

For each depth: fill the context with a document of numbered records plus a question about one record placed a
quarter of the way in, read it, decode DECODE_TOKENS greedily (past EOS, for timing), and check the recalled code.
"""

import json
import os
import pathlib
import random
import sys
import time
import types

import torch
import transformers
from accelerate.hooks import remove_hook_from_module
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.gemma4 import modeling_gemma4 as g4

from w4attn_long import decode_attention, prefill_attention
from w4gemv_v5 import w4_gemv

TEXT = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
EMB4 = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4"
OUT = pathlib.Path(sys.argv[1])
DEPTHS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,4096,16384,32768,65536,131000").split(",")]
DTYPE = torch.float16
MAX_CTX = int(os.environ.get("MAX_CTX", "131072"))
CHUNK = int(os.environ.get("CHUNK", "512"))
DECODE_TOKENS = int(os.environ.get("DECODE_TOKENS", "64"))
SHIFTS = torch.arange(0, 32, 4, device="cuda", dtype=torch.int32)


def out(**kw):
    line = json.dumps(kw)
    print(line, flush=True)
    with OUT.open("a") as f:
        f.write(line + "\n")


def tensor(repo, key, patterns):
    d = pathlib.Path(snapshot_download(repo, allow_patterns=patterns))
    for p in sorted(d.glob("*.safetensors")):
        with safe_open(str(p), "pt") as f:
            if key in f.keys():
                return f.get_tensor(key)
    raise KeyError(key)


# --- int4 linears and tables -----------------------------------------------------------------------------------------
def _dequant(packed, scale):
    o, g = scale.shape
    q = ((packed.unsqueeze(-1) >> SHIFTS) & 15).view(o, g, 32).to(DTYPE) - 8
    return (q * scale.unsqueeze(-1).to(DTYPE)).view(o, g * 32)


torch._dynamo.config.cache_size_limit = 64
_dequant = torch.compile(_dequant, dynamic=False)


def w4a16(x, packed, scale):
    if x.numel() == x.shape[-1]:
        return w4_gemv(x, packed, scale)
    return torch.nn.functional.linear(x, _dequant(packed, scale))


class PackedLinear(torch.nn.Module):
    def __init__(self, packed, scale):
        super().__init__()
        self.packed, self.scale = torch.nn.Buffer(packed), torch.nn.Buffer(scale)

    def forward(self, x):
        return w4a16(x, self.packed, self.scale)


class PackedEmbedding(torch.nn.Module):
    """Gemma4TextScaledWordEmbedding over int4 rows: gather packed rows, unpack, multiply by the embed scale."""

    def __init__(self, packed, scale, embed_scale):
        super().__init__()
        self.packed, self.scale = torch.nn.Buffer(packed), torch.nn.Buffer(scale)
        self.embed_scale = torch.nn.Buffer(torch.tensor(embed_scale, dtype=DTYPE))

    def forward(self, ids):
        p, s = self.packed[ids], self.scale[ids]
        q = ((p.unsqueeze(-1) >> SHIFTS) & 15).to(DTYPE) - 8
        w = q.view(*ids.shape, s.shape[-1], 32) * s.unsqueeze(-1).to(DTYPE)
        return w.view(*ids.shape, -1) * self.embed_scale


with torch.device("meta"):
    meta = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(TEXT))
device_map = {f"model.{n}": 0 for n, _ in meta.model.named_children()}
device_map["model.embed_tokens_per_layer"] = "cpu"
device_map["lm_head"] = 0
del meta
model = AutoModelForCausalLM.from_pretrained(
    TEXT,
    dtype=DTYPE,
    key_mapping={r"^model\.language_model\.": "model."},
    device_map=device_map,
    quantization_config=transformers.CompressedTensorsConfig(run_compressed=True),
).eval()
model.ct_decompress_hook.remove()
tm = model.model
pats, pre = ["model-rest-00[01].safetensors"], "model.language_model."


def packed(name):
    return tensor(EMB4, name + ".weight_packed", pats).cuda(), tensor(EMB4, name + ".weight_scale", pats).cuda()


ple_scale, emb_scale = tm.embed_tokens_per_layer.scalar_embed_scale, tm.embed_tokens.scalar_embed_scale
tm.embed_tokens.weight.data = torch.empty(0, dtype=DTYPE, device=0)  # fp16 table (tied lm_head): free its storage
tm.embed_tokens_per_layer = PackedEmbedding(*packed(pre + "embed_tokens_per_layer"), ple_scale)
tm.embed_tokens = PackedEmbedding(*packed(pre + "embed_tokens"), emb_scale)
model.lm_head = PackedLinear(*packed("lm_head"))
model.config.tie_word_embeddings = False
remove_hook_from_module(model, recurse=True)  # everything is on the GPU; the hooks would only break torch.compile
model.to(DTYPE)
for m in model.modules():
    if hasattr(m, "weight_packed"):
        assert m.bias is None
        m.forward = (lambda mod: lambda x: w4a16(x, mod.weight_packed, mod.weight_scale))(m)


def _merge(mods):
    """Concatenate packed int4 linears along the output dim and free the originals' storage."""
    pk = torch.cat([m.weight_packed for m in mods])
    sc = torch.cat([m.weight_scale for m in mods])
    sizes = [m.weight_packed.shape[0] for m in mods]
    for m in mods:
        m.weight_packed.data = torch.empty(0, dtype=torch.int32, device=0)
        m.weight_scale.data = torch.empty(0, dtype=DTYPE, device=0)
    return pk, sc, sizes


for mod in list(model.modules()):
    if isinstance(mod, g4.Gemma4TextMLP):
        pk, sc, sizes = _merge([mod.gate_proj, mod.up_proj])

        def mlp_forward(x, mod=mod, pk=pk, sc=sc, sizes=sizes):
            g, u = w4a16(x, pk, sc).split(sizes, dim=-1)
            return mod.down_proj(mod.act_fn(g) * u)

        mod.forward = mlp_forward
    elif isinstance(mod, g4.Gemma4TextAttention) and not mod.is_kv_shared_layer:
        pk, sc, sizes = _merge([mod.q_proj, mod.k_proj] + ([mod.v_proj] if mod.v_proj is not None else []))
        stash = []  # q_proj runs first and leaves k (and v) for the calls that follow

        def q_forward(x, pk=pk, sc=sc, sizes=sizes, stash=stash):
            parts = w4a16(x, pk, sc).split(sizes, dim=-1)
            stash[:] = parts[1:]
            return parts[0]

        mod.q_proj.forward = q_forward
        mod.k_proj.forward = lambda x, stash=stash: stash[0]
        if mod.v_proj is not None:
            mod.v_proj.forward = lambda x, stash=stash: stash[1]
torch.cuda.empty_cache()
weights_mib = torch.cuda.memory_allocated() // 2**20

# --- KV cache and attention ---------------------------------------------------------------------------------------
cfg = model.config
WINDOW = cfg.sliding_window
S = types.SimpleNamespace(decode=False, p0=0, pos=torch.zeros(1, dtype=torch.long, device=0))


class KV:
    """K/V for one layer that owns them (KV-shared layers read the owner's). One KV head, so rows are [slots, D]."""

    def __init__(self, sliding, D):
        self.sliding = sliding
        self.cap = WINDOW if sliding else MAX_CTX
        self.k = torch.zeros(self.cap, D, dtype=DTYPE, device=0)
        self.v = torch.zeros(self.cap, D, dtype=DTYPE, device=0)
        self.kpos = torch.full((self.cap,), -1, dtype=torch.int32, device=0) if sliding else None
        self.pk = self.pv = None

    def reset(self):
        if self.sliding:
            self.kpos.fill_(-1)

    def write(self, k, v):
        """k, v [n, D] for positions S.pos (decode) or S.p0.. (prompt chunk)."""
        if S.decode:
            slot = S.pos % self.cap if self.sliding else S.pos
            self.k.index_copy_(0, slot, k)
            self.v.index_copy_(0, slot, v)
            if self.sliding:
                self.kpos.index_copy_(0, slot, S.pos.to(torch.int32))
        elif self.sliding:
            self.pk, self.pv = k, v  # the ring keeps older positions the shared layers still need; flushed after
        else:
            self.k[S.p0 : S.p0 + k.shape[0]] = k
            self.v[S.p0 : S.p0 + v.shape[0]] = v

    def flush(self):
        if self.pk is None:
            return
        n = self.pk.shape[0]
        assert n <= self.cap
        positions = S.p0 + torch.arange(n, device=0)
        slots = positions % self.cap
        self.k.index_copy_(0, slots, self.pk)
        self.v.index_copy_(0, slots, self.pv)
        self.kpos.index_copy_(0, slots, positions.to(torch.int32))
        self.pk = self.pv = None

    def attend(self, q, scale):
        if S.decode:
            return decode_attention(q, self.k, self.v, self.kpos, S.pos, self.sliding, WINDOW, scale)
        n = q.shape[1]
        if self.sliding:
            keys, vals = torch.cat([self.k, self.pk]), torch.cat([self.v, self.pv])
            kpos = torch.cat([self.kpos, (S.p0 + torch.arange(n, device=0)).to(torch.int32)])
            return prefill_attention(q, keys, vals, kpos, S.p0, True, WINDOW, scale)
        L = S.p0 + n
        return prefill_attention(q, self.k[:L], self.v[:L], None, S.p0, False, None, scale)


kvs = []


def attn_forward(
    self, hidden_states, position_embeddings, attention_mask, shared_kv_states, past_key_values=None, **kwargs
):
    b, n, _ = hidden_states.shape
    cos, sin = position_embeddings
    q = self.q_proj(hidden_states).view(b, n, -1, self.head_dim)
    q = g4.apply_rotary_pos_emb(self.q_norm(q), cos, sin, unsqueeze_dim=2)
    if self.is_kv_shared_layer:
        kv = shared_kv_states[self.layer_type]
    else:
        k = self.k_proj(hidden_states).view(b, n, -1, self.head_dim)
        v = self.v_proj(hidden_states).view(b, n, -1, self.head_dim) if self.v_proj is not None else k
        k = g4.apply_rotary_pos_emb(self.k_norm(k), cos, sin, unsqueeze_dim=2)
        v = self.v_norm(v)
        kv = self._kv
        kv.write(k[0, :, 0], v[0, :, 0])
        if self.store_full_length_kv:
            shared_kv_states[self.layer_type] = kv
    o = kv.attend(q, self.scaling)
    return self.o_proj(o.reshape(b, n, -1)), None


for mod in model.modules():
    if isinstance(mod, g4.Gemma4TextAttention):
        assert cfg.per_layer_config[mod.layer_idx].num_key_value_heads == 1
        if not mod.is_kv_shared_layer:
            mod._kv = KV(mod.is_sliding, mod.head_dim)
            kvs.append(mod._kv)
        mod.forward = types.MethodType(attn_forward, mod)
kv_mib = sum(t.numel() * t.element_size() for kv in kvs for t in (kv.k, kv.v)) // 2**20
out(
    step="load",
    weights_mib=weights_mib,
    kv_mib=kv_mib,
    max_ctx=MAX_CTX,
    chunk=CHUNK,
    kv_layers={"full": sum(not kv.sliding for kv in kvs), "sliding": sum(kv.sliding for kv in kvs)},
    cuda_alloc_mib=torch.cuda.memory_allocated() // 2**20,
    torch=torch.__version__,
    transformers=transformers.__version__,
)


# --- prompt read (chunked) and decode (graph) -------------------------------------------------------------------------
def read_prompt(ids):
    """ids [1, N] at positions 0..N-1; returns the next token's logits."""
    for kv in kvs:
        kv.reset()
    S.decode = False
    N = ids.shape[1]
    for p0 in range(0, N, CHUNK):
        chunk = ids[:, p0 : p0 + CHUNK]
        S.p0 = p0
        pos = torch.arange(p0, p0 + chunk.shape[1], device=0).unsqueeze(0)
        logits = model(input_ids=chunk, position_ids=pos, use_cache=False, logits_to_keep=1).logits[:, -1]
        for kv in kvs:
            kv.flush()
    return logits


tok_buf = torch.zeros(1, 1, dtype=torch.long, device=0)


def step():
    return (
        model(input_ids=tok_buf, position_ids=S.pos.view(1, 1), use_cache=False, logits_to_keep=1)
        .logits[:, -1]
        .argmax(-1)
    )


tok = AutoTokenizer.from_pretrained(TEXT)
with torch.inference_mode():
    warm = tok.apply_chat_template(
        [{"role": "user", "content": "Hi"}], add_generation_prompt=True, return_tensors="pt", return_dict=True
    )["input_ids"].to(0)
    first = int(read_prompt(warm).argmax(-1))
    S.decode = True
    tok_buf.fill_(first)
    S.pos.fill_(warm.shape[1])
    step_c = torch.compile(step, mode="default", dynamic=False)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step_c()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.empty_cache()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        next_buf = step_c()
out(
    step="captured",
    cuda_alloc_mib=torch.cuda.memory_allocated() // 2**20,
    cuda_reserved_mib=torch.cuda.memory_reserved() // 2**20,
)


def generate(ids, n_new):
    """Greedy; returns (tokens, prompt seconds, decode seconds). Decodes n_new tokens whatever the EOS."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    toks = [int(read_prompt(ids).argmax(-1))]
    t1 = time.perf_counter()
    S.decode = True
    n = ids.shape[1]
    while len(toks) < n_new:
        tok_buf.fill_(toks[-1])
        S.pos.fill_(n + len(toks) - 1)
        graph.replay()
        toks.append(int(next_buf))
    torch.cuda.synchronize()
    return toks, t1 - t0, time.perf_counter() - t1


# --- the long-context benchmark ---------------------------------------------------------------------------------------
rng = random.Random(1234)
QUESTION = "\n\nQuestion: what is the code for record {r:06d}? Answer with the code only."
eos = model.generation_config.eos_token_id
eos = set(eos if isinstance(eos, list) else [eos])


def build(depth):
    """A chat prompt of about `depth` tokens: numbered records, and a question about one a quarter of the way in."""
    head = tok.apply_chat_template([{"role": "user", "content": "x"}], add_generation_prompt=True, tokenize=False)
    overhead = len(tok(head + QUESTION.format(r=0), add_special_tokens=False)["input_ids"])
    records, used = [], overhead
    codes = {}
    i = 0
    while True:
        i += 1
        code = rng.randrange(100000, 999999)
        line = f"Record {i:06d}: the code for record {i:06d} is {code}.\n"
        cost = len(tok(line, add_special_tokens=False)["input_ids"])
        if used + cost > depth and records:
            break
        records.append(line)
        codes[i] = code
        used += cost
        if depth == 0:
            break
    target = max(1, len(records) // 4)
    content = "".join(records) + QUESTION.format(r=target)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, return_tensors="pt", return_dict=True
    )["input_ids"].to(0)
    return ids, codes[target]


with torch.inference_mode():
    for depth in DEPTHS:
        ids, code = build(depth)
        if ids.shape[1] + DECODE_TOKENS > MAX_CTX:
            out(step="depth", depth=depth, skipped="prompt plus decode exceeds MAX_CTX", prompt_tokens=ids.shape[1])
            continue
        torch.cuda.reset_peak_memory_stats()
        toks, t_prompt, t_dec = generate(ids, DECODE_TOKENS)
        cut = next((i for i, t in enumerate(toks) if t in eos), len(toks))
        answer = tok.decode(toks[:cut], skip_special_tokens=True).strip()
        out(
            step="depth",
            depth=depth,
            prompt_tokens=ids.shape[1],
            prompt_s=round(t_prompt, 2),
            prompt_tok_per_s=round(ids.shape[1] / t_prompt, 1),
            decode_tokens=len(toks) - 1,
            decode_tok_per_s=round((len(toks) - 1) / t_dec, 2),
            peak_cuda_mib=torch.cuda.max_memory_allocated() // 2**20,
            recall_code=code,
            answer=answer[:80],
            recalled=str(code) in answer,
        )
