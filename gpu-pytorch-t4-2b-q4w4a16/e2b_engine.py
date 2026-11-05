"""Gemma 4 E2B QAT int4 in plain PyTorch on a 4 GB Turing GPU: load, read prompts, decode greedily.

    from e2b_engine import Engine
    eng = Engine()                      # ~2.5 GiB of int4 weights + a K/V cache for max_ctx tokens
    ids = eng.render([{"role": "user", "content": "Hi"}])
    out = eng.generate(ids, max_new=128)

Decoder linears are `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text` (Google's QAT grid, int4 group 32); `embed_tokens`,
`lm_head` (one shared copy) and the per-layer table come from `-text-emb4`. Everything sits on the GPU in int4.

- Decode: one step compiled with Inductor and replayed as a CUDA graph; int4 GEMV kernel `w4gemv.py`; split decode
  attention `w4attn.py`. fp16 compute (Inductor skips bf16 on sm_75).
- Prompt reads: CHUNK-token pieces, weights unpacked to fp32 for the multiply (fp16 GEMMs run at about a quarter of
  the fp32 rate on GPUs without tensor cores), memory-efficient SDPA for attention.
- K/V cache: the 3 layers with full attention own max_ctx slots (slot = position), the 12 sliding layers a ring of
  `sliding_window` slots that records each slot's position; the other 20 layers reuse those (Gemma 4 KV sharing).
  transformers' own cache is bypassed: its sliding-window layer keeps a Python-int length that a CUDA graph freezes
  and that makes torch.compile recompile every step.

Measured on a GTX 1650 Ti Max-Q, see benchmarks/runs/2026-10-23-*.
"""

import pathlib
import time
import types

import torch
import transformers
from accelerate.hooks import remove_hook_from_module
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.gemma4 import modeling_gemma4 as g4

from w4attn import decode_attention, prefill_attention
from w4gemv import w4_gemv

TEXT = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
EMB4 = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4"
DTYPE = torch.float16


def _tensor(repo, key, patterns):
    d = pathlib.Path(snapshot_download(repo, allow_patterns=patterns))
    for p in sorted(d.glob("*.safetensors")):
        with safe_open(str(p), "pt") as f:
            if key in f.keys():
                return f.get_tensor(key)
    raise KeyError(key)


class _PackedLinear(torch.nn.Module):
    def __init__(self, eng, packed, scale):
        super().__init__()
        self.eng = eng
        self.packed, self.scale = torch.nn.Buffer(packed), torch.nn.Buffer(scale)

    def forward(self, x):
        return self.eng.w4a16(x, self.packed, self.scale)


class _PackedEmbedding(torch.nn.Module):
    """Gemma4TextScaledWordEmbedding over int4 rows: gather packed rows, unpack, multiply by the embed scale."""

    def __init__(self, shifts, packed, scale, embed_scale):
        super().__init__()
        self.shifts = shifts
        self.packed, self.scale = torch.nn.Buffer(packed), torch.nn.Buffer(scale)
        self.embed_scale = torch.nn.Buffer(torch.tensor(embed_scale, dtype=DTYPE))

    def forward(self, ids):
        p, s = self.packed[ids], self.scale[ids]
        q = ((p.unsqueeze(-1) >> self.shifts) & 15).to(DTYPE) - 8
        w = q.view(*ids.shape, s.shape[-1], 32) * s.unsqueeze(-1).to(DTYPE)
        return w.view(*ids.shape, -1) * self.embed_scale


class _KV:
    """K/V for one layer that owns them. One KV head, so rows are [slots, D]."""

    def __init__(self, st, sliding, D, cap, window):
        self.st, self.sliding, self.cap, self.window = st, sliding, cap, window
        self.k = torch.zeros(cap, D, dtype=DTYPE, device=0)
        self.v = torch.zeros(cap, D, dtype=DTYPE, device=0)
        self.kpos = torch.full((cap,), -1, dtype=torch.int32, device=0) if sliding else None
        self.pk = self.pv = None

    def reset(self):
        if self.sliding:
            self.kpos.fill_(-1)

    def write(self, k, v):
        st = self.st
        if st.decode:
            slot = st.pos % self.cap if self.sliding else st.pos
            self.k.index_copy_(0, slot, k)
            self.v.index_copy_(0, slot, v)
            if self.sliding:
                self.kpos.index_copy_(0, slot, st.pos.to(torch.int32))
        elif self.sliding:
            self.pk, self.pv = k, v  # KV-shared layers later in this pass still need the ring's older positions
        else:
            self.k[st.p0 : st.p0 + k.shape[0]] = k
            self.v[st.p0 : st.p0 + v.shape[0]] = v

    def flush(self):
        if self.pk is None:
            return
        n = self.pk.shape[0]
        positions = self.st.p0 + torch.arange(n, device=0)
        slots = positions % self.cap
        self.k.index_copy_(0, slots, self.pk)
        self.v.index_copy_(0, slots, self.pv)
        self.kpos.index_copy_(0, slots, positions.to(torch.int32))
        self.pk = self.pv = None

    def attend(self, q, scale):
        st = self.st
        if st.decode:
            return decode_attention(q, self.k, self.v, self.kpos, st.pos, self.sliding, self.window, scale)
        n = q.shape[1]
        if self.sliding:
            keys, vals = torch.cat([self.k, self.pk]), torch.cat([self.v, self.pv])
            kpos = torch.cat([self.kpos, (st.p0 + torch.arange(n, device=0)).to(torch.int32)])
            return prefill_attention(q, keys, vals, kpos, st.p0, True, self.window, scale)
        L = st.p0 + n
        return prefill_attention(q, self.k[:L], self.v[:L], None, st.p0, False, None, scale)


def _attn_forward(
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


class Engine:
    def __init__(self, max_ctx=131072, chunk=256, prefill_dtype=torch.float32):
        if not torch.cuda.is_available():
            raise RuntimeError("needs a CUDA GPU")
        self.max_ctx, self.chunk, self.prefill_dtype = max_ctx, chunk, prefill_dtype
        self.shifts = torch.arange(0, 32, 4, device="cuda", dtype=torch.int32)
        torch._dynamo.config.cache_size_limit = 64
        self._dequant = torch.compile(self._dequant_eager, dynamic=False)
        self.tok = AutoTokenizer.from_pretrained(TEXT)
        self._load()
        self._capture()

    # --- int4 linears ---------------------------------------------------------------------------------------------
    def _dequant_eager(self, packed, scale):
        o, g = scale.shape
        q = ((packed.unsqueeze(-1) >> self.shifts) & 15).view(o, g, 32).to(self.prefill_dtype) - 8
        return (q * scale.unsqueeze(-1).to(self.prefill_dtype)).view(o, g * 32)

    def w4a16(self, x, packed, scale):
        if x.numel() == x.shape[-1]:
            return w4_gemv(x, packed, scale)
        return torch.nn.functional.linear(x.to(self.prefill_dtype), self._dequant(packed, scale)).to(x.dtype)

    # --- load -------------------------------------------------------------------------------------------------------
    def _load(self):
        with torch.device("meta"):
            meta = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(TEXT))
        device_map = {f"model.{n}": 0 for n, _ in meta.model.named_children()}
        device_map["model.embed_tokens_per_layer"] = "cpu"  # replaced by the int4 table below; never loaded in fp16
        device_map["lm_head"] = 0
        del meta
        model = AutoModelForCausalLM.from_pretrained(
            TEXT,
            dtype=DTYPE,
            key_mapping={r"^model\.language_model\.": "model."},
            device_map=device_map,
            quantization_config=transformers.CompressedTensorsConfig(run_compressed=True),
        ).eval()
        model.ct_decompress_hook.remove()  # it would unpack every layer to fp16 at the first forward
        tm = model.model
        pats, pre = ["model-rest-00[01].safetensors"], "model.language_model."

        def packed(name):
            return (
                _tensor(EMB4, name + ".weight_packed", pats).cuda(),
                _tensor(EMB4, name + ".weight_scale", pats).cuda(),
            )

        ple_scale, emb_scale = tm.embed_tokens_per_layer.scalar_embed_scale, tm.embed_tokens.scalar_embed_scale
        tm.embed_tokens.weight.data = torch.empty(0, dtype=DTYPE, device=0)
        tm.embed_tokens_per_layer = _PackedEmbedding(self.shifts, *packed(pre + "embed_tokens_per_layer"), ple_scale)
        # -text-emb4 stores embed_tokens and the untied lm_head as identical levels and scales; the model was trained
        # tied, so one copy serves both.
        vocab = packed("lm_head")
        tm.embed_tokens = _PackedEmbedding(self.shifts, *vocab, emb_scale)
        model.lm_head = _PackedLinear(self, *vocab)
        model.config.tie_word_embeddings = False
        remove_hook_from_module(model, recurse=True)  # accelerate's placement hooks break torch.compile graphs
        model.to(DTYPE)
        for m in model.modules():
            if hasattr(m, "weight_packed"):
                assert m.bias is None
                m.forward = (lambda mod: lambda x: self.w4a16(x, mod.weight_packed, mod.weight_scale))(m)
        self._merge_projections(model)
        torch.cuda.empty_cache()
        self.weights_mib = torch.cuda.memory_allocated() // 2**20

        cfg = model.config
        self.st = types.SimpleNamespace(decode=False, p0=0, pos=torch.zeros(1, dtype=torch.long, device=0))
        self.kvs = []
        for mod in model.modules():
            if isinstance(mod, g4.Gemma4TextAttention):
                assert cfg.per_layer_config[mod.layer_idx].num_key_value_heads == 1
                if not mod.is_kv_shared_layer:
                    cap = min(cfg.sliding_window, self.max_ctx) if mod.is_sliding else self.max_ctx
                    mod._kv = _KV(self.st, mod.is_sliding, mod.head_dim, cap, cfg.sliding_window)
                    self.kvs.append(mod._kv)
                mod.forward = types.MethodType(_attn_forward, mod)
        self.kv_mib = sum(t.numel() * t.element_size() for kv in self.kvs for t in (kv.k, kv.v)) // 2**20
        eos = model.generation_config.eos_token_id
        self.stop_ids = set(eos if isinstance(eos, list) else [eos])
        self.model = model

    def _merge_projections(self, model):
        """One int4 GEMV for gate+up in every MLP, and for q+k+v in every attention layer that computes its own K/V."""

        def merge(mods):
            pk = torch.cat([m.weight_packed for m in mods])
            sc = torch.cat([m.weight_scale for m in mods])
            sizes = [m.weight_packed.shape[0] for m in mods]
            for m in mods:
                m.weight_packed.data = torch.empty(0, dtype=torch.int32, device=0)
                m.weight_scale.data = torch.empty(0, dtype=DTYPE, device=0)
            return pk, sc, sizes

        for mod in list(model.modules()):
            if isinstance(mod, g4.Gemma4TextMLP):
                pk, sc, sizes = merge([mod.gate_proj, mod.up_proj])

                def mlp_forward(x, mod=mod, pk=pk, sc=sc, sizes=sizes):
                    g, u = self.w4a16(x, pk, sc).split(sizes, dim=-1)
                    return mod.down_proj(mod.act_fn(g) * u)

                mod.forward = mlp_forward
            elif isinstance(mod, g4.Gemma4TextAttention) and not mod.is_kv_shared_layer:
                pk, sc, sizes = merge([mod.q_proj, mod.k_proj] + ([mod.v_proj] if mod.v_proj is not None else []))
                stash = []  # q_proj runs first and leaves k (and v) for the calls that follow

                def q_forward(x, pk=pk, sc=sc, sizes=sizes, stash=stash):
                    parts = self.w4a16(x, pk, sc).split(sizes, dim=-1)
                    stash[:] = parts[1:]
                    return parts[0]

                mod.q_proj.forward = q_forward
                mod.k_proj.forward = lambda x, stash=stash: stash[0]
                if mod.v_proj is not None:
                    mod.v_proj.forward = lambda x, stash=stash: stash[1]

    # --- prompt read and decode -------------------------------------------------------------------------------------
    def read_prompt(self, ids):
        """ids [1, N] at positions 0..N-1; returns the next token's logits."""
        for kv in self.kvs:
            kv.reset()
        self.st.decode = False
        for p0 in range(0, ids.shape[1], self.chunk):
            chunk = ids[:, p0 : p0 + self.chunk]
            self.st.p0 = p0
            pos = torch.arange(p0, p0 + chunk.shape[1], device=0).unsqueeze(0)
            logits = self.model(input_ids=chunk, position_ids=pos, use_cache=False, logits_to_keep=1).logits[:, -1]
            for kv in self.kvs:
                kv.flush()
        return logits

    def _step(self):
        return (
            self.model(input_ids=self.tok_buf, position_ids=self.st.pos.view(1, 1), use_cache=False, logits_to_keep=1)
            .logits[:, -1]
            .argmax(-1)
        )

    def _capture(self):
        self.tok_buf = torch.zeros(1, 1, dtype=torch.long, device=0)
        with torch.inference_mode():
            warm = self.render([{"role": "user", "content": "Hi"}])
            first = int(self.read_prompt(warm).argmax(-1))
            self.st.decode = True
            self.tok_buf.fill_(first)
            self.st.pos.fill_(warm.shape[1])
            step = torch.compile(self._step, mode="default", dynamic=False)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    step()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.empty_cache()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.next_buf = step()

    def render(self, messages, tools=None, **template_kwargs):
        """Chat template -> input ids [1, N] on the GPU."""
        return self.tok.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, return_tensors="pt", return_dict=True, **template_kwargs
        )["input_ids"].to(0)

    @torch.inference_mode()
    def generate(self, ids, max_new=256, stop_ids=None):
        """Greedy. Returns dict(tokens, finish, prompt_s, decode_s); tokens exclude the stop token."""
        stop = self.stop_ids if stop_ids is None else set(stop_ids)
        n = ids.shape[1]
        max_new = min(max_new, self.max_ctx - n)
        if max_new < 1:
            raise ValueError(f"prompt of {n} tokens leaves no room in a {self.max_ctx}-token context")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        toks = [int(self.read_prompt(ids).argmax(-1))]
        t1 = time.perf_counter()
        self.st.decode = True
        while toks[-1] not in stop and len(toks) < max_new:
            self.tok_buf.fill_(toks[-1])
            self.st.pos.fill_(n + len(toks) - 1)
            self.graph.replay()
            toks.append(int(self.next_buf))
        torch.cuda.synchronize()
        finish = "stop" if toks[-1] in stop else "length"
        if finish == "stop":
            toks = toks[:-1]
        return {"tokens": toks, "finish": finish, "prompt_s": t1 - t0, "decode_s": time.perf_counter() - t1}
