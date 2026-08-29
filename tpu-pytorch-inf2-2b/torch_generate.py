#!/usr/bin/env python3
"""Gemma-4 E2B decode engine for AWS Inferentia2, plus a one-shot CLI driver.

    python3 torch_generate.py --prompt "What is AWS Inferentia?" --stats
    python3 torch_generate.py --device cpu --parity        # CPU vs device, token-for-token
    python3 torch_generate.py --neff-dir /workspace/neff   # reuse saved graphs

NOT the file to confuse with `torchtpu_generate.py`, which sits beside it and
targets a **Google TPU** through `torch.device("tpu")`. This one is Neuron: the
device is reached through `torch_neuronx.trace`, there is no eager device
execution, and none of the TorchTPU rules in CLAUDE.md apply to it.

This is deliberately NOT the serving path -- `torch_openai_server.py` is, and it
imports `NeuronGemmaEngine` from here rather than carrying a second copy. The
KV/mask/position arithmetic below is the part of this rig that fails *silently*
when it is wrong (see `docs/neuron-jax-quirks.md`), so it exists once.

It exists as a separate entry point because the sibling rigs' hardest bug was
only separable by driving the engine outside HTTP: 20 tokens in 0.06 s in-process
against 60-126 s through the server localised the whole cost to process
configuration, which no amount of profiling the model would have found.

WHY THE HOST DOES THE EMBEDDING LOOKUP. `embed_tokens` and `get_per_layer_inputs`
run on the CPU and their outputs are fed into the traced graph as
`inputs_embeds` / `per_layer_inputs`. That is not a convenience. The per-layer
embedding table is 4.70 GB, and a gather that large on a NeuronCore returns an
all-zero tensor rather than raising -- which decodes to token 0, which is the pad
id, which is in the EOS set, so the server answers `200 OK` with zero completion
tokens and nothing anywhere reports a fault. Quirk 1 in
`docs/neuron-jax-quirks.md` has the tensor-by-tensor evidence. Keep the gather on
the host and the failure cannot occur.

Provenance: the graph wrappers, the one-hot KV update and the per-stream position
tensors are the design proved token-exact against a CPU reference in
`quant/qat_e2b_cb_trace.py` (2026-07-27). What is new here is that it is
parameterised rather than hardcoded to `/workspace`, defaults to the **dense
reference** checkpoint rather than the QAT one, and is importable.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("torch_generate")

DEFAULT_MODEL = os.getenv("MODEL_NAME", "google/gemma-4-E2B-it")

# Static-shape budget. Every graph is traced at exactly these sizes and Neuron
# cannot run any other, so they are construction parameters, not request options:
# a caller who varies them is asking for a multi-minute recompile per request.
DEFAULT_MAX_TOTAL = int(os.getenv("KV_MAX", "128"))       # KV rows, prompt + generated
DEFAULT_PROMPT_BUCKET = int(os.getenv("KV_BUCKET", "32"))  # padded prefill length
DEFAULT_BATCH = int(os.getenv("BATCH_SLOTS", "1"))

# `--auto-cast all --auto-cast-type bf16` is what the traced graphs in quant/
# were compiled with and what the parity run validated. bf16 is the format the
# NeuronCore-v2 matmul engine has; fp32 is emulated.
COMPILER_ARGS = ("--model-type", "transformer", "--auto-cast", "all", "--auto-cast-type", "bf16")


def _torch():
    """torch is a SERVING dependency, imported here rather than at module scope.

    The MCP control plane imports this module on a laptop with no Neuron runtime
    and no multi-GB wheels, and must not pay for them to read a default.
    """
    import torch

    return torch


def make_gelu_tanh():
    """Explicit tanh-GELU, substituted for whatever `act_fn` the config selects.

    Placed on every MLP before tracing. The checkpoint asks for the tanh
    approximation and transformers may satisfy that with an activation module
    whose lowering differs from the closed form; writing it out means the traced
    graph contains the arithmetic the CPU reference ran, not a lookalike.
    """
    torch = _torch()

    class GeluTanh(torch.nn.Module):
        def forward(self, x):
            return 0.5 * x * (
                1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))
            )

    return GeluTanh()


class StaticKV:
    """Fixed [B, kv_heads, MAX, head_dim] cache, updated by one-hot blend.

    `is_compileable = False` keeps transformers from trying to torch.compile
    around it. The update is `cache * (1 - onehot) + new * onehot` rather than an
    index write because Neuron traces a static graph: a scatter at a tensor
    position would need dynamic indexing, and the blend is the same result with
    shapes fixed at trace time.

    `get_seq_length()` returns 0 on purpose. The real position rides in the
    `position_ids` / mask tensors the host computes per stream; letting the cache
    report a length would make transformers derive a *second*, conflicting one.
    """

    is_compileable = False

    def __init__(self, key_bufs, val_bufs, onehot, layer_ids):
        self.layer_ids = layer_ids
        self.key = {i: key_bufs[j] for j, i in enumerate(layer_ids)}
        self.val = {i: val_bufs[j] for j, i in enumerate(layer_ids)}
        self.oh = onehot

    def update(self, k, v, idx, *args, **kwargs):
        self.key[idx] = self.key[idx] * (1.0 - self.oh) + k * self.oh
        self.val[idx] = self.val[idx] * (1.0 - self.oh) + v * self.oh
        return self.key[idx], self.val[idx]

    def get_seq_length(self, *args, **kwargs):
        return 0

    def export(self):
        return [self.key[i] for i in self.layer_ids], [self.val[i] for i in self.layer_ids]


def _build_wrappers(lang, lm_head, nonshared, softcap):
    """The two traced graphs: padded prefill, and one decode step.

    Returned as instances rather than classes because both close over the loaded
    model. They are separate graphs because their shapes differ -- prefill runs
    [B, BUCKET] and decode runs [B, 1] -- and Neuron compiles one shape each.
    """
    torch = _torch()
    from transformers import DynamicCache

    def softcap_logits(lg):
        return softcap * torch.tanh(lg / softcap) if softcap else lg

    class PreWrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lang = lang
            self.head = lm_head

        def forward(self, ie, am, ple):
            cache = DynamicCache()
            out = self.lang(inputs_embeds=ie, per_layer_inputs=ple, attention_mask=am,
                            use_cache=True, past_key_values=cache)
            lg = softcap_logits(self.head(out.last_hidden_state))
            ks = [cache.layers[i].keys for i in nonshared]
            vs = [cache.layers[i].values for i in nonshared]
            return (lg, ks, vs)

    class DecWrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lang = lang
            self.head = lm_head

        def forward(self, ie, ple, position_ids, onehot, full_mask, slide_mask,
                    key_bufs, val_bufs):
            cache = StaticKV(key_bufs, val_bufs, onehot, nonshared)
            masks = {"full_attention": full_mask, "sliding_attention": slide_mask}
            out = self.lang(inputs_embeds=ie, per_layer_inputs=ple, position_ids=position_ids,
                            attention_mask=masks, use_cache=True, past_key_values=cache)
            lg = softcap_logits(self.head(out.last_hidden_state))
            ks, vs = cache.export()
            return (lg, ks, vs)

    return PreWrap().eval(), DecWrap().eval()


class NeuronGemmaEngine:
    """Static-shape Gemma-4 E2B decode over traced prefill/decode graphs.

    One instance owns the device. Slots are lockstep: every step runs the decode
    graph once for all `batch` slots whether or not they are occupied, so the
    weights are read once per step regardless of how many streams are live. That
    is why concurrency is close to free here and why `batch` is a serving lever
    rather than a memory cost -- 29.1 ms/step at B=8 against 21.1 at B=1,
    measured on the QAT graphs in `benchmarks/runs/2026-07-31-inf2-serving-perf/`.

    `batch` defaults to 1 anyway, because a graph traced at one B cannot run at
    another and the dense reference build has no measured B>1 result of its own.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        batch: int = DEFAULT_BATCH,
        max_total: int = DEFAULT_MAX_TOTAL,
        prompt_bucket: int = DEFAULT_PROMPT_BUCKET,
        device: str = "neuron",
        neff_dir: str | None = None,
        local_dir: str | None = None,
    ) -> None:
        if prompt_bucket >= max_total:
            raise ValueError(
                f"prompt_bucket {prompt_bucket} must leave room under max_total {max_total}"
            )
        if device not in ("neuron", "cpu"):
            raise ValueError(f"device must be 'neuron' or 'cpu', not {device!r}")
        self.model_id = model_id
        self.batch = batch
        self.max_total = max_total
        self.prompt_bucket = prompt_bucket
        self.device = device
        self.neff_dir = neff_dir
        self.local_dir = local_dir
        # Idle slots write their KV to this row and are capped below it, so a
        # parked slot can never overwrite a live stream's cache.
        self.park = max_total - 1
        self.ready = False
        self.pre = None
        self.dec = None

    # -- load -----------------------------------------------------------------

    def load(self) -> None:
        """Materialise the checkpoint on the HOST. Nothing touches the device here."""
        torch = _torch()
        from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

        source = self.local_dir or self.model_id
        t0 = time.monotonic()
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        probe = self.tokenizer("hello world").input_ids
        if len(set(probe)) < 2:
            raise RuntimeError(
                f"the tokenizer at {source} maps distinct words to one id -- a truncated "
                "or partially downloaded checkpoint. Re-fetch before tracing; a bad "
                "tokenizer produces plausible-looking garbage, not an error."
            )

        model = Gemma4ForConditionalGeneration.from_pretrained(
            source, dtype=torch.bfloat16, attn_implementation="eager"
        ).eval()
        self.model = model
        self.lang = model.model.language_model
        self.cfg = self.lang.config
        self.dtype = model.dtype
        self.sliding_window = self.cfg.sliding_window
        self.neg = torch.finfo(torch.float32).min

        for mod in self.lang.modules():
            if hasattr(mod, "act_fn"):
                mod.act_fn = make_gelu_tanh()

        # Only the layers that own KV get a buffer. Shared-KV layers read a
        # neighbour's, so allocating for them would double the cache for nothing.
        self.nonshared: list[int] = []
        self.layer_info: dict[int, tuple[int, int]] = {}
        for i, layer in enumerate(self.lang.layers[: self.cfg.num_hidden_layers]):
            attn = layer.self_attn
            if not attn.is_kv_shared_layer:
                self.nonshared.append(i)
                self.layer_info[i] = (attn.k_proj.out_features // attn.head_dim, attn.head_dim)

        softcap = getattr(model.config.text_config, "final_logit_softcapping", None)
        self.pre_module, self.dec_module = _build_wrappers(
            self.lang, model.lm_head, self.nonshared, softcap
        )

        eos = model.generation_config.eos_token_id
        self.eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
        self.eos_ids.discard(None)

        logger.info(
            "loaded %s in %.1fs -- %d KV-owning layers of %d, sliding_window=%s, dtype=%s",
            source, time.monotonic() - t0, len(self.nonshared),
            self.cfg.num_hidden_layers, self.sliding_window, self.dtype,
        )

    # -- compile --------------------------------------------------------------

    def _neff_paths(self) -> tuple[str, str]:
        stem = f"e2b_b{self.batch}_m{self.max_total}_p{self.prompt_bucket}"
        base = self.neff_dir or "."
        return os.path.join(base, f"{stem}_prefill.pt"), os.path.join(base, f"{stem}_decode.pt")

    def compile(self) -> None:
        """Trace both graphs for the device, or reload them from `neff_dir`.

        Reuse is keyed on batch/max_total/prompt_bucket because those are exactly
        the axes baked into the graph. A neff compiled at other values will load
        and then fail on shape, so it is better that the filename never matches.
        """
        torch = _torch()
        if self.device == "cpu":
            self.pre, self.dec = self.pre_module, self.dec_module
            self.ready = True
            logger.info("device=cpu: running the wrappers eagerly, nothing compiled")
            return

        pre_path, dec_path = self._neff_paths()
        if self.neff_dir and os.path.exists(pre_path) and os.path.exists(dec_path):
            t0 = time.monotonic()
            self.pre = torch.jit.load(pre_path)
            self.dec = torch.jit.load(dec_path)
            self.ready = True
            logger.info("loaded cached neffs from %s in %.1fs", self.neff_dir,
                        time.monotonic() - t0)
            return

        import torch_neuronx

        ie, ple, am = self._example_prefill_inputs()
        t0 = time.monotonic()
        self.pre = torch_neuronx.trace(self.pre_module, (ie, am, ple),
                                       compiler_args=list(COMPILER_ARGS))
        logger.info("prefill graph compiled in %.0fs", time.monotonic() - t0)

        dec_args = self._example_decode_inputs()
        t0 = time.monotonic()
        self.dec = torch_neuronx.trace(self.dec_module, dec_args,
                                       compiler_args=list(COMPILER_ARGS))
        logger.info("decode graph compiled in %.0fs", time.monotonic() - t0)

        if self.neff_dir:
            os.makedirs(self.neff_dir, exist_ok=True)
            torch.jit.save(self.pre, pre_path)
            torch.jit.save(self.dec, dec_path)
            for path in (pre_path, dec_path):
                logger.info("neff saved: %s (%.2f GB)", path, os.path.getsize(path) / 1e9)
        self.ready = True

    def _example_prefill_inputs(self):
        torch = _torch()
        pads = [[0] * self.prompt_bucket for _ in range(self.batch)]
        ie, ple = self.host_embed(pads)
        am = torch.ones(self.batch, self.prompt_bucket, dtype=torch.long)
        return ie, ple, am

    def _example_decode_inputs(self):
        ie, ple = self.host_embed([[0] for _ in range(self.batch)])
        position_ids, onehot, full_mask, slide_mask = self.host_positions(
            [self.prompt_bucket] * self.batch
        )
        key_bufs, val_bufs = self.zero_kv()
        return (ie, ple, position_ids, onehot, full_mask, slide_mask, key_bufs, val_bufs)

    # -- host-side tensor construction ---------------------------------------

    def host_embed(self, batch_ids: list[list[int]]):
        """Token ids -> (inputs_embeds, per_layer_inputs), computed on the CPU.

        This is the gather that must not run on the device. See the module
        docstring and quirk 1.
        """
        torch = _torch()
        ids = torch.tensor(batch_ids)
        with torch.no_grad():
            ie = self.lang.embed_tokens(ids)
            ple = self.lang.get_per_layer_inputs(ids, ie)
        return ie, ple

    def host_positions(self, positions: list[int]):
        """Per-stream position tensors for one decode step.

        Every stream carries its own position, so slots with different prompt
        lengths and arrival times can share a batch. A mask bug here does not
        crash -- it leaks one stream's context into another's output, which is
        why `--parity` checks duplicate prompts in different slots rather than
        just checking that decoding runs.
        """
        torch = _torch()
        pos = torch.tensor(positions)
        ar = torch.arange(self.max_total)
        onehot = (ar[None, :] == pos[:, None]).view(self.batch, 1, self.max_total, 1).to(self.dtype)
        valid = ar[None, :] <= pos[:, None]
        full = torch.where(valid, 0.0, self.neg).view(self.batch, 1, 1, self.max_total)
        slide = torch.where(
            valid & (ar[None, :] > (pos[:, None] - self.sliding_window)), 0.0, self.neg
        ).view(self.batch, 1, 1, self.max_total)
        return pos.view(self.batch, 1).long(), onehot, full, slide

    def zero_kv(self):
        torch = _torch()
        keys, vals = [], []
        for i in self.nonshared:
            kv_heads, head_dim = self.layer_info[i]
            shape = (self.batch, kv_heads, self.max_total, head_dim)
            keys.append(torch.zeros(*shape, dtype=self.dtype))
            vals.append(torch.zeros(*shape, dtype=self.dtype))
        return keys, vals

    def encode_chat(self, messages: list[dict[str, Any]]) -> list[int]:
        enc = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        return enc["input_ids"][0].tolist()

    def encode_text(self, prompt: str) -> list[int]:
        """Tokenize a raw prompt, forcing BOS.

        Gemma's tokenizer does not prepend <bos> for a plain __call__ even with
        add_special_tokens=True, and the model is trained with it -- without BOS
        greedy decoding degenerates into a repetition loop. apply_chat_template
        inserts it already, so only this raw path needs the fixup.
        """
        ids = self.tokenizer(prompt)["input_ids"]
        bos = self.tokenizer.bos_token_id
        if bos is not None and (not ids or ids[0] != bos):
            ids = [bos, *ids]
        return ids

    def pad_prompts(self, prompt_ids: list[list[int]]):
        """Pad B prompts to the bucket. Returns (padded ids, attention rows, lengths)."""
        pads, masks, lengths = [], [], []
        for ids in prompt_ids:
            if len(ids) > self.prompt_bucket:
                raise ValueError(
                    f"prompt is {len(ids)} tokens; the graph is traced at "
                    f"prompt_bucket={self.prompt_bucket} and cannot take more"
                )
            pads.append(ids + [0] * (self.prompt_bucket - len(ids)))
            # An empty slot still needs one valid column or its softmax is over
            # an all -inf row, which is NaN, which propagates into every slot.
            masks.append([1] * max(len(ids), 1) + [0] * (self.prompt_bucket - max(len(ids), 1)))
            lengths.append(len(ids))
        return pads, masks, lengths

    # -- execution ------------------------------------------------------------

    def prefill(self, pads, masks):
        torch = _torch()
        ie, ple = self.host_embed(pads)
        am = torch.tensor(masks)
        with torch.no_grad():
            return self.pre(ie, am, ple)

    def decode_step(self, last_tokens, positions, key_bufs, val_bufs):
        torch = _torch()
        ie, ple = self.host_embed([[t] for t in last_tokens])
        position_ids, onehot, full_mask, slide_mask = self.host_positions(positions)
        with torch.no_grad():
            return self.dec(ie, ple, position_ids, onehot, full_mask, slide_mask,
                            key_bufs, val_bufs)

    def seed_kv(self, keys, values):
        """Copy the prefill KV into full-length buffers at rows [0, prompt_bucket)."""
        key_bufs, val_bufs = self.zero_kv()
        for j in range(len(self.nonshared)):
            key_bufs[j][:, :, : self.prompt_bucket, :] = keys[j][:, :, : self.prompt_bucket, :]
            val_bufs[j][:, :, : self.prompt_bucket, :] = values[j][:, :, : self.prompt_bucket, :]
        return key_bufs, val_bufs

    def generate_greedy(self, prompt_ids: list[list[int]], max_new: int) -> list[list[int]]:
        """Lockstep greedy decode for `batch` prompts. Returns per-slot token ids.

        Fixed length, no early exit: every slot runs the full budget so the
        device and CPU paths execute the identical number of steps and `--parity`
        compares like with like. The serving path in `torch_openai_server.py`
        stops per stream instead.
        """
        if len(prompt_ids) != self.batch:
            raise ValueError(f"expected {self.batch} prompts, got {len(prompt_ids)}")
        pads, masks, lengths = self.pad_prompts(prompt_ids)
        logits, keys, values = self.prefill(pads, masks)
        seqs = [[int(logits[b, lengths[b] - 1].argmax())] for b in range(self.batch)]
        key_bufs, val_bufs = self.seed_kv(keys, values)

        cur = list(lengths)
        headroom = self.park - max(lengths)
        budget = min(max_new, headroom)
        if budget < 1:
            raise ValueError(
                f"prompt of {max(lengths)} tokens leaves no room to decode under "
                f"max_total={self.max_total}; retrace with a larger --max-total"
            )
        for _ in range(budget - 1):
            logits, key_bufs, val_bufs = self.decode_step(
                [s[-1] for s in seqs], cur, key_bufs, val_bufs
            )
            for b in range(self.batch):
                seqs[b].append(int(logits[b, 0].argmax()))
                cur[b] += 1
        return seqs


# -- CLI ----------------------------------------------------------------------

PARITY_PROMPTS = [
    "What is the capital of France?",
    "Name the largest planet in our solar system.",
    "What is 17 multiplied by 23? Reply with the number only.",
    "In one short sentence, what is AWS Inferentia?",
]


def _build(args, device: str) -> NeuronGemmaEngine:
    engine = NeuronGemmaEngine(
        model_id=args.model, batch=args.batch, max_total=args.max_total,
        prompt_bucket=args.prompt_bucket, device=device, neff_dir=args.neff_dir,
        local_dir=args.local_dir,
    )
    engine.load()
    engine.compile()
    return engine


def run_parity(args) -> int:
    """Decode the same prompts on the device and on the CPU and diff the ids.

    Two assertions, and the second is the one that catches the expensive bug:
    the device must match the CPU token-for-token, AND duplicate prompts placed
    in different slots must emit identical sequences. A per-stream mask error
    passes the first check whenever B=1 and fails the second.
    """
    layout = [i % len(PARITY_PROMPTS) for i in range(args.batch)]
    prompts = [PARITY_PROMPTS[i] for i in layout]

    # ONE engine, run twice. Tracing leaves the eager wrappers intact, so the CPU
    # reference is the same weights in the same process -- and a second engine
    # would mean a second copy of the checkpoint, which does not fit beside the
    # first on the 16 GiB inf2.xlarge.
    engine = _build(args, "neuron")
    prompt_ids = [engine.encode_chat([{"role": "user", "content": p}]) for p in prompts]
    dev = engine.generate_greedy(prompt_ids, args.max_new_tokens)

    traced_pre, traced_dec = engine.pre, engine.dec
    engine.pre, engine.dec = engine.pre_module, engine.dec_module
    try:
        cpu = engine.generate_greedy(prompt_ids, args.max_new_tokens)
    finally:
        engine.pre, engine.dec = traced_pre, traced_dec

    for b in range(args.batch):
        text = engine.tokenizer.decode(
            [t for t in dev[b] if t not in engine.eos_ids], skip_special_tokens=True
        )
        print(f"slot {b} (prompt {layout[b]}): {text!r}")

    dup_ok = all(dev[b] == dev[layout.index(layout[b])] for b in range(args.batch))
    seq_ok = dev == cpu
    print(f"DUP_ISOLATION: {dup_ok}")
    print(f"SEQ_MATCH: {seq_ok}")
    if not dup_ok:
        print("FAIL: duplicate prompts in different slots diverged -- stream isolation broken")
    if not seq_ok:
        print("FAIL: device output diverged from the CPU reference")
    return 0 if (dup_ok and seq_ok) else 1


def run_once(args) -> int:
    engine = _build(args, args.device)
    prompt_ids = engine.encode_chat([{"role": "user", "content": args.prompt}])
    padded = [prompt_ids] + [[engine.tokenizer.bos_token_id or 0]] * (args.batch - 1)

    # Warm at the shape you measure. The first call pays graph load and allocator
    # growth; on the JAX sibling that was 18.77 s cold against 4.35 s warm for the
    # identical request.
    engine.generate_greedy(padded, 2)

    t0 = time.perf_counter()
    seqs = engine.generate_greedy(padded, args.max_new_tokens)
    elapsed = time.perf_counter() - t0

    ids = [t for t in seqs[0] if t not in engine.eos_ids]
    text = engine.tokenizer.decode(ids, skip_special_tokens=True)
    print("\n--- output " + "-" * 49)
    print(text)
    if not text.strip():
        print(
            "\nEMPTY OUTPUT. On this rig that is the signature failure, not an edge "
            "case: a device gather that returns zeros decodes to the pad id, which "
            "is an EOS. Read docs/neuron-jax-quirks.md quirk 1 before tuning anything."
        )
    if args.stats:
        n = len(seqs[0])
        print("-" * 60)
        print(f"{n} tokens in {elapsed:.2f} s  ->  {n / elapsed:.2f} tok/s (warm, batch={args.batch})")
        print(f"per-step: {elapsed / max(n, 1) * 1000:.1f} ms")
    # Token count is not correctness. An isolation run once recorded 20 tokens as
    # a 2700x win; the same configuration returned an empty string end-to-end.
    return 0 if text.strip() else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--local-dir", default=None, help="load from this path instead of the Hub")
    p.add_argument("--prompt", default="In one short sentence, what is AWS Inferentia?")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--max-total", type=int, default=DEFAULT_MAX_TOTAL,
                   help="KV rows the graph is traced for (prompt + generated)")
    p.add_argument("--prompt-bucket", type=int, default=DEFAULT_PROMPT_BUCKET,
                   help="padded prefill length the graph is traced for")
    p.add_argument("--neff-dir", default=os.getenv("NEFF_DIR"),
                   help="reuse/save traced graphs here; tracing takes minutes")
    p.add_argument("--device", default="neuron", choices=("neuron", "cpu"))
    p.add_argument("--parity", action="store_true",
                   help="decode on device AND cpu, then diff ids and check slot isolation")
    p.add_argument("--stats", action="store_true", help="print decode tok/s")
    args = p.parse_args()

    if args.parity:
        if args.batch < 2:
            print("note: --parity checks slot isolation, which needs --batch 2 or more")
        return run_parity(args)
    return run_once(args)


if __name__ == "__main__":
    raise SystemExit(main())
